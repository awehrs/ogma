from src.propagate import propagate
from src.utils import compressed_to_full, stride_inputs

from typing import Any, Optional, Tuple

from einops import rearrange
from jax import nn, vmap
import jax.numpy as jnp


class Decoder:
    """Sparse decoder."""

    def __init__(self, parameters: jnp.array, learning_rate: int):
        self.parameters = parameters
        self.lr = learning_rate
        self.optimizer = None

    def activate(self, h: jnp.array) -> jnp.array:
        """One hot activation."""
        return jnp.argmax(h, axis=1)

    def forward(self, input: jnp.array, parameters: jnp.array) -> jnp.array:
        h = propagate(input, parameters, input_is_one_hot=True, output_is_one_hot=False)
        return nn.softmax(h)

    def loss(self, prediction: jnp.array, target: jnp.array):
        return target - prediction

    def update(
        self,
        input_losses: jnp.array,
        previous_context: jnp.array,
    ) -> jnp.array:
        """
        Args:
            input_losses: array of shape (receptive_area, context_dim)
            previous_context: array of shape (prediction_dim,)

        Returns:
            array of shape (receptive_area, prediction_dim, context_dim)
        """
        return jnp.kron(jnp.transpose(input_losses), previous_context)

    def learn(
        self,
        target: jnp.array,
        prediction: jnp.array,
        prev_context: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        learning_rate: float,
    ) -> jnp.array:
        loss = self.loss(prediction, target)
        loss = stride_inputs(loss, downward_mapping)

        delta = vmap(self.update)(loss, prev_context)

        parameters += learning_rate * delta

        return parameters

    def step(
        self,
        ctx_encoder: jnp.array,
        ctx_decoder: Optional[jnp.array],
        prev_prediction: jnp.array,
        curr_target: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        learning_rate: float,
        learn: bool = True,
    ) -> Optional[Tuple[jnp.array]]:
        if learn:
            assert (prev_prediction is not None) and (curr_target is not None)
            ctx_encoder = compressed_to_full(ctx_encoder, dim=prev_prediction.shape[-1])
            if ctx_decoder is not None:  # E.g., this isn't the top layer.
                # Shape = [num_columns, 2 * col_dimension]:
                prev_context = jnp.concatenate([ctx_decoder, ctx_encoder], axis=1)
                # Shape = [num_columns, receptive_area, 2 * column_dimension]:
                prev_context = stride_inputs(prev_context, downward_mapping)
                # Shape = [num_columns, recepetive_area * 2, column_dimension]:
                prev_context = rearrange(
                    prev_context, "n r (x d) -> n (r x) d", d=prev_prediction.shape[-1]
                )
            self.parameters = self.learn(
                target=compressed_to_full(curr_target, dim=prev_prediction.shape[-1]),
                prediction=prev_prediction,
                prev_context=prev_context,
                parameters=parameters,
                downward_mapping=downward_mapping,
                learning_rate=learning_rate,
            )
        else:
            if ctx_decoder is not None:  # E.g, this isn't the top layer.
                ctx_decoder = self.activate(ctx_decoder)
                # Shape = [num_columns, 2]:
                context = jnp.stack([ctx_decoder, ctx_encoder], axis=1)
                # Shape = [num_columns, recepetive_area, 2]:
                context = stride_inputs(context, downward_mapping)
                # Shape = [num_columns, receptive_area * 2]:
                context = rearrange(context, "n r d -> n (r d)")
            else:
                # Shape = [num_columns, receptive_area]
                context = stride_inputs(ctx_encoder, downward_mapping)

            return self.forward(context, parameters)

    def __call__(
        self,
        *,
        ctx_encoder,
        ctx_decoder,
        downward_mapping: jnp.array,
        learn: bool,
        prev_prediction: Optional[jnp.array] = None,
        curr_target: Optional[jnp.array] = None,
    ) -> Any:
        return self.step(
            ctx_encoder=ctx_encoder,
            ctx_decoder=ctx_decoder,
            prev_prediction=prev_prediction,
            curr_target=curr_target,
            parameters=self.parameters,
            downward_mapping=downward_mapping,
            learning_rate=self.lr,
            learn=learn,
        )
