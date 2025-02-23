# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:32:19 2025

@author: dleon
"""
from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


class CNN(nn.Module):
    max_lag: int
    conv_channels: Sequence[int]
    fc_sizes: Sequence[int]
    final_output_size: int
    conv_kernels: Sequence[int] = None
    dropout_rate: float = 0.1
    deterministic: bool = False

    @nn.compact
    def __call__(self, x, train: bool = None):
        """
        Args:
            x: Input of shape (batch_size, sequence_length, 1)
            train: If provided, overrides the default deterministic setting
        """
        is_training = not self.deterministic if train is None else train

        # Expand 2D input to 3D if necessary
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=-1)

        # Switch input from (batch, seq_len, 1) to (batch, 1, seq_len)
        x = jnp.transpose(x, (0, 2, 1))

        # First conv layer with max_lag kernel size
        x = nn.Conv(
            features=self.conv_channels[0],
            kernel_size=(self.max_lag,),
            padding='SAME',
            name='conv_0'
        )(x)
        x = nn.elu(x)

        # Remaining conv layers
        kernels = self.conv_kernels or [5, 3] * (len(self.conv_channels) - 1)
        for i, (channels, kernel_size) in enumerate(zip(
            self.conv_channels[1:],
            kernels
        ), 1):
            x = nn.Conv(
                features=channels,
                kernel_size=(kernel_size,),
                padding='SAME',
                name=f'conv_{i}'
            )(x)
            x = nn.elu(x)

        # Global pooling (both average and max)
        # avg_pool = jnp.mean(x, axis=2)
        # max_pool = jnp.max(x, axis=2)

        # Concatenate pooled features
        # x = jnp.concatenate([avg_pool, max_pool], axis=1)

        # Fully connected layers
        for i, size in enumerate(self.fc_sizes):
            x = nn.Dense(features=size, name=f'fc_{i}')(x)
            x = nn.elu(x)
            x = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not is_training
            )(x)

        # Final output layer (no activation)
        x = nn.Dense(features=self.final_output_size,
                     name='fc_output')(x)[:, 0, :]

        return x


# Example usage:
if __name__ == "__main__":
    import optax  # Add this import for the training example

    # Initialize model
    model = CNN(
        max_lag=30,
        conv_channels=[32, 64, 32],
        fc_sizes=[64, 32],
        final_output_size=3,
        dropout_rate=0.2
    )

    # Initialize parameters with random key
    key = jax.random.PRNGKey(0)
    # x = jax.random.normal(key, (1, 1000, 1))  # Example input shape
    x = jax.random.normal(key, (32, 1000))  # Example input shape
    params = model.init(key, x)

    # For training (is_training=True) and make sure to pass a new prngkey each time
    output_train = model.apply(
        params,
        x,
        train=True,
        rngs={'dropout': jax.random.PRNGKey(0)}
    )

    # For inference (is_training=False)
    output_inference = model.apply(params, x, train=False)

    # Training setup example:
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    # Create a new PRNG key for training
    rng = jax.random.PRNGKey(0)

    @jax.jit
    def train_step(params, batch, rng):
        """Training step with proper RNG handling"""
        dropout_key = jax.random.fold_in(
            rng, jax.process_index())  # Unique key per process
        dropout_key = jax.random.fold_in(
            dropout_key, jnp.array(0))  # Unique key per step

        def loss_fn(params):
            predictions = model.apply(
                params,
                batch['x'],
                train=True,
                rngs={'dropout': dropout_key}
            )
            loss = jnp.mean((predictions - batch['y']) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
        new_params = optax.apply_updates(params, updates)

        # Generate new RNG key for next step
        new_rng = jax.random.fold_in(rng, jnp.array(1))

        return new_params, new_optimizer_state, loss, new_rng

    @jax.jit
    def eval_step(params, batch):
        predictions = model.apply(
            params,
            batch['x'],
            train=False
        )
        loss = jnp.mean((predictions - batch['y']) ** 2)
        return loss

    # Example training loop
    # for step in range(num_steps):
    #    # Get your batch data here
    #    batch = get_batch()
    #
    #    # Perform training step
    #    params, optimizer_state, loss, rng = train_step(params, batch, rng)
