import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional


class VariableLSTMModel(nn.Module):
    lstm_hidden_size: int
    num_lstm_layers: int
    linear_layer_sizes: Sequence[int]
    increased_size: int
    mean_aggregation: bool
    final_output_size: int
    dropout_rate: float

    def setup(self):
        # LSTM layers
        self.lstm_layers = [
            nn.RNN(
                nn.LSTMCell(features=self.lstm_hidden_size),
                name=f'lstm_{i}'
            )
            for i in range(self.num_lstm_layers)
        ]

        # Theta projector
        self.base_theta_projector = nn.Dense(
            features=self.increased_size,
            name="base_theta_projector"
        )

        # Theta projector
        self.base_mu_stddev_projector = nn.Dense(
            features=self.increased_size,
            name="base_mu_stddev_projector"
        )

        # Linear layers
        self.linear_layers = [
            nn.Dense(features=size) for size in self.linear_layer_sizes
        ]

        # Dropout layer
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Final output layer
        self.output_layer = nn.Dense(features=self.final_output_size)

    def __call__(self, x, theta, x_cache=None, train: bool = False):
        # Expand 2D input to 3D if necessary
        if x_cache is None:
            if x.ndim == 2:
                x = jnp.expand_dims(x, axis=-1)

            x_mu = jnp.mean(x, axis=1, keepdims=True)
            x_std = jnp.std(x, axis=1, keepdims=True)

            x = (x-x_mu) / x_std

            mu_stddev = jnp.concatenate(
                [jnp.squeeze(x_mu, -1), jnp.squeeze(x_std, -1)], axis=1)

            # Process through LSTM layers
            for lstm in self.lstm_layers:
                x = lstm(x)

            # Aggregate outputs
            if self.mean_aggregation:
                x = jnp.mean(x, axis=1)
            else:
                x = x[:, -1, :]

            mu_stddev_projected = self.base_mu_stddev_projector(mu_stddev)
            x_cache = jnp.concatenate([x, mu_stddev_projected], axis=-1)

        # Handle theta if provided
        theta_projected = self.base_theta_projector(theta)
        x = jnp.concatenate([x_cache, theta_projected], axis=-1)

        # Pass through linear layers with ELU activation
        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = nn.elu(x)
            x = self.dropout(x, deterministic=not train)

        # Final linear layer without dropout
        x = self.linear_layers[-1](x)
        x = nn.elu(x)

        return self.output_layer(x), x_cache


# Example usage
if __name__ == "__main__":
    # Model hyperparameters
    lstm_hidden_size = 64
    num_lstm_layers = 2
    linear_layer_sizes = (32, 16, 8)
    mean_aggregation = False
    final_output_size = 5
    dropout_rate = 0.1
    increased_size = 32

    model = VariableLSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate,
        increased_size=increased_size
    )

    # Generate dummy data
    key = jax.random.PRNGKey(0)
    # batch_size, seq_len, fake dimension
    dummy_input = jax.random.normal(key, (32, 10, 1))
    dummy_theta = jax.random.normal(key, (32, 5))  # batch_size, theta_size

    # Initialize parameters
    params = model.init(key, dummy_input, dummy_theta)

    output, stored_x = model.apply(params, dummy_input, dummy_theta)

    # Define JIT-ed apply functions
    @jax.jit
    def apply_model(params, theta, stored_x):
        return model.apply(params, dummy_input, theta, stored_x)

    outputs_with_theta = apply_model(
        params, dummy_theta, stored_x)
