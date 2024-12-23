import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class LSTMModel_with_theta(nn.Module):
    lstm_hidden_size: int
    num_lstm_layers: int
    linear_layer_sizes: Sequence[int]
    mean_aggregation: bool
    final_output_size: int = 1  # Allow customizing final output size

    def setup(self):
        # Define LSTM layers using nn.scan for proper variable management
        self.lstm_layers = [
            nn.scan(
                nn.LSTMCell,
                variable_broadcast='params',
                split_rngs={'params': False},
                in_axes=1,
                out_axes=1,
                reverse=False,
                unroll=1,
            )(features=self.lstm_hidden_size, name=f'lstm_cell_{i}')
            for i in range(self.num_lstm_layers)
        ]

        # Placeholder for theta projector, will be created dynamically
        self.base_theta_projector = nn.Dense(
            features=self.lstm_hidden_size, name="base_theta_projector")

        # Define linear layers
        self.linear_layers = [
            nn.Dense(features=size) for size in self.linear_layer_sizes
        ]

        # Final output layer
        self.output_layer = nn.Dense(features=self.final_output_size)

    def initialize_carry(self, batch_size):
        """Initialize carry states with zeros for all LSTM layers."""
        return [
            (
                # Hidden state (h)
                jnp.zeros((batch_size, self.lstm_hidden_size)),
                # Cell state (c)
                jnp.zeros((batch_size, self.lstm_hidden_size)),
            )
            for _ in range(self.num_lstm_layers)
        ]

    def __call__(self, x, theta, carry=None):
        if carry is None:
            raise ValueError(
                "Carry must be initialized explicitly using `initialize_carry`."
            )

        # Expand 2D input to 3D if necessary
        if x.ndim == 2:
            # [batch_size, sequence_length] -> [batch_size, sequence_length, 1]
            x = jnp.expand_dims(x, axis=-1)

        # Process through LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            carry[i], x = lstm_layer(carry[i], x)

        # Aggregate outputs
        if self.mean_aggregation:
            x = jnp.mean(x, axis=1)  # Average over the sequence
        else:
            x = x[:, -1, :]  # Use the last output

        # Project theta to compatible dimension
        theta_projected = self.base_theta_projector(theta)

        # Concatenate LSTM output with projected theta
        x = jnp.concatenate([x, theta_projected], axis=-1)

        # Pass through linear layers with ELU activation
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
            x = nn.elu(x)

        # Final output layer
        x = self.output_layer(x)

        return x


class LSTMModel_without_theta(nn.Module):
    lstm_hidden_size: int
    num_lstm_layers: int
    linear_layer_sizes: Sequence[int]
    mean_aggregation: bool
    final_output_size: int = 1  # Allow customizing final output size

    def setup(self):
        # Define LSTM layers using nn.scan for proper variable management
        self.lstm_layers = [
            nn.scan(
                nn.LSTMCell,
                variable_broadcast='params',
                split_rngs={'params': False},
                in_axes=1,
                out_axes=1,
                reverse=False,
                unroll=1,
            )(features=self.lstm_hidden_size, name=f'lstm_cell_{i}')
            for i in range(self.num_lstm_layers)
        ]

        # Define linear layers
        self.linear_layers = [
            nn.Dense(features=size) for size in self.linear_layer_sizes
        ]

        # Final output layer
        self.output_layer = nn.Dense(features=self.final_output_size)

    def initialize_carry(self, batch_size):
        """Initialize carry states with zeros for all LSTM layers."""
        return [
            (
                # Hidden state (h)
                jnp.zeros((batch_size, self.lstm_hidden_size)),
                # Cell state (c)
                jnp.zeros((batch_size, self.lstm_hidden_size)),
            )
            for _ in range(self.num_lstm_layers)
        ]

    def __call__(self, x, carry=None):
        if carry is None:
            raise ValueError(
                "Carry must be initialized explicitly using `initialize_carry`."
            )

        # Expand 2D input to 3D if necessary
        if x.ndim == 2:
            # [batch_size, sequence_length] -> [batch_size, sequence_length, 1]
            x = jnp.expand_dims(x, axis=-1)

        # Process through LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            carry[i], x = lstm_layer(carry[i], x)

        # Aggregate outputs
        if self.mean_aggregation:
            x = jnp.mean(x, axis=1)  # Average over the sequence
        else:
            x = x[:, -1, :]  # Use the last output

        # Pass through linear layers with ELU activation
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
            x = nn.elu(x)

        # Final output layer
        x = self.output_layer(x)

        return x


# Example usage
if __name__ == "__main__":
    # Model hyperparameters
    lstm_hidden_size = 64
    num_lstm_layers = 2
    linear_layer_sizes = (32, 16, 8)
    mean_aggregation = False
    final_output_size = 5  # Example of custom output size

    # Initialize model without theta
    model_without_theta = LSTMModel_without_theta(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size
    )

    # Initialize model with theta
    model_with_theta = LSTMModel_with_theta(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size
    )

    # Dummy input: batch of sequences with 10 timesteps
    key = jax.random.PRNGKey(0)

    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(key, (32, 10, 1))

    # Low-dimensional parameter (can be of any size)
    # No need to specify size during initialization
    dummy_theta = jax.random.normal(key, (32, 5))

    # Initialize carry states
    carry_without_theta = model_without_theta.initialize_carry(
        batch_size=dummy_input.shape[0])
    carry_with_theta = model_with_theta.initialize_carry(
        batch_size=dummy_input.shape[0])

    # Initialize parameters
    params_without_theta = model_without_theta.init(
        key, dummy_input, carry_without_theta
    )

    # Explicitly pass carry during initialization
    params_with_theta = model_with_theta.init(
        key, dummy_input, dummy_theta, carry_with_theta  # Pass carry explicitly

    )

    # Define a callable function for applying the model

    @jax.jit
    def apply_model_without_theta(params, x, carry):
        return model_without_theta.apply(params, x, carry)

    @jax.jit
    def apply_model_with_theta(params, x, carry, theta):
        return model_with_theta.apply(params, x, carry, theta)

    # Apply the JIT-ed model
    outputs_without_theta = apply_model_without_theta(
        params_without_theta, dummy_input, carry_without_theta)
    outputs_with_theta = apply_model_with_theta(
        params_with_theta, dummy_input, dummy_theta, carry_with_theta)
    # Should print: (32, final_output_size)
    # print("Model output shape:", outputs.shape)
