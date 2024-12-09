import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class LSTMModel(nn.Module):
    lstm_hidden_size: int
    num_lstm_layers: int
    linear_layer_sizes: Sequence[int]
    mean_aggregation: bool

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
                unroll=1,  # might want to increase this, tradeoff between efficiency and space; not entirely clear to me
            )(features=self.lstm_hidden_size, name=f'lstm_cell_{i}')
            # enumerate([64, 128, 256]) to change the hidden sizes
            for i in range(self.num_lstm_layers)
        ]
        # Define linear layers
        self.linear_layers = [
            nn.Dense(features=size) for size in self.linear_layer_sizes
        ]
        # Final output layer
        self.output_layer = nn.Dense(features=1)

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

        # Pass through linear layers
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
    linear_layer_sizes = [32, 8, 4]
    mean_aggregation = False

    # Initialize model
    model = LSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation
    )

    # Dummy input: batch of sequences with 10 timesteps
    key = jax.random.PRNGKey(0)
    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(key, (32, 10))

    # Initialize carry states
    carry = model.initialize_carry(batch_size=dummy_input.shape[0])

    # Initialize parameters
    params = model.init(key, dummy_input, carry)

    # Apply the model without jit
    # outputs = model.apply(params, dummy_input, carry)

    # Define a callable function for applying the model
    def apply_model(params, x, carry):
        return model.apply(params, x, carry)

    # JIT the apply_model function
    jit_apply_model = jax.jit(apply_model)

    # Apply the JIT-ed model
    outputs = jit_apply_model(params, dummy_input, carry)

    print("Model output shape:", outputs.shape)  # Should print: batch_size
