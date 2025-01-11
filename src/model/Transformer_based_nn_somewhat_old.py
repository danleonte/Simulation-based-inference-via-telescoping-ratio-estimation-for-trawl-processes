import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional, Union, Tuple
import numpy as np


class TimeSeriesTransformerBase(nn.Module):
    """Base class for time series transformer models."""
    hidden_size: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    linear_layer_sizes: Sequence[int]
    dropout_rate: float
    final_output_size: int
    freq_attention: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Input shape: (batch_size, seq_len)
        # Add channel dimension if needed
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=-1)

        # Project input to hidden dimension and add positional embeddings
        x = nn.Dense(self.hidden_size)(x)
        x = PositionalEmbedding(self.hidden_size)(x)

        # Apply transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                freq_attention=self.freq_attention,
                name=f'transformer_block_{i}'
            )(x, train=train)

        # Global average pooling over sequence dimension
        x = jnp.mean(x, axis=1)

        # Final MLP layers
        x = nn.LayerNorm()(x)

        # Pass through linear layers with ELU activation
        for i, size in enumerate(self.linear_layer_sizes):
            x = nn.Dense(features=size, name=f'linear_{i}')(x)
            x = nn.elu(x)
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not train)

        return nn.Dense(features=self.final_output_size)(x)


class TransformerBlock(nn.Module):
    """Transformer block with optional frequency attention."""
    hidden_size: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float
    freq_attention: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # First attention block (time or frequency domain)
        if self.freq_attention:
            attn_output = self.frequency_attention(x, train)
        else:
            attn_output = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                dropout_rate=self.dropout_rate,
                deterministic=not train
            )(x, x, x)

        x = x + nn.Dropout(rate=self.dropout_rate)(attn_output,
                                                   deterministic=not train)
        x = nn.LayerNorm()(x)

        # MLP block
        mlp_output = self.mlp_block(x, train)
        x = x + nn.Dropout(rate=self.dropout_rate)(mlp_output,
                                                   deterministic=not train)
        x = nn.LayerNorm()(x)

        return x

    @nn.compact
    def frequency_attention(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Attention in frequency domain."""
        batch_size, seq_len, channels = x.shape

        # Convert to frequency domain
        x_freq = jnp.fft.rfft(x, axis=1)

        # Reshape to handle real and imaginary parts
        x_freq = jnp.stack([jnp.real(x_freq), jnp.imag(x_freq)], axis=-1)

        # Project to query, key, value
        query = nn.Dense(self.hidden_size)(x_freq)
        key = nn.Dense(self.hidden_size)(x_freq)
        value = nn.Dense(self.hidden_size)(x_freq)

        # Apply attention in frequency domain
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key)
        attn_weights = attn_weights / jnp.sqrt(self.hidden_size)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply attention and combine real/imaginary parts
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)
        attn_output = attn_output[..., 0] + 1j * attn_output[..., 1]

        # Convert back to time domain
        output = jnp.fft.irfft(attn_output, n=seq_len, axis=1)

        # Match the input shape exactly
        if output.ndim < 3:
            output = jnp.expand_dims(output, axis=-1)

        # Ensure output shape matches input shape
        return jnp.broadcast_to(output, (batch_size, seq_len, channels))

    @nn.compact
    def mlp_block(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """MLP block with GELU activation."""
        x = nn.Dense(features=self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(features=self.hidden_size)(x)
        return x


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""
    dim: int
    max_len: int = 10000

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)[None, :, None]
        div_term = jnp.exp(jnp.arange(0, self.dim, 2) *
                           (-jnp.log(10000.0) / self.dim))

        pe = jnp.zeros((1, seq_len, self.dim))
        pe = pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(position * div_term))

        return x + pe
