# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 01:14:55 2024

@author: dleon
"""
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.model.LSTM_based_nn import LSTMModel_with_theta, LSTMModel_without_theta


def get_model(config_file):

    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    if config_file['model_config']['model_name'] == 'LSTMModel':

        trawl_config = config_file['trawl_config']

        seq_len = trawl_config['seq_len']
        batch_size = trawl_config['batch_size']
        theta_size = trawl_config['theta_size']

        model_config = config_file['model_config']
        assert model_config['with_theta'] in [True, False]

        lstm_hidden_size = model_config['lstm_hidden_size']
        num_lstm_layers = model_config['num_lstm_layers']
        linear_layer_sizes = model_config['linear_layer_sizes']
        mean_aggregation = model_config['mean_aggregation']
        final_output_size = model_config['final_output_size']

        # Initialize model
        if model_config['with_theta']:
            LSTMModel = LSTMModel_with_theta
        else:
            LSTMModel = LSTMModel_without_theta

        model = LSTMModel(
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            linear_layer_sizes=linear_layer_sizes,
            mean_aggregation=mean_aggregation,
            final_output_size=final_output_size
        )

        # Dummy input
        # [batch_size, sequence_length, feature_size]
        dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

        # Initialize carry states
        carry = model.initialize_carry(batch_size=dummy_input.shape[0])

        # Low-dimensional parameter (can be of any size)
        if model_config['with_theta']:

            dummy_theta = jnp.random.normal(subkey, (batch_size, theta_size))
            params = model.init(subkey, dummy_input, dummy_theta, carry)

        else:

            params = model.init(subkey, dummy_input, carry)

    return model, params, carry, key
