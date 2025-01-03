# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:37:24 2024

@author: dleon
"""
import os
import yaml
import pickle
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.model.LSTM_based_nn import LSTMModel


def get_model(config_file):

    model_name = config_file['model_config']['model_name']

    if model_name == 'LSTMModel':

        return get_model_LSTM(config_file)

    elif model_name == 'MLP':

        raise ValueError('not yet implemented')
        return get_model_MLP(config_file)

    else:
        raise ValueError('model_name not recognized, please check config file')


def get_model_LSTM(config_file, initialize=True):

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

        # Create model
        model = LSTMModel(
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            linear_layer_sizes=linear_layer_sizes,
            mean_aggregation=mean_aggregation,
            final_output_size=final_output_size
        )

        if not initialize:
            return model

        # Initialize model

        # Dummy input
        # [batch_size, sequence_length, feature_size]
        dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

        # Low-dimensional parameter (can be of any size)
        if model_config['with_theta']:

            dummy_theta = jnp.random.normal(subkey, (batch_size, theta_size))
            params = model.init(subkey, dummy_input, dummy_theta)

        else:

            params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_MLP(config_file):
    pass


###############################################################################
def get_projection_function():

    summary_path = os.path.join("models", "summary_statistics")
    acf_path = os.path.join(summary_path, "learn_acf", "best_model")
    marginal_path = os.path.join(summary_path, "learn_marginal", "best_model")

    # Load configs
    with open(os.path.join(acf_path, "config.yaml"), 'r') as f:
        acf_config = yaml.safe_load(f)

    with open(os.path.join(marginal_path, "config.yaml"), 'r') as f:
        marginal_config = yaml.safe_load(f)

    # Load models
    acf_model, _, __ = get_model(acf_config)
    marginal_model, _, __ = get_model(marginal_config)

    # Load params
    with open(os.path.join(acf_path, "params.pkl"), 'rb') as file:
        acf_params = {'params': pickle.load(file)}

    with open(os.path.join(marginal_path, "params.pkl"), 'rb') as file:
        marginal_params = {'params': pickle.load(file)}

    @jax.jit
    def project(trawl):

        acf_model.apply(acf_params, trawl)
        marginal_model.apply(marginal_params, trawl)
        return jnp.concatenate([acf_model, marginal_model], axis=1)

    return project
