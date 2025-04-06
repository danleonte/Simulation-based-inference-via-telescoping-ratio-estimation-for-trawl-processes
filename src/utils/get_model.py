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
import numpy as np

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.model.LSTM_based_nn import LSTMModel
from src.model.VariableLSTM_based_nn import VariableLSTMModel
from src.model.Dense_model import DenseModel
from src.model.Conv_based_nn import CNN
from src.model.Extended_model_nn import ExtendedModel
from src.model.Transformer_based_nn import TimeSeriesTransformerBase
from statsmodels.tsa.stattools import acf as compute_empirical_acf


def get_model(config_file, initialize=True):
    model_name = config_file['model_config']['model_name']

    if model_name == 'VariableLSTMModel':
        return get_model_VariableLSTM(config_file, initialize)
    elif model_name == 'LSTMModel':
        return get_model_LSTM(config_file, initialize)
    elif model_name == 'CNN':
        return get_model_CNN(config_file, initialize)
    elif model_name == 'TimeSeriesTransformerBase':
        return get_model_transformer(config_file, initialize)
    elif model_name == 'DenseModel':
        return get_model_Dense(config_file, initialize)
    else:
        raise ValueError('model_name not recognized, please check config file')


def get_model_LSTM(config_file, initialize=True):

    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'LSTMModel'
    assert model_config['with_theta'] in [True, False]
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    seq_len = trawl_config['seq_len']
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    lstm_hidden_size = model_config['lstm_hidden_size']
    num_lstm_layers = model_config['num_lstm_layers']
    linear_layer_sizes = model_config['linear_layer_sizes']
    mean_aggregation = model_config['mean_aggregation']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']

    # Create model
    model = LSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

    # Low-dimensional parameter (can be of any size)
    if model_config['with_theta']:

        dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
        params = model.init(subkey, dummy_input, dummy_theta)

    else:

        params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_VariableLSTM(config_file, initialize=True):

    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'VariableLSTMModel'
    assert 'with_theta' not in model_config.keys()
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    # seq_len = trawl_config['seq_len']
    seq_len = 1500  # used to initialize, but shouldn t influence anything
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    lstm_hidden_size = model_config['lstm_hidden_size']
    num_lstm_layers = model_config['num_lstm_layers']
    linear_layer_sizes = model_config['linear_layer_sizes']
    mean_aggregation = model_config['mean_aggregation']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']
    increased_size = model_config['increased_size']

    # Create model
    model = VariableLSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        increased_size=increased_size,
        dropout_rate=dropout_rate
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

    # Low-dimensional parameter (can be of any size)
    # if model_config['with_theta']:

    dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
    params = model.init(subkey, dummy_input, dummy_theta)

    # else:
    #
    #    params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_CNN(config_file, initialize=True):

    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'CNN'
    assert not model_config['with_theta']
    ###########################################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    seq_len = trawl_config['seq_len']
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    max_lag = model_config['max_lag']
    conv_channels = model_config['conv_channels']
    fc_sizes = model_config['fc_sizes']
    conv_kernels = model_config['conv_kernels']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']

    model = CNN(
        max_lag=max_lag,
        conv_channels=conv_channels,
        fc_sizes=fc_sizes,
        conv_kernels=conv_kernels,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

    # Low-dimensional parameter (can be of any size)
    if model_config['with_theta']:

        dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
        params = model.init(subkey, dummy_input, dummy_theta)

    else:

        params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_Dense(config_file, initialize=True):
    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'DenseModel'
    assert model_config['with_theta'] in [True, False]
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    seq_len = trawl_config['input_size']  # 3trawl_config['seq_len']
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    # adjust dimensionality of the input when using summary statistics
    # if 'tre_config' in config_file.keys():
    #    tre_config = config_file['tre_config']
    #    if tre_config['use_summary_statistics']:
    #        seq_len = tre_config['summary_statistics_input_size']

    linear_layer_sizes = model_config['linear_layer_sizes']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']

    # Create model
    model = DenseModel(
        linear_layer_sizes=linear_layer_sizes,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len))

    # Low-dimensional parameter (can be of any size)
    if model_config['with_theta']:
        dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
        params = model.init(subkey, dummy_input, dummy_theta)
    else:
        params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_transformer(config_file, initialize=True):
    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'TimeSeriesTransformerBase'
    assert model_config['with_theta'] in [True, False]
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    seq_len = trawl_config['seq_len']
    batch_size = trawl_config['batch_size']

    # Get transformer-specific parameters from config
    hidden_size = model_config['hidden_size']
    num_heads = model_config['num_heads']
    num_layers = model_config['num_layers']
    mlp_dim = model_config['mlp_dim']
    linear_layer_sizes = model_config['linear_layer_sizes']
    dropout_rate = model_config['dropout_rate']
    final_output_size = model_config['final_output_size']
    # Default to False if not specified
    freq_attention = model_config.get('freq_attention', False)

    # Create model
    model = TimeSeriesTransformerBase(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        linear_layer_sizes=linear_layer_sizes,
        dropout_rate=dropout_rate,
        final_output_size=final_output_size,
        freq_attention=freq_attention
    )

    if not initialize:
        return model

    # Initialize model
    # Dummy input [batch_size, sequence_length]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len))

    # Initialize with deterministic=True for consistency
    params = model.init(subkey, dummy_input)

    return model, params, key
