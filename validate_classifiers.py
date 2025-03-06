from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel
import numpy as np
import datetime
import pickle
import optax
import wandb
import yaml
import os


import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import matplotlib
import tensorflow_probability.substrates.jax as tfp
import corner

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size as ess
import arviz as az


def validate_classifiers(folder_path):

    if 'TRE' in folder_path:
        use_tre = True
    elif 'NRE' in folder_path:
        use_tre = False
    else:
        raise ValueError

    if 'full_trawl' in folder_path:
        use_summary_statistics = False

    elif 'summary_statistics' in folder_path:
        use_summary_statistics = True

    else:
        raise ValueError

    if use_tre:
        classifier_config_file_path = os.path.join(
            folder_path, 'acf', 'config.yaml')
    else:
        classifier_config_file_path = os.path.join(folder_path, 'config.yaml')

    with open(classifier_config_file_path, 'r') as f:
        # an arbitrary config gile; if using TRE, can
        a_classifier_config = yaml.safe_load(f)
        trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']

    dataset_path = os.path.join(os.path.dirname(
        os.path.dirname(folder_path)),  'cal_dataset')
    # load calidation dataset
    cal_trawls_path = os.path.join(dataset_path, 'cal_trawls.npy')
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    cal_trawls_ = jnp.load(cal_trawls_path)
    cal_x = jnp.load(cal_x_path)
    cal_thetas = jnp.load(cal_thetas_path)
    cal_Y = jnp.load(cal_Y_path)

    approximate_log_likelihood_to_evidence, approximate_log_posterior, _ = \
        load_trained_models(folder_path, cal_x[0], trawl_process_type,  # [::-1] not necessary, it s just a dummy, but just to make sure we don t pollute wth true values of some sort
                            use_tre, use_summary_statistics)

    log_r, pred_prob_Y, Y = [], [], []

    for i in range(cal_x.shape[0]):

        log_r.append(approximate_log_likelihood_to_evidence(
            cal_x[i], cal_thetas[i]))
        pred_prob_Y.append(jax.nn.sigmoid(log_r[-1]))
        Y.append(cal_Y)

    log_r = jnp.concatenate(log_r, axis=0)[:, 0]           # num_samples, 1
    pred_prob_Y = jnp.concatenate(
        pred_prob_Y, axis=0)[:, 0]      # num_samples, 1
    Y = jnp.concatenate(Y, axis=0)

    S = 2 * jnp.mean(log_r * Y)
    B = 2 * jnp.mean(pred_prob_Y)
    accuracy = jnp.mean((pred_prob_Y > 0.5).astype(jnp.float32) == Y)
    bce_ = optax.losses.sigmoid_binary_cross_entropy(logits=log_r,
                                                     labels=Y)

    bce_loss, bce_std = bce_.mean(), bce_.std() / len(bce_)**0.5

    with open(os.path.join(folder_path, 'metrics.txt'), "w") as f:
        f.write(f"S: {S}\n")
        f.write(f"B: {B}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"BCE Loss: {bce_loss}\n")
        f.write(f"BCE Std: {bce_std}\n")


if __name__ == '__main__':

    folder_paths = []

    folder_paths.append(
        r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_full_trawl\uncalibrated')
    folder_paths.append(
        r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_full_trawl\beta_calibrated')

    folder_paths.append(
        r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\NRE_full_trawl\uncalibrated')
    folder_paths.append(
        r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\NRE_full_trawl\beta_calibrated')

    # for subfolder in ['acf','beta','mu','sigma']:
    #    for cal in ['uncalibrated','beta_calibrated']:

    for folder_path in folder_paths:

        validate_classifiers(folder_path)
