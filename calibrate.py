# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:45:52 2025

@author: dleon
"""

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration
from jax.scipy.special import logit
import pandas as pd

import optax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function, tre_shuffle
from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import ExtendedModel
import numpy as np
import datetime
import pickle
import yaml
import jax
import os
import netcal
import matplotlib.pyplot as plt
import numpy as np
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.plot_calibration_map import plot_calibration_map


# generate calibration dataset
def generate_dataset(classifier_config, nr_batches):

    # Get params and hyperparams
    tre_config = classifier_config['tre_config']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']
    nlags = tre_config['nlags']

    if use_summary_statistics:
        project_trawl = get_projection_function()

    trawl_config = classifier_config['trawl_config']
    batch_size = trawl_config['batch_size']
    key = jax.random.split(PRNGKey(100), batch_size)

    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        classifier_config)

    # Generate calibration data
    cal_trawls = []
    cal_thetas = []

    for _ in range(nr_batches):

        if _ % 20 == 0:
            print(_)

        theta_acf_cal, key = theta_acf_simulator(key)
        theta_marginal_jax_cal, theta_marginal_tf_cal, key = theta_marginal_simulator(
            key)
        trawl_cal, key = trawl_simulator(
            theta_acf_cal, theta_marginal_tf_cal, key)

        ########################################
        if use_summary_statistics:
            trawl_cal = project_trawl(trawl_cal)

        elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

            trawl_cal = jnp.array([compute_empirical_acf(np.array(trawl_), nlags=nlags)[1:]
                                   for trawl_ in trawl_cal])

            ########################################
        theta_cal = jnp.concatenate(
            [theta_acf_cal, theta_marginal_jax_cal], axis=1)

        trawl_cal, theta_cal, Y = tre_shuffle(
            trawl_cal, theta_cal, jnp.roll(theta_cal, -1, axis=0), classifier_config)

        cal_trawls.append(trawl_cal)
        cal_thetas.append(theta_cal)

    cal_trawls = jnp.array(cal_trawls)  # , axis=0)
    cal_thetas = jnp.array(cal_thetas)  # , axis=0)

    return cal_trawls, cal_thetas, Y


def calibrate(trained_classifier_path, nr_batches):

    # Load config
    with open(os.path.join(trained_classifier_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)

    # load calidation dataset
    cal_trawls_path = os.path.join(trained_classifier_path, 'cal_trawls.npy')
    cal_thetas_path = os.path.join(trained_classifier_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(trained_classifier_path, 'cal_Y.npy')

    if os.path.isfile(cal_trawls_path) and os.path.isfile(cal_thetas_path) and os.path.isfile(cal_Y_path):

        print('Calidation dataset already created')

        cal_trawls = np.load(cal_trawls_path)
        cal_thetas = np.load(cal_thetas_path)
        cal_Y = np.load(cal_Y_path)

    else:

        print('Generating dataset')
        cal_trawls, cal_thetas, cal_Y = generate_dataset(
            classifier_config, nr_batches)
        print('Generated dataset')

        np.save(file=cal_trawls_path, arr=cal_trawls)
        np.save(file=cal_thetas_path, arr=cal_thetas)
        np.save(file=cal_Y_path, arr=cal_Y)

    # Load model
    model, _, _ = get_model(classifier_config)

    trawl_config = classifier_config['trawl_config']
    tre_config = classifier_config['tre_config']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']

    if use_tre:
        assert tre_type in ('beta', 'mu', 'sigma', 'acf')
        if tre_type == 'acf' and (not use_summary_statistics):
            assert replace_acf

        ######  EXTENDED MODEL HERE ########
        # CHECK KEYS ARE UPDATED
        model = ExtendedModel(base_model=model,  trawl_process_type=trawl_config['trawl_process_type'],
                              tre_type=tre_type, use_summary_statistics=use_summary_statistics)

        model.init(PRNGKey(0), cal_trawls[0], cal_thetas[0])

    # First check there's only one pickled file for the params, then load params
    list_params_names = [filename for filename in os.listdir(
        trained_classifier_path) if filename.endswith(".pkl")]
    assert len(list_params_names) == 1

    with open(os.path.join(trained_classifier_path, list_params_names[0]), 'rb') as file:
        params = pickle.load(file)
        # params = {'params': params['params']['base_model']}

    ###########################################################################
    @jax.jit
    def compute_log_r_approx(params, trawl, theta):
        log_r = model.apply(
            variables=params, x=cal_trawls[i], theta=cal_thetas[i], train=False)
        classifier_output = jax.nn.sigmoid(log_r)

        return log_r, classifier_output

    def compute_metrics(log_r, classifier_output, Y):

        bce_loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(
            logits=log_r, labels=Y))

        # half of them are 0s, half of them are 1, so we have to x2
        S = 2 * jnp.mean(log_r * Y)
        B = 2 * jnp.mean(classifier_output)
        accuracy = jnp.mean(
            (classifier_output > 0.5).astype(jnp.float32) == Y)

        return bce_loss, S, B

    ######################## post training regression ########################

    # perform isotonic regression, Beta and Plat scaling
    lr = LogisticRegression(C=99999999999)
    iso = IsotonicRegression(y_min=0.001, y_max=0.999)
    bc = BetaCalibration(parameters="abm")

    log_r, pred_prob_Y, Y = [], [], []

    for i in range(cal_trawls.shape[0]):

        a, b = compute_log_r_approx(params, cal_trawls[i], cal_thetas[i])
        log_r.append(a)
        pred_prob_Y.append(b)
        Y.append(cal_Y)

    log_r = jnp.concatenate(log_r, axis=0)           # num_samples, 1
    pred_prob_Y = jnp.concatenate(pred_prob_Y, axis=0)      # num_samples, 1
    Y = jnp.concatenate(Y, axis=0)

    lr.fit(pred_prob_Y, np.array(Y))
    iso.fit(pred_prob_Y, np.array(Y))
    bc.fit(pred_prob_Y,  np.array(Y))

    linspace = np.linspace(0, 1, 100)
    pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1],
          iso.predict(linspace), bc.predict(linspace)]
    methods_text = ['logistic', 'isotonic', 'beta']
    # indeces_to_plot = np.random.randint(low=0, high = Y.shape[0], size = 1000)

    ############ UNCALIBRATED plots ############

    # General classifier histogram

    hist_beta, ax = plt.subplots()
    ax.hist(
        pred_prob_Y[Y == 1].squeeze(), label='Y=1', alpha=0.5, density=True)
    ax.hist(
        pred_prob_Y[Y == 0].squeeze(), label='Y=0', alpha=0.5, density=True)
    ax.set_title(
        r'Histogram of $c(\mathbf{x},\mathbf{\theta})$ classifier')
    ax.legend(loc='upper center')
    hist_beta.savefig(os.path.join(
        trained_classifier_path, 'Uncalibrated_hist.pdf'))

    # Reliability 1
    diagram_un = ReliabilityDiagram(
        15, equal_intervals=True)
    fig_un = diagram_un.plot(
        np.array(pred_prob_Y), np.array(Y)).get_figure()

    fig_un.savefig(os.path.join(
        trained_classifier_path, 'uncalibrated_unequal.pdf'))

    # reliability 2

    try:
        diagram_eq = ReliabilityDiagram(6, equal_intervals=False)
        fig_eq = diagram_eq.plot(
            np.array(pred_prob_Y), np.array(Y)).get_figure()

        fig_eq.savefig(os.path.join(
            trained_classifier_path, 'uncalibrated_equal.pdf'))

    except:
        print('one reliability map not possible to do')

    ################### Calibration curves ###########################
    fig_map = plot_calibration_map(
        pr, [None, None, linspace], methods_text)  # alpha
    fig_map.savefig(os.path.join(
        trained_classifier_path, 'calibration_map.pdf'))

    # get calibrated datasets
    calibrated_pr = [lr.predict_proba(pred_prob_Y)[:, 1],
                     iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

    ################## CALIBRATED reliability diagrams ####################

    # General classifier histogram
    for i in range(3):
        try:
            hist_beta, ax = plt.subplots()
            ax.hist(
                calibrated_pr[i][Y == 1].squeeze(), label='Y=1', alpha=0.5, density=True, bins=15)
            ax.hist(
                calibrated_pr[i][Y == 0].squeeze(), label='Y=0', alpha=0.5, density=True, bins=15)
            ax.set_title(
                r'Histogram of $c(\mathbf{x},\mathbf{\theta})$ classifier')
            ax.legend(loc='upper center')
            hist_beta.savefig(os.path.join(
                trained_classifier_path, f'Calibrated_hist_{methods_text[i]}.pdf'))
        except:
            print('problems')

    for i in range(3):

        diagram_un = ReliabilityDiagram(
            15, equal_intervals=True)
        fig_un = diagram_un.plot(
            calibrated_pr[i], np.array(Y)).get_figure()

        fig_un.savefig(os.path.join(trained_classifier_path,
                       f'calibrated_unequal_{methods_text[i]}.pdf'))

    for i in range(3):
        try:

            diagram_eq = ReliabilityDiagram(
                6, equal_intervals=False)
            fig_eq = diagram_eq.plot(
                calibrated_pr[i], np.array(Y)).get_figure()

            fig_eq.savefig(os.path.join(trained_classifier_path,
                           f'calibrated_equal_{methods_text[i]}.pdf'))
        except:
            print('one reliability map not possible to do')

    ##################### compute metrics ########################
    metrics = []
    metrics.append(compute_metrics(log_r.squeeze(), pred_prob_Y.squeeze(), Y))
    for i in range(3):

        metrics.append(compute_metrics(
            logit(calibrated_pr[i]), calibrated_pr[i], Y))

    df = pd.DataFrame(np.array(metrics).transpose(),
                      columns=('uncal', ' lr', 'isotonic', 'beta'),
                      index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(trained_classifier_path, 'BCE_S_B.xlsx'))


if __name__ == '__main__':

    # trained_classifier_path = os.path.join(os.getcwd(),'models','classifier',
    #                            'NRE_summary_statistics','best_model')
    nr_batches = 500
    trained_classifier_path = os.path.join(os.getcwd(), 'models', 'classifier',
                                           'TRE_summary_statistics', 'beta', 'best_model_degenerate')

    calibrate(trained_classifier_path, nr_batches)
    # TRE options: acf, beta, mu, sigma

    # calibrate
