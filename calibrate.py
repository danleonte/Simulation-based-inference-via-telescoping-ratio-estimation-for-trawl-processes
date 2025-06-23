# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:45:52 2025

@author: dleon
"""
from netcal.metrics.confidence import ECE, MCE, ACE
from src.utils.plot_calibration_map import plot_calibration_map
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration
from jax.scipy.special import logit
import pandas as pd
import distrax
import optax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.get_model import get_model
# from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
# , model_apply_wrapper
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function, tre_shuffle
from src.utils.monotone_spline_post_training_calibration import fit_spline
from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import VariableExtendedModel  # ,ExtendedModel
from jax.nn import sigmoid
import numpy as np
import datetime
import pickle
import yaml
import jax
import os
import netcal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


plt.rcParams["text.usetex"] = False


def compact_logit(x, epsilon):
    """
    Implements the compact logit function G_ε(x)

    Args:
        x: Input values in [0, 1]
        epsilon: Parameter controlling the transition regions (0 < epsilon < 0.5)

    Returns:
        Transformed values according to the compact logit function
    """
    # Constants based on epsilon
    scale_factor = (1 - 2 * epsilon) / (2 * jnp.log((1 - epsilon) / epsilon))

    # Conditions for the piecewise function
    in_middle = (x >= epsilon) & (x <= 1 - epsilon)

    # Calculate the logit transformation for the middle region
    logit_part = scale_factor * jnp.log(x / (1 - x)) + 0.5

    # Return the piecewise function using where
    return jnp.where(in_middle, logit_part, x)


# generate calibration dataset
def generate_dataset_from_Y_equal_1(classifier_config, nr_batches):

    # Get params and
    tre_config = classifier_config['tre_config']
    use_tre = tre_config['use_tre']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']
    tre_type = tre_config['tre_type']

    assert not use_summary_statistics
    assert (not replace_acf) or (not (use_tre and tre_type == 'acf'))

    if use_summary_statistics:
        project_trawl = get_projection_function()

    trawl_config = classifier_config['trawl_config']
    batch_size = trawl_config['batch_size']
    key = jax.random.split(PRNGKey(np.random.randint(1, 10000)), batch_size)

    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        classifier_config)

    # Generate calibration data
    # cal_trawls = []
    cal_thetas = []
    cal_x = []

    for _ in range(nr_batches):

        if _ % 50 == 0:
            print(_)

        theta_acf_cal, key = theta_acf_simulator(key)
        theta_marginal_jax_cal, theta_marginal_tf_cal, key = theta_marginal_simulator(
            key)
        trawl_cal, key = trawl_simulator(
            theta_acf_cal, theta_marginal_tf_cal, key)

        ########################################
        if use_summary_statistics:
            x_cal = project_trawl(trawl_cal)

        elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

            raise ValueError
            # x_cal = jnp.array([compute_empirical_acf(np.array(trawl_), nlags=nlags)[1:]
            #                   for trawl_ in trawl_cal])
        else:
            x_cal = trawl_cal

            ########################################
        theta_cal = jnp.concatenate(
            [theta_acf_cal, theta_marginal_jax_cal], axis=1)

        ### DO THE SHUFFLING WHEN CALIBRATING; THE DATASET WILL JUST ###
        ### CONTAIN SAMPLES  FROM THE JOINT  ###

        # x_cal, theta_cal, Y = tre_shuffle(
        #    x_cal, theta_cal, jnp.roll(theta_cal, -1, axis=0), classifier_config)

        # cal_trawls.append(trawl_cal)
        cal_thetas.append(np.array(theta_cal))
        cal_x.append(np.array(x_cal))

    # cal_trawls = jnp.array(cal_trawls)  # , axis=0)
    cal_x = np.array(cal_x)       # , axis=0)
    cal_thetas = np.array(cal_thetas)  # , axis=0)

    return cal_x, cal_thetas  # cal_trawls, cal_x, cal_thetas, Y


def OLDDDD___calibrated_the_NRE_of_a_calibrated_TRE___OLDDDDD(trained_classifier_path, seq_len):

    assert 'TRE' in trained_classifier_path
    use_tre = True

    if 'full_trawl' in trained_classifier_path:
        use_summary_statistics = False

    elif 'summary_statistics' in trained_classifier_path:
        use_summary_statistics = True

    else:
        raise ValueError

    assert use_tre
    classifier_config_file_path = os.path.join(
        trained_classifier_path, 'acf', 'config.yaml')
    # else:
    #    classifier_config_file_path = os.path.join(folder_path, 'config.yaml')

    with open(classifier_config_file_path, 'r') as f:
        # an arbitrary config gile; if using TRE, can
        a_classifier_config = yaml.safe_load(f)
        trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']

    dataset_path = os.path.join(os.path.dirname(
        os.path.dirname(trained_classifier_path)),  f'cal_dataset_{seq_len}')
    # load calidation dataset
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')
    cal_Y = jnp.load(cal_Y_path)

    # cal_trawls_ = jnp.load(cal_trawls_path)
    # cal_x = jnp.load(cal_x_path)
    # cal_thetas = jnp.load(cal_thetas_path)

    # approximate_log_likelihood_to_evidence,  _ = \
    #    load_trained_models(trained_classifier_path, cal_x[0], trawl_process_type,  # [::-1] not necessary, it s just a dummy, but just to make sure we don t pollute wth true values of some sort
    #                        use_tre, use_summary_statistics, f'spline_calibration_{seq_len}.pkl')

    log_r = 0.0

    for TRE_component, best_model_name in zip(['acf', 'beta', 'mu', 'sigma'], ['04_12_12_36_45', '04_12_04_26_56', '04_12_00_32_46', '04_12_04_28_49']):

        TRE_component_path = os.path.join(os.path.dirname(
            trained_classifier_path), TRE_component, best_model_name, 'best_model')
        spline_params = jnp.load(os.path.join(TRE_component_path,
                                              f'spline_calibration_{seq_len}.npy'))

        spline = distrax.RationalQuadraticSpline(boundary_slopes='identity',
                                                 params=spline_params, range_min=0.0, range_max=1.0)
        log_r_to_add = jnp.load(os.path.join(
            TRE_component_path, f'final_log_r_{seq_len}.npy'))
        log_r_to_add = logit(spline.forward(jax.nn.sigmoid(log_r_to_add)))
        log_r += log_r_to_add

    Y = jnp.concatenate([cal_Y] * (len(log_r) // len(cal_Y)))

    pred_prob_Y = jax.nn.sigmoid(log_r)
    max_pred_Y = jnp.max(pred_prob_Y)
    min_pred_Y = jnp.min(pred_prob_Y)

    if seq_len <= 2000:
        num_bins = 10
        epsilon = None

    else:
        num_bins = 15

        if seq_len == 2500:
            epsilon = 10**(-4)
        else:
            raise ValueError

    spline_params = fit_spline(probs=jnp.array(pred_prob_Y).squeeze(),
                               labels=jnp.array(Y), params=None, num_bins=num_bins)

    spline = distrax.RationalQuadraticSpline(boundary_slopes='identity',
                                             params=spline_params, range_min=0.0, range_max=1.0)

    if epsilon is None:

        spline_calibrated_classifier_output = spline.forward(
            jax.nn.sigmoid(log_r))
    else:
        spline_calibrated_classifier_output = spline.forward(
            compact_logit(jax.nn.sigmoid(log_r)), epsilon)

    spline_calibrated_log_r = logit(spline_calibrated_classifier_output)

    def compute_metrics(log_r, classifier_output, Y):

        extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=log_r, labels=Y)

        # this is due to numerical instability in the logit function and should be 0
        mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)

        # Replace values where mask is True with 0, otherwise keep original values
        extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)

        bce_loss = jnp.mean(extended_bce_loss)

        # half of them are 0s, half of them are 1, so we have to x2
        # S = 2 * jnp.mean(log_r * Y)
        S = jnp.mean(log_r[Y == 1])
        B = 2 * jnp.mean(classifier_output)
        accuracy = jnp.mean(
            (classifier_output > 0.5).astype(jnp.float32) == Y)

        return bce_loss, S, B

    pre_cal = compute_metrics(
        log_r.squeeze(), jax.nn.sigmoid(log_r).squeeze(), Y)
    post_cal = compute_metrics(
        spline_calibrated_log_r.squeeze(), spline_calibrated_classifier_output, Y)
    double_calibration_path = os.path.join(
        trained_classifier_path, f'overall_NRE_spline_cal_of_TRE_{seq_len}')
    os.makedirs(double_calibration_path, exist_ok=True)
    np.save(arr=spline_params, file=os.path.join(
        double_calibration_path, 'double_cal_spline_params.npy'))
    print(seq_len)
    print(pre_cal)
    print(post_cal)

# def fit_splines(trained_classifier_path, seq_len):


def calibrate_new(trained_classifier_path, nr_batches, seq_len):
    best_model_path = os.path.join(trained_classifier_path, 'best_model')

    # Load config
    with open(os.path.join(best_model_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)

    if 'TRE' in trained_classifier_path:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(trained_classifier_path)))), 'cal_dataset', f'cal_dataset_{seq_len}')

    elif 'NRE' in trained_classifier_path:

        dataset_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(best_model_path)))), 'cal_dataset', f'cal_dataset_{seq_len}')

    # load validation dataset
    cal_x_path = os.path.join(dataset_path, 'cal_x_joint.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas_joint.npy')

    if os.path.isfile(cal_x_path) and os.path.isfile(cal_thetas_path):
        print('Validation dataset already created')
        # Don't load the entire arrays at once - we'll load and process in batches later
    else:
        from copy import deepcopy
        print('Generating dataset')
        classifier_config_ = deepcopy(classifier_config)
        classifier_config_['trawl_config']['seq_len'] = seq_len
        cal_x, cal_thetas = generate_dataset_from_Y_equal_1(
            classifier_config_, nr_batches)
        print('Generated dataset')

        np.save(file=cal_x_path, arr=cal_x)
        np.save(file=cal_thetas_path, arr=cal_thetas)

    trawl_config = classifier_config['trawl_config']
    tre_config = classifier_config['tre_config']
    trawl_process_type = trawl_config['trawl_process_type']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']

    assert not use_summary_statistics
    assert (not replace_acf) or (not use_tre or tre_type != 'acf')

    # Load model
    # model, _, _ = get_model(classifier_config)
    model, params, _, __ = load_one_tre_model_only_and_prior_and_bounds(best_model_path,
                                                                        jnp.ones(
                                                                            [1, seq_len]),
                                                                        trawl_process_type, tre_type)

    # save x_cache
    cal_x_cache_path = os.path.join(
        best_model_path, f'x_cache_{tre_type}_{seq_len}.npy')

    if os.path.isfile(cal_x_cache_path):

        cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
        cal_x_cache_array = np.load(cal_x_cache_path, mmap_mode='r')
        print('cached x is already saved')

    else:

        # SAVE x_cache
        cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
        cal_x_array = np.load(cal_x_path, mmap_mode='r')

        nr_batches, batch_size, _ = cal_x_array.shape
        assert _ == seq_len

        # dummy to get x_cache shape
        dummy_x_ = jnp.ones([1, seq_len])
        dummy_theta_ = jnp.ones([1, 5])
        _, dummy_x_cache_batch = model.apply(params, dummy_x_, dummy_theta_)

        x_cache_shape = dummy_x_cache_batch.shape[-1]
        full_shape = (nr_batches, batch_size, x_cache_shape)

        cal_x_cache_array = np.lib.format.open_memmap(cal_x_cache_path, mode='w+',
                                                      dtype=np.float32, shape=full_shape)

        for i in range(nr_batches):

            cal_thetas_batch = jnp.array(cal_thetas_array[i])
            cal_x_batch = jnp.array(cal_x_array[i])

            _, x_cache_batch = model.apply(
                params, cal_x_batch, cal_thetas_batch)
            cal_x_cache_array[i] = np.array(x_cache_batch)

            if i % 50 == 0:

                cal_x_cache_array.flush()

        cal_x_cache_array.flush()

        print('finished caching x')
        del cal_x_array
        del cal_x_cache_array
        del cal_thetas_array

    ###################### DO THE CALIBRATION     #############################
    log_r_path = os.path.join(
        best_model_path, f'log_r_{seq_len}_{tre_type}.npy')
    pred_prob_Y_path = os.path.join(
        best_model_path, f'pred_prob_Y_{seq_len}_{tre_type}.npy')
    Y_path = os.path.join(best_model_path, f'Y_{seq_len}_{tre_type}.npy')

    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):

        cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
        cal_x_cache_array = np.load(cal_x_cache_path, mmap_mode='r')
        log_r = []
        Y = []
        pred_prob_Y = []

        for i in range(nr_batches):

            cal_theta_batch = jnp.array(cal_thetas_array[i])
            cal_x_cache_batch = jnp.array(cal_x_cache_array[i])

            cal_x_cache_batch, cal_theta_batch, cal_Y_to_append = tre_shuffle(
                cal_x_cache_batch, cal_theta_batch, jnp.roll(cal_theta_batch, -1, axis=0), classifier_config)

            log_r_to_append, _ = model.apply(
                params, None, cal_theta_batch, x_cache=cal_x_cache_batch)
            pred_prob_Y_to_append = jax.nn.sigmoid(log_r_to_append)

            log_r.append(log_r_to_append)
            Y.append(cal_Y_to_append)
            pred_prob_Y.append(pred_prob_Y_to_append)

        np.save(arr=np.concatenate(log_r, axis=0),       file=log_r_path)
        np.save(arr=np.concatenate(pred_prob_Y, axis=0), file=pred_prob_Y_path)
        np.save(arr=np.concatenate(Y, axis=0),           file=Y_path)

        del cal_x_cache_array

    # else:

    log_r = np.load(log_r_path)
    pred_prob_Y = np.load(pred_prob_Y_path)
    Y = np.load(Y_path)

    print(f"pred_prob_Y shape: {pred_prob_Y.shape}")
    print(f"pred_prob_Y dtype: {pred_prob_Y.dtype}")
    print(f"Y shape: {np.array(Y).shape}")

    def compute_metrics(log_r, classifier_output, Y):
        extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=log_r, labels=Y)
        mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)
        extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)
        bce_loss = jnp.mean(extended_bce_loss)
        S = jnp.mean(log_r[Y == 1])
        B = 2 * jnp.mean(classifier_output)
        accuracy = jnp.mean(
            (classifier_output > 0.5).astype(jnp.float32) == Y)
        return bce_loss, S, B

    # perform isotonic regression, Beta and Plat scaling
    lr = LogisticRegression(C=99999999999)
    iso = IsotonicRegression(y_min=0.0001, y_max=0.9999)
    bc = BetaCalibration(parameters="abm")

    # These sklearn methods work with numpy arrays
    lr.fit(pred_prob_Y, np.array(Y))
    iso.fit(pred_prob_Y, np.array(Y))
    bc.fit(pred_prob_Y, np.array(Y))

    beta_calibration_dict = {'use_beta_calibration': True,
                             'params': bc.calibrator_.map_}

    # open a text file
    with open(os.path.join(best_model_path, f'beta_calibration_{seq_len}_{tre_type}.pkl'), 'wb') as f:
        # serialize the list
        pickle.dump(beta_calibration_dict, f)

    linspace = np.linspace(0.0001, 0.9999, 200)
    pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1],
          iso.predict(linspace), bc.predict(linspace)]  # , spline.forward(linspace)]
    methods_text = ['logistic', 'isotonic', 'beta']  # , 'splines']

    # get calibrated datasets
    calibrated_pr = [lr.predict_proba(pred_prob_Y)[:, 1],
                     iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

    ### spline calibration ###
    num_bins_to_try = (2, 3, 4, 5, 6, 8, 10, 12)

    for num_bins_for_splines in num_bins_to_try:
        # load initial params, if available; otherwise, they will be generated randomly
        initial_spline_params_path = os.path.join(
            best_model_path, f'spline_calibration_{seq_len-500}_{num_bins_for_splines}_bins.npy')

        spline_calibration_already_done = True

        if not spline_calibration_already_done:

            if os.path.isfile(initial_spline_params_path):
                initial_spline_params = jnp.load(initial_spline_params_path)
            else:
                initial_spline_params = None

            # fit spline
            spline_params = fit_spline(probs=jnp.array(pred_prob_Y).squeeze(), num_bins=num_bins_for_splines,
                                       labels=jnp.array(Y), params=initial_spline_params)

            jnp.save(file=os.path.join(best_model_path,
                     f'spline_calibration_{seq_len}_{num_bins_for_splines}_bins.npy'), arr=spline_params)

        else:
            spline_params = jnp.load(os.path.join(best_model_path,
                                                  f'spline_calibration_{seq_len}_{num_bins_for_splines}_bins.npy'))

        spline = distrax.RationalQuadraticSpline(
            params=spline_params, range_min=0.0, range_max=1.0, boundary_slopes='identity')

        calibrated_pr.append(spline.forward(pred_prob_Y.squeeze(-1)))
        methods_text.append(f'splines_{num_bins_for_splines}')

    log_r_jax = jnp.array(log_r.squeeze())
    pred_prob_Y_jax = jnp.array(pred_prob_Y.squeeze())
    Y_jax = jnp.array(Y)

    metrics = []
    metrics.append(compute_metrics(log_r_jax, pred_prob_Y_jax, Y_jax))

    if False:
        ece_false = []
        ece_true = []
        mce_true = []
        mce_false = []
        ace_true = []
        ace_false = []

        ece_false.append(ECE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ece_true.append(ECE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        mce_false.append(MCE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        mce_true.append(MCE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ace_false.append(ACE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ace_true.append(ACE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))

        for i in range(len(methods_text)):
            # Convert one calibrated result at a time
            calibrated_pr_jax = jnp.array(calibrated_pr[i])
            logit_calibrated = logit(calibrated_pr_jax)
            metrics.append(compute_metrics(
                logit_calibrated, calibrated_pr_jax, Y_jax))
            ece_false.append(ECE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            ece_true.append(ECE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))

            mce_false.append(MCE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            mce_true.append(MCE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))

            ace_false.append(ACE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            ace_true.append(ACE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            # Free memory
            # del calibrated_pr_jax
            # del logit_calibrated

    df = pd.DataFrame(np.array(metrics).transpose(),
                      columns=['uncal', ' lr', 'isotonic',
                               'beta'] + methods_text[3:],
                      index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(trained_classifier_path,
                f'BCE_S_B_{seq_len}_{tre_type}_with_splines.xlsx'))

    # Optionally add the visualization code here if needed (set if False: to if True:)

    # Optionally add the visualization code here if needed (set if False: to if True:)
    # indeces_to_plot = np.random.randint(low=0, high = Y.shape[0], size = 1000)

    # calibration_results_path = os.path.join(trained_classifier_path,'calibration_results')
    # os.makedirs(calibration_results_path, exist_ok=True)

    ############ UNCALIBRATED plots ############
    do_calibration_plots = False
    if do_calibration_plots:

        hist_beta, ax = plt.subplots()
        ax.hist(
            pred_prob_Y[Y == 1].squeeze(), label='Y=1', alpha=0.5, density=True,
            bins=30)
        ax.hist(
            pred_prob_Y[Y == 0].squeeze(), label='Y=0', alpha=0.5, density=True)
        ax.set_title(
            'Histogram of c(x,theta) classifier')
        ax.legend(loc='upper center')
        hist_beta.savefig(os.path.join(
            trained_classifier_path, f'Uncalibrated_hist_{seq_len}_{tre_type}.pdf'))

        # Reliability 1
        diagram_un = ReliabilityDiagram(
            20, equal_intervals=True)
        fig_un = diagram_un.plot(
            np.array(pred_prob_Y), np.array(Y)).get_figure()

        fig_un.savefig(os.path.join(
            trained_classifier_path, f'uncalibrated_unequal_{seq_len}_{tre_type}.pdf'))

        # reliability 2

        try:
            diagram_eq = ReliabilityDiagram(9, equal_intervals=False)
            fig_eq = diagram_eq.plot(
                np.array(pred_prob_Y), np.array(Y)).get_figure()

            fig_eq.savefig(os.path.join(
                trained_classifier_path, f'uncalibrated_equal_{seq_len}_{tre_type}.pdf'))

        except:
            print('one reliability map not possible to do')

        ################### Calibration curves ###########################
        fig_map = plot_calibration_map(
            pr, [None, None, linspace], methods_text[:3])  # alpha
        fig_map.savefig(os.path.join(
            trained_classifier_path, f'calibration_map_{seq_len}_{tre_type}.pdf'))

        ################## CALIBRATED reliability diagrams ####################

        # General classifier histogram
        for i in range(4):
            try:
                hist_beta, ax = plt.subplots()
                ax.hist(
                    calibrated_pr[i][Y == 1].squeeze(), label='Y=1', alpha=0.5, density=True, bins=15)
                ax.hist(
                    calibrated_pr[i][Y == 0].squeeze(), label='Y=0', alpha=0.5, density=True, bins=15)
                ax.set_title(
                    'Histogram of c(x,theta) classifier')
                ax.legend(loc='upper center')
                hist_beta.savefig(os.path.join(
                    trained_classifier_path, f'Calibrated_hist_{methods_text[i]}_{seq_len}_{tre_type}.pdf'))
            except:
                print('problems')

        for i in range(4):

            diagram_un = ReliabilityDiagram(
                20, equal_intervals=True)
            fig_un = diagram_un.plot(
                np.array(calibrated_pr[i]), np.array(Y)).get_figure()

            fig_un.savefig(os.path.join(trained_classifier_path,
                           f'calibrated_unequal_{methods_text[i]}_{seq_len}_{tre_type}.pdf'))

        for i in range(4):
            try:
                diagram_eq = ReliabilityDiagram(
                    5, equal_intervals=False)
                fig_eq = diagram_eq.plot(
                    np.array(calibrated_pr[i]), np.array(Y)).get_figure()

                fig_eq.savefig(os.path.join(trained_classifier_path,
                               f'calibrated_equal_{methods_text[i]}_{seq_len}_{tre_type}.pdf'))

                diagram_eq = ReliabilityDiagram(
                    6, equal_intervals=False)
                fig_eq = diagram_eq.plot(
                    np.array(calibrated_pr[i]), np.array(Y)).get_figure()

                fig_eq.savefig(os.path.join(trained_classifier_path,
                               f'calibrated_equal_{methods_text[i]}_{seq_len}_{tre_type}.pdf'))

                diagram_eq = ReliabilityDiagram(
                    7, equal_intervals=False)
                fig_eq = diagram_eq.plot(
                    np.array(calibrated_pr[i]), np.array(Y)).get_figure()

                fig_eq.savefig(os.path.join(trained_classifier_path,
                               f'calibrated_equal_{methods_text[i]}_{seq_len}_{tre_type}.pdf'))

                diagram_eq = ReliabilityDiagram(
                    8, equal_intervals=False)
                fig_eq = diagram_eq.plot(
                    np.array(calibrated_pr[i]), np.array(Y)).get_figure()

                fig_eq.savefig(os.path.join(trained_classifier_path,
                               f'calibrated_equal_{methods_text[i]}_{seq_len}_{tre_type}.pdf'))
            except:
                print('one reliability map not possible to do')


def validate_new(trained_classifier_path, nr_batches, seq_len):
    best_model_path = os.path.join(trained_classifier_path, 'best_model')

    # Load config
    with open(os.path.join(best_model_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(trained_classifier_path)))), 'val_dataset', f'val_dataset_{seq_len}')
    # load validation dataset
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    if os.path.isfile(val_x_path) and os.path.isfile(val_thetas_path):
        print('Validation dataset already created')
        # Don't load the entire arrays at once - we'll load and process in batches later
    else:
        from copy import deepcopy
        print('Generating dataset')
        classifier_config_ = deepcopy(classifier_config)
        classifier_config_['trawl_config']['seq_len'] = seq_len
        val_x, val_thetas = generate_dataset_from_Y_equal_1(
            classifier_config_, nr_batches)
        print('Generated dataset')

        np.save(file=val_x_path, arr=val_x)
        np.save(file=val_thetas_path, arr=val_thetas)

    trawl_config = classifier_config['trawl_config']
    tre_config = classifier_config['tre_config']
    trawl_process_type = trawl_config['trawl_process_type']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']

    assert not use_summary_statistics
    assert (not replace_acf) or (not use_tre or tre_type != 'acf')

    # Load model
    # model, _, _ = get_model(classifier_config)
    model, params, _, __ = load_one_tre_model_only_and_prior_and_bounds(best_model_path,
                                                                        jnp.ones(
                                                                            [1, seq_len]),
                                                                        trawl_process_type, tre_type)
    val_data_results_path = os.path.join(best_model_path, 'val_data_results')
    os.makedirs(val_data_results_path, exist_ok=True)
    # save x_cache
    val_x_cache_path = os.path.join(
        val_data_results_path, f'x_cache_{tre_type}_{seq_len}.npy')

    if os.path.isfile(val_x_cache_path):

        val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
        val_x_cache_array = np.load(val_x_cache_path, mmap_mode='r')
        print('cached x is already saved')

    else:

        # SAVE x_cache
        val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
        val_x_array = np.load(val_x_path, mmap_mode='r')

        nr_batches, batch_size, _ = val_x_array.shape
        assert _ == seq_len

        # dummy to get x_cache shape
        dummy_x_ = jnp.ones([1, seq_len])
        dummy_theta_ = jnp.ones([1, 5])
        _, dummy_x_cache_batch = model.apply(params, dummy_x_, dummy_theta_)

        x_cache_shape = dummy_x_cache_batch.shape[-1]
        full_shape = (nr_batches, batch_size, x_cache_shape)

        val_x_cache_array = np.lib.format.open_memmap(val_x_cache_path, mode='w+',
                                                      dtype=np.float32, shape=full_shape)

        for i in range(nr_batches):

            val_thetas_batch = jnp.array(val_thetas_array[i])
            val_x_batch = jnp.array(val_x_array[i])

            _, x_cache_batch = model.apply(
                params, val_x_batch, val_thetas_batch)
            val_x_cache_array[i] = np.array(x_cache_batch)

            if i % 50 == 0:

                val_x_cache_array.flush()

        val_x_cache_array.flush()

        print('finished caching x')
        del val_x_array
        del val_x_cache_array
        del val_thetas_array

    ###################### DO THE CALIBRATION     #############################
    log_r_path = os.path.join(
        val_data_results_path, f'log_r_{seq_len}_{tre_type}.npy')
    pred_prob_Y_path = os.path.join(
        val_data_results_path, f'pred_prob_Y_{seq_len}_{tre_type}.npy')
    Y_path = os.path.join(val_data_results_path, f'Y_{seq_len}_{tre_type}.npy')

    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):

        val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
        val_x_cache_array = np.load(val_x_cache_path, mmap_mode='r')
        log_r = []
        Y = []
        pred_prob_Y = []

        for i in range(nr_batches):

            val_theta_batch = jnp.array(val_thetas_array[i])
            val_x_cache_batch = jnp.array(val_x_cache_array[i])

            val_x_cache_batch, val_theta_batch, val_Y_to_append = tre_shuffle(
                val_x_cache_batch, val_theta_batch, jnp.roll(val_theta_batch, -1, axis=0), classifier_config)

            log_r_to_append, _ = model.apply(
                params, None, val_theta_batch, x_cache=val_x_cache_batch)
            pred_prob_Y_to_append = jax.nn.sigmoid(log_r_to_append)

            log_r.append(log_r_to_append)
            Y.append(val_Y_to_append)
            pred_prob_Y.append(pred_prob_Y_to_append)

        np.save(arr=np.concatenate(log_r, axis=0),       file=log_r_path)
        np.save(arr=np.concatenate(pred_prob_Y, axis=0), file=pred_prob_Y_path)
        np.save(arr=np.concatenate(Y, axis=0),           file=Y_path)

        del val_x_cache_array

    # else:

    log_r = np.load(log_r_path)
    pred_prob_Y = np.load(pred_prob_Y_path)
    Y = np.load(Y_path)

    print(f"pred_prob_Y shape: {pred_prob_Y.shape}")
    print(f"pred_prob_Y dtype: {pred_prob_Y.dtype}")
    print(f"Y shape: {np.array(Y).shape}")

    def compute_metrics(log_r, classifier_output, Y):
        extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=log_r, labels=Y)
        mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)
        extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)
        bce_loss = jnp.mean(extended_bce_loss)
        S = jnp.mean(log_r[Y == 1])
        B = 2 * jnp.mean(classifier_output)
        accuracy = jnp.mean(
            (classifier_output > 0.5).astype(jnp.float32) == Y)
        return bce_loss, S, B

    # open a text file
    with open(os.path.join(best_model_path, f'beta_calibration_{seq_len}_{tre_type}.pkl'), 'rb') as f:
        # serialize the list
        beta_calibration_dict = pickle.load(f)

    methods_text = ['beta']  # , 'splines']

    # get calibrated datasets
    beta_cal_log_r = beta_calibrate_log_r(
        log_r, beta_calibration_dict['params']).squeeze()

    calibrated_pr = [sigmoid(beta_cal_log_r).squeeze()]
    # [lr.predict_proba(pred_prob_Y)[:, 1],
    # iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

    ### spline calibration ###
    num_bins_to_try = (2, 3, 4, 5, 6, 8, 10, 12)

    for num_bins_for_splines in num_bins_to_try:

        spline_params = jnp.load(os.path.join(best_model_path,
                                              f'spline_calibration_{seq_len}_{num_bins_for_splines}_bins.npy'))

        spline = distrax.RationalQuadraticSpline(
            params=spline_params, range_min=0.0, range_max=1.0, boundary_slopes='identity')

        calibrated_pr.append(spline.forward(pred_prob_Y.squeeze(-1)))
        methods_text.append(f'splines_{num_bins_for_splines}')

    log_r_jax = jnp.array(log_r.squeeze())
    pred_prob_Y_jax = jnp.array(pred_prob_Y.squeeze())
    Y_jax = jnp.array(Y)

    metrics = []
    metrics.append(compute_metrics(log_r_jax, pred_prob_Y_jax, Y_jax))

    if True:
        # ece_false = []
        # ece_true = []
        # mce_true = []
        # mce_false = []
        # ace_true = []
        # ace_false = []

        # ece_false.append(ECE(bins=5, equal_intervals=False).measure(
        #    np.array(pred_prob_Y_jax), np.array(Y_jax)))
        # ece_true.append(ECE(bins=20, equal_intervals=True).measure(
        #    np.array(pred_prob_Y_jax), np.array(Y_jax)))
        # mce_false.append(MCE(bins=5, equal_intervals=False).measure(
        #    np.array(pred_prob_Y_jax), np.array(Y_jax)))
        # mce_true.append(MCE(bins=20, equal_intervals=True).measure(
        #    np.array(pred_prob_Y_jax), np.array(Y_jax)))
        # ace_false.append(ACE(bins=5, equal_intervals=False).measure(
        #    np.array(pred_prob_Y_jax), np.array(Y_jax)))
        # ace_true.append(ACE(bins=20, equal_intervals=True).measure(
        #    np.array(pred_prob_Y_jax), np.array(Y_jax)))

        for i in range(len(methods_text)):
            # Convert one calibrated result at a time
            calibrated_pr_jax = jnp.array(calibrated_pr[i])
            logit_calibrated = logit(calibrated_pr_jax)
            metrics.append(compute_metrics(
                logit_calibrated, calibrated_pr_jax, Y_jax))
            # ece_false.append(ECE(bins=5, equal_intervals=False).measure(
            #    np.array(calibrated_pr_jax), np.array(Y_jax)))
            # ece_true.append(ECE(bins=20, equal_intervals=True).measure(
            #    np.array(calibrated_pr_jax), np.array(Y_jax)))

            # mce_false.append(MCE(bins=5, equal_intervals=False).measure(
            #    np.array(calibrated_pr_jax), np.array(Y_jax)))
            # mce_true.append(MCE(bins=20, equal_intervals=True).measure(
            #    np.array(calibrated_pr_jax), np.array(Y_jax)))

            # ace_false.append(ACE(bins=5, equal_intervals=False).measure(
            #    np.array(calibrated_pr_jax), np.array(Y_jax)))
            # ace_true.append(ACE(bins=20, equal_intervals=True).measure(
            #    np.array(calibrated_pr_jax), np.array(Y_jax)))
            # Free memory
            # del calibrated_pr_jax
            # del logit_calibrated

    df = pd.DataFrame(np.array(metrics).transpose(),
                      columns=['uncal'] + methods_text,
                      index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(val_data_results_path,
                f'BCE_S_B_{seq_len}_{tre_type}_with_splines.xlsx'))


if __name__ == '__main__':
    nr_batches = 5000

    folder_names = {'NRE_full_trawl': ['04_12_04_25_37']  # , '04_11_23_17_38','04_12_05_18_53','04_12_12_34_53','04_12_00_24_16','04_11_23_37_05','04_12_02_18_27','04_12_08_21_48',
                    # '04_11_20_15_48','04_12_11_52_49','04_12_11_21_25'],########['03_03_10_00_51', '03_03_13_37_12', '03_03_19_26_42', '03_04_02_34_16', '03_04_08_50_44', '03_04_13_48_26'],
                    # 'acf': ['04_12_00_27_16', '04_11_20_26_03', '04_12_12_36_45', '04_12_08_35_46', '04_12_04_29_36', '04_12_03_36_28', '04_12_09_57_22', '04_12_00_25_13', '04_11_20_30_16',
                    #        '04_12_13_11_32',  '04_12_06_47_42',   '04_11_23_40_38', '04_12_09_14_24', '04_11_20_32_11'],
                    # 'beta': ['04_12_04_26_56','04_12_12_35_41','04_12_00_25_49','04_12_05_27_48','04_11_23_24_46','04_12_12_17_28','04_12_08_31_07','04_11_23_37_34','04_12_11_30_54',                                '04_11_20_30_16'],
                    # 'mu': ['04_12_04_41_11','04_12_12_59_45','04_12_00_32_46','04_11_20_26_03','04_12_08_53_27','04_12_08_53_57','04_12_05_42_50','04_12_12_21_06'],
                    # 'sigma': ['04_12_04_28_49','04_12_00_26_44','04_12_12_37_42','04_12_05_36_55','04_12_12_37_35','04_12_11_18_04','04_12_08_35_55','04_12_09_33_30','04_12_05_59_51',                                '04_11_20_26_03']
                    # 'acf': ['04_12_12_36_45'],
                    # 'beta': ['04_12_04_26_56'],
                    # 'mu': ['04_12_00_32_46'],
                    # 'sigma': ['04_12_04_28_49'],
                    # 'acf':['02_26_18_30_52', '02_28_16_37_11', '03_01_09_30_39', '03_01_21_17_21', '03_02_21_06_17','03_02_06_41_57'],
                    # 'beta':['02_26_15_56_48', '02_26_16_02_10', '02_26_19_29_54', '02_26_19_37_09', '02_26_23_14_03', '02_27_02_50_03'],
                    # 'mu':['03_03_16_41_47', '03_03_16_45_26', '03_03_18_35_58', '03_03_21_29_04', '03_04_01_33_54', '03_04_01_46_31'],
                    # 'sigma':['03_03_16_56_52', '03_03_23_18_48', '03_04_02_42_13', '03_04_07_34_58', '03_04_12_28_46', '03_04_21_43_47']
                    }

    for key in folder_names:
        for value in folder_names[key]:

            if key == 'NRE_full_trawl':

                trained_classifier_path = os.path.join(
                    os.getcwd(), 'models', 'new_classifier', 'NRE_full_trawl', value)
            else:
                trained_classifier_path = os.path.join(
                    os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl', key, value)  # 'NRE_full_trawl '

            if True:
                calibrate_new(trained_classifier_path, nr_batches, 1000)
                # calibrate_new(trained_classifier_path, nr_batches, 1500)
            # calibrate_new(trained_classifier_path, nr_batches, 2000)
            # calibrate_new(trained_classifier_path, nr_batches, 2500)
            # calibrate(trained_classifier_path, nr_batches, 3000)
            # calibrate(trained_classifier_path, nr_batches, 3500)

    # TRE options: acf, beta, mu, sigma

    # calibrate
            if True:
                validate_new(trained_classifier_path, nr_batches, 1000)
                # validate_new(trained_classifier_path, nr_batches, 1500)
                # validate_new(trained_classifier_path, nr_batches, 2000)
                # validate_new(trained_classifier_path, nr_batches, 2500)

        # calibrated_the_NRE_of_a_calibrated_TRE(
        #    double_trained_classifier_path, 2000)
        # calibrated_the_NRE_of_a_calibrated_TRE(
        #    double_trained_classifier_path, 1500)
        # calibrated_the_NRE_of_a_calibrated_TRE(
        #    double_trained_classifier_path, 1000)
        # calibrated_the_NRE_of_a_calibrated_TRE(
        #    double_trained_classifier_path, 1500)
        # calibrated_the_NRE_of_a_calibrated_TRE(
        #    double_trained_classifier_path, 2000)
        # calibrated_the_NRE_of_a_calibrated_TRE(
        #    double_trained_classifier_path, 1000)
