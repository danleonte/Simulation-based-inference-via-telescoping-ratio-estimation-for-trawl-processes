# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:45:52 2025

@author: dleon
"""

from src.utils.plot_calibration_map import plot_calibration_map
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
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function, tre_shuffle
from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import ExtendedModel, VariableExtendedModel
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
    #cal_trawls = []
    cal_thetas = []
    cal_x = []

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
            x_cal = project_trawl(trawl_cal)

        elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

            x_cal = jnp.array([compute_empirical_acf(np.array(trawl_), nlags=nlags)[1:]
                               for trawl_ in trawl_cal])
        else:
            x_cal = trawl_cal

            ########################################
        theta_cal = jnp.concatenate(
            [theta_acf_cal, theta_marginal_jax_cal], axis=1)

        x_cal, theta_cal, Y = tre_shuffle(
            x_cal, theta_cal, jnp.roll(theta_cal, -1, axis=0), classifier_config)

        #cal_trawls.append(trawl_cal)
        cal_thetas.append(np.array(theta_cal))
        cal_x.append(np.array(x_cal))

    #cal_trawls = jnp.array(cal_trawls)  # , axis=0)
    cal_x = np.array(cal_x)       # , axis=0)
    cal_thetas = np.array(cal_thetas)  # , axis=0)

    return cal_x, cal_thetas, Y #cal_trawls, cal_x, cal_thetas, Y


def calibrated_the_NRE_of_a_calibrated_TRE(trained_classifier_path, seq_len):

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
    #cal_trawls_path = os.path.join(dataset_path, 'cal_trawls.npy')
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    #cal_trawls_ = jnp.load(cal_trawls_path)
    cal_x = jnp.load(cal_x_path)
    cal_thetas = jnp.load(cal_thetas_path)
    cal_Y = jnp.load(cal_Y_path)

    approximate_log_likelihood_to_evidence,  _ = \
        load_trained_models(trained_classifier_path, cal_x[0], trawl_process_type,  # [::-1] not necessary, it s just a dummy, but just to make sure we don t pollute wth true values of some sort
                            use_tre, use_summary_statistics, f'calibration_{seq_len}.pkl')

    double_calibration_path = os.path.join(
        trained_classifier_path, f'double_cal_{seq_len}')
    os.makedirs(double_calibration_path, exist_ok=True)

    log_r_path = os.path.join(double_calibration_path,
                              f'double_cal_log_r_{seq_len}.npy')
    pred_prob_Y_path = os.path.join(
        double_calibration_path, f'double_cal_pred_prob_Y_{seq_len}.npy')
    Y_path = os.path.join(double_calibration_path,
                          f'double_cal_Y_{seq_len}.npy')

    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):

        log_r, pred_prob_Y, Y = [], [], []

        for i in range(cal_x.shape[0]):

            log_r_to_append = approximate_log_likelihood_to_evidence(
                cal_x[i], cal_thetas[i])
            pred_prob_Y_to_append = jax.nn.sigmoid(log_r_to_append)

            log_r.append(log_r_to_append)
            pred_prob_Y.append(pred_prob_Y_to_append)
            Y.append(cal_Y)

        log_r = jnp.concatenate(log_r, axis=0)           # num_samples, 1
        pred_prob_Y = jnp.concatenate(
            pred_prob_Y, axis=0)      # num_samples, 1
        Y = jnp.concatenate(Y, axis=0)

        np.save(file=log_r_path, arr=log_r)
        np.save(file=pred_prob_Y_path, arr=pred_prob_Y)
        np.save(file=Y_path, arr=Y)

    else:

        log_r = np.load(log_r_path)
        pred_prob_Y = np.load(pred_prob_Y_path)
        Y = jnp.load(Y_path)

    # perform isotonic regression, Beta and Plat scaling
    lr = LogisticRegression(C=99999999999)
    iso = IsotonicRegression(y_min=0.001, y_max=0.999)
    bc = BetaCalibration(parameters="abm")

    lr.fit(pred_prob_Y, np.array(Y))
    iso.fit(pred_prob_Y, np.array(Y))
    bc.fit(pred_prob_Y,  np.array(Y))

    calibration_dict = {'use_beta_calibration': True,
                        'params': bc.calibrator_.map_}
    # open a text file
    with open(os.path.join(double_calibration_path, f'double_cal_{seq_len}.pkl'), 'wb') as f:
        # serialize the list
        pickle.dump(calibration_dict, f)

    linspace = np.linspace(0, 1, 100)
    pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1],
          iso.predict(linspace), bc.predict(linspace)]
    methods_text = ['logistic', 'isotonic', 'beta']

    # get calibrated datasets
    calibrated_pr = [lr.predict_proba(pred_prob_Y)[:, 1],
                     iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

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

    metrics = []
    metrics.append(compute_metrics(log_r.squeeze(), pred_prob_Y.squeeze(), Y))
    for i in range(3):

        metrics.append(compute_metrics(
            logit(calibrated_pr[i]), calibrated_pr[i], Y))

    df = pd.DataFrame(np.array(metrics).transpose(),
                      columns=('uncal', ' lr', 'isotonic', 'beta'),
                      index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(double_calibration_path,
                f'double_cal_BCE_S_B_{seq_len}.xlsx'))

    ################### Calibration curves ###########################
    if False: 
     try:
        fig_map = plot_calibration_map(
            pr, [None, None, linspace], methods_text)  # alpha
        fig_map.savefig(os.path.join(
            double_calibration_path, f'double_calibration_map_{seq_len}.pdf'))

        # General classifier histogram

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
            double_calibration_path, f'Uncalibrated_hist_{seq_len}.pdf'))

     except:
        pass
    print(min(pred_prob_Y[Y == 1]), max(pred_prob_Y[Y == 0]))

def calibrate(trained_classifier_path, nr_batches, seq_len):
    # Load config
    with open(os.path.join(trained_classifier_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(trained_classifier_path)))),  f'cal_dataset_{seq_len}')
    # load validation dataset
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    if os.path.isfile(cal_x_path) and os.path.isfile(cal_thetas_path) and os.path.isfile(cal_Y_path):
        print('Validation dataset already created')
        # Don't load the entire arrays at once - we'll load and process in batches later
    else:
        from copy import deepcopy
        print('Generating dataset')
        classifier_config_ = deepcopy(classifier_config)
        classifier_config_['trawl_config']['seq_len'] = seq_len
        cal_x, cal_thetas, cal_Y = generate_dataset(
            classifier_config_, nr_batches)
        print('Generated dataset')

        np.save(file=cal_x_path, arr=cal_x)
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
        assert tre_type in ('beta', 'mu', 'sigma', 'scale', 'acf')
        
        # The model expects a specific shape for initialization
        # Get first sample from first batch with correct dimensions
        sample_x = np.load(cal_x_path, mmap_mode='r')[0]  # Get first sample
        sample_thetas = np.load(cal_thetas_path, mmap_mode='r')[0]  # Get first sample
        
        # Convert to JAX for initialization only
        sample_x_jax = jnp.array(sample_x)
        sample_thetas_jax = jnp.array(sample_thetas)
        
        model = VariableExtendedModel(base_model=model, trawl_process_type=trawl_config['trawl_process_type'],
                                      tre_type=tre_type, use_summary_statistics=use_summary_statistics)
        
        # Initialize with single samples, not batches
        model.init(PRNGKey(0), sample_x_jax, sample_thetas_jax)

    # Load best model params
    with open(os.path.join(trained_classifier_path, "best_model_info.txt"), "r") as file:
        lines = file.readlines()

    # Extract the number from the first line
    first_line = lines[0]
    best_model_number = int(first_line.split(":")[-1].strip())
    best_model_path = os.path.join(os.path.dirname(
        trained_classifier_path), 'params_iter_' + str(best_model_number) + '.pkl')

    with open(best_model_path, 'rb') as file:
        params = pickle.load(file)

    @jax.jit
    def compute_log_r_approx(params, x, theta):
        log_r, _ = model.apply(
            variables=params, x=x, theta=theta, train=False)
        classifier_output = jax.nn.sigmoid(log_r)
        return log_r, classifier_output

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

    log_r_path = os.path.join(trained_classifier_path, f'final_log_r_{seq_len}.npy')
    pred_prob_Y_path = os.path.join(trained_classifier_path, f'final_pred_prob_Y_{seq_len}.npy')
    Y_path = os.path.join(trained_classifier_path, f'final_Y_{seq_len}.npy')

    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):
        # Load the data in batches and process
        # Use memmap for initial loading without pulling all data into memory
        cal_x_array = np.load(cal_x_path, mmap_mode='r')
        cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
        cal_Y_array = np.load(cal_Y_path, mmap_mode='r')
        
        # Get file info first to understand the data structure
        cal_x_info = np.load(cal_x_path, mmap_mode='r')
        batch_count = cal_x_info.shape[0]
        samples_per_batch = cal_x_info.shape[1]
        total_samples = batch_count * samples_per_batch
        
        # Initialize lists to collect results (we'll convert to arrays later)
        log_r_list = []
        pred_prob_Y_list = []
        Y_list = []
        
        # Process one batch at a time
        for i in range(batch_count):
            if i % 10 == 0:
                print(f"Processing batch {i} out of {batch_count}")
                
            # Load the current batch into memory
            x_batch = np.array(cal_x_array[i])  # Make a copy to ensure it's in memory
            theta_batch = np.array(cal_thetas_array[i])
            y_batch = np.array(cal_Y_array)
            
            # Convert to JAX arrays for processing
            x_batch_jax = jnp.array(x_batch)
            theta_batch_jax = jnp.array(theta_batch)
            
            # Process with JAX
            log_r_batch, pred_prob_Y_batch = compute_log_r_approx(params, x_batch_jax, theta_batch_jax)
            
            # Convert results back to numpy and store
            log_r_list.append(np.array(log_r_batch))
            pred_prob_Y_list.append(np.array(pred_prob_Y_batch))
            Y_list.append(np.array(y_batch))
            
            # Explicitly delete JAX arrays to free memory
            del x_batch_jax
            del theta_batch_jax
        
        # Concatenate all results
        log_r_np = np.concatenate(log_r_list, axis=0)
        pred_prob_Y_np = np.concatenate(pred_prob_Y_list, axis=0)
        Y_np = np.concatenate(Y_list, axis=0)
            
        # Explicitly delete JAX arrays to free memory
        #del x_batch_jax
        #del theta_batch_jax
        #del log_r_batch
        #del pred_prob_Y_batch
            
        # Save results
        np.save(file=log_r_path, arr=log_r_np)
        np.save(file=pred_prob_Y_path, arr=pred_prob_Y_np)
        np.save(file=Y_path, arr=Y_np)
        
        # Use numpy arrays for calibration
        log_r = log_r_np
        pred_prob_Y = pred_prob_Y_np
        Y = Y_np
    else:
        # Load existing results
        log_r = np.load(log_r_path)
        pred_prob_Y = np.load(pred_prob_Y_path)
        Y = np.load(Y_path)

    # perform isotonic regression, Beta and Plat scaling
    lr = LogisticRegression(C=99999999999)
    iso = IsotonicRegression(y_min=0.0001, y_max=0.9999)
    bc = BetaCalibration(parameters="abm")

    # These sklearn methods work with numpy arrays
    lr.fit(pred_prob_Y, np.array(Y))
    iso.fit(pred_prob_Y, np.array(Y))
    bc.fit(pred_prob_Y, np.array(Y))

    calibration_dict = {'use_beta_calibration': True,
                        'params': bc.calibrator_.map_}
    # open a text file
    with open(os.path.join(trained_classifier_path, f'calibration_{seq_len}.pkl'), 'wb') as f:
        # serialize the list
        pickle.dump(calibration_dict, f)

    linspace = np.linspace(0.0001, 0.9999, 100)
    pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1],
          iso.predict(linspace), bc.predict(linspace)]
    methods_text = ['logistic', 'isotonic', 'beta']

    # get calibrated datasets
    calibrated_pr = [lr.predict_proba(pred_prob_Y)[:, 1],
                     iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

    # We need to convert to JAX for the metrics calculation
    # Convert in chunks if needed for large datasets
    # For demonstration, we'll convert all at once but you could do this in batches too
    log_r_jax = jnp.array(log_r.squeeze())
    pred_prob_Y_jax = jnp.array(pred_prob_Y.squeeze())
    Y_jax = jnp.array(Y)
    
    metrics = []
    metrics.append(compute_metrics(log_r_jax, pred_prob_Y_jax, Y_jax))
    
    for i in range(3):
        # Convert one calibrated result at a time
        calibrated_pr_jax = jnp.array(calibrated_pr[i])
        logit_calibrated = logit(calibrated_pr_jax)
        metrics.append(compute_metrics(logit_calibrated, calibrated_pr_jax, Y_jax))
        # Free memory
        #del calibrated_pr_jax
        #del logit_calibrated

    df = pd.DataFrame(np.array(metrics).transpose(),
                    columns=('uncal', ' lr', 'isotonic', 'beta'),
                    index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(trained_classifier_path, f'BCE_S_B_{seq_len}.xlsx'))
    
    # Optionally add the visualization code here if needed (set if False: to if True:)
    
    # Optionally add the visualization code here if needed (set if False: to if True:)
    # indeces_to_plot = np.random.randint(low=0, high = Y.shape[0], size = 1000)

    # calibration_results_path = os.path.join(trained_classifier_path,'calibration_results')
    # os.makedirs(calibration_results_path, exist_ok=True)

    ############ UNCALIBRATED plots ############

    if False:

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
            trained_classifier_path, f'Uncalibrated_hist_{seq_len}.pdf'))

        # Reliability 1
        diagram_un = ReliabilityDiagram(
            15, equal_intervals=True)
        fig_un = diagram_un.plot(
            np.array(pred_prob_Y), np.array(Y)).get_figure()

        fig_un.savefig(os.path.join(
            trained_classifier_path, f'uncalibrated_unequal_{seq_len}.pdf'))

        # reliability 2

        try:
            diagram_eq = ReliabilityDiagram(9, equal_intervals=False)
            fig_eq = diagram_eq.plot(
                np.array(pred_prob_Y), np.array(Y)).get_figure()

            fig_eq.savefig(os.path.join(
                trained_classifier_path, f'uncalibrated_equal_{seq_len}.pdf'))

        except:
            print('one reliability map not possible to do')

        ################### Calibration curves ###########################
        fig_map = plot_calibration_map(
            pr, [None, None, linspace], methods_text)  # alpha
        fig_map.savefig(os.path.join(
            trained_classifier_path, f'calibration_map_{seq_len}.pdf'))

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
                    'Histogram of c(x,theta) classifier')
                ax.legend(loc='upper center')
                hist_beta.savefig(os.path.join(
                    trained_classifier_path, f'Calibrated_hist_{methods_text[i]}_{seq_len}.pdf'))
            except:
                print('problems')

        for i in range(3):

            diagram_un = ReliabilityDiagram(
                15, equal_intervals=True)
            fig_un = diagram_un.plot(
                calibrated_pr[i], np.array(Y)).get_figure()

            fig_un.savefig(os.path.join(trained_classifier_path,
                           f'calibrated_unequal_{methods_text[i]}_{seq_len}.pdf'))

        for i in range(3):
            try:

                diagram_eq = ReliabilityDiagram(
                    8, equal_intervals=False)
                fig_eq = diagram_eq.plot(
                    calibrated_pr[i], np.array(Y)).get_figure()

                fig_eq.savefig(os.path.join(trained_classifier_path,
                               f'calibrated_equal_{methods_text[i]}_{seq_len}.pdf'))
            except:
                print('one reliability map not possible to do')


if __name__ == '__main__':
    nr_batches = 5500

    folder_names = {  # 'NRE_full_trawl': ['04_12_04_25_37', '04_11_23_17_38','04_12_05_18_53','04_12_12_34_53','04_12_00_24_16','04_11_23_37_05','04_12_02_18_27','04_12_08_21_48',
        # '04_11_20_15_48','04_12_11_52_49','04_12_11_21_25'],########['03_03_10_00_51', '03_03_13_37_12', '03_03_19_26_42', '03_04_02_34_16', '03_04_08_50_44', '03_04_13_48_26'],
        # 'acf': ['04_12_00_27_16', '04_11_20_26_03', '04_12_12_36_45', '04_12_08_35_46', '04_12_04_29_36', '04_12_03_36_28', '04_12_09_57_22', '04_12_00_25_13', '04_11_20_30_16',
        #        '04_12_13_11_32',  '04_12_06_47_42',   '04_11_23_40_38', '04_12_09_14_24', '04_11_20_32_11'],
        # 'beta': ['04_12_04_26_56','04_12_12_35_41','04_12_00_25_49','04_12_05_27_48','04_11_23_24_46','04_12_12_17_28','04_12_08_31_07','04_11_23_37_34','04_12_11_30_54',                                '04_11_20_30_16'],
        # 'mu': ['04_12_04_41_11','04_12_12_59_45','04_12_00_32_46','04_11_20_26_03','04_12_08_53_27','04_12_08_53_57','04_12_05_42_50','04_12_12_21_06'],
        # 'sigma': ['04_12_04_28_49','04_12_00_26_44','04_12_12_37_42','04_12_05_36_55','04_12_12_37_35','04_12_11_18_04','04_12_08_35_55','04_12_09_33_30','04_12_05_59_51',                                '04_11_20_26_03']
        'acf': ['04_12_12_36_45'],
        'beta': ['04_12_04_26_56'],
        'mu': ['04_12_00_32_46'],
        'sigma': ['04_12_04_28_49'],
        # 'acf':['02_26_18_30_52', '02_28_16_37_11', '03_01_09_30_39', '03_01_21_17_21', '03_02_21_06_17','03_02_06_41_57'],
        # 'beta':['02_26_15_56_48', '02_26_16_02_10', '02_26_19_29_54', '02_26_19_37_09', '02_26_23_14_03', '02_27_02_50_03'],
        # 'mu':['03_03_16_41_47', '03_03_16_45_26', '03_03_18_35_58', '03_03_21_29_04', '03_04_01_33_54', '03_04_01_46_31'],
        # 'sigma':['03_03_16_56_52', '03_03_23_18_48', '03_04_02_42_13', '03_04_07_34_58', '03_04_12_28_46', '03_04_21_43_47']
    }
    if True:
        for key in folder_names:
            for value in folder_names[key]:

                if key == 'NRE_full_trawl':

                    trained_classifier_path = os.path.join(
                        os.getcwd(), 'models', 'new_classifier', 'NRE_full_trawl', value, 'best_model')
                else:
                    trained_classifier_path = os.path.join(
                        os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl', key, value, 'best_model')  # 'NRE_full_trawl '

                
                calibrate(trained_classifier_path, nr_batches, 1000)
                calibrate(trained_classifier_path, nr_batches, 1500)
                calibrate(trained_classifier_path, nr_batches, 2500)
                #calibrate(trained_classifier_path, nr_batches, 2000)
                #calibrate(trained_classifier_path, nr_batches, 3000)
                #calibrate(trained_classifier_path, nr_batches, 3500)


    # TRE options: acf, beta, mu, sigma

    # calibrate
    double_trained_classifier_path = os.path.join(
        os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl', 'selected_models')

    #calibrated_the_NRE_of_a_calibrated_TRE(
    #    double_trained_classifier_path, 1500)
