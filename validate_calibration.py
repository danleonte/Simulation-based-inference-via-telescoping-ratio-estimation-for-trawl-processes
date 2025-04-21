# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:18:28 2025

@author: dleon
"""
import pandas as pd
import jax.numpy as jnp
import jax
import distrax
import yaml
import os
import pickle
import matplotlib
import numpy as np
from jax.scipy.special import logit
from jax.nn import sigmoid
from src.utils.get_model import get_model
from src.model.Extended_model_nn import ExtendedModel, VariableExtendedModel
from jax.random import PRNGKey
import optax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt
    from calibrate import generate_dataset


def validate_calibrator(trained_classifier_path, nr_batches, seq_len):

    # Load config
    with open(os.path.join(trained_classifier_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(trained_classifier_path)))),  'validation_dataset_for_calibration')

    # load validation dataset
    val_x_path = os.path.join(dataset_path, 'cal_x.npy')
    val_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    val_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    if os.path.isfile(val_x_path) and os.path.isfile(val_thetas_path) and os.path.isfile(val_Y_path):
        print('Validation dataset already created')
    else:
        from copy import deepcopy
        print('Generating dataset')
        classifier_config_ = deepcopy(classifier_config)
        classifier_config_['trawl_config']['seq_len'] = 3000
        val_x, val_thetas, val_Y = generate_dataset(
            classifier_config_, nr_batches)
        print('Generated dataset')

        np.save(file=val_x_path, arr=val_x)
        np.save(file=val_thetas_path, arr=val_thetas)
        np.save(file=val_Y_path, arr=val_Y)

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
        sample_x = np.load(val_x_path, mmap_mode='r')[0]  # Get first sample
        sample_thetas = np.load(val_thetas_path, mmap_mode='r')[
            0]  # Get first sample

        # Convert to JAX for initialization only
        sample_x_jax = jnp.array(sample_x)
        sample_thetas_jax = jnp.array(sample_thetas)

        model = VariableExtendedModel(base_model=model, trawl_process_type=trawl_config['trawl_process_type'],
                                      tre_type=tre_type, use_summary_statistics=use_summary_statistics)

        # Initialize with single samples, not batches
        model.init(PRNGKey(0), sample_x_jax, sample_thetas_jax)

    del sample_x_jax
    del sample_x
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

    val_results_path = os.path.join(trained_classifier_path, 'val_results')
    os.makedirs(val_results_path, exist_ok=True)

    log_r_path = os.path.join(val_results_path,
                              f'val_log_r_{seq_len}.npy')
    pred_prob_Y_path = os.path.join(
        val_results_path, f'val_pred_prob_Y_{seq_len}.npy')
    Y_path = os.path.join(val_results_path, f'val_Y_{seq_len}.npy')

    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):
        # Load the data in batches and process
        # Use memmap for initial loading without pulling all data into memory
        val_x_array = np.load(val_x_path, mmap_mode='r')
        val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
        val_Y_array = np.load(val_Y_path, mmap_mode='r')

        # Get file info first to understand the data structure
        val_x_info = np.load(val_x_path, mmap_mode='r')
        batch_count = val_x_info.shape[0]
        samples_per_batch = val_x_info.shape[1]
        total_samples = batch_count * samples_per_batch

        # Initialize lists to collect results (we'll convert to arrays later)
        log_r_list = []
        pred_prob_Y_list = []
        Y_list = []

        # Process one batch at a time
        for i in range(batch_count):
            if i % 20 == 0:
                print(f"Processing batch {i} out of {batch_count}")

            # Load the current batch into memory
            # Make a copy to ensure it's in memory
            x_batch = np.array(val_x_array[i])
            theta_batch = np.array(val_thetas_array[i])
            y_batch = np.array(val_Y_array)

            if i == 0:
                assert x_batch.shape[-1] == 3000
                # assert theta_batch.shape[-1] == 3000
                # assert y_batch.shape[-1] == 3000

            # Convert to JAX arrays for processing
            x_batch_jax = jnp.array(x_batch)[:, :seq_len]
            theta_batch_jax = jnp.array(theta_batch)

            # Process with JAX
            log_r_batch, pred_prob_Y_batch = compute_log_r_approx(
                params, x_batch_jax, theta_batch_jax)

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
        # del x_batch_jax
        # del theta_batch_jax
        # del log_r_batch
        # del pred_prob_Y_batch

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

    if False:
        # perform isotonic regression, Beta and Plat scaling
        lr = LogisticRegression(C=99999999999)
        iso = IsotonicRegression(y_min=0.0001, y_max=0.9999)
        bc = BetaCalibration(parameters="abm")

        # These sklearn methods work with numpy arrays
        lr.fit(pred_prob_Y, np.array(Y))
        iso.fit(pred_prob_Y, np.array(Y))
        bc.fit(pred_prob_Y, np.array(Y))

        linspace = np.linspace(0.000199, 0.999999, 1000)
        pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1],
              iso.predict(linspace), bc.predict(linspace)]  # , spline.forward(linspace)]
        methods_text = ['logistic', 'isotonic', 'beta']  # , 'splines']

        # get calibrated datasets
        calibrated_pr = [lr.predict_proba(pred_prob_Y)[:, 1],
                         iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

        num_bins_to_try = (2, 3, 4, 5, 6, 8, 10, 12, 15)

        for num_bins_for_splines in num_bins_to_try:

            spline_params = jnp.load(os.path.join(trained_classifier_path,
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

        for i in range(len(methods_text)):
            # Convert one calibrated result at a time
            calibrated_pr_jax = jnp.array(calibrated_pr[i])
            logit_calibrated = logit(calibrated_pr_jax)
            metrics.append(compute_metrics(
                logit_calibrated, calibrated_pr_jax, Y_jax))
            # Free memory
            # del calibrated_pr_jax
            # del logit_calibrated

        df = pd.DataFrame(np.array(metrics).transpose(),
                          columns=['uncal', ' lr', 'isotonic',
                                   'beta'] + methods_text[3:],
                          index=('BCE', 'S', 'B'))

        df.to_excel(os.path.join(trained_classifier_path,
                    f'val_cal_BCE_S_B_{seq_len}_with_splines.xlsx'))


if __name__ == '__main__':
    nr_batches = 5500

    folder_names = {  # 'NRE_full_trawl': ['04_12_04_25_37', '04_11_23_17_38','04_12_05_18_53','04_12_12_34_53','04_12_00_24_16','04_11_23_37_05','04_12_02_18_27','04_12_08_21_48',
        # '04_11_20_15_48','04_12_11_52_49','04_12_11_21_25'],########['03_03_10_00_51', '03_03_13_37_12', '03_03_19_26_42', '03_04_02_34_16', '03_04_08_50_44', '03_04_13_48_26'],
        # 'acf': ['04_12_00_27_16', '04_11_20_26_03', '04_12_12_36_45', '04_12_08_35_46', '04_12_04_29_36', '04_12_03_36_28', '04_12_09_57_22', '04_12_00_25_13', '04_11_20_30_16',
        #        '04_12_13_11_32',  '04_12_06_47_42',   '04_11_23_40_38', '04_12_09_14_24', '04_11_20_32_11'],
        # 'beta': ['04_12_04_26_56','04_12_12_35_41','04_12_00_25_49','04_12_05_27_48','04_11_23_24_46','04_12_12_17_28','04_12_08_31_07','04_11_23_37_34','04_12_11_30_54',                                '04_11_20_30_16'],
        # 'mu': ['04_12_04_41_11','04_12_12_59_45','04_12_00_32_46','04_11_20_26_03','04_12_08_53_27','04_12_08_53_57','04_12_05_42_50','04_12_12_21_06'],
        # 'sigma': ['04_12_04_28_49','04_12_00_26_44','04_12_12_37_42','04_12_05_36_55','04_12_12_37_35','04_12_11_18_04','04_12_08_35_55','04_12_09_33_30','04_12_05_59_51',                                '04_11_20_26_03']
        # 'acf': ['04_12_12_36_45'],
        # 'beta': ['04_12_04_26_56'],
        # 'mu': ['04_12_00_32_46'],
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

                validate_calibrator(trained_classifier_path, nr_batches, 1000)
                validate_calibrator(trained_classifier_path, nr_batches, 1500)
                # validate_calibrator(trained_classifier_path, nr_batches, 2000)
                # validate_calibrator(trained_classifier_path, nr_batches, 2500)
                # validate_calibrator(trained_classifier_path, nr_batches, 3000)
                # validate_calibrator(trained_classifier_path, nr_batches, 3500)

                # validate_calibrator(trained_classifier_path, nr_batches, seq_len)
