# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:33:40 2025

@author: dleon
"""
from reconstruct_beta_calibration import beta_calibrate_log_r
import jax.numpy as jnp
import numpy as np
import jax
from jax.nn import sigmoid
import os
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import optax
import pandas as pd


def compute_metrics(log_r, Y):

    extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
        logits=log_r, labels=Y)

    # this is due to numerical instability in the logit function and should be 0
    mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)

    # Replace values where mask is True with 0, otherwise keep original values
    extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)

    bce_loss = jnp.mean(extended_bce_loss)

    classifier_output = sigmoid(log_r)
    # half of them are 0s, half of them are 1, so we have to x2
    # S = 2 * jnp.mean(log_r * Y)
    S = jnp.mean(log_r[Y == 1])
    B = 2 * jnp.mean(classifier_output)
    accuracy = jnp.mean(
        (classifier_output > 0.5).astype(jnp.float32) == Y)

    return bce_loss, S, B  # , accuracy


def load_log_r_Y_TRE(seq_len, calibration):
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                             'models', 'new_classifier', 'TRE_full_trawl'
                             )
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    model_numberings = ['04_12_12_36_45', '04_12_04_26_56',
                        '04_12_00_32_46', '04_12_04_28_49']

    log_r = 0

    for tre_type, model_number in zip(tre_types, model_numberings):

        best_model_path = os.path.join(base_path, tre_type, model_number,
                                       'best_model', 'val_data_results')

        log_r_path = os.path.join(
            best_model_path, f'log_r_{seq_len}_{tre_type}.npy')
        # pred_prob_Y_path = os.path.join(
        #    best_model_path, f'pred_prob_Y_{seq_len}_{tre_type}.npy')
        Y_path = os.path.join(best_model_path, f'Y_{seq_len}_{tre_type}.npy')
        log_r_to_add = jnp.load(log_r_path).squeeze()

        if calibration:
            calibration_file_path = os.path.join(os.path.dirname(
                best_model_path), f'beta_calibration_{seq_len}_{tre_type}.pkl')
            cal_params = jnp.load(calibration_file_path, allow_pickle=True)
            log_r_to_add = beta_calibrate_log_r(
                log_r_to_add, cal_params['params'])

        log_r += log_r_to_add

        if tre_type == 'acf':
            Y = jnp.load(Y_path)
        else:
            Y_new = jnp.load(Y_path)
            assert jnp.all(Y == Y_new)

    return log_r, Y


def load_NRE_BCE_S_B(seq_len):
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                             'models', 'new_classifier', 'NRE_full_trawl',
                             '04_12_04_25_37', 'best_model', 'val_data_results')

    df_NRE = pd.read_excel(os.path.join(base_path, f'BCE_S_B_{seq_len}.xlsx'),
                           header=0,     # Use first row as column names
                           index_col=0)  # Use first column as row index)

    df_NRE = df_NRE[['uncal', 'beta']]
    df_NRE.columns = ['uncal', 'cal']
    return df_NRE


def final_metrics_NRE_and_TRE_validation(seq_len):
    """this will read the validaton NRE_metrics, produce the TRE metrics,
    because we validated each of the individual NREs within TREs but not
    the TRE: BCE, S, B. The  
    'NRE_TRE_coverage_figures_and_ecdf_metrics.py' script will produce the deviations
    from uniform posterior cdf
    """

    # load NRE BCE,S,B metrics
    df_NRE = load_NRE_BCE_S_B(seq_len)

    # load TRE BCE,S,B,mertics
    uncal_log_r_TRE, Y_TRE = load_log_r_Y_TRE(seq_len, False)
    cal_log_r_TRE, _ = load_log_r_Y_TRE(seq_len, True)

    uncal_TRE = jnp.array(compute_metrics(uncal_log_r_TRE, Y_TRE))
    cal_TRE = jnp.array(compute_metrics(cal_log_r_TRE, Y_TRE))

    df_TRE = pd.DataFrame(data=np.stack([uncal_TRE, cal_TRE], axis=1), index=('BCE', 'S', 'B'),
                          columns=('uncal', 'cal'))
    df_combined = pd.concat([df_NRE, df_TRE], axis=1, keys=['NRE', 'TRE'])

    return df_combined


if __name__ == '__main__':

    base_path_to_save_to = os.path.join(os.path.dirname(
        os.path.dirname(os.getcwd())), 'models', 'new_classifier')

    for seq_len in (1000, 1500, 2000):

        df = final_metrics_NRE_and_TRE_validation(seq_len)
        file_name = f'BCE_S_B_for_NRE_and_TRE_{seq_len}.xlsx'
        df.to_excel(os.path.join(base_path_to_save_to, file_name))
