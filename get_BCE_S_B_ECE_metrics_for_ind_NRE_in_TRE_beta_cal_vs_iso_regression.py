# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 00:19:54 2025

@author: dleon
"""

import pandas as pd
import numpy as np
import os
from get_ecdf_statistics import compare_uncal_cal_ranks  # load_NRE_within_TRE_ranks
# from NRE_TRE_coverage_figures_and_ecdf_metrics import load_uncal_and_beta_cal_NRE_TRE_ranks


def load_beta_cal_and_iso_regression_TRE_ranks(seq_len):

    base = os.path.join(os.getcwd(), 'models', 'new_classifier',
                        'coverage_check_ranks_NRE_and_TRE')

    beta_ranks = np.load(os.path.join(
        base, f'seq_sampling_TRE_{seq_len}_beta_128_160.npy'))
    iso_ranks = np.load(os.path.join(
        base, f'seq_sampling_TRE_{seq_len}_isotonic_128_160.npy'))

    return beta_ranks, iso_ranks


def load_NRE_within_TRE_ranks_beta_vs_iso(tre_type, seq_len, N):

    beta_path = os.path.join(os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl',
                             'selected_models', 'per_classifier_coverage_check', tre_type)
    beta_ranks = np.load(os.path.join(
        beta_path, f'{tre_type}_cal_ranks_beta_seq_len_{seq_len}_N_{N}.npy'))
    iso_ranks = np.load(os.path.join(
        beta_path, f'{tre_type}_cal_ranks_isotonic_seq_len_{seq_len}_N_{N}.npy'))

    return beta_ranks, iso_ranks


def get_BCE_S_B_ECE_classifier_metrics_for_individual_NRE_within_TRE(seq_len):

    # d = {1000: 9, 1500: 7, 2000: 5}
    d = {1000: 5, 1500: 5, 2000: 5}
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    model_numbers = ('04_12_12_36_45', '04_12_04_26_56',
                     '04_12_00_32_46', '04_12_04_28_49')
    base_path = os.path.join(os.getcwd(), 'models', 'new_classifier')

    # TRE_ paths
    df_list = []
    for tre_type, model_number in zip(tre_types, model_numbers):
        TRE_path = os.path.join(base_path, 'TRE_full_trawl',
                                tre_type, model_number, 'best_model', 'val_data_results')
        df = pd.read_excel(os.path.join(TRE_path, f'BCE_S_B_{seq_len}_{tre_type}_with_splines_{d[seq_len]}_{20}.xlsx'),
                           header=0).set_index('Unnamed: 0')
        df.index.name = None
        # df = df.rename(columns={calibration_type: 'cal'})

        # Then subset
        df_list.append(
            df.loc[['BCE', 'S', 'B', 'ECE_t'], ['beta', 'iso']])

    return pd.concat(df_list, keys=tre_types, axis=1)


if __name__ == '__main__':

    seq_lengths = (1000, 1500, 2000)
    ecdf_metric_to_use = 'w1'
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    N = 128
    # calibration_type = 'beta'
    df_list_1 = []  # posterior ecdf deviation
    df_list_2 = []  # BCE S B ECE

    # index_NRE_TRE = [('NRE', 'uncal'),('NRE',   'cal'), ('TRE', 'uncal'),('TRE',   'cal')]
    # index_acf

    # deviations from uniiform cdf of posterior samples
    for seq_len in seq_lengths:

        # NRE TRE: uncal first
        TRE_ = compare_uncal_cal_ranks(
            *load_beta_cal_and_iso_regression_TRE_ranks(seq_len))[ecdf_metric_to_use].values

        df_ = pd.DataFrame(np.array(TRE_).reshape(1, -1),
                           columns=pd.MultiIndex.from_product(
                               [['TRE'], ['beta', 'iso']]),
                           index=[ecdf_metric_to_use])

        # individual NRES within TRE
        l = []

        for tre_type in tre_types:
            l.append(compare_uncal_cal_ranks(
                *load_NRE_within_TRE_ranks_beta_vs_iso(tre_type, seq_len, N))[ecdf_metric_to_use].values)

        df__ = pd.DataFrame(np.concatenate(l).reshape(1, -1),
                            columns=pd.MultiIndex.from_product(
                                [tre_types,  ['beta', 'iso']]),
                            index=[ecdf_metric_to_use])

        df_list_1.append(pd.concat([df_, df__], axis=1))

    # BCE S B ECE F
    for seq_len in seq_lengths:
        # BCE, S, B, ECE_f for NRE and TRE
        path_2_beta = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                   f'BCE_S_B_for_NRE_and_TRE_{seq_len}_beta.xlsx')
        path_2_isotonic = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                       f'BCE_S_B_for_NRE_and_TRE_{seq_len}_isotonic.xlsx')

        df_beta = pd.read_excel(path_2_beta, header=[0, 1],  index_col=0)
        df_isotonic = pd.read_excel(
            path_2_isotonic, header=[0, 1],  index_col=0)

        df_ = pd.DataFrame(np.transpose([df_beta.TRE.cal.values, df_isotonic.TRE.cal.values]),
                           index=df_isotonic.index,
                           columns=pd.MultiIndex.from_product(
            [['TRE'], ['beta', 'iso']]
        ))

        # BCE, S, B, ECE_f for individual NRES within TRE
        df__ = get_BCE_S_B_ECE_classifier_metrics_for_individual_NRE_within_TRE(
            seq_len)

        df_list_2.append(pd.concat([df_, df__], axis=1))

    df_list = []
    for i in (0, 1, 2):
        df_list.append(pd.concat([df_list_2[i], df_list_1[i]], axis=0))

    df = pd.concat(df_list, keys=seq_lengths, axis=0)

#    # Reorder columns
#    new_order = [('acf', 'uncal'), ('acf', 'cal'),
#                 ('beta', 'uncal'), ('beta', 'cal'),
#                 ('mu', 'uncal'), ('mu', 'cal'),
#                 ('sigma', 'uncal'), ('sigma', 'cal'),
#                 ('NRE', 'uncal'), ('NRE', 'cal'),
#                 ('TRE', 'uncal'), ('TRE', 'cal')]
#    df = df[new_order]

    # Reorder rows
    metric_order = ['BCE', 'S', 'B', ecdf_metric_to_use, 'ECE_t']
    df = df.reindex(metric_order, level=1)

    excel_save_path = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                   f'final_table_{ecdf_metric_to_use}_beta_vs_iso.xlsx')
    tex_save_path = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                 f'final_table_{ecdf_metric_to_use}_beta_vs_iso.tex')
    import math

    # Excel: rounds automatically
    df.to_excel(excel_save_path, float_format="%.3f")

    # LaTeX: custom formatter with NaN handling
    df.to_latex(
        tex_save_path,
        float_format=lambda x: f"{x:.3f}" if pd.notnull(x) else "--"
    )
