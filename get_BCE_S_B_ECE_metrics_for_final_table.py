# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 00:19:54 2025

@author: dleon
"""

import pandas as pd
import numpy as np
import os
from get_ecdf_statistics import load_NRE_within_TRE_ranks, compare_uncal_cal_ranks
from NRE_TRE_coverage_figures_and_ecdf_metrics import load_NRE_TRE_ranks


def get_BCE_S_B_ECE_classifier_metrics_for_individual_NRE_within_TRE(seq_len, calibration_type):

    d = {1000: 12, 1500: 8, 2000: 5}
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
        df = df.rename(columns={calibration_type: 'cal'})

        # Then subset
        df_list.append(
            df.loc[['BCE', 'S', 'B', 'ECE_f'], ['uncal', 'cal']])

    return pd.concat(df_list, keys=tre_types, axis=1)


if __name__ == '__main__':

    seq_lengths = (1000, 1500, 2000)
    ecdf_metric_to_use = 'w1'
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    N = 128
    calibration_type = 'iso'
    df_list_1 = []  # posterior ecdf deviation
    df_list_2 = []  # BCE S B ECE

    # index_NRE_TRE = [('NRE', 'uncal'),('NRE',   'cal'), ('TRE', 'uncal'),('TRE',   'cal')]
    # index_acf

    # deviations from uniiform cdf of posterior samples
    for seq_len in seq_lengths:

        # NRE TRE: uncal first
        NRE_ = compare_uncal_cal_ranks(
            *load_NRE_TRE_ranks('NRE', seq_len))[ecdf_metric_to_use].values
        TRE_ = compare_uncal_cal_ranks(
            *load_NRE_TRE_ranks('TRE', seq_len))[ecdf_metric_to_use].values

        df_ = pd.DataFrame(np.concatenate([NRE_, TRE_]).reshape(1, -1),
                           columns=pd.MultiIndex.from_product(
                               [['NRE', 'TRE'], ['uncal', 'cal']]),
                           index=[ecdf_metric_to_use])

        # individual NRES within TRE
        l = []

        for tre_type in tre_types:
            l.append(compare_uncal_cal_ranks(
                *load_NRE_within_TRE_ranks(tre_type, seq_len, N))[ecdf_metric_to_use].values)

        df__ = pd.DataFrame(np.concatenate(l).reshape(1, -1),
                            columns=pd.MultiIndex.from_product(
                                [tre_types,  ['uncal', 'cal']]),
                            index=[ecdf_metric_to_use])

        df_list_1.append(pd.concat([df_, df__], axis=1))

    # BCE S B ECE F
    for seq_len in seq_lengths:
        # BCE, S, B, ECE_f for NRE and TRE
        path_2 = os.path.join(os.getcwd(), 'models', 'new_classifier',
                              f'BCE_S_B_for_NRE_and_TRE_{seq_len}_cal_type_{calibration_type}.xlsx')
        df_ = pd.read_excel(path_2, header=[0, 1],  index_col=0)

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
    metric_order = ['BCE', 'S', 'B', ecdf_metric_to_use, 'ECE_f']
    df = df.reindex(metric_order, level=1)

    excel_save_path = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                   f'final_table_{ecdf_metric_to_use}_cal_type_{calibration_type}.xlsx')
    tex_save_path = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                 f'final_table_{ecdf_metric_to_use}_cal_type_{calibration_type}.tex')
    df.to_excel(excel_save_path)
    df.to_latex(tex_save_path, float_format='%.3f')
