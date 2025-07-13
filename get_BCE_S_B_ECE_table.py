# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 00:19:54 2025

@author: dleon
"""

import pandas as pd
import numpy as np
import os


def get_classifier_metrics(seq_len):

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

        # Then subset
        df_list.append(
            df.loc[['BCE', 'S', 'B', 'ECE_f'], ['uncal', 'beta']])

    return pd.concat(df_list, keys=tre_types, axis=1)


if __name__ == '__main__':

    seq_lengths = (1000, 1500, 2000)
    df_list = []
    for seq_len in seq_lengths:
        df_list.append(get_classifier_metrics(seq_len))

    df = pd.concat(df_list, keys=seq_lengths, axis=0)

    # load NRE results
