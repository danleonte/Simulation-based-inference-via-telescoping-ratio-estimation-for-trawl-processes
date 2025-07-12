# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 21:12:08 2025

@author: dleon
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from get_ecdf_statistics import summarize_ecdf_metrics, compare_uncal_cal_ranks


def load_ranks(classifier_type, seq_len):

    base = os.path.join(os.getcwd(), 'models', 'new_classifier',
                        'coverage_check_ranks_NRE_and_TRE')

    uncal_ranks = np.load(os.path.join(
        base, f'{classifier_type}_{seq_len}no_calibration.npy'))
    cal_ranks = np.load(os.path.join(
        base, f'{classifier_type}_{seq_len}beta_calibration.npy'))

    return uncal_ranks, cal_ranks


if __name__ == 'main__':

    d = dict()
    plot_difference = True

    for seq_len in (1000, 1500, 2000):  # 'acf',
        f, ax = plt.subplots()

        for classifier_type in ('NRE', 'TRE'):

            uncal_ranks, cal_ranks = load_ranks(classifier_type, seq_len)

            d[(classifier_type, seq_len)] = compare_uncal_cal_ranks(
                uncal_ranks, cal_ranks)

            num_ranks_uncal = len(uncal_ranks)
            num_ranks_cal = len(cal_ranks)

            sorted_uncal_ranks = np.sort(uncal_ranks)
            sorted_cal_ranks = np.sort(cal_ranks)
            ecdf_uncal = np.arange(1, num_ranks_uncal + 1) / num_ranks_uncal
            ecdf_cal = np.arange(1, num_ranks_cal + 1) / num_ranks_cal

            if plot_difference:

                ax.plot(sorted_uncal_ranks, ecdf_uncal - sorted_uncal_ranks,
                        label=f'{classifier_type}_uncal', linewidth=1, alpha=0.9)

                ax.plot(sorted_cal_ranks, ecdf_cal - sorted_cal_ranks,
                        label=f'{classifier_type}_cal', linewidth=1, alpha=0.9)

            ax.plot(np.linspace(0, 1, 100), np.zeros(100), alpha=0.9,
                    linewidth=1, linestyle='dashed')

        plt.title(seq_len)
        plt.legend()
