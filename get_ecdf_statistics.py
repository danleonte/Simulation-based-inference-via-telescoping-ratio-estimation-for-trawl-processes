# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:47:52 2025

@author: dleon
"""
import pandas as pd
import numpy as np
import os
from src.utils.ecdf_distances_from_samples import check_samples, wasserstein_1_analytical,\
    kolmogorov_smirnov_uniform, cramer_von_mises_uniform, anderson_darling_uniform
import matplotlib.pyplot as plt


def load_ranks(tre_type, seq_len, N):

    beta_path = os.path.join(os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl',
                             'selected_models', 'per_classifier_coverage_check', tre_type)
    uncal_ranks = np.load(os.path.join(
        beta_path, f'{tre_type}_uncal_ranks_seq_len_{seq_len}_N_{N}.npy'))
    cal_ranks = np.load(os.path.join(
        beta_path, f'{tre_type}_cal_ranks_seq_len_{seq_len}_N_{N}.npy'))

    return uncal_ranks, cal_ranks


def summarize_ecdf_metrics(ranks):

    return [wasserstein_1_analytical(ranks)] + list(kolmogorov_smirnov_uniform(ranks)[:2]) + \
        [cramer_von_mises_uniform(
            ranks), np.nan] + [anderson_darling_uniform(ranks, True)[1]['ad_alt'], np.nan]


def compare_uncal_cal_ranks(uncal_ranks, cal_ranks):

    columns = ['w1', 'ks', 'p-ks', 'cvm', 'p-cvm', 'ad', 'p-ad']
    rows = ['uncal', 'cal']
    data = [summarize_ecdf_metrics(
        uncal_ranks), summarize_ecdf_metrics(cal_ranks)]
    return pd.DataFrame(data, columns=columns, index=rows)


if __name__ == '__main__':

    d = dict()
    N = 256
    plot_difference = True

    # group by classifier

    for tre_type in ('beta', 'mu', 'sigma'):  # 'acf',
        f, ax = plt.subplots()
        for seq_len in (1000, 1500, 2000):

            uncal_ranks, cal_ranks = load_ranks(tre_type, seq_len, N)
            d[(tre_type, seq_len)] = compare_uncal_cal_ranks(
                uncal_ranks, cal_ranks)
            assert len(uncal_ranks) == len(cal_ranks)

            num_ranks = len(uncal_ranks)
            sorted_uncal_ranks = np.sort(uncal_ranks)
            sorted_cal_ranks = np.sort(cal_ranks)
            ecdf = np.arange(1, num_ranks + 1) / num_ranks

            if plot_difference:

                ax.plot(sorted_uncal_ranks, ecdf - sorted_uncal_ranks,
                        label=f'{seq_len}_uncal', linewidth=1, alpha=0.9)

                ax.plot(sorted_cal_ranks, ecdf - sorted_cal_ranks,
                        label=f'{seq_len}_cal', linewidth=1, alpha=0.9)

                ax.plot(np.linspace(0, 1, 100), np.zeros(100), alpha=0.9,
                        linewidth=1, linestyle='dashed')

            else:

                ax.plot(sorted_uncal_ranks, ecdf,
                        label=f'{seq_len}_uncal', linewidth=1, alpha=0.9)

                ax.plot(sorted_cal_ranks, ecdf,
                        label=f'{seq_len}_cal', linewidth=1, alpha=0.9)

                ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), alpha=0.9,
                        linewidth=1, linestyle='dashed')

        plt.title(tre_type)
        plt.legend()

    # group by length

    for seq_len in (1500,):  # (1000, 1500, 2000):  # 'acf',
        f, ax = plt.subplots()
        for tre_type in ('beta', 'mu', 'sigma', 'acf'):

            if tre_type == 'acf':
                N = 128
            uncal_ranks, cal_ranks = load_ranks(tre_type, seq_len, N)

            N = 256
            d[(tre_type, seq_len)] = compare_uncal_cal_ranks(
                uncal_ranks, cal_ranks)
            assert len(uncal_ranks) == len(cal_ranks)

            num_ranks = len(uncal_ranks)
            sorted_uncal_ranks = np.sort(uncal_ranks)
            sorted_cal_ranks = np.sort(cal_ranks)
            ecdf = np.arange(1, num_ranks + 1) / num_ranks

            if plot_difference:

                ax.plot(sorted_uncal_ranks, ecdf - sorted_uncal_ranks,
                        label=f'{tre_type}_uncal', linewidth=1, alpha=0.9)

                ax.plot(sorted_cal_ranks, ecdf - sorted_cal_ranks,
                        label=f'{tre_type}_cal', linewidth=1, alpha=0.9)

                ax.plot(np.linspace(0, 1, 100), np.zeros(100), alpha=0.9,
                        linewidth=1, linestyle='dashed')

            else:

                ax.plot(sorted_uncal_ranks, ecdf,
                        label=f'{tre_type}_uncal', linewidth=1, alpha=0.9)

                ax.plot(sorted_cal_ranks, ecdf,
                        label=f'{tre_type}_cal', linewidth=1, alpha=0.9)

                ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), alpha=0.9,
                        linewidth=1, linestyle='dashed')

        plt.title(seq_len)
        plt.legend()

    d[('beta', 1000)], d[('beta', 1500)], d[('beta', 2000)]
    d[('mu', 1000)], d[('mu', 1500)], d[('mu', 2000)]
    d[('sigma', 1000)], d[('sigma', 1500)], d[('sigma', 2000)]
