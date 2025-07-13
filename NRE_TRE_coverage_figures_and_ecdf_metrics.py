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


if __name__ == '__main__':

    d = dict()
    plot_difference = True

    # Define colors for the two classifier types
    colors = {'NRE': '#1f77b4', 'TRE': '#ff7f0e'}
    seq_lengths = [1000, 1500, 2000]

    # Create figure with 3 subplots using constrained_layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                             constrained_layout=True)

    for i, seq_len in enumerate(seq_lengths):
        ax = axes[i]

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
                # Uncalibrated: solid line, lighter alpha
                ax.plot(sorted_uncal_ranks, ecdf_uncal - sorted_uncal_ranks,
                        label=f'{classifier_type} uncal',
                        color=colors[classifier_type],
                        linestyle='-',
                        linewidth=1.5,
                        alpha=0.6)

                # Calibrated: prominent dashed line, full alpha
                ax.plot(sorted_cal_ranks, ecdf_cal - sorted_cal_ranks,
                        label=f'{classifier_type} cal',
                        color=colors[classifier_type],
                        linestyle=(0, (8, 4)),
                        linewidth=2,
                        alpha=0.9)

        # Reference line at y=0
        ax.plot(np.linspace(0, 1, 100), np.zeros(100),
                color='black', alpha=0.4, linewidth=1, linestyle=':')

        # Styling improvements
        ax.set_title(rf'$k$={seq_len}', fontsize=12, pad=10)

        # Only add x-label to the middle plot
        if i == 1:  # Middle subplot (index 1)
            ax.set_xlabel(r'Credible level $\alpha$', fontsize=10)

        # Only add y-label to the leftmost plot
        if i == 0:
            ax.set_ylabel(
                r'Coverage deviation $\alpha - \mathcal{C}_{\alpha}$', fontsize=10)

        # Only show legend on the rightmost plot, inside the plot area
        if i == len(seq_lengths) - 1:
            ax.legend(loc='upper right', frameon=True, fancybox=True,
                      shadow=True, fontsize=9, framealpha=0.9)

        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=9)

    path_to_save = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                'coverage_check_ranks_NRE_and_TRE',
                                'calibration_comparison.pdf')
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0.05)
