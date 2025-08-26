# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 23:41:26 2025

@author: dleon
"""

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
def load_beta_cal_and_iso_regression_TRE_ranks(seq_len):

    base = os.path.join(os.getcwd(), 'models', 'new_classifier',
                        'coverage_check_ranks_NRE_and_TRE')

    beta_ranks = np.load(os.path.join(
        base, f'seq_sampling_TRE_{seq_len}_beta_128_160.npy'))
    iso_ranks = np.load(os.path.join(
        base, f'seq_sampling_TRE_{seq_len}_isotonic_128_160.npy'))

    return beta_ranks, iso_ranks


if __name__ == '__main__':

    d = dict()
    plot_difference = True

    # Define colors for the two classifier types
    colors = {'beta': '#1f77b4', 'isotonic': '#ff7f0e'}
    seq_lengths = [1000, 1500, 2000]

    # Create figure with 3 subplots using constrained_layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                             constrained_layout=True)

    # Store line objects for custom legend
    legend_elements = []

    for i, seq_len in enumerate(seq_lengths):
        ax = axes[i]

        beta_ranks, iso_ranks = load_beta_cal_and_iso_regression_TRE_ranks(
            seq_len)
       # d[('TRE', seq_len)] = compare_uncal_cal_ranks(
       #     beta_ranks, iso_ranks)

        num_ranks_beta = len(beta_ranks)
        num_ranks_iso = len(iso_ranks)
        sorted_beta_ranks = np.sort(beta_ranks)
        sorted_iso_ranks = np.sort(iso_ranks)
        ecdf_beta = np.arange(0, num_ranks_beta) / num_ranks_beta
        ecdf_iso = np.arange(0, num_ranks_iso) / num_ranks_iso

        if plot_difference:
            # Uncalibrated: solid line, lighter alpha
            line_uncal = ax.plot(sorted_beta_ranks, ecdf_beta - sorted_beta_ranks,
                                 label=f'beta-calibrated',
                                 color=colors['beta'],
                                 linestyle='-',
                                 linewidth=2,
                                 alpha=0.6)[0]

            # Calibrated: prominent dashed line, full alpha
            line_cal = ax.plot(sorted_iso_ranks, ecdf_iso - sorted_iso_ranks,
                               label=f'isotonic regression',
                               color=colors['isotonic'],
                               linestyle='--',  # Use simple dashed style
                               linewidth=2.5,
                               alpha=0.9)[0]

            # Store lines for legend (only from first subplot to avoid duplicates)
            if i == 0:
                legend_elements.append(line_uncal)
                legend_elements.append(line_cal)

                # ax.axhline(y=0.025, color='gray', linestyle=':',
                #           alpha=0.3, linewidth=0.8)
                # ax.axhline(y=-0.025, color='gray', linestyle=':',
                #           alpha=0.3, linewidth=0.8)

        # Reference line at y=0
        ax.plot(np.linspace(0, 1, 100), np.zeros(100),
                color='black', alpha=0.4, linewidth=1, linestyle=':')

        # Styling improvements
        ax.set_title(rf'$k$={seq_len}', fontsize=13, pad=10)

        # Only add x-label to the middle plot
        if i == 1:  # Middle subplot (index 1)
            ax.set_xlabel(r'Theoretical coverage level $\alpha$', fontsize=13)

        # Only add y-label to the leftmost plot
        if i == 0:
            ax.set_ylabel(
                r'Coverage deviation $\mathcal{C}_{\alpha} - \alpha$', fontsize=13)

        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=13)

    # Create a custom legend on the rightmost subplot with proper line styles
    labels = ['beta-calibrated TRE', 'isotonic regression TRE']
    axes[-1].legend(legend_elements, labels,
                    loc='lower right',
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    fontsize=12,
                    framealpha=0.95,
                    handlelength=3.0,  # Make legend lines longer to show dash pattern
                    handletextpad=0.8)  # Add some space between line and text

    fig.suptitle(
        'TRE Coverage comparison for beta and isotonic calibration methods', fontsize=14)

    path_to_save = os.path.join(os.getcwd(), 'models', 'new_classifier',
                                'coverage_check_ranks_NRE_and_TRE', 'TRE'
                                'calibration_comparison_beta_iso.pdf')
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0.05, dpi=900)
    plt.show()  # Add this to see the plot
