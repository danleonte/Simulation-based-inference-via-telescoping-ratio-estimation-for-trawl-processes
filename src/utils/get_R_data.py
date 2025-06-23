# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 20:56:48 2025

@author: dleon
"""

from statsmodels.tsa.stattools import acf as compute_empirical_acf
import matplotlib.pyplot as plt
from src.utils.acf_functions import get_acf
import numpy as np
import os

# run from sbi_ambit\SBI_for_trawl_processes_and_ambit_fields


def save_first_5000_rows_for_r():

    val_path = os.path.join(
        os.getcwd(), 'models', 'val_dataset')
    r_folder_path = os.path.join(val_path, 'R_folder_nonparametric')
    os.makedirs(r_folder_path, exist_ok=True)

    for seq_len in (1000, 1500, 2000):

        folder_path = os.path.join(val_path, f'val_dataset_{seq_len}')

        val_thetas_joint = np.load(os.path.join(folder_path, 'val_thetas_joint.npy'),
                                   mmap_mode='r')[:80]
        val_x_joint = np.load(os.path.join(folder_path, 'val_x_joint.npy'),
                              mmap_mode='r')[:80]

        val_x_joint = val_x_joint.reshape(-1, seq_len)[:5000]
        val_thetas_joint = val_thetas_joint.reshape(
            -1, val_thetas_joint.shape[-1])[:5000]

        np.savetxt(os.path.join(r_folder_path,
                                f'val_thetas_joint_{seq_len}.csv'), val_thetas_joint, delimiter=',')
        np.savetxt(os.path.join(r_folder_path,
                                f'val_x_joint_{seq_len}.csv'), val_x_joint, delimiter=',')


# nr_to_use = 1
# f,ax = plt.subplots()
# true_theta = val_thetas_joint[nr_to_use]
# true_x = val_x_joint[nr_to_use]
# H = np.arange(1, 21)
# acf_func = get_acf('sup_IG')
# theoretical_acf = acf_func(H, true_theta[:2])
# ax.plot(H,theoretical_acf)
# ax.plot(H,compute_empirical_acf(true_x,nlags = 20)[1:])
save_first_5000_rows_for_r()
