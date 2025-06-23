# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 00:37:01 2025

@author: dleon
"""

import os
import numpy as np

uncal_acf_2000 = np.load(os.path.join(os.getcwd(
), 'results_seq_len_2000_num_rows_160', 'MLE_acf_estimation_error.npy'), allow_pickle=True)
uncal_mar_2000 = np.load(os.path.join(os.getcwd(
), 'results_seq_len_2000_num_rows_160', 'MLE_marginal_estimation_error.npy'), allow_pickle=True)


uncal_acf_1500 = np.load(os.path.join(os.getcwd(), 'results_seq_len_1500_num_rows_160',
                         'MLE_acf_estimation_error_1500_15.npy'), allow_pickle=True)
uncal_mar_1500 = np.load(os.path.join(os.getcwd(), 'results_seq_len_1500_num_rows_160',
                         'MLE_marginal_estimation_error_1500_15.npy'), allow_pickle=True)


cal_acf_2000 = np.load(os.path.join(os.getcwd(), 'point_estimators', 'calibrated_TRE',
                       'results_seq_len_2000_num_rows_160', 'MLE_acf_estimation_error.npy'), allow_pickle=True)
cal_mar_2000 = np.load(os.path.join(os.getcwd(), 'point_estimators', 'calibrated_TRE',
                       'results_seq_len_2000_num_rows_160', 'MLE_marginal_estimation_error.npy'), allow_pickle=True)

cal_acf_1500 = np.load(os.path.join(os.getcwd(), 'point_estimators', 'calibrated_TRE',
                       'results_seq_len_1500_num_rows_160', 'MLE_acf_estimation_error_1500_15.npy'), allow_pickle=True)
cal_mar_1500 = np.load(os.path.join(os.getcwd(), 'point_estimators', 'calibrated_TRE',
                       'results_seq_len_1500_num_rows_160', 'MLE_marginal_estimation_error_1500_15.npy'), allow_pickle=True)

cal_acf_1000 = np.load(os.path.join(os.getcwd(), 'point_estimators', 'calibrated_TRE',
                       'results_seq_len_1000_num_rows_160', 'MLE_acf_estimation_error.npy'), allow_pickle=True)
cal_mar_1000 = np.load(os.path.join(os.getcwd(), 'point_estimators', 'calibrated_TRE',
                       'results_seq_len_1000_num_rows_160', 'MLE_marginal_estimation_error.npy'), allow_pickle=True)


gmm_1000_15 = np.load(os.path.join(os.getcwd(), 'point_estimators',
                      'GMM', 'GMM_acf_estimation_error_1000_15.npy'), allow_pickle=True)


gmm_1500_15 = np.load(os.path.join(os.getcwd(), 'point_estimators',
                      'GMM', 'GMM_acf_estimation_error_1500_15.npy'), allow_pickle=True)
