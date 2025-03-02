# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:56:11 2025

@author: dleon
"""


import seaborn as sns
from scipy import stats
import numpy as np
pred_beta_x = iso.predict(flatten_beta_x)

f, ax = plt.subplots(figsize=(12, 12))
ax.scatter(flatten_beta_theta, pred_beta_x)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.show()


def distribution_matching(y_pred, y_true, new_predictions):
    """
    Calibrate new predictions using distribution matching

    Parameters:
    -----------
    y_true : array-like
        Ground truth values from training data
    y_pred : array-like
        Predicted values from training data (uncalibrated)
    new_predictions : array-like
        New predictions to calibrate

    Returns:
    --------
    array-like
        Calibrated predictions
    """
    # Create empirical CDFs
    def empirical_cdf(x, values):
        return np.mean(values <= x)

    # Find percentiles of new predictions in the training predictions
    percentiles = np.array([stats.percentileofscore(y_pred, pred) / 100.0
                           for pred in new_predictions])

    # Map percentiles to the true distribution
    y_true_sorted = np.sort(y_true)
    calibrated = np.array([y_true_sorted[int(p * len(y_true_sorted))
                          if p < 1.0 else len(y_true_sorted)-1]
                          for p in percentiles])

    return calibrated


pred_beta_x_with_cdf = distribution_matching(
    flatten_beta_x, flatten_beta_theta, flatten_beta_x)

f, ax = plt.subplots(figsize=(12, 12))
ax.scatter(flatten_beta_theta, pred_beta_x_with_cdf)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.show()


sns.kdeplot(x=flatten_beta_theta, y=pred_beta_x_with_cdf)
