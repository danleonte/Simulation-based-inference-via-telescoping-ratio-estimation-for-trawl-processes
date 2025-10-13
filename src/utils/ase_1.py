# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:48:26 2025

@author: dleon
"""

import jax
import jax.numpy as jnp
import os
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

seq_len = 2000
nlags = 10

data = jnp.load(os.path.join(
    f'val_dataset_{seq_len}', 'val_x_joint.npy'))[:, 0, :]
pacf_results = jnp.array([pacf(i, nlags=nlags)[1:] for i in data])

# Put into a DataFrame
df = pd.DataFrame(pacf_results, columns=[f"Col {i+1}" for i in range(nlags)])

# Plot with seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xlabel("Columns")
plt.ylabel("Values")
plt.title("Boxplot of Each Column")
plt.show()
