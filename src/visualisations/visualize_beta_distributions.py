# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:34:40 2024

@author: dleon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.linspace(0, 1, 500)

# Parameters
alpha_beta_flat = [1.3, 1.3]  # Very flat in the middle
alpha_beta_skew = [1.1, 1.1]  # Slight skew

# Compute PDFs
pdf_flat = beta.pdf(x, *alpha_beta_flat)
pdf_skew = beta.pdf(x, *alpha_beta_skew)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_flat,
         label=f"α={alpha_beta_flat[0]}, β={alpha_beta_flat[1]} (Flat Middle)")
plt.plot(x, pdf_skew,
         label=f"α={alpha_beta_skew[0]}, β={alpha_beta_skew[1]} (Slight Skew)")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Beta Distribution with Flat Middle and Rapid Decay")
plt.legend()
plt.show()
