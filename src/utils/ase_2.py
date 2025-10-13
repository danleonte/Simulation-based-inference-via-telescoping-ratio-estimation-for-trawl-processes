# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:49:46 2025

@author: dleon
"""

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

# Define the stock tickers
tickers = ['AAPL', 'MSFT', 'PLTR', 'NVDA']

# Download the stock data for the last 2 years
print("Downloading 2 years of stock data...")

stock_data = {}
for ticker in tickers:
    stock_data[ticker] = yf.download(ticker, period="2y")

# Extract close prices and create a DataFrame
prices = pd.DataFrame()
for ticker in tickers:
    prices[ticker] = stock_data[ticker]['Close']

# Remove any NaN values
prices = prices.dropna()


# Calculate daily returns
returns = prices.pct_change().dropna()

print(f"\nReturns data shape: {returns.shape}")
print(f"Returns summary statistics:")
print(returns.describe())

# Compute the covariance matrix
covariance_matrix = returns.cov()

# Annualize the covariance matrix (multiply by 252 trading days)
annualized_covariance_matrix = covariance_matrix * 252

print(f"\nDaily Covariance Matrix:")
print(covariance_matrix)

print(f"\nAnnualized Covariance Matrix:")
print(annualized_covariance_matrix)

# Compute the vector of expected returns (mean returns)
expected_daily_returns = returns.mean()

# Annualize expected returns
expected_annual_returns = expected_daily_returns * 252

print(f"\nExpected Daily Returns:")
print(expected_daily_returns)

print(f"\nExpected Annual Returns:")
print(expected_annual_returns)

# Convert to numpy arrays for easier manipulation
cov_matrix_np = annualized_covariance_matrix.values
expected_returns_np = expected_annual_returns.values

print(f"\nCovariance Matrix (NumPy array):")
print(cov_matrix_np)

print(f"\nExpected Returns Vector (NumPy array):")
print(expected_returns_np)

# Optional: Display correlation matrix for additional insight
correlation_matrix = returns.corr()
print(f"\nCorrelation Matrix:")
print(correlation_matrix)

# Optional: Plot the cumulative returns

cumulative_returns = (1 + returns).cumprod()
plt.figure(figsize=(12, 8))
for ticker in tickers:
    plt.plot(cumulative_returns.index,
             cumulative_returns[ticker], label=ticker, linewidth=2)

plt.title('Cumulative Returns - Last 2 Years')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save results to CSV files (optional)
returns.to_csv('daily_returns.csv')
covariance_matrix.to_csv('covariance_matrix.csv')
expected_daily_returns.to_csv('expected_returns.csv')
