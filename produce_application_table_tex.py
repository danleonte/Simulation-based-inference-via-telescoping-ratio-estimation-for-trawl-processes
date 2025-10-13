import pandas as pd

# Read the CSV
df = pd.read_csv('summary_MAP_estimates.csv')  # Replace with your file path

# Generate LaTeX table
latex_code = r"""\begin{table}[htbp]
\centering
\caption{Parameter Estimates with 95\% Confidence Intervals}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Dataset & $\tau_{\text{decorr}}$ & ACF(1) & $\mu$ & $\sigma$ & $\beta$ \\
\midrule
"""

for _, row in df.iterrows():
    dataset = row['Dataset']
    decorr_time = int(row['Effective_Decorr_Time'])
    
    # Format each parameter with CI
    acf_lag1 = f"{row['Lag1_ACF']:.3f} [{row['Lag1_ACF_CI_lower']:.3f}, {row['Lag1_ACF_CI_upper']:.3f}]"
    mu = f"{row['Mu']:.0f} [{row['Mu_CI_lower']:.0f}, {row['Mu_CI_upper']:.0f}]"
    sigma = f"{row['Sigma']:.0f} [{row['Sigma_CI_lower']:.0f}, {row['Sigma_CI_upper']:.0f}]"
    beta = f"{row['Beta']:.3f} [{row['Beta_CI_lower']:.3f}, {row['Beta_CI_upper']:.3f}]"
    
    latex_code += f"{dataset} & {decorr_time} & {acf_lag1} & {mu} & {sigma} & {beta} \\\\\n"

latex_code += r"""\bottomrule
\end{tabular}%
}
\label{tab:parameters}
\end{table}"""

print(latex_code)
