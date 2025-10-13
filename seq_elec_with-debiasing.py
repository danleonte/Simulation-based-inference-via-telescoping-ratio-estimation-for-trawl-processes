import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from src.utils.get_model import get_model
from src.utils.estimate_bias import estimate_bias_from_MAP, estimate_bias_from_posterior_samples_batched
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.KL_divergence import convert_3_to_4_param_nig
import os
import pickle
from tqdm import tqdm
import numpy as np
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.chebyshev_utils import chebint_ab, interpolation_points_domain, integrate_from_sampled, polyfit_domain,  \
    vec_polyfit_domain, sample_from_coeff, chebval_ab_jax, vec_sample_from_coeff,\
    chebval_ab_for_one_x, vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes,\
    vec_integrate_from_samples
import statsmodels as sm
from sequential_posteror_sampling import create_parameter_sweep_fn, model_apply_wrapper, predict_2d, \
    apply_calibration, estimate_first_density_enclosure, get_cond_prob_at_true_value
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@jax.jit
def corr_sup_ig_envelope(h, params):
    gamma, eta = params
    return jnp.exp(eta * (1 - jnp.sqrt(1 + 2*h/gamma**2)))

def run_analysis_for_dataset(dataset_name, data_type, base_path):
    """
    Run analysis for a single dataset and data type (MSTL or R_deseasonalized)
    """
    tre_types_list = ['acf', 'mu', 'sigma', 'beta']
    calibration_seq_len = 2000  # For loading calibration file
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_samples = 10**3
    key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.split(vec_key, num_samples)
    calibration_type = 'beta'
    nlags = 40  # Number of lags for ACF
    
    # Load data based on type
    if data_type == 'MSTL':
        data_path = os.path.join(base_path, 'MSTL_results_14', dataset_name, f'{dataset_name}_MSTL.csv')
        data_df = pd.read_csv(data_path)
        data_column = data_df['resid']
    elif data_type == 'R_deseasonalized':
        data_path = os.path.join(base_path, 'R_deseasonalized_data', dataset_name, f'{dataset_name}_residuals_R.csv')
        data_df = pd.read_csv(data_path)
        data_column = data_df['residuals']
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Prepare data - use full series
    data_log = data_column
    data_log = data_log.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Use full series
    x = data_log.values
    x[np.isnan(x)] = np.nanmean(x)
    x = jnp.array(x).reshape(1, -1)  # Shape: (1, seq_len)
    
    seq_len = x.shape[1]
    print(f"Processing {dataset_name} ({data_type}) - Series length: {seq_len}")
    
    # Create dummy_x with actual sequence length for model loading
    dummy_x = jnp.ones([1, seq_len])
    
    # Load models
    models_dict = dict()
    apply_fn_dict = dict()
    appl_fn_to_get_x_cache_dict = dict()
    parameter_sweeps_dict = dict()
    calibration_dict = dict()
    x_cache_dict = dict()
    bounds_dict = {'acf': [10., 20.], 'beta': [-5., 5.], 'mu': [-1., 1.], 'sigma': [0.5, 1.5]}
    
    for tre_type in tre_types_list:
        trained_classifier_path = f'D:\\sbi_trawls\\SBI_for_trawl_processes_and_ambit_fields\\models\\new_classifier\\TRE_full_trawl\\selected_models\\{tre_type}'
        model, params, _, __bounds = load_one_tre_model_only_and_prior_and_bounds(
            trained_classifier_path, dummy_x, trawl_process_type, tre_type)
        
        models_dict[tre_type] = model
        apply_fn_to_get_x_cache, apply_fn = model_apply_wrapper(model, params)
        apply_fn_dict[tre_type] = apply_fn
        appl_fn_to_get_x_cache_dict[tre_type] = apply_fn_to_get_x_cache
        
        # Load calibration with length_2000
        calibration_file = os.path.join(trained_classifier_path, f'beta_calibration_{calibration_seq_len}.pkl')
        with open(calibration_file, 'rb') as file:
            calibration_dict[tre_type] = pickle.load(file)['params']
        
        parameter_sweeps_dict[tre_type] = create_parameter_sweep_fn(
            tre_type, apply_fn_dict, bounds_dict, N+1)
        
        # Compute x_cache for the full series
        x_normalized = (x - jnp.mean(x, axis=1, keepdims=True)) / jnp.std(x, axis=1, keepdims=True)
        _, x_cache = apply_fn_to_get_x_cache(x_normalized, jnp.ones((1, 5)))
        x_cache_dict[tre_type] = x_cache
    
    def acf_integrate_partial(samples):
        return integrate_from_sampled(samples, a=bounds_dict['acf'][0], b=bounds_dict['acf'][1])
    vec_integrate_2nd_component_acf_from_sampled = jax.jit(jax.vmap(acf_integrate_partial))
    
    estimate_first_density = estimate_first_density_enclosure('acf', parameter_sweeps_dict, bounds_dict, N)
    
    # Run inference for the single time series
    print(f"Running inference for {dataset_name} ({data_type})...")
    
    tre_type = 'acf'
    true_x_cache = x_cache_dict[tre_type][0]  # Only one time series
    
    two_d_log_prob = estimate_first_density(true_x_cache)
    two_d_prob = apply_calibration(two_d_log_prob, tre_type, calibration_type, calibration_dict)
    f_x = vec_integrate_2nd_component_acf_from_sampled(two_d_prob)
    cheb_coeff_f_x = polyfit_domain(f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
    
    key, subkey = jax.random.split(key)
    first_comp_samples = sample_from_coeff(cheb_coeff_f_x, subkey, bounds_dict[tre_type][0], bounds_dict[tre_type][1], num_samples)
    normalizing_constant_acf = integrate_from_sampled(f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
    first_comp_densities = chebval_ab_jax(first_comp_samples, cheb_coeff_f_x,
                                          bounds_dict[tre_type][0], bounds_dict[tre_type][1]) / normalizing_constant_acf
    
    thetas_ = jnp.zeros([num_samples, 5])
    thetas_ = thetas_.at[:, 0].set(first_comp_samples)
    sample_densities = jnp.copy(first_comp_densities)
    
    for col_index, tre_type in enumerate(tre_types_list, 1):
        x_cache_to_use = x_cache_dict[tre_type][0]
        x_cache_to_use_expanded = jnp.broadcast_to(x_cache_to_use, (num_samples, x_cache_to_use.shape[-1]))
        log_conditional_prob_at_cheb_knots = parameter_sweeps_dict[tre_type](thetas_, x_cache_to_use_expanded)
        
        conditional_prob_at_cheb_knots = apply_calibration(log_conditional_prob_at_cheb_knots, tre_type, calibration_type, calibration_dict)
        conditional_density_cheb_coeff = vec_polyfit_domain(conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        
        split_keys = jax.vmap(lambda k: jax.random.split(k, num=2))(vec_key)
        last_component_samples = vec_sample_from_coeff(conditional_density_cheb_coeff, vec_key, bounds_dict[tre_type][0], bounds_dict[tre_type][1], 1)
        vec_key = split_keys[:, 0]
        
        normalizing_constants = vec_integrate_from_samples(conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        conditional_prob = vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes(
            last_component_samples, conditional_density_cheb_coeff, bounds_dict[tre_type][0], bounds_dict[tre_type][1]).squeeze() / normalizing_constants
        
        sample_densities *= conditional_prob
        thetas_ = thetas_.at[:, col_index].set(last_component_samples.squeeze())
    
    # Transform back to original scale
    result_samples = np.array(thetas_)
    mean_x, std_x = np.mean(x), np.std(x)
    
    result_samples[:, 2] = result_samples[:, 2] * std_x + mean_x  # mu
    result_samples[:, 3] = result_samples[:, 3] * std_x  # sigma
    
    # Get MAP estimate
    argmax = int(np.argmax(sample_densities))
    results_MAP = result_samples[argmax]
    
    # Estimate bias using MAP parameters
    print(f"Estimating bias for {dataset_name} ({data_type})...")
    key, bias_key = jax.random.split(key)
    # Use normalized data for bias estimation
    observed_trawl = x_normalized.squeeze()
    num_replications_for_bias = 1000  # Adjust as needed
    mean_bias, lower_ci_bias, upper_ci_bias = estimate_bias_from_MAP(
        results_MAP, observed_trawl, num_replications_for_bias, nlags, bias_key
    )
    
    # Save results
    if data_type == 'MSTL':
        save_path = os.path.join(base_path, 'MSTL_results_14', dataset_name)
    else:
        save_path = os.path.join(base_path, 'R_deseasonalized_data', dataset_name)
    
    np.save(os.path.join(save_path, 'posterior_samples.npy'), result_samples)
    np.save(os.path.join(save_path, 'MAP_results.npy'), results_MAP)
    np.save(os.path.join(save_path, 'sample_densities.npy'), sample_densities)
    np.save(os.path.join(save_path, 'bias_estimates.npy'), {'mean': mean_bias, 'lower_ci': lower_ci_bias, 'upper_ci': upper_ci_bias})
    
# Create ACF plots
    h = jnp.arange(1, nlags+1, 1)
    
    # Calculate empirical ACF from the original series
    empirical_acf = sm.tsa.stattools.acf(np.array(x.squeeze()), nlags=nlags)[1:]
    
    # Calculate ACF from posterior samples
    param_samples = result_samples[:, :2]
    acf_curves = jax.vmap(lambda pars: corr_sup_ig_envelope(h, pars))(param_samples)
    posterior_mean = np.mean(np.array(acf_curves), axis=0)
    posterior_median = np.median(np.array(acf_curves), axis=0)
    
    # MAP ACF
    map_params = results_MAP[:2]
    map_acf = corr_sup_ig_envelope(h, map_params)
    
    # Compute bias using posterior samples
    print(f"Estimating bias from posterior samples for {dataset_name} ({data_type})...")
    key, posterior_bias_key = jax.random.split(key)
    mean_bias_posterior, lower_ci_bias_posterior, upper_ci_bias_posterior = estimate_bias_from_posterior_samples_batched(
        result_samples, observed_trawl, nlags, posterior_bias_key, 100
    )
    
    # Compute 95% CI from posterior ACF samples
    acf_lower_95 = np.percentile(np.array(acf_curves), 2.5, axis=0)
    acf_upper_95 = np.percentile(np.array(acf_curves), 97.5, axis=0)
    
    # Debiased empirical ACFs
    debiased_empirical_acf_map = empirical_acf - mean_bias[1:]
    debiased_empirical_acf_posterior = empirical_acf - mean_bias_posterior[1:]
    
    # Empirical CI bounds for MAP debiasing
    empirical_lower_map = empirical_acf - upper_ci_bias[1:]
    empirical_upper_map = empirical_acf - lower_ci_bias[1:]
    
    # Empirical CI bounds for posterior debiasing
    empirical_lower_posterior = empirical_acf - upper_ci_bias_posterior[1:]
    empirical_upper_posterior = empirical_acf - lower_ci_bias_posterior[1:]
    
    # Create the two debiasing methods comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: MAP-based debiasing
    ax1 = axes[0]
    # Debiased empirical with MAP and its CI
    ax1.plot(h, debiased_empirical_acf_map, 'o-', alpha=0.9, label='Empirical ACF'' #(MAP-debiased)', 
             color='darkblue', markersize=4, zorder=3)
    ax1.fill_between(h, empirical_lower_map, empirical_upper_map, alpha=0.2, color='darkblue', 
                     label='95% CI (empirical)', zorder=1)
    
    # MAP ACF with theoretical CI from posterior
    ax1.plot(h, map_acf, '-', linewidth=2, label='MAP ACF', color='red', zorder=3)
    ax1.fill_between(h, acf_lower_95, acf_upper_95, alpha=0.2, color='red', 
                     label='95% CI (theoretical)', zorder=1)
    
    # Posterior mean and median
    ax1.plot(h, posterior_mean, '--', alpha=0.7, label='Posterior mean', color='orange', linewidth=1.5)
    ax1.plot(h, posterior_median, ':', alpha=0.7, label='Posterior median', color='green', linewidth=1.5)
    
    ax1.set_xlabel('Lag', fontsize=12)
    ax1.set_ylabel('ACF', fontsize=12)
    ax1.set_title(f'MAP-based debiasing - {dataset_name} ({data_type})', fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.0)
    
    # Plot 2: Posterior-based debiasing
    ax2 = axes[1]
    # Debiased empirical with posterior and its CI
    ax2.plot(h, debiased_empirical_acf_posterior, 's-', alpha=0.9, 
             label='Empirical ACF (posterior-debiased)', color='darkgreen', markersize=4, zorder=3)
    ax2.fill_between(h, empirical_lower_posterior, empirical_upper_posterior, alpha=0.2, color='darkgreen', 
                     label='95% CI (empirical)', zorder=1)
    
    # MAP ACF with theoretical CI from posterior
    ax2.plot(h, map_acf, '-', linewidth=2, label='MAP ACF', color='red', zorder=3)
    ax2.fill_between(h, acf_lower_95, acf_upper_95, alpha=0.2, color='red', 
                     label='95% CI (theoretical)', zorder=1)
    
    # Posterior mean and median
    ax2.plot(h, posterior_mean, '--', alpha=0.7, label='Posterior mean', color='orange', linewidth=1.5)
    ax2.plot(h, posterior_median, ':', alpha=0.7, label='Posterior median', color='green', linewidth=1.5)
    
    ax2.set_xlabel('Lag', fontsize=12)
    ax2.set_ylabel('ACF', fontsize=12)
    ax2.set_title(f'Posterior-based debiasing - {dataset_name} ({data_type})', fontsize=13)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'acf_comparison_two_debiasing_methods.png'), dpi=150)
    plt.close()
    
    print(f"MAP parameters: gamma={results_MAP[0]:.3f}, eta={results_MAP[1]:.3f}, mu={results_MAP[2]:.3f}, sigma={results_MAP[3]:.3f}, beta={results_MAP[4]:.3f}")
    print(f"Mean bias at lag 1 - MAP method: {mean_bias[1]:.4f}")
    print(f"Mean bias at lag 1 - Posterior method: {mean_bias_posterior[1]:.4f}")
#####################
    
    # Create parameter posterior plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    param_names = ['gamma (ACF)', 'eta (ACF)', 'mu', 'sigma', 'beta']
    for i, param_name in enumerate(param_names):
        ax = axes[i // 3, i % 3]
        ax.hist(result_samples[:, i], bins=50, alpha=0.7, density=True)
        ax.axvline(results_MAP[i], color='red', linestyle='--', label='MAP')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior - {param_name}')
        ax.legend()
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.suptitle(f'Parameter Posteriors - {dataset_name} ({data_type})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'parameter_posteriors.png'), dpi=150)
    plt.close()
    
    print(f"Completed analysis for {dataset_name} ({data_type})")
    print(f"MAP parameters: gamma={results_MAP[0]:.3f}, eta={results_MAP[1]:.3f}, mu={results_MAP[2]:.3f}, sigma={results_MAP[3]:.3f}, beta={results_MAP[4]:.3f}")
    print(f"Mean bias at lag 1: {mean_bias[1]:.4f}")
    
    return result_samples, results_MAP

def main():
    """
    Main function to run analysis for all datasets
    """
    # Set base paths
    project_base = 'D:\\sbi_trawls\\SBI_for_trawl_processes_and_ambit_fields'
    data_base = 'D:\\sbi_trawls\\data\\electricity'
    
    # Change to project directory to ensure imports work
    os.chdir(project_base)
    
    # List of datasets to process
    datasets = ['CISO', 'ERCO', 'NYIS', 'PJM', 'MISO', 'FPL', 'DUK', 'AZPS', 'BPAT']
    
    # Process each dataset for both data types
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        all_results[dataset_name] = {}
        
        # Process MSTL results
        try:
            print(f"\nProcessing MSTL data for {dataset_name}...")
            samples_mstl, map_mstl = run_analysis_for_dataset(dataset_name, 'MSTL', data_base)
            all_results[dataset_name]['MSTL'] = {'samples': samples_mstl, 'MAP': map_mstl}
        except Exception as e:
            print(f"Error processing MSTL data for {dataset_name}: {e}")
        
        ## Process R_deseasonalized data
        #try:
        #    print(f"\nProcessing R_deseasonalized data for {dataset_name}...")
        #    samples_r, map_r = run_analysis_for_dataset(dataset_name, 'R_deseasonalized', data_base)
        #    all_results[dataset_name]['R_deseasonalized'] = {'samples': samples_r, 'MAP': map_r}
        #except Exception as e:
        #    print(f"Error processing R_deseasonalized data for {dataset_name}: {e}")
    
    # Create summary plots comparing all datasets
    print(f"\n{'='*60}")
    print("Creating summary plots...")
    print(f"{'='*60}")
    
    # Summary plot for ACF parameters
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for data_type_idx, data_type in enumerate(['MSTL', 'R_deseasonalized']):
        ax = axes[data_type_idx]
        for dataset_name in datasets:
            if dataset_name in all_results and data_type in all_results[dataset_name]:
                map_result = all_results[dataset_name][data_type]['MAP']
                ax.scatter(map_result[0], map_result[1], s=100, alpha=0.7, label=dataset_name)
        
        ax.set_xlabel('Gamma (ACF param 1)')
        ax.set_ylabel('Eta (ACF param 2)')
        ax.set_title(f'MAP ACF Parameters - {data_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_base, 'summary_acf_parameters.png'), dpi=150)
    plt.close()
    
    # Summary table of MAP estimates
    summary_data = []
    for dataset_name in datasets:
        for data_type in ['MSTL', 'R_deseasonalized']:
            if dataset_name in all_results and data_type in all_results[dataset_name]:
                map_result = all_results[dataset_name][data_type]['MAP']
                summary_data.append({
                    'Dataset': dataset_name,
                    'Data Type': data_type,
                    'Gamma': map_result[0],
                    'Eta': map_result[1],
                    'Mu': map_result[2],
                    'Sigma': map_result[3],
                    'Beta': map_result[4]
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(data_base, 'summary_MAP_estimates.csv'), index=False)
    print("\nSummary of MAP estimates saved to 'summary_MAP_estimates.csv'")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()