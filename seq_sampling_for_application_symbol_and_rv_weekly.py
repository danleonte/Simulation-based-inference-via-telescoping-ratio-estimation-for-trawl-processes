import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from src.utils.get_model import get_model
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
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

def aggregate_weekly(data, date_column, value_column):
    data = data.copy()
    print(f"Date column dtype before conversion: {data[date_column].dtype}")
    
    try:
        data[date_column] = pd.to_datetime(data[date_column], utc=True)
    except:
        data[date_column] = pd.to_datetime(data[date_column])
    
    data['week'] = data[date_column].dt.isocalendar().week
    data['year'] = data[date_column].dt.year
    data['year_week'] = data['year'].astype(str) + '_' + data['week'].astype(str)
    
    weekly_data = data.groupby(['Symbol', 'year_week'])[value_column].sum().reset_index()
    return weekly_data.pivot(index='year_week', columns='Symbol', values=value_column)

def run_analysis_for_symbol(symbol, data_column, column_name, base_path, is_weekly=False, seq_len=1500, is_differenced=False):
    tre_types_list = ['acf', 'mu', 'sigma', 'beta']
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_samples = 10**3
    batch_size_for_evaluating_x_cache = 1
    key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.split(vec_key, num_samples)
    dummy_x = jnp.ones([1, seq_len])
    calibration_type = 'beta'
    
    print(f"Using seq_len: {seq_len} for symbol {symbol}")
    
    # Prepare data
    data_log = data_column.apply(np.log)
    data_log = data_log.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Apply lag-5 differencing if requested
    if is_differenced:
        data_log = data_log.diff(5).dropna()
        print(f"After lag-5 differencing, length: {len(data_log)}")
    
    if is_weekly:
        # Use full length as single window
        x = jnp.array([data_log.values])
        window_indices = [0]  # Only one window
    else:
        window_size = seq_len
        step_size = 250
        first_end = seq_len
        ends = np.arange(first_end, len(data_log), step_size)
        starts = ends - seq_len
        window_indices = [0, 8, 16]
        
        x = []
        for start, end in zip(starts, ends):
            x_to_add = data_log.iloc[start:end].values
            x_to_add[np.isnan(x_to_add)] = np.nanmean(x_to_add)
            x.append(x_to_add)
        x = jnp.array(x).squeeze()
    
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
        
        with open(os.path.join(trained_classifier_path, f'beta_calibration_1000.pkl'), 'rb') as file:
            calibration_dict[tre_type] = pickle.load(file)['params']
        
        parameter_sweeps_dict[tre_type] = create_parameter_sweep_fn(
            tre_type, apply_fn_dict, bounds_dict, N+1)
        
        # Compute x_cache
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
        
        num_batches = x.shape[0] // batch_size_for_evaluating_x_cache
        x_batches = np.array_split(x, max(1, num_batches))
        x_cache_list = []
        
        for x_batch in x_batches:
            x_batch = (x_batch - jnp.mean(x_batch,axis=1,keepdims=True)) / jnp.std(x_batch,axis=1,keepdims=True)
            _, x_cache_to_append = apply_fn_to_get_x_cache(x_batch, jnp.ones((x_batch.shape[0],5)))
            x_cache_list.append(x_cache_to_append)
        
        x_cache_dict[tre_type] = jnp.concatenate(x_cache_list)
    
    def acf_integrate_partial(samples):
        return integrate_from_sampled(samples, a=bounds_dict['acf'][0], b=bounds_dict['acf'][1])
    vec_integrate_2nd_component_acf_from_sampled = jax.jit(jax.vmap(acf_integrate_partial))
    
    estimate_first_density = estimate_first_density_enclosure('acf', parameter_sweeps_dict, bounds_dict, N)
    
    # Run inference
    result_sample_list = []
    result_sample_MAP = []
    
    for i in tqdm(range(x.shape[0]), desc=f"Processing {symbol}"):
        tre_type = 'acf'
        true_x_cache = x_cache_dict[tre_type][i]
        
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
            x_cache_to_use = x_cache_dict[tre_type][i]
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
        
        result_sample_list.append(thetas_)
        result_sample_MAP.append(sample_densities)
    
    result_samples_array = np.array(result_sample_list)
    means, stds = np.mean(x, axis=1, keepdims=True), np.std(x, axis=1, keepdims=True)
    
    result_samples_array[:,:,2] = result_samples_array[:,:,2] * stds
    result_samples_array[:,:,2] = result_samples_array[:,:,2] + means
    result_samples_array[:,:,3] = result_samples_array[:,:,3] * stds
    
    argmax_list = [int(np.argmax(i)) for i in result_sample_MAP]
    results_MAP = np.array([result_samples_array[index][argmax_list[index]] for index in range(len(result_sample_list))])
    
    print(f"Results MAP shape: {results_MAP.shape}")  # Debug
    
    # Save results
    folder_suffix = '_weekly_agg' if is_weekly else ''
    if is_differenced:
        folder_suffix += '_diff5'
    symbol_path = os.path.join(base_path, column_name + folder_suffix, symbol)
    os.makedirs(symbol_path, exist_ok=True)
    
    np.save(os.path.join(symbol_path, 'posterior_samples.npy'), result_samples_array)
    np.save(os.path.join(symbol_path, 'MAP_results.npy'), results_MAP)
    
    # Save length info
    with open(os.path.join(symbol_path, 'length_info.txt'), 'w') as f:
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Sequence length used: {seq_len}\n")
        f.write(f"Number of windows: {x.shape[0]}\n")
        f.write(f"Data type: {'Weekly aggregated' if is_weekly else 'Daily'}\n")
        f.write(f"Differenced: {'Lag-5 differenced' if is_differenced else 'No differencing'}\n")
    
    # Create ACF comparison plots
    h = jnp.arange(1, 21, 1)
    
    # Plot 1: Posterior mean vs empirical
    plt.figure(figsize=(10, 6))
    for idx in window_indices:
        if idx < len(x):
            x_window = x[idx]
            empirical_acf = sm.tsa.stattools.acf(np.array(x_window), nlags=20)[1:]
            
            param_samples = result_samples_array[idx, :, :2]
            acf_curves = jax.vmap(lambda pars: corr_sup_ig_envelope(h, pars))(param_samples)
            posterior_mean = np.mean(np.array(acf_curves), axis=0)
            
            plt.plot(h, empirical_acf, 'o-', alpha=0.7, label=f'Empirical ACF (win {idx})')
            plt.plot(h, posterior_mean, '--', alpha=0.7, label=f'Posterior mean ACF (win {idx})')
    
    diff_suffix = ' (Diff5)' if is_differenced else ''
    plt.title(f'ACF Comparison (Mean) - {symbol} (seq_len={seq_len}){diff_suffix}')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(symbol_path, 'acf_comparison_mean.png'), dpi=150)
    plt.close()
    
    # Plot 2: Posterior median vs empirical
    plt.figure(figsize=(10, 6))
    for idx in window_indices:
        if idx < len(x):
            x_window = x[idx]
            empirical_acf = sm.tsa.stattools.acf(np.array(x_window), nlags=20)[1:]
            
            param_samples = result_samples_array[idx, :, :2]
            acf_curves = jax.vmap(lambda pars: corr_sup_ig_envelope(h, pars))(param_samples)
            posterior_median = np.median(np.array(acf_curves), axis=0)
            
            plt.plot(h, empirical_acf, 'o-', alpha=0.7, label=f'Empirical ACF (win {idx})')
            plt.plot(h, posterior_median, '--', alpha=0.7, label=f'Posterior median ACF (win {idx})')
    
    plt.title(f'ACF Comparison (Median) - {symbol} (seq_len={seq_len}){diff_suffix}')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(symbol_path, 'acf_comparison_median.png'), dpi=150)
    plt.close()
    
    # Plot 3: MAP vs empirical
    plt.figure(figsize=(10, 6))
    for idx in window_indices:
        if idx < len(x):
            x_window = x[idx]
            empirical_acf = sm.tsa.stattools.acf(np.array(x_window), nlags=20)[1:]
            
            map_params = results_MAP[idx][:2]
            map_acf = corr_sup_ig_envelope(h, map_params)
            
            plt.plot(h, empirical_acf, 'o-', alpha=0.7, label=f'Empirical ACF (win {idx})')
            plt.plot(h, map_acf, '--', alpha=0.7, label=f'MAP ACF (win {idx})')
    
    plt.title(f'ACF Comparison (MAP) - {symbol} (seq_len={seq_len}){diff_suffix}')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(symbol_path, 'acf_comparison_MAP.png'), dpi=150)
    plt.close()
    
    return result_samples_array, results_MAP

if __name__ == '__main__':
    column_to_use = 'bv_ss'
    base_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RV_panel')
    
    # Read main data file
    data = pd.read_excel(os.path.join(base_path, 'ox_man_vol_indices.xlsx'))
    date_column = data.columns[0]
    
    # Weekly aggregation
    weekly_data = aggregate_weekly(data, date_column, column_to_use)
    symbols = weekly_data.columns
    
    # Run analysis for both regular and differenced data
    for is_differenced in [False, True]:
        diff_suffix = '_diff5' if is_differenced else ''
        print(f"\n{'='*50}")
        print(f"Running {'DIFFERENCED' if is_differenced else 'REGULAR'} analysis")
        print(f"{'='*50}")
        
        all_results = {}
        
        for symbol in symbols:
            if symbol not in weekly_data.columns:
                continue
            symbol_data = weekly_data[symbol].dropna()
            
            if is_differenced and len(symbol_data) < 105:  # Need extra for differencing
                print(f"Skipping {symbol}: length {len(symbol_data)} too short for differencing")
                continue
            elif not is_differenced and len(symbol_data) < 100:
                print(f"Skipping {symbol}: length {len(symbol_data)} too short")
                continue
            
            seq_len = len(symbol_data) - (5 if is_differenced else 0)  # Account for differencing
            
            result_samples, results_MAP = run_analysis_for_symbol(
                symbol, symbol_data, column_to_use, base_path, 
                is_weekly=True, seq_len=seq_len, is_differenced=is_differenced
            )
            all_results[symbol] = {'samples': result_samples, 'MAP': results_MAP}
        
        # Create summary plots
        if all_results:
            column_path = os.path.join(base_path, column_to_use + '_weekly_agg' + diff_suffix)
            os.makedirs(column_path, exist_ok=True)
            
            # Plot lengths
            lengths_dict = {symbol: len(results['MAP']) for symbol, results in all_results.items()}
            
            plt.figure(figsize=(12, 6))
            plt.bar(lengths_dict.keys(), lengths_dict.values())
            plt.title(f'Data Length by Symbol - {column_to_use}{diff_suffix}')
            plt.xlabel('Symbol')
            plt.ylabel('Length')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(column_path, 'data_lengths.png'), dpi=150)
            plt.close()
            
            # 2D ACF parameters plot
            plt.figure(figsize=(10, 8))
            for symbol, results in all_results.items():
                map_results = results['MAP']
                if len(map_results) > 0:
                    plt.scatter(map_results[:, 0], map_results[:, 1], alpha=0.6, label=symbol, s=20)
            plt.xlabel('Gamma (ACF param 1)')
            plt.ylabel('Eta (ACF param 2)')
            plt.title(f'MAP ACF Parameters (Weekly{diff_suffix}) - {column_to_use}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(column_path, 'MAP_acf_parameters_2d.png'), dpi=150)
            plt.close()
            
            # 1D plots for means, variances, betas
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            for symbol, results in all_results.items():
                map_results = results['MAP']
                if len(map_results) > 0:
                    axes[0].plot(map_results[:, 2], alpha=0.7, label=symbol)
                    axes[1].plot(map_results[:, 3], alpha=0.7, label=symbol)
                    axes[2].plot(map_results[:, 4], alpha=0.7, label=symbol)
            
            axes[0].set_title(f'Means (Weekly{diff_suffix}) - {column_to_use}')
            axes[0].set_ylabel('Mean')
            axes[0].legend()
            
            axes[1].set_title(f'Variances (Weekly{diff_suffix}) - {column_to_use}')
            axes[1].set_ylabel('Variance')
            axes[1].legend()
            
            axes[2].set_title(f'Betas (Weekly{diff_suffix}) - {column_to_use}')
            axes[2].set_ylabel('Beta')
            axes[2].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(column_path, 'means_variances_betas_1d.png'), dpi=150)
            plt.close()