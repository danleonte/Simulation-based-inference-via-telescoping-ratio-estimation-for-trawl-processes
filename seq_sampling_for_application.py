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
    

from sequential_posteror_sampling import create_parameter_sweep_fn, model_apply_wrapper, predict_2d, \
    apply_calibration, estimate_first_density_enclosure, get_cond_prob_at_true_value
    
    
if __name__ == '__main__':


    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd 

    tre_types_list = ['acf', 'mu', 'sigma', 'beta']
    seq_len = 1000
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_samples = 5 * 10**3
    batch_size_for_evaluating_x_cache = 1
    key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.split(vec_key, num_samples)

    dummy_x = jnp.ones([1, seq_len])
    calibration_type = 'beta'

    assert calibration_type in ('None', 'beta', 'isotonic')

    ### load data ###
    #lead_to_use   = 0 
    #chanel_to_use = 'T3'
    #window_size = seq_len ############# do not redefine window_size to be different
    #step_size = 100
    #dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data') 
    #data = pd.read_csv(os.path.join(dataset_path, 'application_data.csv'))
    #data = data[data.Lead== lead_to_use][chanel_to_use]
    
    window_size = seq_len
    step_size = 168
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data') 
    data = pd.read_csv(os.path.join(dataset_path, 'd_h_temp_data_1999-01-01_2019-01-01_deseazonalized_both.csv'))
                                    #'VIX_25_y.csv'))['resid']#'normalized_temperature_data1.csv'))
    
    first_end = 2000
    ends = np.arange(first_end, len(data), step_size)
    starts = ends - seq_len 
    #last_end = ((len(data) - 2000)) // step_size * step_size
    #ends  = np.arange()
    
    x = []
    
    for start, end in zip(starts, ends):
        x.append(data['AIR_TEMPERATURE'].iloc[start:end].values)
        
    x = jnp.array(x).squeeze()

    ### load models and precompute x_cache ###
    models_dict = dict()
    apply_fn_dict = dict()
    appl_fn_to_get_x_cache_dict = dict()
    parameter_sweeps_dict = dict()
    #if calibration_type == 'beta':
    #    beta_calibration_params = dict()
#
    #elif calibration_type == 'isotonic':
    #    iso_calibration_dict = dict()
    calibration_dict = dict()

    bounds_dict = {'acf': [10., 20.], 'beta': [-5., 5.],
                   'mu': [-1., 1.], 'sigma': [0.5, 1.5]}
    


    for tre_type in tre_types_list:

        # trained_classifier_path = f'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/new_classifier/TRE_full_trawl/selected_models/{tre_type}'
        trained_classifier_path = f'D:\\sbi_trawls\\SBI_for_trawl_processes_and_ambit_fields\\models\\new_classifier\\TRE_full_trawl\\selected_models\\{tre_type}'
        model, params, _, __bounds = load_one_tre_model_only_and_prior_and_bounds(
            trained_classifier_path, dummy_x, trawl_process_type, tre_type)

        # load model
        models_dict[tre_type] = model

        # load apply_fn
        apply_fn_to_get_x_cache, apply_fn = model_apply_wrapper(model, params)
        apply_fn_dict[tre_type] = apply_fn
        appl_fn_to_get_x_cache_dict[tre_type] = apply_fn_to_get_x_cache

        # load calibratitons params
        if calibration_type == 'beta':
            with open(os.path.join(trained_classifier_path, f'beta_calibration_{seq_len}.pkl'), 'rb') as file:
                #beta_calibration_params[tre_type] = pickle.load(file)['params']
                calibration_dict[tre_type] = pickle.load(file)['params']

        elif calibration_type == 'isotonic':
            # Load the model
            with open(os.path.join(trained_classifier_path, f'fitted_iso_{seq_len}_{tre_type}.pkl'), 'rb') as file:
                #iso_calibration_dict[tre_type] = pickle.load(file)
                calibration_dict[tre_type] = pickle.load(file)


        # create parameter sweeps here:
        parameter_sweeps_dict[tre_type] = create_parameter_sweep_fn(
            tre_type, apply_fn_dict, bounds_dict, N+1)  # +1 just to make sure we do not confuse the first two acf components

        # Load x_cache or precmpue it and save it
        x_cache_path = os.path.join(
            dataset_path, f'x_cache_{tre_type}_window_size_{window_size}_step_size_{step_size}.npy')
        
        if os.path.exists(x_cache_path):
            x_cache = jnp.load(x_cache_path)

        else:
            
            num_batches = x.shape[0] // batch_size_for_evaluating_x_cache
            assert x.shape[0] % num_batches == 0, 'total number of datapoints is not divisible by the batch size'
            x_batches = np.array_split(x, num_batches)
            x_cache_list = []

            for x_batch in x_batches:
                
                x_batch = (x_batch - jnp.mean(x_batch,axis=1,keepdims=True)) / jnp.std(x_batch,axis=1,keepdims=True) 
                #don't forget to transform posterior samples at the end
                _, x_cache_to_append = apply_fn_to_get_x_cache(
                    x_batch, jnp.ones((batch_size_for_evaluating_x_cache,5)))
                x_cache_list.append(x_cache_to_append)

            x_cache_list = jnp.concatenate(x_cache_list)
            np.save(file= x_cache_path, arr= x_cache_list)
            del x_cache_list

    # load x_cache for each tre_type
    x_cache_dict = dict()
    for tre_type in tre_types_list:
        x_cache_dict[tre_type] = jnp.load(os.path.join(
            dataset_path, f'x_cache_{tre_type}_window_size_{window_size}_step_size_{step_size}.npy'))

    # to use with acf  ### ignore for now

    def acf_integrate_partial_enclosure(bounds_dict):
        
        def acf_integrate_partial(samples):
            return integrate_from_sampled(samples, a=bounds_dict['acf'][0], b=bounds_dict['acf'][1])
        vec_integrate_2nd_component_acf_from_sampled = jax.jit(
            jax.vmap(acf_integrate_partial))
        
        return acf_integrate_partial, vec_integrate_2nd_component_acf_from_sampled
    
    #added enclosures so i can call these functions from a different script, for the application
    estimate_first_density = estimate_first_density_enclosure(tre_type, parameter_sweeps_dict, bounds_dict, N)
    acf_integrate_partial, vec_integrate_2nd_component_acf_from_sampled  = acf_integrate_partial_enclosure(bounds_dict)

    #########################
    
    result_sample_list = []
    result_sample_MAP  = []
    
    for i in tqdm(range(x.shape[0])):

        # ACF sampling
        tre_type = 'acf'
        true_x_cache = x_cache_dict[tre_type][i]

        # get 2d log probabilities on a 2d cheb gridi
        two_d_log_prob = estimate_first_density(true_x_cache)
        # calibrate 2d grid probabilities
        two_d_prob = apply_calibration(
            two_d_log_prob, tre_type, calibration_type, calibration_dict) 
        # get 1d prob for the first component
        f_x = vec_integrate_2nd_component_acf_from_sampled(
            two_d_prob)  # vec_integrate_from_sampled(two_d_prob)
        # get 1d coeff
        cheb_coeff_f_x = polyfit_domain(
            f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        # sample from 1st dimension
        key, subkey = jax.random.split(key)
        first_comp_samples = sample_from_coeff(
            cheb_coeff_f_x, subkey, bounds_dict[tre_type][0], bounds_dict[tre_type][1], num_samples)
        normalizing_constant_acf = integrate_from_sampled(
            f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        first_comp_densities = chebval_ab_jax(first_comp_samples, cheb_coeff_f_x,
                                              bounds_dict[tre_type][0],
                                              bounds_dict[tre_type][1]) / normalizing_constant_acf

        #true_density = chebval_ab_for_one_x(true_theta[0, 0], cheb_coeff_f_x,
        #                                   bounds_dict[tre_type][0],
        #                                    bounds_dict[tre_type][1]) / normalizing_constant_acf
        # sampling of first component finished
        thetas_ = jnp.zeros([num_samples, 5])
        col_index = 0
        thetas_ = thetas_.at[:, col_index].set(first_comp_samples)
        sample_densities = jnp.copy(first_comp_densities)
        del true_x_cache

        # sequentially sample starting the 2nd acf component, then mu sigma beta
        for tre_type in tre_types_list:

            x_cache_to_use = x_cache_dict[tre_type][i]
            x_cache_to_use_expanded = jnp.broadcast_to(
                x_cache_to_use, (num_samples, x_cache_to_use.shape[-1]))
            log_conditional_prob_at_cheb_knots = parameter_sweeps_dict[tre_type](
                thetas_, x_cache_to_use_expanded)

            conditional_prob_at_cheb_knots = apply_calibration(
                log_conditional_prob_at_cheb_knots, tre_type, calibration_type, calibration_dict) 

            conditional_density_cheb_coeff = vec_polyfit_domain(
                conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])

            split_keys = jax.vmap(
                lambda k: jax.random.split(k, num=2))(vec_key)
            last_component_samples = vec_sample_from_coeff(
                conditional_density_cheb_coeff, vec_key, bounds_dict[tre_type][0], bounds_dict[tre_type][1], 1)
            vec_key = split_keys[:, 0]  # Use first split from each key

            normalizing_constants = vec_integrate_from_samples(
                conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
            conditional_prob = vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes(last_component_samples,
                                                                                                conditional_density_cheb_coeff,
                                                                                                bounds_dict[tre_type][0], bounds_dict[tre_type][1]).squeeze() / normalizing_constants
            # get true conditional probability
            #true_conditional_density = get_cond_prob_at_true_value(
            #    true_theta, x_cache_to_use, tre_type, parameter_sweeps_dict, bounds_dict, calibration_type, calibration_dict)

            sample_densities *= conditional_prob
            #true_density *= true_conditional_density

            col_index += 1
            thetas_ = thetas_.at[:, col_index].set(
                last_component_samples.squeeze())
            
        result_sample_list.append(thetas_)
        result_sample_MAP.append(sample_densities)

        #del true_density
        del sample_densities
        del conditional_prob
        
    result_samples_array = np.array(result_sample_list)
    
    means, stds = np.mean(x,axis=1, keepdims = True), np.std(x,axis=1, keepdims = True)
    # undo the transformation (x-mu) / std = y -> x = std * y +  mu
    
    result_samples_array[:,:,2] = result_samples_array[:,:,2] * stds
    result_samples_array[:,:,2] = result_samples_array[:,:,2] + means
    result_samples_array[:,:,3] = result_samples_array[:,:,3] * stds
    
    argmax_list = [int(np.argmax(i)) for i in result_sample_MAP]
    results_MAP = [result_samples_array[index][argmax_list[index]] for index in range(len(result_sample_list))]
    
    np.save(os.path.join(dataset_path,f'results_{seq_len}.npy'), result_samples_array)
    np.save(os.path.join(dataset_path,f'results_MAP_{seq_len}.npy'), results_MAP)

    #to be careful to only modify the mu and sigma columns
    #for i in range(result_sample_list):
        
        #x[i].mean() + x[i].std() * result_sample_list[i]
        
        #result_samples_array.append()
        
    #result_samples_array = jnp.stack(result_samples_array)
    #argmax_indicator = jnp.argmax(result_sample_MAP,axis=1)
        
        
        #jnp.stack(result_sample_list)
    #result_sample_MAP = jnp.stack(result_sample_MAP)
    #result_sample_MAP = [result_sample_MAP[i][jnp.argmax(result_sample_MAP,axis=1)[i]] for i in range(result_sample_MAP.shape[0])]
        