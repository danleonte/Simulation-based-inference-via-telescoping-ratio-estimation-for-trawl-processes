import os
import yaml
import pickle
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator


###############################################################################
###################### LOAD TRAINED MODELS FOR INFERENCE ######################
###################### POSTERIOR SAMPLING AND CHECKS ##########################
###############################################################################


def load_trained_models_for_posterior_inference(folder_path, dummy_x, trawl_process_type,
                                                use_tre, use_summary_statistics):
    """
    Loads the trained models, may it be NRE or TRE, and returns two functions,
    which approximate the log likelihood and log posterior. There are multiple
    possibilities: NRE / TRE, and within TRE, we may chose to use the full trawls
    or the summary statistics. If we use the full trawls, the TRE ACF uses
    the empirical acf as input, which complicates the loading of the trained models.

    In the training script, we passed the correct x to the model full trawl,
    summary_statistics or empirical_acf. We do the same here. to this end, we 
    first do some sanity checks and then load all models. we have an indicator
    use_empirical_acf. if set to true, we modify the input to the TRE ACF model.

    I m doing it like this because in MCMC / posterior inference / checks i 
    will use this for a fixed value fo x, and i don t want to recompute the 
    empirical acf each time, and also i don t want to return a list of models and deal with
    that in the posterior inference script.
    """

    assert dummy_x.ndim == 2  # should actually be a traw
    if trawl_process_type == 'sup_ig_nig_5p':
        dummy_theta = jnp.ones((dummy_x.shape[0], 5))
    else:
        raise ValueError

    if use_summary_statistics:

        project_trawl = get_projection_function()
        dummy_x = project_trawl(dummy_x)

    # if using tre, check there are exactly 4 subfolders, each with exactly one
    # set of params
    if use_tre:

        # List all items in the directory and filter only folders
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(
            os.path.join(folder_path, f))]
        assert len(folders) == 4 and set(folders) == set(
            ['acf', 'beta', 'mu', 'scale']) and folders[0] == 'acf'

        folders = [os.path.join(folder_path, f) for f in folders]

    else:
        folders = [folder_path]

    config_list = []
    calibration_list = []
    params_list = []
    model_list = []

    for i in range(len(folders)):
        folder = folders[i]

        params_path = [f for f in os.listdir(
            folder) if f.startswith("params") and f.endswith(".pkl")]
        assert len(params_path) == 1

        # Then load params, config and model
        with open(os.path.join(folder, params_path), 'rb') as file:
            params_list.append(pickle.load(file))

        with open(os.path.join(folder, 'calibration.pkl'), 'rb') as file:
            calibration_list.append(pickle.load(file))

        with open(os.path.join(folder, 'config.yaml'), 'r') as file:
            config_list.append(yaml.safe_load(file))
            # sanity check
            assert use_tre == config_list[-1]['tre_config']['use_tre']
            assert use_summary_statistics == config_list[-1]['tre_config']['use_summary_statistics']
            assert trawl_process_type == config_list[-1]['trawl_config']['trawl_process_type']

        model_ = get_model(config_list[-1], False)

        if use_tre:
            model_ = ExtendedModel(base_model=model_, trawl_process_type=trawl_process_type,
                                   tre_type=config_list[-1]['tre_config']['tre_type'],
                                   use_summary_statistics=use_summary_statistics)

        use_empirical_acf = False
        # Initialize
        if i == 0 and use_tre:
            acf_config = config_list[-1]
            acf_tre_config = acf_config['tre_config']
            assert folder[-3:] == 'acf'
            # can assume we are doing acf
            assert acf_tre_config['tre_type'] == 'acf'
            n_lags = acf_tre_config['nlags']

            if use_tre and (not use_summary_statistics) and acf_tre_config['replace_full_trawl_with_acf']:

                use_empirical_acf = True

                # empirical_acf_x = jnp.array(
                #    compute_empirical_acf(np.array(dummy_x[0]), nlags=n_lags)[1:])[jnp.newaxis, :]
                empirical_dummy_x = jnp.ones((dummy_x.shape[0], n_lags))

                _ = model_.init(PRNGKey(0), empirical_dummy_x, dummy_theta)

                # TO CHECK

            else:
                _ = model_.init(PRNGKey(0), dummy_x, dummy_theta)

        #############
        model_list.append(model_)
        #############

    ################ get bounds, assuming they're all the same ################
    acf_prior_hyperparams = config_list[-1]['trawl_config']['acf_prior_hyperparams']
    eta_bounds = acf_prior_hyperparams['eta_prior_hyperparams']
    gamma_bounds = acf_prior_hyperparams['gamma_prior_hyperparams']

    marginal_distr_hyperparams = config_list[-1]['trawl_config']['marginal_distr_hyperparams']
    mu_bounds = marginal_distr_hyperparams['loc_prior_hyperparams']
    scale_bounds = marginal_distr_hyperparams['scale_prior_hyperparams']
    beta_bounds = marginal_distr_hyperparams['beta_prior_hyperparams']
    # gamma, eta, mu, scale , beta
    bounds = (gamma_bounds, eta_bounds, mu_bounds, scale_bounds, beta_bounds)
    lower_bounds = jnp.array([i[0] for i in bounds])
    upper_bounds = jnp.array([i[1] for i in bounds])
    total_mass = jnp.prod(upper_bounds - lower_bounds)

    ###################### get functions ######################################

    if use_empirical_acf:

        @jax.jit
        def approximate_log_likelihood_to_evidence(x, empirical_acf_x, theta):

            log_r = 0

            for i in range(len(model_list)):

                model = model_list[i]
                params = params_list[i]
                config = config_list[i]
                calibration_dict = calibration_list[i]

                if i == 0 and use_empirical_acf:
                    x_to_use = empirical_acf_x

                else:
                    x_to_use = x

                log_r_to_add = model.apply(variables=params, x=x_to_use,
                                           theta=theta, train=False)

                if calibration_dict['use_beta_calibration']:

                    log_r_to_add = beta_calibrate_log_r(
                        log_r_to_add, calibration_dict['params'])

                log_r += log_r_to_add

            return log_r

        @jax.jit
        def approximate_log_posterior(x, empirical_acf_x, theta):

            log_likelihood = approximate_log_likelihood_to_evidence(
                x, empirical_acf_x, theta)
            in_bounds = jnp.all((theta > lower_bounds) &
                                (theta < upper_bounds))
            log_prior = jnp.where(in_bounds, - jnp.log(total_mass), -jnp.inf)
            return log_likelihood + log_prior

    else:

        @jax.jit
        def approximate_log_likelihood_to_evidence(x, theta):

            log_r = 0

            for i in range(len(model_list)):

                model = model_list[i]
                params = params_list[i]
                config = config_list[i]
                calibration_dict = calibration_list[i]

                log_r_to_add = model.apply(variables=params, x=x,
                                           theta=theta, train=False)

                if calibration_dict['use_beta_calibration']:

                    log_r_to_add = beta_calibrate_log_r(
                        log_r_to_add, calibration_dict['params'])

                log_r += log_r_to_add

            return log_r

        @jax.jit
        def approximate_log_posterior(x, theta):

            log_likelihood = approximate_log_likelihood_to_evidence(
                x, theta)
            in_bounds = jnp.all((theta > lower_bounds) &
                                (theta < upper_bounds))
            log_prior = jnp.where(in_bounds, - jnp.log(total_mass), -jnp.inf)
            return log_likelihood + log_prior

    return approximate_log_likelihood_to_evidence, approximate_log_posterior, \
        use_empirical_acf


if __name__ == '__main__':

    folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_summary_statistics\trial_set1'
    trawl_process_type = 'sup_ig_nig_5p'
    use_tre = True
    use_summary_statistics = True
    dummy_x = jnp.load('trawl.npy')
    dummy_theta = jnp.ones([dummy_x.shape[0], 5])
    # x = ....
