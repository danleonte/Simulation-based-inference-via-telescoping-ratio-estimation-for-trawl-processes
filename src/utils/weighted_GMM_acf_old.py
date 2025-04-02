if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.acf_functions import get_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as compute_empirical_acf
# from src.utils.modified_GMM_class import GMM
from statsmodels.sandbox.regression.gmm import GMM

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def acf_moment_conditions(params, trawl, num_lags, acf_func):
    acf_gamma, acf_eta = params
    # print(f"ACF params: gamma={acf_gamma}, eta={acf_eta}")

    # Compute demeaned series for ACF calculation
    demeaned_trawl = trawl - np.mean(trawl)
    variance = np.var(trawl)
    # print(f"Trawl variance: {variance}")

    # Check for NaN or Inf in demeaned_trawl
   # print(f"demeaned_trawl contains NaN: {np.any(np.isnan(demeaned_trawl))}")
    # print(f"demeaned_trawl contains Inf: {np.any(np.isinf(demeaned_trawl))}")

    # Initialize array for ACF errors
    acf_errors = np.zeros((len(trawl) - num_lags, num_lags))

    for k in range(1, num_lags + 1):
        # Calculate product of lagged values
        prod = demeaned_trawl[:-k] * demeaned_trawl[k:]
        # print(f"Lag {k} - prod contains NaN: {np.any(np.isnan(prod))}")
        # print(f"Lag {k} - prod contains Inf: {np.any(np.isinf(prod))}")

        # Calculate empirical products
        empirical_products = prod / variance
        # print(
        #    f"Lag {k} - empirical_products contains NaN: {np.any(np.isnan(empirical_products))}")
        # print(
        #    f"Lag {k} - empirical_products contains Inf: {np.any(np.isinf(empirical_products))}")

        # Calculate theoretical ACF
        theoretical_acf = acf_func(k, np.array([acf_gamma, acf_eta]))
        # print(f"Lag {k} - theoretical_acf: {theoretical_acf}")
        # print(f"Lag {k} - theoretical_acf is NaN: {np.isnan(theoretical_acf)}")
        # print(f"Lag {k} - theoretical_acf is Inf: {np.isinf(theoretical_acf)}")

        # Calculate error
        error = empirical_products[:len(trawl) - num_lags] - theoretical_acf
        # print(f"Lag {k} - error contains NaN: {np.any(np.isnan(error))}")
        # print(f"Lag {k} - error contains Inf: {np.any(np.isinf(error))}")

        acf_errors[:, k - 1] = error

    # Final check
   # print(f"acf_errors contains NaN: {np.any(np.isnan(acf_errors))}")
    # print(f"acf_errors contains Inf: {np.any(np.isinf(acf_errors))}")

    return acf_errors


class ACFGMM(GMM):
    def __init__(self, endog, exog, instrument, num_lags, trawl_function_name):
        super().__init__(endog, exog, instrument)
        self.num_lags = num_lags
        self.acf_func = get_acf(trawl_function_name)

    def momcond(self, params):
        try:
            moment_errors = acf_moment_conditions(
                params, self.endog, self.num_lags, self.acf_func)

            # Check more thoroughly and print debug info
            has_nan = np.any(np.isnan(moment_errors))
            has_inf = np.any(np.isinf(moment_errors))

            if has_nan or has_inf:
                print(
                    f"WARNING: Found NaN ({has_nan}) or Inf ({has_inf}) in moment errors with params {params}")
                # Return a large but FINITE penalty
                return 1e6 * np.ones_like(moment_errors)

            return np.array(moment_errors)
        except Exception as e:
            print(f"EXCEPTION in momcond with params {params}: {str(e)}")
            # Return a large but FINITE penalty
            return 1e6 * np.ones((len(self.endog), self.num_lags))


# num_lags=30, trawl_function_name='sup_IG'):
# , num_lags, trawl_function_name, initial_guess=None):
def estimate_acf_parameters(trawl, config, initial_guess=None):
    """
    Estimate ACF parameters using GMM.

    Parameters:
    -----------
    trawl : array-like
        Observed data
    num_lags : int, optional
        Number of ACF lags to compare (default=30)
    trawl_function_name : str, optional
        Type of trawl process to use (default='sup_IG')
    """

    num_lags = config['loss_config']['nr_acf_lags']
    trawl_function_name = config['trawl_config']['acf']

    if initial_guess is None:

        if trawl_function_name == 'sup_IG':
            initial_guess = np.array(
                [17.0, 15.0])  # Initial guess for [acf_gamma, acf_eta]
        else:
            raise ValueError

    # Update instruments matrix to account for ACF moments only
    instruments = np.ones((len(trawl), num_lags))
    exog = np.ones((len(trawl), 1))

    gmm_model = ACFGMM(endog=np.array(trawl),
                       exog=exog,
                       instrument=instruments,
                       num_lags=num_lags,
                       trawl_function_name=trawl_function_name)

    try:
        # gmm_model.bounds = gmm_model.bounds[:2]
        # gmm_model.bounds = ((10.0, 20.0), (10.0, 20.0))
        result = gmm_model.fit(start_params=initial_guess,  # optim_method='bfgs',
                               maxiter=5)
        acf_gamma, acf_eta = result.params

        # get final acf errors
        # acf_moment_conditions(params, trawl, num_lags, acf_func)

        return result, np.std(gmm_model.momcond(result.params), axis=0)
        # return {
        #    "acf_gamma": acf_gamma,
        #    "acf_eta": acf_eta
        # }
    except Exception as e:
        return None


if __name__ == '__main__':
    # num_lags = 30
    # trawl_function_name = 'sup_IG'
    import yaml
    config_file_path = 'config_files/summary_statistics/LSTM/acf\\config3.yaml'

    # Load config file
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    trawl = np.load('trawl.npy')[5]
    theta_acf = np.load('theta_acf.npy')[5]
    num_lags = config['loss_config']['nr_acf_lags']
    trawl_function_name = config['trawl_config']['acf']
    acf_func = get_acf(trawl_function_name)

    # plotting
    H = np.arange(1, num_lags + 1)
    theoretical_acf = acf_func(H, theta_acf)
    empirical_acf = compute_empirical_acf(trawl, nlags=num_lags)[1:]
    gmm_params = estimate_acf_parameters(
        trawl, config)  # num_lags=num_lags, trawl_function_name=trawl_function_name)
    gmm_acf = acf_func(H, gmm_params)

    f, ax = plt.subplots()
    ax.scatter(H, theoretical_acf, label='Theoretical')
    ax.scatter(H, empirical_acf, label='Empirical')
    ax.scatter(H, gmm_acf, label='GMM')

    plt.legend()
    plt.show()
