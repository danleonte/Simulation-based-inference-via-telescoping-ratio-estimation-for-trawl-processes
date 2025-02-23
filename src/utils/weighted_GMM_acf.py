if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.acf_functions import get_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from statsmodels.sandbox.regression.gmm import GMM
import numpy as np
import matplotlib.pyplot as plt


def acf_moment_conditions(params, trawl, num_lags, acf_func):
    """
    Calculate moment conditions for ACF parameters only.

    Parameters:
    -----------
    params : array-like
        Parameters [acf_gamma, acf_eta]
    trawl : array-like
        Observed data
    num_lags : int
        Number of ACF lags to compare
    acf_func : function
        Function to compute theoretical ACF based on trawl_function_name
    """
    acf_gamma, acf_eta = params

    # Compute demeaned series for ACF calculation
    demeaned_trawl = trawl - np.mean(trawl)
    variance = np.var(trawl)

    # Initialize array for ACF errors
    acf_errors = np.zeros((len(trawl) - num_lags, num_lags))

    for k in range(1, num_lags + 1):
        empirical_products = (
            demeaned_trawl[:-k] * demeaned_trawl[k:]) / variance
        theoretical_acf = acf_func(k, np.array([acf_gamma, acf_eta]))
        acf_errors[:, k -
                   1] = empirical_products[:len(trawl) - num_lags] - theoretical_acf

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
            if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
                return np.inf * np.ones_like(moment_errors)
            return np.array(moment_errors)
        except:
            return np.inf * np.ones((len(self.endog), self.num_lags))


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
                [15.0, 15.0])  # Initial guess for [acf_gamma, acf_eta]
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
        result = gmm_model.fit(start_params=initial_guess,
                               maxiter=1000)
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
    config_file_path = 'config_files/summary_statistics/LSTM\\config1.yaml'

    # Load config file
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    trawl = np.load('trawl.npy')[1]
    theta_acf = np.load('theta_acf.npy')[1]
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
