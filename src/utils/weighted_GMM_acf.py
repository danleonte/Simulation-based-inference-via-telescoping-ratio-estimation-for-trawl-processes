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
        Function to compute theoretical ACF based on trawl_type
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
    def __init__(self, endog, exog, instrument, num_lags, trawl_type):
        super().__init__(endog, exog, instrument)
        self.num_lags = num_lags
        self.acf_func = get_acf(trawl_type)

    def momcond(self, params):
        try:
            moment_errors = acf_moment_conditions(
                params, self.endog, self.num_lags, self.acf_func)
            if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
                return np.inf * np.ones_like(moment_errors)
            return np.array(moment_errors)
        except:
            return np.inf * np.ones((len(self.endog), self.num_lags))


def estimate_acf_parameters(trawl, num_lags=30, trawl_type='sup_IG'):
    """
    Estimate ACF parameters using GMM.

    Parameters:
    -----------
    trawl : array-like
        Observed data
    num_lags : int, optional
        Number of ACF lags to compare (default=30)
    trawl_type : str, optional
        Type of trawl process to use (default='sup_IG')
    """
    initial_guess = np.array(
        [15.0, 15.0])  # Initial guess for [acf_gamma, acf_eta]

    # Update instruments matrix to account for ACF moments only
    instruments = np.ones((len(trawl), num_lags))
    exog = np.ones((len(trawl), 1))

    gmm_model = ACFGMM(endog=np.array(trawl),
                       exog=exog,
                       instrument=instruments,
                       num_lags=num_lags,
                       trawl_type=trawl_type)

    try:
        result = gmm_model.fit(start_params=initial_guess, maxiter=1000)
        acf_gamma, acf_eta = result.params
        return {
            "acf_gamma": acf_gamma,
            "acf_eta": acf_eta
        }
    except Exception as e:
        raise ValueError(f"Optimization failed: {e}")


if __name__ == '__main__':
    num_lags = 30
    trawl_type = 'sup_IG'
    trawl = np.load('trawl.npy')[1]
    theta_acf = np.load('theta_acf.npy')[1]

    result = estimate_acf_parameters(
        trawl, num_lags=num_lags, trawl_type=trawl_type)

    H = np.arange(1, num_lags + 1)
    acf_func = get_acf(trawl_type)
    theoretical_acf = acf_func(H, theta_acf)
    empirical_acf = compute_empirical_acf(trawl, nlags=num_lags)[1:]

    inferred_params = np.array([result['acf_gamma'], result['acf_eta']])
    inferred_acf = acf_func(H, inferred_params)

    f, ax = plt.subplots()
    ax.scatter(H, theoretical_acf, label='Theoretical')
    ax.scatter(H, empirical_acf, label='Empirical')
    ax.scatter(H, inferred_acf, label='Inferred')
    plt.legend()
    plt.show()
