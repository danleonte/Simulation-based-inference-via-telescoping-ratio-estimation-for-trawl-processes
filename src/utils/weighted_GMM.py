if True:
    from path_setup import setup_sys_path
    setup_sys_path()


from src.utils.acf_functions import get_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from statsmodels.sandbox.regression.gmm import GMM
import scipy
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def transform_to_tf_params(jax_mu, jax_scale, beta):
    """"returns mu, dleta, gamma, beta

    NOT

    mu, delta, alpha ,beta"""
    gamma = 1 + jnp.abs(beta) / 5
    alpha = jnp.sqrt(gamma**2 + beta**2)
    tf_delta = jax_scale**2 * gamma**3 / alpha**2
    tf_mu = jax_mu - beta * tf_delta / gamma
    return tf_mu, tf_delta, gamma, beta


def transform_to_jax_params(tf_mu, tf_delta, gamma, beta):
    """
    Transform TensorFlow parameters back to JAX parameters.
    """
    alpha = jnp.sqrt(gamma**2 + beta**2)
    jax_scale = jnp.sqrt(tf_delta * alpha**2 / gamma**3)
    jax_mu = tf_mu + beta * tf_delta / gamma
    return jax_mu, jax_scale, beta


def nig_moments(mu, delta, gamma, beta):

    # alpha, beta, mu, delta =  a/scale, b/scale, loc, scale
    alpha = (gamma**2+beta**2)**0.5
    a = alpha * delta
    b = beta * delta
    loc = mu
    scale = delta

    moments = [scipy.stats.norminvgauss(
        a=a, b=b, loc=loc, scale=scale).moment(i) for i in (1, 2, 3, 4)]
    return np.array(moments)


class JAXGMM(GMM):
    def __init__(self, endog, exog, instrument, num_lags, trawl_type):
        super().__init__(endog, exog, instrument)
        self.num_lags = num_lags
        self.trawl_type = trawl_type
        # Get the appropriate ACF function based on trawl_type
        self.acf_func = get_acf(trawl_type)

    def momcond(self, params):
        try:
            moment_errors = moment_conditions_jax(
                params, self.endog, self.num_lags, self.acf_func)
            if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
                return np.inf * np.ones_like(moment_errors)
            return np.array(moment_errors)
        except:
            return np.inf * np.ones((len(self.endog), 4 + self.num_lags))


def moment_conditions_jax(theta_jax, trawl, num_lags, acf_func):
    """
    Calculate moment conditions for GMM estimation with theta_jax.
    Includes both distribution moments and ACF comparisons.
    Trims num_lags observations from all moments for consistent length.

    Parameters:
    -----------
    theta_jax : array-like
        Parameters [jax_mu, jax_scale, beta, acf_gamma, acf_eta]
    trawl : array-like
        Observed data
    num_lags : int
        Number of ACF lags to compare
    acf_func : function
        Function to compute theoretical ACF based on trawl_type
    """
    try:
        jax_mu, jax_scale, beta, acf_gamma, acf_eta = theta_jax

        # Transform parameters and compute distributional moments
        tf_mu, tf_delta, gamma, _ = transform_to_tf_params(
            jax_mu, jax_scale, beta)
        model_moments = nig_moments(tf_mu, tf_delta, gamma, beta)

        # Trim the series to have consistent length for all moments
        trimmed_length = len(trawl) - num_lags
        trimmed_trawl = trawl[:trimmed_length]

        # Compute marginal moment conditions with trimmed series
        marginal_errors = jnp.array([
            trimmed_trawl - model_moments[0],
            trimmed_trawl**2 - model_moments[1],
            trimmed_trawl**3 - model_moments[2],
            trimmed_trawl**4 - model_moments[3]
        ]).T

        # Compute demeaned series for ACF calculation
        demeaned_trawl = trawl - np.mean(trawl)
        variance = np.var(trawl)

        # Initialize array for ACF errors
        acf_errors = np.zeros((trimmed_length, num_lags))

        # Compute ACF errors for each lag while maintaining individual observations
        for k in range(1, num_lags + 1):
            # For each lag k, we can use trimmed_length observations
            empirical_products = (
                demeaned_trawl[:-k] * demeaned_trawl[k:]) / variance
            theoretical_acf = acf_func(k, np.array([acf_gamma, acf_eta]))

            # Store only the first trimmed_length values
            acf_errors[:, k-1] = empirical_products[:trimmed_length] - \
                theoretical_acf

        # Combine marginal and ACF errors
        moment_errors = jnp.concatenate([marginal_errors, acf_errors], axis=1)

        return moment_errors
    except:
        # Return appropriate sized array of infinities
        return jnp.inf * jnp.ones((len(trawl) - num_lags, 4 + num_lags))


def estimate_jax_parameters(trawl, num_lags=35, trawl_type='sup_IG'):
    """
    Estimate parameters using GMM with theta_jax, including ACF comparisons.

    Parameters:
    -----------
    trawl : array-like
        Observed data
    num_lags : int, optional
        Number of ACF lags to compare (default=5)
    trawl_type : str, optional
        Type of trawl process to use (default='exp')
    """
    initial_guess = np.array([
        np.mean(trawl),  # jax_mu
        np.std(trawl),   # jax_scale
        0.0,            # jax_beta
        15.0,           # acf_gamma
        15.0            # acf_eta
    ])

    # Update instruments matrix to account for additional ACF moments
    instruments = np.ones((len(trawl), 4 + num_lags))
    exog = np.ones((len(trawl), 1))

    gmm_model = JAXGMM(endog=np.array(trawl),
                       exog=exog,
                       instrument=instruments,
                       num_lags=num_lags,
                       trawl_type=trawl_type)

    try:
        result = gmm_model.fit(start_params=initial_guess)  # , maxiter=10)
        jax_mu, jax_scale, jax_beta, acf_gamma, acf_eta = result.params
        return {
            "jax_mu": jax_mu,
            "jax_scale": jax_scale,
            "jax_beta": jax_beta,
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
    theta_marginal = np.load('theta_marginal_jax.npy')[1]
    theta_jax = jnp.concatenate([theta_marginal, theta_acf])
    result = estimate_jax_parameters(
        trawl, num_lags=num_lags, trawl_type=trawl_type)

    H = np.arange(1, num_lags+1)
    acf_func = get_acf(trawl_type)
    theoretical_acf = acf_func(H, theta_acf)
    empirical_acf = compute_empirical_acf(trawl, nlags=num_lags)[1:]

    infered_params = np.array([result['acf_gamma'], result['acf_eta']])
    infered_acf = acf_func(H, infered_params)

    f, ax = plt.subplots()
    ax.scatter(H, theoretical_acf, label='theoretical')
    ax.scatter(H, empirical_acf, label='empirical')
    ax.scatter(H, infered_acf, label='infered')
    plt.legend()
