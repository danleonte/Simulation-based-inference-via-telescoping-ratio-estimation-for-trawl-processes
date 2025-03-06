from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel
import numpy as np
import datetime
import pickle
import optax
import wandb
import yaml
import os
import time


import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import matplotlib
import tensorflow_probability.substrates.jax as tfp
import corner

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


import numpyro
from numpyro.infer import MCMC, NUTS, HMC
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size as ess
import arviz as az

folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_full_trawl\beta_calibrated'

if 'TRE' in folder_path:
    use_tre = True
elif 'NRE' in folder_path:
    use_tre = False
else:
    raise ValueError

if 'full_trawl' in folder_path:
    use_summary_statistics = False

elif 'summary_statistics' in folder_path:
    use_summary_statistics = True

else:
    raise ValueError

if use_tre:
    classifier_config_file_path = os.path.join(
        folder_path, 'acf', 'config.yaml')
else:
    classifier_config_file_path = os.path.join(folder_path, 'config.yaml')

with open(classifier_config_file_path, 'r') as f:
    # an arbitrary config gile; if using TRE, can
    a_classifier_config = yaml.safe_load(f)
    trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']

dataset_path = os.path.join(os.path.dirname(
    os.path.dirname(folder_path)),  'cal_dataset')
# load calidation dataset
# cal_trawls_path = os.path.join(dataset_path, 'cal_trawls.npy')
cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

cal_Y = jnp.load(cal_Y_path)
true_trawls = jnp.load(cal_x_path)[0][cal_Y == 1]
true_thetas = jnp.load(cal_thetas_path)[0][cal_Y == 1]
del cal_Y

approximate_log_likelihood_to_evidence, approximate_log_posterior, _ = \
    load_trained_models(folder_path, true_trawls[:, ::-1], trawl_process_type,  # [::-1] not necessary, it s just a dummy, but just to make sure we don t pollute wth true values of some sort
                        use_tre, use_summary_statistics)


test_index = -1
test_trawl = true_trawls[test_index, :]
test_theta = true_thetas[test_index, :]


def model_vec():
    eta = numpyro.sample("eta", dist.Uniform(10, 20))
    gamma = numpyro.sample("gamma", dist.Uniform(10, 20))
    mu = numpyro.sample("mu", dist.Uniform(-1, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0.5, 1.5))
    beta = numpyro.sample("beta", dist.Uniform(-5, 5))

    params = jnp.array([eta, gamma, mu, sigma, beta])[jnp.newaxis, :]
    batch_size = params.shape[0]  # Should be `num_chains`
    x_tiled = jnp.tile(test_trawl, (batch_size, 1))
    numpyro.factor("likelihood", approximate_log_likelihood_to_evidence(x_tiled,
                                                                        params))  # Include log-likelihood in inference


num_samples = 5000
num_warmup = 2000
num_chains = 2  # Vectorized MCMC

rng_key = jax.random.PRNGKey(42)
chain_keys = jax.random.split(rng_key, num_chains)

hmc_kernel = HMC(
    model_vec,
    step_size=0.1,            # Initial step size (will be adapted)
    adapt_step_size=True,     # Enables step size adaptation
    adapt_mass_matrix=True,   # Enables mass matrix adaptation
    dense_mass=True,          # Uses a dense mass matrix (full covariance)
)

mcmc = MCMC(
    hmc_kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method='vectorized',
    progress_bar=True
)


start_time = time.time()
mcmc.run(chain_keys)
end_time = time.time()
print(start_time - end_time)

posterior_samples = mcmc.get_samples(group_by_chain=True)
az_data = az.from_numpyro(mcmc)
az.plot_trace(az_data)
ess = az.ess(az_data)
print(ess)


az.plot_pair(
    az_data,
    var_names=["eta", "gamma", "mu", "sigma", "beta"],
    marginals=True,
    kind='kde',
    figsize=(10, 10),
    reference_values={"eta": test_theta[0],
                      "gamma": test_theta[1],
                      "mu": test_theta[2],
                      "sigma": test_theta[3],
                      "beta": test_theta[4]},  # Add true values
    reference_values_kwargs={"color": "r", "marker": "o"}
)
plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')
