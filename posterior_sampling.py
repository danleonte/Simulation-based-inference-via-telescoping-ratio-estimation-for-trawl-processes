
import jax.numpy as jnp
from jax.random import PRNGKey
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel
from netcal.presentation import ReliabilityDiagram
from scipy.optimize import minimize
import numpy as np
import datetime
import pickle
import optax
import wandb
import yaml
import jax
import os
import netcal
import matplotlib
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt

project_trawl = get_projection_function()
folder_path = os.path.join(
    os.getcwd(), 'models', 'classifier', 'NRE_summary_statistics', 'best_model')


Y = np.load(os.path.join(folder_path, 'cal_Y.npy'))
trawls = np.load(os.path.join(folder_path, 'cal_trawls.npy'))[0][Y == 1]
# val_trawls = jnp.array([project_trawl(trawl_batch) for trawl_batch in val_trawls])
thetas = np.load(os.path.join(folder_path, 'cal_thetas.npy'))[0][Y == 1]

true_trawl = trawls[15, :]
true_theta = thetas[15, :]

with open(os.path.join(folder_path, 'config.yaml'), 'r') as f:
    classifier_config = yaml.safe_load(f)

model, _, _ = get_model(classifier_config)
with open(os.path.join(folder_path, "params_iter_43200.pkl"), 'rb') as file:
    params = pickle.load(file)

trawl_config = classifier_config['trawl_config']
acf_prior_hyperparams = trawl_config['acf_prior_hyperparams']
eta_bounds = acf_prior_hyperparams['eta_prior_hyperparams']
gamma_bounds = acf_prior_hyperparams['gamma_prior_hyperparams']

marginal_distr_hyperparams = trawl_config['marginal_distr_hyperparams']
mu_bounds = marginal_distr_hyperparams['loc_prior_hyperparams']
scale_bounds = marginal_distr_hyperparams['scale_prior_hyperparams']
beta_bounds = marginal_distr_hyperparams['beta_prior_hyperparams']
# gamma, eta, mu, scale , beta
bounds = (gamma_bounds, eta_bounds, mu_bounds, scale_bounds, beta_bounds)
lower_bounds = jnp.array([i[0] for i in bounds])
upper_bounds = jnp.array([i[1] for i in bounds])
total_mass = jnp.prod(upper_bounds - lower_bounds)


@jax.jit
def approx_posterior_log_pdf(theta):

    log_likelihood = model.apply(params, true_trawl, theta, train=False)
    in_bounds = jnp.all((theta >= lower_bounds) & (theta <= upper_bounds))
    log_prior = jnp.where(in_bounds, - jnp.log(total_mass), -jnp.inf)
    return (log_likelihood + log_prior)


result = minimize(lambda theta: - approx_posterior_log_pdf(theta).item(),
                  true_theta, method='BFGS', options={'disp': True})
print(result.x)

plot_marginals(
    true_theta[2:], jnp.array(result.x[2:]), classifier_config)

plot_marginals(
    true_theta[2:], jnp.array(true_trawl[2:]), classifier_config)
