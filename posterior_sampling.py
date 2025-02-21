import jax.numpy as jnp
from jax.random import PRNGKey
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
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
import tensorflow_probability.substrates.jax as tfp
import corner
from numpyro.diagnostics import effective_sample_size as ess

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt

import blackjax
######### Inputs ##########

folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_summary_statistics\trial_set1'
trawl_process_type = 'sup_ig_nig_5p'
use_tre = True
use_summary_statistics = True
generate_dataset = True

assert use_summary_statistics


##############################################################

################### GENERATE DATA SET #######################################
if use_tre:
    classifier_config_file_path = os.path.join(
        folder_path, 'acf', 'config.yaml')
else:
    classifier_config_file_path = os.path.join(folder_path, 'config.yaml')


with open(classifier_config_file_path, 'r') as f:
    # an arbitrary config gile; if using TRE, can
    acf_config_file = yaml.safe_load(f)


true_trawls_path = os.path.join(folder_path, 'true_trawls.npy')
true_thetas_path = os.path.join(folder_path, 'true_thetas.npy')
true_summaries_path = os.path.join(folder_path, 'true_summaries.npy')

if os.path.isfile(true_trawls_path) and os.path.isfile(true_thetas_path) and os.path.isfile(true_summaries_path):

    true_trawls = np.load(true_trawls_path)
    true_thetas = np.load(true_thetas_path)
    true_summaries = np.load(true_summaries_path)

else:

    batch_size = acf_config_file['trawl_config']['batch_size']
    key = jax.random.split(
        PRNGKey(np.random.randint(low=1, high=10**6)), batch_size)

    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        acf_config_file)

    true_theta_acf_, key = theta_acf_simulator(key)
    true_theta_marginal_jax_, true_theta_marginal_tf_, key = theta_marginal_simulator(
        key)
    true_trawls, key = trawl_simulator(
        true_theta_acf_, true_theta_marginal_tf_, key)

    true_thetas = jnp.concatenate(
        [true_theta_acf_, true_theta_marginal_jax_], axis=1)

    np.save(file=true_trawls_path, arr=true_trawls)
    np.save(file=true_thetas_path, arr=true_thetas)

    if use_summary_statistics:

        project_trawl = get_projection_function()

        true_summaries = project_trawl(true_trawls)

        np.save(file=true_summaries_path, arr=true_summaries)


#####################    LOAD MODELS    #######################################
true_iteration = 19
true_trawl = true_trawls[true_iteration, :]
true_theta = true_thetas[true_iteration, :]
true_s = true_summaries[true_iteration, :]
approximate_log_likelihood_to_evidence, approximate_log_posterior, _ = \
    load_trained_models(folder_path, true_trawls[:, ::-1], trawl_process_type,  # [::-1] not necessary, it s just a dummy, but just to make sure we don t pollute wth true values of some sort
                        use_tre, use_summary_statistics)

assert not _


########################### MCMC #######################################
step_size = 0.05
num_chains = 10
num_samples = 500000
num_burnin_steps = 5000
num_adaptation_steps = 5000
rng_key = jax.random.PRNGKey(42)
num_params = true_thetas.shape[1]
mass_matrix_adaptation = blackjax.adaptation.mass_matrix_adaptation(num_params)
mass_matrix_state = mass_matrix_adaptation.init()

################## TO CHANGE OT TRUE TRAWL IF USING EMPIRICAL ACF NOT TRUES #############


def vec_log_density_fn(theta): return approximate_log_posterior(jnp.repeat(true_s[jnp.newaxis, :], repeats=num_chains, axis=0),
                                                                theta)


def log_density_fn(theta): return approximate_log_posterior(
    true_s[jnp.newaxis, :], theta[jnp.newaxis, :]).squeeze()


# initial_states = jnp.repeat(jnp.array([15.,15.,0.,1.,0.])[jnp.newaxis,:],axis=0, repeats = num_chains)
initial_position = jnp.array([15., 15., 0., 1., 0.])  # shape (5,)
barker_state = blackjax.mcmc.barker.init(
    position=initial_position, logdensity_fn=log_density_fn)
barker_kernel = blackjax.barker.build_kernel()
barker_kernel(rng_key=rng_key, state=barker_state,
              logdensity_fn=log_density_fn, step_size=0.1)


@jax.jit
def one_step(state, subkey):
    new_state, info = barker_kernel(subkey, state, log_density_fn, step_size)
    return new_state, state.position


# blackjax.mcmc.barker.build_kernel(jax.random.PRNGKey(42),barker_state)
keys = jax.random.split(rng_key, num_samples)
result = jax.lax.scan(one_step, barker_state, keys)


# adaptation
# https://academic.oup.com/biomet/article/110/3/579/6764577?utm_source=chatgpt.com&login=false

adapt_kernel = blackjax.adaptation.window_adaptation(algorithm=barker_kernel,
                                                     logdensity_fn=log_density_fn)


# Create MALA kernel - using approximate_log_posterior directly
# kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
#    target_log_prob_fn=lambda theta: approximate_log_posterior(theta)[0],
#    step_size=step_size
# )
#
# samples, kernel_results = tfp.mcmc.sample_chain(
#    num_results=num_results,
#    current_state=true_thetas[1],
#    kernel=kernel,
#    num_burnin_steps=num_burnin_steps,
#    seed=jax.random.PRNGKey(np.random.randint(1, 100000))
# )
#
# Calculate acceptance rate
# acceptance_rate = jnp.mean(kernel_results.is_accepted)
# print(f"Acceptance Rate: {acceptance_rate:.4f}")
#
# Convert to numpy and flatten chains
# reshaped_samples = np.array(samples).reshape(-1, samples.shape[-1])[::75]
#
# Create the corner plot
# figure = corner.corner(
#    reshaped_samples,
#    labels=[r"$\theta_1$", r"$\theta_2$",
#            r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"],
#    quantiles=[0.05, 0.5, 0.95],
#    show_titles=True,
#    truths=true_theta,  # Add the true values
#    title_kwargs={"fontsize": 12},
# )


##################### COMPARE WITH OTHER THINGS ##############################


def two_d_func(theta_acf_):
    return approximate_log_posterior(jnp.concatenate([theta_acf_, true_theta[2:]]))


points = 200  # Number of points in each dimension
x = np.linspace(10.1, 19.9, points)
y = np.linspace(10.1, 19.9, points)
X, Y = np.meshgrid(x, y)

# Calculate Z values by applying function to each point
Z = np.zeros((points, points))
for i in range(points):
    for j in range(points):
        Z[i, j] = np.exp(two_d_func(jnp.array([X[i, j], Y[i, j]])).item())


# Create the plot
plt.figure(figsize=(12, 10))

# Create heatmap with distinct cells and no interpolation
im = plt.pcolormesh(X, Y, Z, cmap='viridis', shading='nearest')
plt.colorbar(im, label='f(x,y)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Fine Grid Heatmap')

plt.show()

# Create the plot
# plt.figure(figsize=(10, 8))
# plt.contour(X, Y, Z, levels=20)  # Create contour lines
# plt.contourf(X, Y, Z, levels=20, cmap='viridis')  # Fill contours with colors
# plt.colorbar(label='f(x,y)')

# Customize the plot
# plt.grid(True)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('2D Function Plot')

# Show the plot
# plt.show()


result1 = minimize(lambda theta: - approximate_log_posterior(theta).item(),
                   true_theta, method='BFGS', options={'disp': True})

bfgs_start_2 = jnp.array(
    [15, 15, jnp.mean(true_trawl), jnp.std(true_trawl), 0.1])
result2 = minimize(lambda theta: - approximate_log_posterior(theta).item(),
                   bfgs_start_2, method='BFGS', options={'disp': True})
print(result2.x)

plot_marginals(
    true_theta[2:], jnp.array(result.x[2:]), acf_config_file)

plot_marginals(
    true_theta[2:], jnp.array(summary_s[0, 2:]), acf_config_file)
