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

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


######### Inputs ##########

folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_summary_statistics\trial_set1'
trawl_process_type = 'sup_ig_nig_5p'
use_tre = True
use_summary_statistics = True
generate_dataset = True


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

if os.path.isfile(true_trawls_path) and os.path.isfile(true_thetas_path):

    true_trawls = np.load(true_trawls_path)
    true_thetas = np.load(true_thetas_path)

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


#####################    LOAD MODELS    #######################################

approximate_log_likelihood_to_evidence, approximate_log_posterior = load_trained_models(folder_path, true_trawls[[30], :], trawl_process_type, use_tre,
                                                                                        use_summary_statistics)


########################### MCMC #######################################
step_size = 0.01
num_chains = 10
num_results = 600000
num_burnin_steps = 5000


# Create MALA kernel - using approximate_log_posterior directly
kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
    target_log_prob_fn=lambda theta: approximate_log_posterior(theta)[0],
    step_size=step_size
)

samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    current_state=true_thetas[0],
    kernel=kernel,
    num_burnin_steps=num_burnin_steps,
    seed=jax.random.PRNGKey(np.random.randint(1, 100000))
)

# Calculate acceptance rate
acceptance_rate = jnp.mean(kernel_results.is_accepted)
print(f"Acceptance Rate: {acceptance_rate:.4f}")

# Convert to numpy and flatten chains
reshaped_samples = np.array(samples).reshape(-1, samples.shape[-1])[::75]

# Create the corner plot
figure = corner.corner(
    reshaped_samples,
    labels=[r"$\theta_1$", r"$\theta_2$",
            r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"],
    quantiles=[0.05, 0.5, 0.95],
    show_titles=True,
    truths=true_thetas[30],  # Add the true values
    title_kwargs={"fontsize": 12},
)


##################### COMPARE WITH OTHER THINGS ##############################


if use_summary_statistics:

    # DO NOT USE THIS BEFORE PASSING X INTO THE APPROXIMATE INFERENCE FUNCTIONS
    project_trawl = get_projection_function()
    summary_s = project_trawl(true_trawls[[25], :])
    # WE ALREADY APPLY THE SUMMARY STATSITCS ? REPLACE WITH ACF INSIDE THE FUNCTIONS

##############################################################


result = minimize(lambda theta: - approximate_log_posterior(theta).item(),
                  true_thetas[0], method='BFGS', options={'disp': True})
print(result.x)

plot_marginals(
    true_theta[2:], jnp.array(result.x[2:]), classifier_config)

plot_marginals(
    true_theta[2:], jnp.array(true_trawl[2:]), classifier_config)
