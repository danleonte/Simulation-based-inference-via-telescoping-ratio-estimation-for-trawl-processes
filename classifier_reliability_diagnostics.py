# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:23:50 2025

@author: dleon
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:16:40 2025

@author: dleon
"""

import jax.numpy as jnp
from jax.random import PRNGKey
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel
from netcal.presentation import ReliabilityDiagram
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
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt

project_trawl = get_projection_function()
folder_path = os.path.join(
    os.getcwd(), 'models', 'classifier', 'NRE_summary_statistics', '2_08_19_13_40')
validation_data_path = os.path.join(folder_path, 'validation_data')

val_trawls = np.load(os.path.join(validation_data_path, 'val_trawls.npy'))
# val_trawls = jnp.array([project_trawl(trawl_batch) for trawl_batch in val_trawls])
val_thetas = np.load(os.path.join(validation_data_path, 'val_thetas.npy'))

with open(os.path.join(validation_data_path, 'config.yaml'), 'r') as f:
    classifier_config = yaml.safe_load(f)

model, _, _ = get_model(classifier_config)
with open(os.path.join(folder_path, "params_iter_9000.pkl"), 'rb') as file:
    params = pickle.load(file)


@jax.jit
def process_sample(params, trawl_val, theta_val):
    """JIT-compiled function to process a single validation sample."""
    batch_size = theta_val.shape[0]

    # Shuffle
    trawl_val = jnp.vstack([trawl_val, trawl_val])  # normal, normal
    theta_val = jnp.vstack(
        [theta_val, jnp.roll(theta_val, -1, axis=0)])  # normal, shuffled
    Y_val = jnp.vstack(
        [jnp.ones([batch_size, 1]), jnp.zeros([batch_size, 1])])  # 1, then 0

    pred_Y = model.apply(variables=params, x=trawl_val,
                         theta=theta_val, train=False).squeeze()

    return jax.nn.sigmoid(pred_Y)


all_classifier_outputs = []
batch_size = val_trawls.shape[1]

for i in range(val_trawls.shape[0]):
    theta_val = val_thetas[i]
    trawl_val = val_trawls[i]

    all_classifier_outputs.append(process_sample(params, trawl_val, theta_val))

# Convert classifier outputs to JAX array
all_classifier_outputs = jnp.concatenate(
    all_classifier_outputs, axis=0)

Y_calibration = jnp.hstack(
    [jnp.ones([batch_size]), jnp.zeros([batch_size])])
Y_calibration = np.concatenate(
    [Y_calibration] * val_trawls.shape[0])

all_classifier_outputs = np.array(
    all_classifier_outputs)


hist_beta, ax = plt.subplots()
ax.hist(
    all_classifier_outputs[Y_calibration == 1], label='Y=1', alpha=0.5, density=True)
ax.hist(
    all_classifier_outputs[Y_calibration == 0], label='Y=0', alpha=0.5, density=True)
ax.set_title(
    r'Histogram of $c(\mathbf{x},\mathbf{\theta})$ classifier')
ax.legend(loc='upper center')
# hist_beta.canvas.draw()  # Force render


# Reliability diagram with equal intervals
diagram_un = ReliabilityDiagram(
    25, equal_intervals=True)
fig_un = diagram_un.plot(
    all_classifier_outputs, Y_calibration).get_figure()
fig_un.canvas.draw()  # Force render
# wandb.log({"Diagram eq": wandb.Image(fig_eq)},
#         step=iteration)  # Add step
