import jax.numpy as jnp
from jax.numpy.fft import fft
from jax.numpy.fft import ifft
import jax
import numpy as np
from functools import partial


def create_parameter_sweep_fn_final(apply_fn, bounds_for_integration, N):
    """
    FINAL VERSION: Create a parameter sweep function that integrates over the LAST parameter (index 4).
    """
    param_idx = 4  # Last parameter in 5-element theta
    param_values = interpolation_points_domain(
        N, bounds_for_integration[0], bounds_for_integration[1])

    def process_param(p_val, thetas, x_cache):
        batch_size = thetas.shape[0]
        modified = thetas.at[:, param_idx].set(jnp.full(batch_size, p_val))
        results, _ = apply_fn(modified, x_cache)

        # DIMENSION FIX: Remove extra dimensions from (1, 1) to (1,)
        while results.ndim > 1 and results.shape[-1] == 1:
            results = jnp.squeeze(results, axis=-1)

        return results

    vectorized_process = jax.vmap(process_param, in_axes=(0, None, None))

    @jax.jit
    def parameter_sweep(thetas, x_cache):
        all_results = vectorized_process(param_values, thetas, x_cache)
        # all_results shape: [N, batch_size] = [129, 1]
        # Transpose to get [batch_size, N] = [1, 129]
        result = jnp.transpose(all_results)
        return result

    return parameter_sweep


def dct_via_fft(data: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a DCT-like transform using FFT, JAX version.
    """
    N = data.shape[0] // 2
    fftdata = fft(data, axis=0)[:N + 1] / N
    fftdata = fftdata.at[0].set(fftdata[0] / 2.)
    fftdata = fftdata.at[-1].set(fftdata[-1] / 2.)

    result = jnp.real(fftdata) if jnp.isrealobj(data) else fftdata
    return result


def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    Return: array of length 2(N-1)
    For instance, [0,1,2,3,4] --> [0,1,2,3,4,3,2,1]
    """
    return jnp.concatenate([data, data[-2:0:-1]])


def polyfit_domain_final(sampled, a, b):
    """
    FINAL VERSION: Fixed polyfit_domain that handles all shape cases.
    """
    asampled = jnp.asarray(sampled)

    # Handle scalar case
    if asampled.ndim == 0:
        return jnp.array([asampled])

    # Handle 1D case
    if asampled.shape[0] == 1:
        return asampled

    evened = even_data(asampled)
    coeffs = dct_via_fft(evened)
    return coeffs


def integrate_from_sampled_final(sampled, a, b):
    """
    FINAL VERSION: Single integration with proper shape handling.
    """
    # Ensure sampled is 1D for polyfit_domain
    if sampled.ndim > 1:
        if sampled.shape[0] == 1:
            sampled = sampled[0]  # [1, 129] -> [129]
        else:
            raise ValueError(f"Expected batch size 1, got {sampled.shape[0]}")

    coeffs = polyfit_domain_final(sampled, a, b)

    # Ensure coeffs is 1D for chebint_ab
    if coeffs.ndim > 1:
        if coeffs.shape[0] == 1:
            coeffs = coeffs[0]  # [1, 129] -> [129]

    coeffs_int = chebint_ab(coeffs, a, b)

    # Evaluate antiderivative at domain endpoints
    endpoints = jnp.array([a, b])
    results = chebval_ab_jax(endpoints, coeffs_int, a, b)

    return results[1] - results[0]


# Vectorized version
vec_integrate_from_sampled_final = jax.vmap(
    integrate_from_sampled_final, in_axes=(0, None, None))


def integrate_posteriors_final(theta_values_batch, x_values_batch, integration_bounds):
    """
    FINAL VERSION: Integrate over the last parameter (index 4).
    """
    a, b = integration_bounds

    _, x_cache_values_batch = apply_model_with_x(
        x_values_batch, theta_values_batch)

    # Evaluate log probabilities at Chebyshev knots for the last parameter
    log_prob_envelope = evaluate_at_chebyshev_knots_final(
        theta_values_batch, x_cache_values_batch)

    cal_log_prob_envelope = beta_calibrate_log_r(
        log_prob_envelope, calibration_params['params'])

    # Compute integrals
    cal_int = vec_integrate_from_sampled_final(
        jnp.exp(cal_log_prob_envelope), a, b)
    uncal_int = vec_integrate_from_sampled_final(
        jnp.exp(log_prob_envelope), a, b)

    return uncal_int, cal_int


def load_single_sample(val_x_path, val_thetas_path, sample_idx=0):
    """Load a single sample from the validation data."""
    # Load with memory mapping to avoid loading everything
    val_x_full = np.load(val_x_path, mmap_mode='r')
    val_thetas_full = np.load(val_thetas_path, mmap_mode='r')

    # Get dimensions
    num_batches, batch_dim, seq_len = val_x_full.shape
    _, _, theta_dim = val_thetas_full.shape

    # Extract one sample
    batch_idx = sample_idx // batch_dim
    within_batch_idx = sample_idx % batch_dim

    x_sample = val_x_full[batch_idx, within_batch_idx, :]  # Shape: [seq_len]
    theta_sample = val_thetas_full[batch_idx,
                                   within_batch_idx, :]  # Shape: [5]

    return x_sample, theta_sample


def create_parameter_sweep_plots_final(x_sample, base_theta, n_points=50):
    """
    FINAL VERSION: Create parameter sweep plots with all fixes.
    """
    # Integration bounds for the last parameter (index 4)
    integration_bounds = (-5.0, 5.0)

    # Parameter ranges to vary
    param3_values = np.linspace(-1, 1, n_points)    # 3rd component (index 2)
    param4_values = np.linspace(0.5, 1.5, n_points)  # 4th component (index 3)

    # Prepare x for batch processing
    x_batch = jnp.expand_dims(x_sample, axis=0)  # Shape: [1, seq_len]

    # Results storage
    areas_param3_cal = []
    areas_param3_uncal = []
    areas_param4_cal = []
    areas_param4_uncal = []

    print("Computing parameter sweep for component 3 (index 2)...")
    print(
        f"  Varying from -1 to 1, integrating over component 5 (index 4) from {integration_bounds[0]} to {integration_bounds[1]}")

    for i, param_val in enumerate(param3_values):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_points}")

        # Create modified theta: vary component 3 (index 2), keep others fixed
        theta_modified = base_theta.copy()
        theta_modified[2] = param_val  # 3rd component (index 2)
        theta_batch = jnp.expand_dims(theta_modified, axis=0)  # Shape: [1, 5]

        # Integrate over the last parameter (index 4)
        uncal_area, cal_area = integrate_posteriors_final(
            theta_batch, x_batch, integration_bounds)
        areas_param3_uncal.append(float(uncal_area[0]))
        areas_param3_cal.append(float(cal_area[0]))

    print("Computing parameter sweep for component 4 (index 3)...")
    print(
        f"  Varying from 0.5 to 1.5, integrating over component 5 (index 4) from {integration_bounds[0]} to {integration_bounds[1]}")

    for i, param_val in enumerate(param4_values):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_points}")

        # Create modified theta: vary component 4 (index 3), keep others fixed
        theta_modified = base_theta.copy()
        theta_modified[3] = param_val  # 4th component (index 3)
        theta_batch = jnp.expand_dims(theta_modified, axis=0)  # Shape: [1, 5]

        # Integrate over the last parameter (index 4)
        uncal_area, cal_area = integrate_posteriors_final(
            theta_batch, x_batch, integration_bounds)
        areas_param4_uncal.append(float(uncal_area[0]))
        areas_param4_cal.append(float(cal_area[0]))

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Component 3 - Uncalibrated
    ax1.plot(param3_values, areas_param3_uncal, 'b-', linewidth=2,
             label='Uncalibrated', marker='o', markersize=4)
    ax1.set_xlabel('Component 3 (index 2)')
    ax1.set_ylabel('Integrated Area (over component 5)')
    ax1.set_title('Parameter Sweep: Component 3 (Uncalibrated)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Component 3 - Calibrated
    ax2.plot(param3_values, areas_param3_cal, 'r-', linewidth=2,
             label='Calibrated', marker='o', markersize=4)
    ax2.set_xlabel('Component 3 (index 2)')
    ax2.set_ylabel('Integrated Area (over component 5)')
    ax2.set_title('Parameter Sweep: Component 3 (Calibrated)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Component 4 - Uncalibrated
    ax3.plot(param4_values, areas_param4_uncal, 'b-', linewidth=2,
             label='Uncalibrated', marker='o', markersize=4)
    ax3.set_xlabel('Component 4 (index 3)')
    ax3.set_ylabel('Integrated Area (over component 5)')
    ax3.set_title('Parameter Sweep: Component 4 (Uncalibrated)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Component 4 - Calibrated
    ax4.plot(param4_values, areas_param4_cal, 'r-', linewidth=2,
             label='Calibrated', marker='o', markersize=4)
    ax4.set_xlabel('Component 4 (index 3)')
    ax4.set_ylabel('Integrated Area (over component 5)')
    ax4.set_title('Parameter Sweep: Component 4 (Calibrated)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Save the plot
    results_path = os.path.join(os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl',
                                'selected_models', 'per_classifier_coverage_check', str(tre_type))
    plt.savefig(os.path.join(
        results_path, 'parameter_sweep_plots_final.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print results summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(
        f"Component 3 integral range: [{min(areas_param3_uncal):.6f}, {max(areas_param3_uncal):.6f}] (uncal)")
    print(
        f"Component 3 integral range: [{min(areas_param3_cal):.6f}, {max(areas_param3_cal):.6f}] (cal)")
    print(
        f"Component 4 integral range: [{min(areas_param4_uncal):.6f}, {max(areas_param4_uncal):.6f}] (uncal)")
    print(
        f"Component 4 integral range: [{min(areas_param4_cal):.6f}, {max(areas_param4_cal):.6f}] (cal)")

    return {
        'param3_values': param3_values,
        'areas_param3_uncal': areas_param3_uncal,
        'areas_param3_cal': areas_param3_cal,
        'param4_values': param4_values,
        'areas_param4_uncal': areas_param4_uncal,
        'areas_param4_cal': areas_param4_cal,
        'integration_bounds': integration_bounds
    }


# MAIN EXECUTION CODE
if __name__ == '__main__':
    # Your existing setup code here...
    # tre_type = 'beta'
    # trained_classifier_path = ...
    # ... all your model loading code ...

    # Integration setup
    integration_bounds = (-5.0, 5.0)  # Bounds for the last parameter (index 4)
    N = 128  # Number of integration points

    # Create the final evaluation function
    evaluate_at_chebyshev_knots_final = create_parameter_sweep_fn_final(
        apply_model_with_x_cache, integration_bounds, N+1)

    # Load sample data
    sample_idx = 0  # Change this to try different samples
    x_sample, theta_sample = load_single_sample(
        val_x_path, val_thetas_path, sample_idx)

    print(f"Loaded sample {sample_idx}:")
    print(f"  theta values: {theta_sample}")
    print(f"  Component 1 (index 0): {theta_sample[0]:.3f} - FIXED")
    print(f"  Component 2 (index 1): {theta_sample[1]:.3f} - FIXED")
    print(
        f"  Component 3 (index 2): {theta_sample[2]:.3f} - WILL VARY from -1 to 1")
    print(
        f"  Component 4 (index 3): {theta_sample[3]:.3f} - WILL VARY from 0.5 to 1.5")
    print(
        f"  Component 5 (index 4): {theta_sample[4]:.3f} - INTEGRATION VARIABLE (bounds: [-5, 5])")

    # Run the final parameter sweep
    print(f"\nStarting parameter sweep...")
    results = create_parameter_sweep_plots_final(
        x_sample, theta_sample, n_points=50)

    print(f"\n✓ Parameter sweep completed successfully!")
    print(f"Results saved to: parameter_sweep_plots_final.png")
