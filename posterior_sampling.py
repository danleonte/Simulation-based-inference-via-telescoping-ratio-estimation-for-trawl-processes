import sys
import os
import yaml
import pickle
import jax.numpy as jnp
from posterior_sampling_utils import run_mcmc_for_trawl, save_results, create_and_save_plots
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models


def main(start_idx, end_idx, seq_len):

    # Load configuration
    # folder_path = r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/TRE_full_trawl/beta_calibrated'
    folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\TRE_full_trawl\selected_models'

    # Set up model configuration
    use_tre = 'TRE' in folder_path
    if not (use_tre or 'NRE' in folder_path):
        raise ValueError("Path must contain 'TRE' or 'NRE'")

    use_summary_statistics = 'summary_statistics' in folder_path
    if not (use_summary_statistics or 'full_trawl' in folder_path):
        raise ValueError(
            "Path must contain 'full_trawl' or 'summary_statistics'")

    if use_tre:
        classifier_config_file_path = os.path.join(
            folder_path, 'acf', 'config.yaml')
    else:
        classifier_config_file_path = os.path.join(folder_path, 'config.yaml')

    with open(classifier_config_file_path, 'r') as f:
        a_classifier_config = yaml.safe_load(f)
        trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(
        os.path.dirname(folder_path)), f'cal_dataset_{seq_len}')
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    cal_Y = jnp.load(cal_Y_path)
    true_trawls = jnp.load(cal_x_path)[:, cal_Y == 1].reshape(-1, seq_len)
    true_thetas = jnp.load(cal_thetas_path)
    true_thetas = true_thetas[:, cal_Y == 1].reshape(-1, true_thetas.shape[-1])
    del cal_Y

    # Load approximate likelihood function
    approximate_log_likelihood_to_evidence, _ = load_trained_models(
        folder_path, true_trawls[[0], ::-1], trawl_process_type,
        use_tre, use_summary_statistics, f'calibration_{seq_len}.pkl'
    )

    # MCMC parameters
    num_samples = 6000  # 7500
    num_warmup = 2000  # 2500
    num_burnin = 2000  # 2500
    num_chains = 25  # 25
    seed = 25246  # this gets chaged inside the posterior_sampling_utils

    if end_idx <= start_idx:
        print(f": No trawls assigned.")
        return

    # Create results directory
    results_dir = f"mcmc_results_{trawl_process_type}"
    results_dir = os.path.join(folder_path, results_dir, str(seq_len))
    os.makedirs(results_dir, exist_ok=True)

    for idx in range(start_idx, end_idx):
        # Create directory for this trawl
        trawl_dir = os.path.join(results_dir, f"trawl_{idx}")
        os.makedirs(trawl_dir, exist_ok=True)

        # Skip if already completed
        if os.path.exists(os.path.join(trawl_dir, "results.pkl")):
            print(f"Trawl {idx} already processed, skipping...")
            continue

        try:
            # Run MCMC for this trawl
            results, posterior_samples = run_mcmc_for_trawl(
                trawl_idx=idx,
                true_trawls=true_trawls,
                true_thetas=true_thetas,
                approximate_log_likelihood_to_evidence=approximate_log_likelihood_to_evidence,
                seed=seed + idx**2,
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_burnin=num_burnin,
                num_chains=num_chains
            )

            # Add true theta to results
            results['true_theta'] = true_thetas[idx].tolist()
            results['true_trawl'] = true_trawls[idx].tolist()

            # Save results
            save_results(results, os.path.join(trawl_dir, "results.pkl"))

            try:
                create_and_save_plots(
                    posterior_samples=posterior_samples,
                    true_theta=results['true_theta'],
                    save_dir=trawl_dir
                )
            except Exception as e:
                print(
                    f"Error creating plots for trawl {idx}: {str(e)}")

            # Save memory by clearing results
            del results

            print(f"Completed trawl {idx}")

        except Exception as e:
            print(f"Error processing trawl {idx}: {str(e)}")
            # Save the error to a file
            with open(os.path.join(trawl_dir, "error.txt"), 'w') as f:
                f.write(f"Error processing trawl {idx}: {str(e)}")

    print(f"Completed all assigned trawls")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <start_idx> <end_idx>")
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    main(start_idx, end_idx)
