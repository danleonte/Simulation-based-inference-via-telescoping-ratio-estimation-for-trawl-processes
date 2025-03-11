import sys
import os
import yaml
import pickle
import jax.numpy as jnp
import traceback  # Added for better error reporting
import time  # Added for timing information
from posterior_sampling_utils import run_mcmc_for_trawl, save_results, create_and_save_plots
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models


def main(gpu_id, num_gpus, num_experiments_to_do):
    try:
        #print(f"=== Starting posterior sampling on GPU {gpu_id} ===")
        #start_time = time.time()
        #
        #assert num_experiments_to_do is not None
#
        # Set GPU
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ.get("SLURM_LOCALID", gpu_id))
        
        
        
        print(f"Setting visible GPU to {gpu_id}")
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_id = int(os.environ.get("SLURM_LOCALID", gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        
        # Configure JAX to use only one GPU
        print("Configuring JAX...")
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Available devices before configuration: {jax.devices()}")
        
        
        
        
        # Added - check JAX can see GPUs
        print(f"GPU {gpu_id}: Checking if JAX can see GPUs...")
        try:
            from jax.lib import xla_bridge
            print(f"GPU {gpu_id}: JAX platform: {xla_bridge.get_backend().platform}")
        except Exception as e:
            print(f"GPU {gpu_id}: Error checking JAX platform: {str(e)}")

        # Load configuration
        # folder_path = r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/NRE_full_trawl/beta_calibrated'
        folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\NRE_full_trawl\beta_calibrated'

        
        print(f"GPU {gpu_id}: Using folder path: {folder_path}")
        print(f"GPU {gpu_id}: Checking if folder exists: {os.path.exists(folder_path)}")
        
        # Set up model configuration
        use_tre = 'TRE' in folder_path
        if not (use_tre or 'NRE' in folder_path):
            raise ValueError("Path must contain 'TRE' or 'NRE'")

        use_summary_statistics = 'summary_statistics' in folder_path
        if not (use_summary_statistics or 'full_trawl' in folder_path):
            raise ValueError("Path must contain 'full_trawl' or 'summary_statistics'")

        if use_tre:
            classifier_config_file_path = os.path.join(folder_path, 'acf', 'config.yaml')
        else:
            classifier_config_file_path = os.path.join(folder_path, 'config.yaml')
            
        print(f"GPU {gpu_id}: Using config file: {classifier_config_file_path}")
        print(f"GPU {gpu_id}: Checking if config file exists: {os.path.exists(classifier_config_file_path)}")

        with open(classifier_config_file_path, 'r') as f:
            a_classifier_config = yaml.safe_load(f)
            trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']
            seq_len = a_classifier_config['trawl_config']['seq_len']
            
        print(f"GPU {gpu_id}: Loaded config. trawl_process_type={trawl_process_type}, seq_len={seq_len}")

        # Load dataset
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(folder_path)), 'cal_dataset')
        cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
        cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
        cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')
        
        print(f"GPU {gpu_id}: Dataset path: {dataset_path}")
        print(f"GPU {gpu_id}: Checking data files exist:")
        print(f"  - cal_x.npy: {os.path.exists(cal_x_path)}")
        print(f"  - cal_thetas.npy: {os.path.exists(cal_thetas_path)}")
        print(f"  - cal_Y.npy: {os.path.exists(cal_Y_path)}")

        print(f"GPU {gpu_id}: Loading data files...")
        try:
            cal_Y = jnp.load(cal_Y_path)
            print(f"GPU {gpu_id}: Loaded cal_Y with shape {cal_Y.shape}")
            
            # Add filtering info
            print(f"GPU {gpu_id}: Number of positive examples in cal_Y: {jnp.sum(cal_Y == 1)}")
            
            true_trawls = jnp.load(cal_x_path)[:, cal_Y == 1].reshape(-1, seq_len)
            print(f"GPU {gpu_id}: Loaded true_trawls with shape {true_trawls.shape}")
            
            true_thetas = jnp.load(cal_thetas_path)
            true_thetas = true_thetas[:, cal_Y == 1].reshape(-1, true_thetas.shape[-1])
            print(f"GPU {gpu_id}: Loaded true_thetas with shape {true_thetas.shape}")
            
            del cal_Y
        except Exception as e:
            print(f"GPU {gpu_id}: Error loading data: {str(e)}")
            traceback.print_exc()
            return

        # Limit dataset if requested
        if num_experiments_to_do is not None:
            print(f"GPU {gpu_id}: Limiting to {num_experiments_to_do} experiments")
            true_trawls = true_trawls[:num_experiments_to_do]
            true_thetas = true_thetas[:num_experiments_to_do]
            print(f"GPU {gpu_id}: After limiting: true_trawls shape {true_trawls.shape}, true_thetas shape {true_thetas.shape}")

        # Load approximate likelihood function
        print(f"GPU {gpu_id}: Loading trained models...")
        try:
            approximate_log_likelihood_to_evidence, _, _ = load_trained_models(
                folder_path, true_trawls[[0], ::-1], trawl_process_type,
                use_tre, use_summary_statistics
            )
            print(f"GPU {gpu_id}: Successfully loaded trained models")
        except Exception as e:
            print(f"GPU {gpu_id}: Error loading trained models: {str(e)}")
            traceback.print_exc()
            return

        # MCMC parameters
        num_samples = 750  # 7500
        num_warmup = 500  # 2500
        num_burnin = 500  # 2500
        num_chains = 20  # 25
        seed = 1411  # this gets changed inside the posterior_sampling_utils
        
        print(f"GPU {gpu_id}: MCMC parameters: num_samples={num_samples}, num_warmup={num_warmup}, " + 
              f"num_burnin={num_burnin}, num_chains={num_chains}, seed={seed}")

        # Calculate this GPU's workload
        total_trawls = true_trawls.shape[0]
        trawls_per_gpu = (total_trawls + num_gpus - 1) // num_gpus  # Ceiling division
        start_idx = gpu_id * trawls_per_gpu
        end_idx = min((gpu_id + 1) * trawls_per_gpu, total_trawls)
        
        # Add detailed logging
        print(f"GPU {gpu_id}: Total trawls: {total_trawls}")
        print(f"GPU {gpu_id}: Number of GPUs: {num_gpus}")
        print(f"GPU {gpu_id}: Trawls per GPU: {trawls_per_gpu}")
        print(f"GPU {gpu_id}: This GPU will process trawls from {start_idx} to {end_idx-1}")
        print(f"GPU {gpu_id}: Assigned {end_idx - start_idx} trawls")
        
        # If end_idx <= start_idx, this GPU has no work to do
        if end_idx <= start_idx:
            print(f"GPU {gpu_id}: No trawls assigned to this GPU. Exiting.")
            return

        # Create results directory
        results_dir = f"mcmc_results_{trawl_process_type}"
        results_dir = os.path.join(folder_path, results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"GPU {gpu_id}: Results directory: {results_dir}")
        print(f"GPU {gpu_id}: Checking if results directory exists: {os.path.exists(results_dir)}")
        print(f"GPU {gpu_id}: Checking if results directory is writable: {os.access(results_dir, os.W_OK)}")

        # Add sanity checks
        try:
            test_file_path = os.path.join(results_dir, f"test_write_gpu_{gpu_id}.txt")
            with open(test_file_path, 'w') as f:
                f.write(f"Test write from GPU {gpu_id} at {time.time()}")
            print(f"GPU {gpu_id}: Successfully wrote test file: {test_file_path}")
            
            # Check if file was actually written
            if os.path.exists(test_file_path):
                print(f"GPU {gpu_id}: Confirmed test file exists")
                with open(test_file_path, 'r') as f:
                    print(f"GPU {gpu_id}: Test file content: {f.read()}")
            else:
                print(f"GPU {gpu_id}: WARNING: Test file does not exist after writing!")
        except Exception as e:
            print(f"GPU {gpu_id}: Error writing test file: {str(e)}")
            traceback.print_exc()

        # Process assigned trawls
        print(f"GPU {gpu_id}: Processing trawls {start_idx} to {end_idx-1} out of {total_trawls}")

        for idx in range(start_idx, end_idx):
            trawl_start_time = time.time()
            
            # Create directory for this trawl
            trawl_dir = os.path.join(results_dir, f"trawl_{idx}")
            os.makedirs(trawl_dir, exist_ok=True)
            
            print(f"GPU {gpu_id}: Created directory for trawl {idx}: {trawl_dir}")
            print(f"GPU {gpu_id}: Checking if trawl directory exists: {os.path.exists(trawl_dir)}")
            print(f"GPU {gpu_id}: Checking if trawl directory is writable: {os.access(trawl_dir, os.W_OK)}")

            # Skip if already completed
            results_file = os.path.join(trawl_dir, "results.pkl")
            if os.path.exists(results_file):
                print(f"GPU {gpu_id}: Trawl {idx} already processed, skipping...")
                continue

            print(f"GPU {gpu_id}: Processing trawl {idx}/{total_trawls-1}")

            try:
                # Add sanity check for output directory
                test_trawl_file = os.path.join(trawl_dir, "processing_started.txt")
                with open(test_trawl_file, 'w') as f:
                    f.write(f"Started processing trawl {idx} at {time.time()}")
                
                # Run MCMC for this trawl
                print(f"GPU {gpu_id}: Starting MCMC for trawl {idx}...")
                
                
                try:
                    results = run_mcmc_for_trawl(
                        trawl_idx=idx,
                        true_trawls=true_trawls,
                        true_thetas=true_thetas,
                        approximate_log_likelihood_to_evidence=approximate_log_likelihood_to_evidence,
                        seed=seed,
                        num_samples=num_samples,
                        num_warmup=num_warmup,
                        num_burnin=num_burnin,
                        num_chains=num_chains
                    )
                    print(f"GPU {gpu_id}: MCMC completed for trawl {idx}")
                except Exception as e:
                    print(f"GPU {gpu_id}: Error in run_mcmc_for_trawl for trawl {idx}: {str(e)}")
                    traceback.print_exc()
                    with open(os.path.join(trawl_dir, "mcmc_error.txt"), 'w') as f:
                        f.write(f"Error in MCMC: {str(e)}\n\n{traceback.format_exc()}")
                    continue

                # Add true theta to results
                print(f"GPU {gpu_id}: Adding true theta and trawl to results for trawl {idx}")
                results['true_theta'] = true_thetas[idx].tolist()
                results['true_trawl'] = true_trawls[idx].tolist()

                # Save results
                print(f"GPU {gpu_id}: Saving results for trawl {idx} to {results_file}")
                try:
                    save_results(results, results_file)
                    print(f"GPU {gpu_id}: Results saved successfully for trawl {idx}")
                    # Verify file was written
                    if os.path.exists(results_file):
                        print(f"GPU {gpu_id}: Confirmed results file exists: {results_file}")
                        print(f"GPU {gpu_id}: File size: {os.path.getsize(results_file)} bytes")
                    else:
                        print(f"GPU {gpu_id}: WARNING: Results file does not exist after saving!")
                except Exception as e:
                    print(f"GPU {gpu_id}: Error saving results for trawl {idx}: {str(e)}")
                    traceback.print_exc()
                    with open(os.path.join(trawl_dir, "save_error.txt"), 'w') as f:
                        f.write(f"Error saving results: {str(e)}\n\n{traceback.format_exc()}")
                
                # Create and save plots
                print(f"GPU {gpu_id}: Creating plots for trawl {idx}")
                try:
                    create_and_save_plots(
                        results=results,
                        save_dir=trawl_dir
                    )
                    print(f"GPU {gpu_id}: Plots created successfully for trawl {idx}")
                except Exception as e:
                    print(f"GPU {gpu_id}: Error creating plots for trawl {idx}: {str(e)}")
                    traceback.print_exc()
                    with open(os.path.join(trawl_dir, "plot_error.txt"), 'w') as f:
                        f.write(f"Error creating plots: {str(e)}\n\n{traceback.format_exc()}")

                # Save memory by clearing results
                del results

                trawl_end_time = time.time()
                trawl_duration = trawl_end_time - trawl_start_time
                print(f"GPU {gpu_id}: Completed trawl {idx} in {trawl_duration:.2f} seconds")

            except Exception as e:
                print(f"GPU {gpu_id}: Error processing trawl {idx}: {str(e)}")
                traceback.print_exc()
                # Save the error to a file
                with open(os.path.join(trawl_dir, "error.txt"), 'w') as f:
                    f.write(f"Error processing trawl {idx}: {str(e)}\n\n{traceback.format_exc()}")

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"GPU {gpu_id}: Completed all assigned trawls in {total_duration:.2f} seconds")
    
    except Exception as e:
        print(f"GPU {gpu_id}: Critical error in main function: {str(e)}")
        traceback.print_exc()
        
        

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python posterior_sampling.py <gpu_id> <num_gpus> [<num_experiments>]")
        sys.exit(1)

    # Required arguments
    gpu_id = int(sys.argv[1])
    num_gpus = int(sys.argv[2])

    # Optional argument
    num_experiments_to_do = 10  # Default value
    #if len(sys.argv) > 3:
    #    num_experiments_to_do = int(sys.argv[3])

    # Call main with parsed arguments
    main(gpu_id, num_gpus, num_experiments_to_do)

