import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import gc  # Import garbage collector

# Load configuration
calibrate_bool = True
seq_len = 5000
calibrate_suffix = '_calibrated' if calibrate_bool else '_uncalibrated'
folder_path = r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/new_classifier/TRE_full_trawl/selected_models/mcmc_results_sup_ig_nig_5p_' + str(seq_len) + calibrate_suffix
trawl_folders = [item for item in os.listdir(folder_path) 
               if os.path.isdir(os.path.join(folder_path, item))]
               
print(f"Total folders found: {len(trawl_folders)}")
ranks = []
skipped_folders = 0

for i, trawl_folder in enumerate(trawl_folders):
    if i % 250 == 0:  # Print status every 250 files
        print(f"Processing folder {i}/{len(trawl_folders)}")
    
    # Check if results.pkl exists in the folder before attempting to open it
    result_file_path = os.path.join(folder_path, trawl_folder, 'results.pkl')
    if not os.path.isfile(result_file_path):
        skipped_folders += 1
        continue
    
    try:
        with open(result_file_path, 'rb') as f:
            # Load the dictionary
            results = pickle.load(f)
            # Extract only what we need
            coverage_value = results['coverage'].item()
            # Immediately delete the reference to the large dictionary
            del results
            ranks.append(coverage_value)
            
        # Force garbage collection every 100 iterations
        if i % 25 == 0:
            gc.collect()
    except Exception as e:
        print(f"Error with {trawl_folder}: {e}")
        
print(f"Skipped {skipped_folders} folders that didn't contain results.pkl")
print(f"Successfully processed {len(ranks)} folders")

# Continue only if we have data
if ranks:
    ranks = np.array(ranks)  # Convert to numpy array for vector operations
    
    num_points = 21
    theoretical_coverages = np.linspace(0, 1, num_points)
    empirical_coverages = []
    for alpha in theoretical_coverages:
        empirical_coverages.append(np.mean(ranks > 1 - alpha))
        
    plt.figure(figsize=(8, 6))
    plt.plot(theoretical_coverages, empirical_coverages, 'o-', label='Empirical calibration')
    plt.plot(theoretical_coverages, theoretical_coverages, '--', label='Perfect calibration')
    plt.xlabel('Theoretical coverage')
    plt.ylabel('Empirical coverage')
    plt.title('Expected Coverage Check')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure to parent directory
    parent_dir = os.path.dirname(folder_path)
    output_path = os.path.join(parent_dir, f'coverage_check_{num_points}_{seq_len}_{calibrate_suffix}.pdf')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
else:
    print("No valid data found to create plot.")