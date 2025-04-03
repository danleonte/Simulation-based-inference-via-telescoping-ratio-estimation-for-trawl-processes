import os


def find_min_loss_summary_stats(root_folder):
    min_loss = float('inf')
    params_path = None
    config_path = None

    for subfolder in os.scandir(root_folder):
        if subfolder.is_dir():
            validation_folder = os.path.join(subfolder.path, "best_model")
            best_model_file = os.path.join(
                validation_folder, "best_model_info.txt")

            if os.path.exists(best_model_file):
                try:
                    with open(best_model_file, "r") as file:
                        lines = file.readlines()
                        best_model_iteration = None
                        best_validation_loss = None

                        for line in lines:
                            if "Best model iteration" in line:
                                best_model_iteration = int(
                                    line.split(":")[1].strip())
                            elif "Best validation loss" in line:
                                best_validation_loss = float(
                                    line.split(":")[1].strip())

                        if best_model_iteration is not None and best_validation_loss is not None:
                            if best_validation_loss < min_loss:
                                min_loss = best_validation_loss
                                params_path = os.path.join(
                                    subfolder.path, f'params_iter_{best_model_iteration}.pkl')
                                config_path = os.path.join(
                                    validation_folder, 'config.yaml')

                except Exception as e:
                    print(f"Error reading {best_model_file}: {e}")

    return (params_path, config_path, min_loss) if min_loss != float('inf') else None


if __name__ == '__main__':
    paths_to_check = []

    root_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())), 'models', 'summary_statistics', 'learn_marginal','direct')

    result = find_min_loss_summary_stats(root_folder)
    output_file = os.path.join(root_folder,   os.path.join(
        root_folder, "best_model_summary.txt"))

    if result:
        params_path, config_path, min_loss = result
        print(f"Best Params Path: {params_path}")
        print(f"Best Config Path: {config_path}")
        print(f"Minimum Validation Loss: {min_loss}")

        # Save the best model information to a file
        with open(output_file, "w") as f:
            f.write(f"Best Params Path: {params_path}\n")
            f.write(f"Best Config Path: {config_path}\n")
            f.write(f"Minimum Validation Loss: {min_loss}\n")

        print(f"Best model information saved to: {output_file}")
    else:
        print("No valid model found.")
