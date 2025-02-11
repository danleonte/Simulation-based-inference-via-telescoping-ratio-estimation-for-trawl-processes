import os


def find_min_loss_model(root_folder):
    min_loss = float('inf')
    params_path = None
    config_path = None

    for subfolder in os.scandir(root_folder):
        if subfolder.is_dir():
            validation_folder = os.path.join(subfolder.path, "validation_data")
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

    classifier_models_path = os.path.join(
        os.path.dirname(os.getcwd()), 'models', 'classifier')

    # NRE WITH SUMMARY
    paths_to_check.append(os.path.join(classifier_models_path,
                                       'NRE_summary_statistics'))

    # TRE WITH SUMMARY
    tre_super_folder_path = os.path.join(
        classifier_models_path, 'TRE_summary_statistics')
    paths_to_check += [f.path for f in os.scandir(
        tre_super_folder_path) if f.is_dir()]

    # TO ADD TRE WITHOUT SUMMARY

    results = []
    for root_path in paths_to_check:
        results.append(find_min_loss_model(root_path))

    for i in range(len(results)):

        group_name = paths_to_check[i].split("classifier", 1)[-1].lstrip("\\/")
        individual_result = results[i]
        if results[i] == None:

            print(group_name, ': ', None)
            print('\n')

        else:
            print(group_name, ': ')
            print(individual_result[0].split(
                "classifier", 1)[-1].lstrip("\\/"))
            print('BCE Loss is ', individual_result[-1])
            print('\n')
