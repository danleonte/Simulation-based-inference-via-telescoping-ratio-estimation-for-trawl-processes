import os
import heapq


def find_min_loss_classifier(root_folder):
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


def find_top_k_classifiers(root_folder, k=1):
    """
    Find the top k models with the lowest validation loss in the given folder.

    Args:
        root_folder: Path to the folder containing model subfolders
        k: Number of top models to return

    Returns:
        List of tuples (params_path, config_path, loss) for the top k models
    """
    # Use a max heap of size k to keep track of models with smallest loss
    models_heap = []

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
                            params_path = os.path.join(
                                subfolder.path, f'params_iter_{best_model_iteration}.pkl')
                            config_path = os.path.join(
                                validation_folder, 'config.yaml')

                            # Use negative loss for max heap to simulate min heap
                            item = (-best_validation_loss, params_path,
                                    config_path, best_validation_loss)

                            if len(models_heap) < k:
                                heapq.heappush(models_heap, item)
                            elif -best_validation_loss > models_heap[0][0]:
                                # If current model loss is less than the largest in our heap
                                heapq.heappushpop(models_heap, item)

                except Exception as e:
                    print(f"Error reading {best_model_file}: {e}")

    # Convert heap to sorted list of results
    results = []
    while models_heap:
        _, params_path, config_path, loss = heapq.heappop(models_heap)
        # Insert at beginning to maintain ascending order
        results.insert(0, (params_path, config_path, loss))

    return results if results else None


if __name__ == '__main__':

    best_only = False

    if best_only:

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
            results.append(find_min_loss_classifier(root_path))

        for i in range(len(results)):

            group_name = paths_to_check[i].split(
                "classifier", 1)[-1].lstrip("\\/")
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

    else:
        # Number of top models to return
        k = 15  # You can change this to any value you want

        paths_to_check = []
        classifier_models_path = os.path.join(
            os.path.dirname(os.getcwd()), 'models', 'new_classifier')

        # NRE WITH SUMMARY
        paths_to_check.append(os.path.join(classifier_models_path,
                                           'NRE_full_trawl'))

        # TRE WITH SUMMARY
        tre_super_folder_path = os.path.join(
            classifier_models_path, 'TRE_full_trawl')
        paths_to_check += [f.path for f in os.scandir(
            tre_super_folder_path) if f.is_dir()]

        # TO ADD TRE WITHOUT SUMMARY

        # Process each path and get top k models
        for root_path in paths_to_check:
            group_name = root_path.split("classifier", 1)[-1].lstrip("/")
            print(f"\nTop {k} models for {group_name}:")

            top_models = find_top_k_classifiers(root_path, k)

            if top_models is None:
                print("No models found")
            else:
                for i, (params_path, config_path, loss) in enumerate(top_models):
                    relative_path = params_path.split(
                        "classifier", 1)[-1].lstrip("/")
                    model_dir = os.path.dirname(relative_path)
                    print(f"Rank {i+1}:")
                    print(f"  Model: {model_dir}")
                    print(f"  Path: {relative_path}")
                    print(f"  BCE Loss: {loss}")
                    print('\n')

        print('\n')
        print('\n')
        print('\n')
