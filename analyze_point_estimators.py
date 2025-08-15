if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from src.utils.acf_functions import get_acf
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig


def convert_numpy_string_to_array(x):
    """
    Convert numpy string format '[1.2 3.4 5.6]' to actual numpy array
    """
    if pd.isna(x):
        return np.array([])

    # Convert to string and clean up
    x = str(x).strip()

    # Remove brackets
    if x.startswith('[') and x.endswith(']'):
        x = x[1:-1]

    # Split on whitespace and convert to float
    try:
        values = [float(val) for val in x.split()]
        return np.array(values)
    except:
        return np.array([])


def compare_acf_point_estimators(true_theta, infered_theta_acf, mle_or_gmm, num_lags):

    # acf errors
    H = np.arange(1, num_lags + 1)
    # this should be num_trawls, num_lags shaped
    theoretical_acf = acf_func(H, true_theta[:, :2])
    infered_acf = acf_func(H, infered_theta_acf)  # infered_theta[:, :2])
    acf_differences = np.abs(theoretical_acf - infered_acf)
    # might want to convert these to integrals or to sums
    med_L1_acf = np.median(np.mean(acf_differences, axis=1))
    L1_acf = np.mean(np.mean(acf_differences, axis=1))
    L2_acf = np.mean(np.sqrt(np.mean(acf_differences**2, axis=1)))
    rMSE_acf = np.mean(np.mean(acf_differences**2, axis=1))**0.5

    acf_estimation_error = {
        'med_L1_acf': med_L1_acf,
        'L1_acf': L1_acf,
        'L2_acf': L2_acf,
        'rMSE_acf': rMSE_acf
    }

    np.save(os.path.join(results_path, mle_or_gmm + f'_acf_estimation_error_{seq_len}_{num_lags}.npy'),
            acf_estimation_error)


def compare_marginal_point_estimators(true_theta, infered_theta, mle_or_gmm):

    # marginal errors
    marginal_medAEs = np.median(
        np.abs((true_theta[:, 2:] - infered_theta)), axis=0)
    marginal_MAEs = np.mean(
        np.abs((true_theta[:, 2:] - infered_theta)), axis=0)
    marginal_MSEs = np.mean(
        np.square((true_theta[:, 2:] - infered_theta)), axis=0)**0.5

    print('Starting KL divergence calculations')

    batched_true_thetas = [jnp.array(true_theta[i:i + 100, 2:])
                           for i in range(0, len(true_theta), 100)]
    batched_inferred_thetas = [
        jnp.array(infered_theta[i:i + 100]) for i in range(0, len(infered_theta), 100)]

    forward_kl = []
    rev_kl = []

    for i in range(len(batched_true_thetas)):

        num_samples = 7500
        params1 = batched_true_thetas[i]
        params2 = batched_inferred_thetas[i]
        vec_key = jax.random.split(jax.random.PRNGKey(12414), len(params1))

        forward_kl.append(vec_monte_carlo_kl_3_param_nig(
            params1, params2, vec_key, num_samples)[0].mean().item())
        rev_kl.append(vec_monte_carlo_kl_3_param_nig(
            params2, params1, vec_key, num_samples)[0].mean().item())

    marginal_estimation_error = {
        'marginal_medAE': marginal_medAEs,
        'marginal_MAEs': marginal_MAEs,
        'marginal_MSEs': marginal_MSEs,
        'forward_kl': np.mean(forward_kl),
        'rev_kl': np.mean(rev_kl),
    }

    np.save(os.path.join(results_path, mle_or_gmm + f'_marginal_estimation_error_{seq_len}.npy'),
            marginal_estimation_error)
    return marginal_estimation_error
    # np.load(os.path.join(results_path,'acf_estimation_error.npy'),allow_pickle=True)


def compare_point_estimators(true_theta, infered_theta, mle_or_gmm, num_lags):

    # acf errors
    H = np.arange(1, num_lags + 1)
    # this should be num_trawls, num_lags shaped
    theoretical_acf = acf_func(H, true_theta[:, :2])
    infered_acf = acf_func(H, infered_theta[:, :2])
    acf_differences = np.abs(theoretical_acf - infered_acf)
    # might want to convert these to integrals or to sums
    med_L1_acf = np.median(np.mean(acf_differences, axis=1))
    L1_acf = np.mean(np.mean(acf_differences, axis=1))
    L2_acf = np.mean(np.sqrt(np.mean(acf_differences**2, axis=1)))
    rMSE_acf = np.mean(np.mean(acf_differences**2, axis=1))**0.5

    # marginal errors
    marginal_medAEs = np.median(
        np.abs((true_theta - infered_theta)[:, 2:]), axis=0)
    marginal_MAEs = np.mean(
        np.abs((true_theta - infered_theta)[:, 2:]), axis=0)
    marginal_MSEs = np.mean(
        np.square((true_theta - infered_theta)[:, 2:]), axis=0)**0.5

    print('Starting KL divergence calculations')

    batched_true_thetas = [jnp.array(true_theta[i:i + 100, 2:])
                           for i in range(0, len(true_theta), 100)]
    batched_inferred_thetas = [
        jnp.array(infered_theta[i:i + 100, 2:]) for i in range(0, len(infered_acf), 100)]

    forward_kl = []
    rev_kl = []

    for i in range(len(batched_true_thetas)):

        num_samples = 7500
        params1 = batched_true_thetas[i]
        params2 = batched_inferred_thetas[i]
        vec_key = jax.random.split(jax.random.PRNGKey(12414), len(params1))

        forward_kl.append(vec_monte_carlo_kl_3_param_nig(
            params1, params2, vec_key, num_samples)[0].mean().item())
        rev_kl.append(vec_monte_carlo_kl_3_param_nig(
            params2, params1, vec_key, num_samples)[0].mean().item())

    acf_estimation_error = {
        'med_L1_acf': med_L1_acf,
        'L1_acf': L1_acf,
        'L2_acf': L2_acf,
        'rMSE_acf': rMSE_acf
    }

    marginal_estimation_error = {
        'marginal_medAE': marginal_medAEs,
        'marginal_MAEs': marginal_MAEs,
        'marginal_MSEs': marginal_MSEs,
        'forward_kl': np.mean(forward_kl),
        'rev_kl': np.mean(rev_kl),
    }

    np.save(os.path.join(results_path, mle_or_gmm + f'_acf_estimation_error_{seq_len}_{num_lags}.npy'),
            acf_estimation_error)
    np.save(os.path.join(results_path, mle_or_gmm + f'_marginal_estimation_error_{seq_len}.npy'),
            marginal_estimation_error)
    # np.load(os.path.join(results_path,'acf_estimation_error.npy'),allow_pickle=True)


if __name__ == '__main__':

    seq_len = 2000
    num_lags = 35

    acf_func = jax.vmap(get_acf('sup_IG'), in_axes=(None, 0))

    MLE_TRE = False
    MLE_NRE = False
    GMM = False
    NBE = True
    assert MLE_TRE + MLE_NRE + GMM + NBE == 1

    if MLE_TRE and not MLE_NRE and not GMM and not NBE:

        folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\point_estimators\calibrated_TRE'
        results_path = os.path.join(folder_path, f'TRE_{seq_len}_num_rows_160')
        os.makedirs(results_path, exist_ok=True)

        df = pd.read_pickle(os.path.join(
            results_path, f'TRE_{seq_len}.pkl'))
        true_theta = np.array([np.array(i) for i in df.true_theta.values])
        infered_theta = np.array([np.array(i) for i in df.MLE.values])
        compare_point_estimators(true_theta, infered_theta, 'MLE', num_lags)

    elif MLE_NRE and not MLE_TRE and not GMM and not NBE:

        folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\point_estimators\NRE'
        results_path = os.path.join(folder_path, f'NRE_{seq_len}_num_rows_160')
        os.makedirs(results_path, exist_ok=True)

        df = pd.read_pickle(os.path.join(
            results_path, f'NRE_{seq_len}.pkl'))

        true_theta = np.array([np.array(i) for i in df.true_theta.values])
        infered_theta = np.array([np.array(i) for i in df.MLE.values])
        compare_point_estimators(true_theta, infered_theta, 'MLE', num_lags)

    elif NBE and not MLE_NRE and not MLE_TRE and not GMM:

        pass

    elif GMM:

        # folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\TRE_full_trawl\selected_models\point_estimators\GMM'
        folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\point_estimators\GMM'

        results_path = folder_path

        ####  do marginal ####
        df_marginal = pd.read_csv(os.path.join(
            results_path, f'marginal_GMM_seq_len_{seq_len}_num_trawls_to_use_10000.csv'))  # f'ACF_{seq_len}_{num_lags}.pkl'))

        df_marginal = df_marginal.replace({None: np.nan}).dropna()

        # there are problems reading the pickle saved from the cluster because of differentt
        # numpy version. to this end, i save as a csv and then change the strings to arrays after
        # reading
        df_marginal['true_theta'] = df_marginal['true_theta'].apply(
            convert_numpy_string_to_array)
        df_marginal['GMM'] = df_marginal['GMM'].apply(
            convert_numpy_string_to_array)

        true_marginal_theta = np.array([np.array(i)
                                        for i in df_marginal.true_theta.values])
        infered_marginal_theta = np.vstack(
            [np.array(i) for i in df_marginal.GMM.values])
        compare_marginal_point_estimators(
            true_marginal_theta, infered_marginal_theta, 'GMM')

        print('only did marginal GMM, acf already done before')
        raise ValueError
        #### do acf #####
        df_acf = pd.read_pickle(os.path.join(
            results_path, f'ACF_{seq_len}_{num_lags}.pkl'))
        df_acf = df_acf.replace({None: np.nan}).dropna()

        true_acf_theta = np.array([np.array(i)
                                   for i in df_acf.true_theta.values])
        infered_acf_theta = np.vstack([np.array(i) for i in df_acf.GMM.values])
        compare_acf_point_estimators(
            true_acf_theta, infered_acf_theta, 'GMM', num_lags)

    else:
        raise ValueError
