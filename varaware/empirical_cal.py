# varaware/calibration/empirical_null.py

import numpy as np
import pandas as pd


def build_null_distribution(
    results,
    gamma_true_col="gamma_true",
    gamma_est_col="gamma_est",
    null_gamma=0.0
):
    """
    Extract null gamma estimates from simulation results.

    Parameters
    ----------
    results : pd.DataFrame
        Simulation results dataframe.

    gamma_true_col : str
        Column containing true gamma values.

    gamma_est_col : str
        Column containing estimated gamma values.

    null_gamma : float
        Value defining the null coupling regime.

    Returns
    -------
    np.ndarray
        Null gamma estimates.
    """

    null_estimates = results.loc[
        results[gamma_true_col] == null_gamma,
        gamma_est_col
    ].values

    if len(null_estimates) == 0:
        raise ValueError(
            "No null simulations found. "
            "Check gamma_true values."
        )

    return np.asarray(null_estimates, dtype=float)


def empirical_null_p_value(
    gamma_est,
    null_estimates
):
    """
    Empirical null exceedance probability.

    Computes:

        p = (1 + #{|null| >= |obs|}) / (N + 1)

    Parameters
    ----------
    gamma_est : float
        Observed gamma estimate.

    null_estimates : array-like
        Null gamma estimates.

    Returns
    -------
    float
        Empirical p-value.
    """

    null_estimates = np.asarray(null_estimates, dtype=float)

    exceed = np.sum(
        np.abs(null_estimates) >= abs(gamma_est)
    )

    return (exceed + 1) / (len(null_estimates) + 1)


def add_empirical_p_values(
    results,
    gamma_est_col="gamma_est",
    gamma_true_col="gamma_true",
    null_gamma=0.0,
    output_col="empirical_p"
):
    """
    Add empirical p-values to simulation results.

    Parameters
    ----------
    results : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    results = results.copy()

    null_estimates = build_null_distribution(
        results,
        gamma_true_col=gamma_true_col,
        gamma_est_col=gamma_est_col,
        null_gamma=null_gamma
    )

    results[output_col] = results[gamma_est_col].apply(
        lambda g: empirical_null_p_value(
            gamma_est=g,
            null_estimates=null_estimates
        )
    )

    return results


def classify_reliability(
    p,
    highly_trustworthy=0.01,
    trustworthy=0.05,
    suggestive=0.10
):
    """
    Convert empirical p-values into reliability labels.
    """

    if p <= highly_trustworthy:
        return "highly_trustworthy"

    elif p <= trustworthy:
        return "trustworthy"

    elif p <= suggestive:
        return "suggestive"

    else:
        return "not_trustworthy"


def add_reliability_labels(
    results,
    p_col="empirical_p",
    output_col="reliability"
):
    """
    Add trustworthiness labels to results.
    """

    results = results.copy()

    results[output_col] = results[p_col].apply(
        classify_reliability
    )

    return results