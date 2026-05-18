# varaware/calibration/empirical_null.py
import numpy as np
import pandas as pd


class EmpiricalNullCalibration:
    """
    Empirical null calibration framework for state-coupled
    stochastic volatility models.

    Computes empirical null p-values and reliability labels
    from simulation results.
    """

    def __init__(
        self,
        results,
        gamma_true_col="gamma_true",
        gamma_est_col="gamma_est",
        null_gamma=0.0,
        highly_trustworthy=0.01,
        trustworthy=0.05,
        suggestive=0.10,
    ):
        self.results = results.copy()

        self.gamma_true_col = gamma_true_col
        self.gamma_est_col = gamma_est_col
        self.null_gamma = null_gamma

        self.highly_trustworthy = highly_trustworthy
        self.trustworthy = trustworthy
        self.suggestive = suggestive

        self.null_estimates_ = None
        self.is_fitted_ = False

    def build_null_distribution(self):
        """
        Extract null gamma estimates.
        """

        null_estimates = self.results.loc[
            self.results[self.gamma_true_col] == self.null_gamma,
            self.gamma_est_col
        ].values

        if len(null_estimates) == 0:
            raise ValueError(
                "No null simulations found. "
                "Check gamma_true values."
            )

        self.null_estimates_ = np.asarray(
            null_estimates,
            dtype=float
        )

        return self.null_estimates_

    def empirical_p_value(self, gamma_est):
        """
        Compute empirical null p-value.
        """

        if self.null_estimates_ is None:
            raise RuntimeError(
                "Null distribution has not been built. "
                "Call build_null_distribution() first."
            )

        exceed = np.sum(
            np.abs(self.null_estimates_) >= abs(gamma_est)
        )

        return (
            exceed + 1
        ) / (
            len(self.null_estimates_) + 1
        )

    def add_empirical_p_values(
        self,
        output_col="empirical_p"
    ):
        """
        Add empirical p-values to results dataframe.
        """

        if self.null_estimates_ is None:
            self.build_null_distribution()

        self.results[output_col] = self.results[
            self.gamma_est_col
        ].apply(self.empirical_p_value)

        return self.results

    def classify_reliability(self, p):
        """
        Convert empirical p-values into reliability labels.
        """

        if p <= self.highly_trustworthy:
            return "highly_trustworthy"

        elif p <= self.trustworthy:
            return "trustworthy"

        elif p <= self.suggestive:
            return "suggestive"

        else:
            return "not_trustworthy"

    def add_reliability_labels(
        self,
        p_col="empirical_p",
        output_col="reliability"
    ):
        """
        Add reliability labels to results dataframe.
        """

        if p_col not in self.results.columns:
            raise ValueError(
                f"{p_col} not found in results."
            )

        self.results[output_col] = self.results[
            p_col
        ].apply(self.classify_reliability)

        return self.results

    def fit(self):
        """
        Full calibration pipeline.
        """

        self.build_null_distribution()
        self.add_empirical_p_values()
        self.add_reliability_labels()

        self.is_fitted_ = True

        return self

    def summary(self):
        """
        Summary statistics for calibration behavior.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() before summary()."
            )

        summary_df = self.results.groupby(
            self.gamma_true_col
        ).agg({
            self.gamma_est_col: ["mean", "std"],
            "empirical_p": ["mean", "median"],
        })

        return summary_df

    def get_results(self):
        """
        Return calibrated results dataframe.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() before accessing results."
            )

        return self.results.copy()
