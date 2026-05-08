import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


class VarianceAwareModel:
    """
    Two-stage variance-aware residual model.

    Stage 1:
        Fit subject-level AR(1) dynamics to the observed signal.

    Stage 2:
        Extract residuals and estimate whether structured feature z
        explains residual variability.
    """

    def __init__(self):
        self.fitted_ = False
        self.data_ = None
        self.subject_estimates_ = None
        self.id_col_ = None
        self.time_col_ = None
        self.y_col_ = None
        self.z_col_ = None

    def fit(self, data, id_col="id", time_col="time", y_col="y", z_col="z"):
        """
        Fit the two-stage model.

        Parameters
        ----------
        data : pandas.DataFrame
            Long-format data.
        id_col : str
            Subject identifier column.
        time_col : str
            Time column.
        y_col : str
            Observed signal column.
        z_col : str
            Structured residual feature column.
        """

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.y_col_ = y_col
        self.z_col_ = z_col

        required = [id_col, time_col, y_col, z_col]
        missing = [col for col in required if col not in data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        fitted_rows = []
        estimates = []

        for sid, d in data.groupby(id_col):
            d = d.sort_values(time_col).copy()

            if len(d) < 3:
                continue

            y = d[y_col].to_numpy()
            z = d[z_col].to_numpy()

            y_t = y[:-1]
            y_next = y[1:]

            X_ar = sm.add_constant(y_t)
            ar_model = sm.OLS(y_next, X_ar).fit()

            c_hat = ar_model.params[0]
            phi_hat = ar_model.params[1]

            x_hat = np.empty(len(d))
            x_hat[0] = y[0]

            for t in range(1, len(d)):
                x_hat[t] = c_hat + phi_hat * y[t - 1]

            resid = y - x_hat

            X_z = sm.add_constant(z)
            z_model = sm.OLS(resid, X_z).fit()

            gamma_hat = z_model.params[1]
            gamma_se = z_model.bse[1]
            gamma_p = z_model.pvalues[1]

            d["x_hat"] = x_hat
            d["resid_struct"] = resid
            d["phi_hat"] = phi_hat
            d["gamma_hat"] = gamma_hat

            estimates.append(
                {
                    id_col: sid,
                    "n_time": len(d),
                    "c_hat": c_hat,
                    "phi_hat": phi_hat,
                    "gamma_hat": gamma_hat,
                    "gamma_se": gamma_se,
                    "gamma_p": gamma_p,
                    "resid_sd": np.std(resid, ddof=1),
                }
            )

            fitted_rows.append(d)

        if not fitted_rows:
            raise ValueError("No subjects had enough observations to fit the model.")

        self.data_ = pd.concat(fitted_rows, ignore_index=True)
        self.subject_estimates_ = pd.DataFrame(estimates)
        self.fitted_ = True

        return self

    def summary(self):
        """
        Return a summary of fitted subject-level parameters.
        """

        self._check_fitted()

        est = self.subject_estimates_

        summary = {
            "n_subjects": len(est),
            "mean_phi_hat": est["phi_hat"].mean(),
            "sd_phi_hat": est["phi_hat"].std(),
            "mean_gamma_hat": est["gamma_hat"].mean(),
            "sd_gamma_hat": est["gamma_hat"].std(),
            "median_gamma_p": est["gamma_p"].median(),
            "proportion_gamma_p_lt_0_05": np.mean(est["gamma_p"] < 0.05),
        }

        return pd.DataFrame([summary])

    def get_subject_estimates(self):
        """
        Return subject-level parameter estimates.
        """
        self._check_fitted()
        return self.subject_estimates_.copy()

    def get_fitted_data(self):
        """
        Return original data plus fitted latent trend and residuals.
        """
        self._check_fitted()
        return self.data_.copy()

    def recovery_summary(
        self,
        phi_true_col="phi_true",
        gamma_true_col="gamma_true",
    ):
        """
        If true simulation parameters exist, return recovery correlations.
        """

        self._check_fitted()

        est = self.subject_estimates_.copy()
        data = self.data_

        if phi_true_col not in data.columns or gamma_true_col not in data.columns:
            raise ValueError("True parameter columns not found in fitted data.")

        truth = (
            data.groupby(self.id_col_)[[phi_true_col, gamma_true_col]]
            .first()
            .reset_index()
        )

        merged = est.merge(truth, on=self.id_col_, how="left")

        if merged[phi_true_col].nunique() > 1:
            phi_corr = spearmanr(merged[phi_true_col], merged["phi_hat"]).statistic
        else:
            phi_corr = np.nan

        if merged[gamma_true_col].nunique() > 1:
            gamma_corr = spearmanr(
                merged[gamma_true_col], merged["gamma_hat"]
            ).statistic
        else:
            gamma_corr = np.nan

        return pd.DataFrame(
            [
                {
                    "phi_spearman": phi_corr,
                    "gamma_spearman": gamma_corr,
                    "gamma_hat_mean": merged["gamma_hat"].mean(),
                    "gamma_true_mean": merged[gamma_true_col].mean(),
                    "gamma_sign_accuracy": np.mean(
                        np.sign(merged["gamma_hat"])
                        == np.sign(merged[gamma_true_col])
                    ),
                }
            ]
        )
    def variance_decomposition(self):
     """
    Estimate variance decomposition of the observed signal.

    Decomposes:
        total variance =
            latent trend variance +
            structured residual variance +
            unexplained residual variance

    Returns
    -------
    pandas.DataFrame
        Subject-level variance decomposition.
    """

     self._check_fitted()

     results = []

     for sid, d in self.data_.groupby(self.id_col_):

        y = d[self.y_col_].to_numpy()
        x_hat = d["x_hat"].to_numpy()
        z = d[self.z_col_].to_numpy()

        gamma_hat = d["gamma_hat"].iloc[0]

        structured_component = gamma_hat * z

        residual_noise = y - x_hat - structured_component

        total_var = np.var(y, ddof=1)

        latent_var = np.var(x_hat, ddof=1)
        structured_var = np.var(structured_component, ddof=1)
        unexplained_var = np.var(residual_noise, ddof=1)

        results.append({
            self.id_col_: sid,

            "total_variance": total_var,

            "latent_variance": latent_var,
            "structured_variance": structured_var,
            "unexplained_variance": unexplained_var,

            "latent_fraction":
                latent_var / total_var if total_var > 0 else np.nan,

            "structured_fraction":
                structured_var / total_var if total_var > 0 else np.nan,

            "unexplained_fraction":
                unexplained_var / total_var if total_var > 0 else np.nan
        })

     return pd.DataFrame(results)
    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet.")
   