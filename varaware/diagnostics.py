import matplotlib.pyplot as plt


def plot_latent_fit(model, subject_id):
    """
    Plot observed signal and fitted latent AR trend for one subject.
    """

    data = model.get_fitted_data()
    id_col = model.id_col_
    time_col = model.time_col_
    y_col = model.y_col_

    d = data[data[id_col] == subject_id].sort_values(time_col)

    if d.empty:
        raise ValueError(f"No data found for subject_id={subject_id}")

    plt.figure(figsize=(8, 4))
    plt.plot(d[time_col], d[y_col], label="Observed")
    plt.plot(d[time_col], d["x_hat"], label="Fitted latent trend")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.title(f"Latent Fit: Subject {subject_id}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residual_structure(model):
    """
    Scatterplot of structured feature z against residuals.
    """

    data = model.get_fitted_data()
    z_col = model.z_col_

    plt.figure(figsize=(6, 5))
    plt.scatter(data[z_col], data["resid_struct"], alpha=0.4)
    plt.xlabel("Structured feature z")
    plt.ylabel("Residual after latent fit")
    plt.title("Residual Structure")
    plt.tight_layout()
    plt.show()


def plot_gamma_distribution(model):
    """
    Histogram of subject-level gamma estimates.
    """

    est = model.get_subject_estimates()

    plt.figure(figsize=(6, 4))
    plt.hist(est["gamma_hat"], bins=20)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Estimated gamma")
    plt.ylabel("Count")
    plt.title("Distribution of Gamma Estimates")
    plt.tight_layout()
    plt.show()