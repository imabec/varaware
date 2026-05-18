import numpy as np
import pandas as pd


def simulate_variance_aware(
    n_subjects=50,
    T=100,
    phi_mean=0.75,
    phi_sd=0.08,
    gamma_mean=1.0,
    gamma_sd=0.25,
    process_sd=0.35,
    obs_sd=0.50,
    rho_xz=0.0,
    structured_type="gaussian",
    nonlinear_latent=False,
    seed=123,
):
    """
    Simulate longitudinal data from a variance-aware dynamical system.

    Model:
        x_t = phi * x_{t-1} + process noise
        y_t = x_t + gamma * z_t + observation noise

    Parameters
    ----------
    rho_xz : float
        Correlation between latent state x and structured feature z.
    structured_type : str
        One of: "gaussian", "sinusoidal", "bursty", "regime_shift".
    nonlinear_latent : bool
        If True, simulate x_t using a weak nonlinear cubic term.
    """

    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_subjects):
        phi_i = np.clip(rng.normal(phi_mean, phi_sd), 0.1, 0.98)
        gamma_i = rng.normal(gamma_mean, gamma_sd)

        x = np.zeros(T)
        x[0] = rng.normal(0, 1)

        for t in range(1, T):
            if nonlinear_latent:
                x[t] = (
                    phi_i * x[t - 1]
                    - 0.08 * x[t - 1] ** 3
                    + rng.normal(0, process_sd)
                )
            else:
                x[t] = phi_i * x[t - 1] + rng.normal(0, process_sd)

        x_std = (x - x.mean()) / (x.std() + 1e-8)

        if structured_type == "gaussian":
            noise = rng.normal(0, 1, T)

        elif structured_type == "sinusoidal":
            time = np.arange(T)
            noise = np.sin(2 * np.pi * time / 12) + rng.normal(0, 0.5, T)

        elif structured_type == "bursty":
            noise = rng.normal(0, 0.5, T)
            burst_idx = rng.choice(T, size=max(1, T // 10), replace=False)
            noise[burst_idx] += rng.normal(0, 3.0, len(burst_idx))

        elif structured_type == "regime_shift":
            noise = rng.normal(0, 1, T)
            noise[T // 2:] += 1.0

        else:
            raise ValueError(
                "structured_type must be one of: "
                "'gaussian', 'sinusoidal', 'bursty', 'regime_shift'"
            )

        noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        z = rho_xz * x_std + np.sqrt(max(0, 1 - rho_xz ** 2)) * noise
        z = z - z.mean()

        y = x + gamma_i * z + rng.normal(0, obs_sd, T)

        for t in range(T):
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "x_true": x[t],
                    "z": z[t],
                    "y": y[t],
                    "phi_true": phi_i,
                    "gamma_true": gamma_i,
                    "rho_xz": rho_xz,
                    "structured_type": structured_type,
                    "nonlinear_latent": nonlinear_latent,
                }
            )

    return pd.DataFrame(rows)
