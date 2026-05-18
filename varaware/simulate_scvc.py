# varaware/simulation/scsv.py

import numpy as np
import pandas as pd


def simulate_scsv(
    T=200,
    mu=0.0,
    phi=0.6,
    alpha=-2.5,
    gamma=0.8,
    r=0.05,
    x0=0.0,
    seed=1,
    q_min=1e-8,
    clip_log_q=(-20, 10)
):
    """
    Simulate a state-coupled stochastic volatility latent state-space model.

    Model:
        x_t = mu + phi * (x_{t-1} - mu) + w_t
        w_t ~ N(0, q_t)

        log(q_t) = alpha + gamma * |x_{t-1} - mu|

        y_t = x_t + v_t
        v_t ~ N(0, r)

    Returns
    -------
    pd.DataFrame
        Columns:
        time, x_true, y, z_true, q_true, log_q_true
    """

    if T < 2:
        raise ValueError("T must be at least 2.")

    if r <= 0:
        raise ValueError("Observation variance r must be positive.")

    if q_min <= 0:
        raise ValueError("q_min must be positive.")

    if not (-1 < phi < 1):
        raise ValueError("phi should be between -1 and 1 for stable simulations.")

    rng = np.random.default_rng(seed)

    x = np.zeros(T)
    y = np.zeros(T)
    z = np.zeros(T)
    q = np.zeros(T)
    log_q = np.zeros(T)

    x[0] = x0
    y[0] = x[0] + rng.normal(0, np.sqrt(r))

    z[0] = abs(x[0] - mu)
    log_q[0] = alpha + gamma * z[0]
    log_q[0] = np.clip(log_q[0], *clip_log_q)
    q[0] = max(np.exp(log_q[0]), q_min)

    for t in range(1, T):
        z[t] = abs(x[t - 1] - mu)

        log_q[t] = alpha + gamma * z[t]
        log_q[t] = np.clip(log_q[t], *clip_log_q)
        q[t] = max(np.exp(log_q[t]), q_min)

        x_mean = mu + phi * (x[t - 1] - mu)
        x[t] = x_mean + rng.normal(0, np.sqrt(q[t]))

        y[t] = x[t] + rng.normal(0, np.sqrt(r))

    return pd.DataFrame({
        "time": np.arange(T),
        "x_true": x,
        "y": y,
        "z_true": z,
        "q_true": q,
        "log_q_true": log_q,
        "mu": mu,
        "phi": phi,
        "alpha": alpha,
        "gamma": gamma,
        "r": r,
        "seed": seed
    })


# Backward-compatible alias
simulate_scsv_no_rho = simulate_scsv