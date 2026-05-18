# varaware/simulation/scsv.py
import numpy as np
import pandas as pd


class SCSVSimulator:
    """
    Simulator for a state-coupled stochastic volatility latent state-space model.

    Model:
        x_t = mu + phi * (x_{t-1} - mu) + w_t
        w_t ~ N(0, q_t)

        log(q_t) = alpha + gamma * |x_{t-1} - mu|

        y_t = x_t + v_t
        v_t ~ N(0, r)
    """

    def __init__(
        self,
        T=200,
        mu=0.0,
        phi=0.6,
        alpha=-2.5,
        gamma=0.8,
        r=0.05,
        x0=0.0,
        seed=1,
        q_min=1e-8,
        clip_log_q=(-20, 10),
    ):
        self.T = T
        self.mu = mu
        self.phi = phi
        self.alpha = alpha
        self.gamma = gamma
        self.r = r
        self.x0 = x0
        self.seed = seed
        self.q_min = q_min
        self.clip_log_q = clip_log_q

        self._validate_params()

    def _validate_params(self):
        if self.T < 2:
            raise ValueError("T must be at least 2.")

        if self.r <= 0:
            raise ValueError("Observation variance r must be positive.")

        if self.q_min <= 0:
            raise ValueError("q_min must be positive.")

        if not (-1 < self.phi < 1):
            raise ValueError("phi should be between -1 and 1 for stable simulations.")

        if (
            not isinstance(self.clip_log_q, tuple)
            or len(self.clip_log_q) != 2
            or self.clip_log_q[0] >= self.clip_log_q[1]
        ):
            raise ValueError("clip_log_q must be a tuple like (-20, 10).")

    def simulate(self):
        rng = np.random.default_rng(self.seed)

        x = np.zeros(self.T)
        y = np.zeros(self.T)
        z = np.zeros(self.T)
        q = np.zeros(self.T)
        log_q = np.zeros(self.T)

        x[0] = self.x0
        y[0] = x[0] + rng.normal(0, np.sqrt(self.r))

        z[0] = abs(x[0] - self.mu)
        log_q[0] = self.alpha + self.gamma * z[0]
        log_q[0] = np.clip(log_q[0], *self.clip_log_q)
        q[0] = max(np.exp(log_q[0]), self.q_min)

        for t in range(1, self.T):
            z[t] = abs(x[t - 1] - self.mu)

            log_q[t] = self.alpha + self.gamma * z[t]
            log_q[t] = np.clip(log_q[t], *self.clip_log_q)
            q[t] = max(np.exp(log_q[t]), self.q_min)

            x_mean = self.mu + self.phi * (x[t - 1] - self.mu)
            x[t] = x_mean + rng.normal(0, np.sqrt(q[t]))

            y[t] = x[t] + rng.normal(0, np.sqrt(self.r))

        return pd.DataFrame({
            "time": np.arange(self.T),
            "x_true": x,
            "y": y,
            "z_true": z,
            "q_true": q,
            "log_q_true": log_q,
            "mu": self.mu,
            "phi": self.phi,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "r": self.r,
            "seed": self.seed,
        })

    def summary(self):
        return {
            "T": self.T,
            "mu": self.mu,
            "phi": self.phi,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "r": self.r,
            "x0": self.x0,
            "seed": self.seed,
            "q_min": self.q_min,
            "clip_log_q": self.clip_log_q,
        }


# Optional functional wrapper for backward compatibility
def simulate_scsv(**kwargs):
    return SCSVSimulator(**kwargs).simulate()
