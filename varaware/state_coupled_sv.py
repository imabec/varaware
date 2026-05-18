import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize


class StateCoupledSV:
    """
    State-coupled stochastic volatility model.

    Model:
        x_t = mu + phi * (x_{t-1} - mu) + w_t
        w_t ~ N(0, q_t)

        log q_t = alpha + gamma * |x_{t-1} - mu|

        y_t = x_t + v_t
        v_t ~ N(0, r)

    Current implementation estimates alpha and gamma using particle EM,
    while mu, phi, and r are treated as fixed.
    """

    def __init__(
        self,
        mu=0.0,
        phi=0.6,
        r=0.05,
        alpha_start=-2.0,
        gamma_start=0.3,
        n_particles=3000,
        n_paths=300,
        max_iter=20,
        tol=1e-3,
        step_size=0.5,
        seed=123,
        clip_log_q=(-20, 10),
        verbose=False
    ):
        self.mu = mu
        self.phi = phi
        self.r = r
        self.alpha_start = alpha_start
        self.gamma_start = gamma_start
        self.n_particles = n_particles
        self.n_paths = n_paths
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.seed = seed
        self.clip_log_q = clip_log_q
        self.verbose = verbose

        self.alpha_ = None
        self.gamma_ = None
        self.history_ = None
        self.is_fitted_ = False

    @staticmethod
    def _systematic_resample(weights, rng):
        n = len(weights)
        positions = (rng.random() + np.arange(n)) / n
        indexes = np.zeros(n, dtype=int)

        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0

        while i < n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        return indexes

    def _particle_filter(self, y, alpha, gamma, seed):
        rng = np.random.default_rng(seed)

        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]

        T = len(y)
        N = self.n_particles

        particles = np.zeros((T, N))
        ancestors = np.zeros((T, N), dtype=int)
        log_weights = np.zeros((T, N))

        particles[0] = rng.normal(
            loc=y[0],
            scale=np.sqrt(self.r),
            size=N
        )

        log_weights[0] = -0.5 * (
            np.log(2 * np.pi * self.r)
            + ((y[0] - particles[0]) ** 2) / self.r
        )
        log_weights[0] -= logsumexp(log_weights[0])

        loglik = logsumexp(log_weights[0])

        for t in range(1, T):
            weights_prev = np.exp(log_weights[t - 1])

            ancestor_idx = self._systematic_resample(weights_prev, rng)
            ancestors[t] = ancestor_idx

            x_prev = particles[t - 1, ancestor_idx]

            z = np.abs(x_prev - self.mu)
            log_q = alpha + gamma * z
            log_q = np.clip(log_q, *self.clip_log_q)
            q = np.exp(log_q)

            x_mean = self.mu + self.phi * (x_prev - self.mu)

            particles[t] = rng.normal(
                loc=x_mean,
                scale=np.sqrt(q)
            )

            log_w = -0.5 * (
                np.log(2 * np.pi * self.r)
                + ((y[t] - particles[t]) ** 2) / self.r
            )

            log_norm = logsumexp(log_w)
            log_weights[t] = log_w - log_norm

            loglik += log_norm - np.log(N)

        return {
            "particles": particles,
            "ancestors": ancestors,
            "log_weights": log_weights,
            "loglik": loglik
        }

    def _sample_particle_paths(self, pf_out, seed):
        rng = np.random.default_rng(seed)

        particles = pf_out["particles"]
        ancestors = pf_out["ancestors"]
        log_weights = pf_out["log_weights"]

        T, N = particles.shape
        M = self.n_paths

        paths = np.zeros((M, T))
        final_weights = np.exp(log_weights[-1])

        for m in range(M):
            idx = rng.choice(N, p=final_weights)
            paths[m, T - 1] = particles[T - 1, idx]

            for t in range(T - 1, 0, -1):
                idx = ancestors[t, idx]
                paths[m, t - 1] = particles[t - 1, idx]

        return paths

    def _estimate_alpha_gamma_from_paths(self, paths, eps=1e-8):
        Z_all = []
        R2_all = []

        for x in paths:
            z = np.abs(x[:-1] - self.mu)

            x_mean = self.mu + self.phi * (x[:-1] - self.mu)
            resid = x[1:] - x_mean

            Z_all.append(z)
            R2_all.append(resid ** 2 + eps)

        z_all = np.concatenate(Z_all)
        r2_all = np.concatenate(R2_all)

        def loss(params):
            alpha, gamma = params

            log_q = alpha + gamma * z_all
            log_q = np.clip(log_q, *self.clip_log_q)
            q = np.exp(log_q)

            return 0.5 * np.sum(log_q + r2_all / q)

        fit = minimize(
            loss,
            x0=np.array([self.alpha_start, self.gamma_start]),
            method="L-BFGS-B",
            bounds=[(-8, 4), (-5, 5)]
        )

        return {
            "alpha_est": fit.x[0],
            "gamma_est": fit.x[1],
            "success": fit.success,
            "fun": fit.fun
        }

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]

        alpha = self.alpha_start
        gamma = self.gamma_start

        history = []

        for it in range(self.max_iter):
            pf = self._particle_filter(
                y=y,
                alpha=alpha,
                gamma=gamma,
                seed=self.seed + 10 * it
            )

            paths = self._sample_particle_paths(
                pf,
                seed=self.seed + 10 * it + 1
            )

            ag_fit = self._estimate_alpha_gamma_from_paths(paths)

            alpha_new = ag_fit["alpha_est"]
            gamma_new = ag_fit["gamma_est"]

            delta_alpha = abs(alpha_new - alpha)
            delta_gamma = abs(gamma_new - gamma)

            history.append({
                "iter": it,
                "alpha_old": alpha,
                "gamma_old": gamma,
                "alpha_new": alpha_new,
                "gamma_new": gamma_new,
                "delta_alpha": delta_alpha,
                "delta_gamma": delta_gamma,
                "particle_loglik": pf["loglik"],
                "m_step_success": ag_fit["success"],
                "m_step_fun": ag_fit["fun"]
            })

            if self.verbose:
                print(
                    f"iter {it:02d} | "
                    f"alpha {alpha:.4f} -> {alpha_new:.4f} | "
                    f"gamma {gamma:.4f} -> {gamma_new:.4f} | "
                    f"delta_gamma={delta_gamma:.5f}"
                )

            alpha = (1 - self.step_size) * alpha + self.step_size * alpha_new
            gamma = (1 - self.step_size) * gamma + self.step_size * gamma_new

            if delta_alpha < self.tol and delta_gamma < self.tol:
                break

        self.alpha_ = alpha
        self.gamma_ = gamma
        self.history_ = pd.DataFrame(history)
        self.is_fitted_ = True

        return self

    def summary(self):
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary().")

        return {
            "alpha_est": self.alpha_,
            "gamma_est": self.gamma_,
            "mu": self.mu,
            "phi": self.phi,
            "r": self.r,
            "n_particles": self.n_particles,
            "n_paths": self.n_paths,
            "n_iter": len(self.history_),
            "converged": (
                self.history_["delta_alpha"].iloc[-1] < self.tol
                and self.history_["delta_gamma"].iloc[-1] < self.tol
            )
        }