import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from varaware import VarianceAwareModel


def simulate_pure_case(
    n_subjects=50,
    T=100,
    phi=0.8,
    latent_sd=0.5,
    noise_sd=0.3,
    scenario="pure_latent",
    seed=123,
):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_subjects):
        x = np.zeros(T)
        x[0] = rng.normal(0, latent_sd)

        for t in range(1, T):
            x[t] = phi * x[t - 1] + rng.normal(0, latent_sd)

        eps = rng.normal(0, noise_sd, T)

        if scenario == "pure_latent":
            y = x + eps
        elif scenario == "pure_noise":
            y = eps
        else:
            raise ValueError("scenario must be 'pure_latent' or 'pure_noise'")

        for t in range(T):
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "y": y[t],
                    "z": 0.0,
                    "scenario": scenario,
                }
            )

    return pd.DataFrame(rows)


def run_fluctuation_test(df):
    model = VarianceAwareModel()
    model.fit(df, id_col="id", time_col="time", y_col="y", z_col="z")
    fd = model.fluctuation_decomposition()
    return model, fd


def main():
    latent_df = simulate_pure_case(scenario="pure_latent", seed=1)
    noise_df = simulate_pure_case(scenario="pure_noise", seed=2)

    _, latent_fd = run_fluctuation_test(latent_df)
    _, noise_fd = run_fluctuation_test(noise_df)

    print("PURE LATENT")
    print(latent_fd[["latent_fraction", "fluctuation_fraction"]].describe())

    print("\nPURE NOISE")
    print(noise_fd[["latent_fraction", "fluctuation_fraction"]].describe())

    plot_df = pd.concat(
        [
            latent_fd.assign(scenario="pure_latent"),
            noise_fd.assign(scenario="pure_noise"),
        ]
    )

    plot_df.boxplot(
        column=["latent_fraction", "fluctuation_fraction"],
        by="scenario",
        figsize=(8, 5),
    )

    plt.suptitle("")
    plt.title("Fluctuation Decomposition: Pure Latent vs Pure Noise")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()