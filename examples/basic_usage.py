from varaware import VarianceAwareModel
from varaware import simulate_variance_aware

df = simulate_variance_aware(
    n_subjects=50,
    T=100,
    seed=42
)

model = VarianceAwareModel()

model.fit(
    df,
    id_col="id",
    time_col="time",
    y_col="y",
    z_col="z"
)

print(model.summary())

print(model.recovery_summary())

vd = model.variance_decomposition()

print(vd.head())

print(vd[[
    "latent_fraction",
    "structured_fraction",
    "unexplained_fraction"
]].mean())