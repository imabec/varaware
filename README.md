# varaware

`varaware` is a Python framework for structured residual analysis in longitudinal biological dynamical systems.

The package is designed for settings in which variability may contain meaningful biological or dynamical organization rather than representing purely nuisance noise. Examples include neural signals, physiological trajectories, behavioral time series, and other longitudinal biological systems.

The framework implements a two-stage variance-aware approach:

1. latent dynamical modeling of dominant system trajectories
2. structured residual decomposition and analysis

After estimating latent trajectory dynamics, the framework evaluates whether residual variability contains organized structure beyond the primary latent process.

Current functionality includes:
- longitudinal dynamical simulation
- latent AR trajectory estimation
- structured residual analysis
- variance decomposition
- parameter recovery simulations
- diagnostic visualization

The package is intended primarily for exploratory and methodological analysis of biological systems in which residual variability may itself carry interpretable dynamical information.
# Model Functions, Use Cases, and Limitations

## 1. Dominant Temporal / Fluctuation Decomposition

### Function
`fluctuation_decomposition()`

### Purpose

This model separates biological time-series variability into:

1. dominant temporal organization
2. fluctuation-driven variability

The decomposition is:

$$
y_t = \hat{x}_t + r_t
$$

where:

- $\(\hat{x}_t\)$ = fitted dominant temporal component
- $\(r_t\)$ = fluctuation component

The temporal component is currently estimated using a subject-level AR(1)-style approximation.

---

## What this model is good at

This model works best when:

- the signal contains meaningful temporal persistence,
- biological dynamics evolve gradually over time,
- fluctuations around the dominant trajectory are biologically interesting,
- and the primary question is:

> how much variability is attributable to smooth temporal organization versus fluctuations?

Examples include:

- EEG coherence trajectories
- physiological regulation
- longitudinal behavioral dynamics
- repeated biological measurements
- slowly evolving biological systems

---

## What this model can estimate well

The model appears reasonably capable of estimating:

- whether signals are fluctuation-dominated,
- whether strong temporal dependence exists,
- relative contributions of temporal organization versus residual variability,
- and differences in fluctuation dominance across subjects or groups.

---

## What this model is NOT designed for

This model is not intended to:

- recover exact hidden biological states,
- infer mechanistic neural generators,
- estimate causal dynamics,
- or fully model nonlinear state-space behavior.

The fitted temporal component should therefore be interpreted as:

> a dominant temporal approximation

rather than:

> the true hidden latent state.

---

## Settings where this model may fail

This model may perform poorly when:

- the signal is highly nonlinear,
- rapid oscillations dominate over smooth temporal trends,
- abrupt regime shifts occur,
- temporal dependence changes dynamically over time,
- or multiple overlapping timescales exist.

Performance may also degrade when:

- very short time series are used,
- noise dominates the signal,
- or temporal organization is weak.

---

## Important limitation

The current implementation uses a simple AR-style approximation:

$$
y_{t+1} = c + \phi y_t + \eta_t
$$

As a result:

- latent estimates may underestimate true persistence,
- fluctuation variance may absorb unmodeled temporal structure,
- and the decomposition should be interpreted as exploratory rather than exact.

---

# 2. Structured Variance Decomposition

### Function
`variance_decomposition()`

### Purpose

This model extends the temporal decomposition by testing whether a user-specified feature \(z_t\) explains residual variability.

The decomposition is approximately:

$$
y_t = \hat{x}_t + \gamma z_t + \epsilon_t
$$

where:

- $\(\hat{x}_t\)$ = fitted temporal component
- $\(z_t\)$ = candidate structured feature
- $\(\epsilon_t\)$ = unexplained residual variability

---

## What this model is good at

This model works best when:

- a biologically meaningful feature is available,
- the feature is not strongly redundant with temporal persistence,
- and the question is:

> does this candidate feature explain variability beyond the dominant temporal trajectory?

Examples of useful $\(z_t\)$ variables include:

- entropy
- oscillatory phase
- burst states
- local volatility
- behavioral covariates
- physiological states
- experimental conditions

---

## What this model can estimate well

The model can estimate:

- whether a supplied feature explains residual variability,
- approximate feature-associated variance fractions,
- and whether candidate structures differ across groups or sessions.

The model is especially useful for:

- exploratory biological hypothesis testing,
- feature screening,
- and variance attribution.

---

## What this model is NOT designed for

This model is not intended to:

- automatically discover all hidden biological structure,
- identify unknown latent states,
- infer causality,
- or fully separate overlapping dynamical systems.

The model only tests:

> whether the supplied feature $\(z_t\)$ explains variance.

---

## Settings where this model may fail

This model may fail when:

- $\(z_t\)$ is highly correlated with temporal persistence,
- latent and structured processes overlap strongly,
- structure is nonlinear,
- structure changes over time,
- or the chosen feature poorly represents the underlying biology.

The model also struggles when:

- both latent dynamics and residual dynamics are autoregressive,
- because these processes become difficult to identify separately.

---

## Important limitation

Structured variance is conditional on the selected feature representation.

Therefore:

- low structured variance does NOT imply absence of biological structure,
- it only implies that the supplied feature did not explain much variance.

Different choices of $\(z_t\)$ may produce substantially different decompositions.

---

# 3. Simulation Recovery Functions

### Function
`recovery_summary()`

### Purpose

Evaluates whether the model can recover known simulated parameters.

---

## What this function is good at

Useful for:

- simulation benchmarking,
- testing identifiability,
- evaluating sensitivity to noise,
- and comparing decomposition performance across scenarios.

---

## What this function cannot guarantee

Good simulation recovery does NOT guarantee:

- biological correctness,
- causal validity,
- or generalization to all real-world systems.

Simulation recovery only demonstrates:

> the model can recover structure under the assumed simulation conditions.

---
# 4. Variance Aware Latent State Model

The implemented latent dynamical system is:

\[
x_t = \mu + \phi(x_{t-1}-\mu) + w_t
\]

with

\[
w_t \sim \mathcal{N}(0,q_t)
\]

and state-coupled volatility

\[
\log q_t
=
\alpha + \gamma |x_{t-1}-\mu|.
\]

Observed data are generated according to

\[
y_t = x_t + v_t,
\qquad
v_t \sim \mathcal{N}(0,r).
\]

The parameter \(\gamma\) controls how strongly latent-state displacement influences latent process variability.

---

# Features

## Latent simulation

- state-coupled stochastic volatility simulation
- configurable latent persistence
- configurable observation noise
- reproducible simulation pipelines

## Particle filtering and smoothing

- bootstrap particle filtering
- systematic resampling
- backward trajectory sampling
- smoothed latent trajectory inference

## Particle EM estimation

- Monte Carlo EM inference
- iterative recovery of:
  - baseline volatility (\(\alpha\))
  - state-coupling strength (\(\gamma\))
- damped EM updates for improved stability

## Empirical calibration framework

The package includes a simulation-calibrated empirical null framework for assessing reliability of estimated coupling structure.

Given a null distribution of coupling estimates generated under

\[
\gamma_{\text{true}} = 0,
\]

empirical null probabilities are computed as:

\[
p_{\mathrm{emp}}
=
\frac{
1 +
\sum
\mathbf{1}
(
|\hat{\gamma}_{null}|
\ge
|\hat{\gamma}_{obs}|
)
}{
N+1
}.
\]

This provides a simulation-based reliability score quantifying how frequently the estimator produces coupling estimates at least as extreme as an observed estimate under null-coupling conditions.

---

# Overall Recommended Use

The framework is currently strongest as:

> an exploratory variance decomposition tool for biological time-series data.

The most reliable current use case is:

- decomposing temporal versus fluctuation-driven variability.

The structured variance extension should currently be interpreted as:

- hypothesis-driven,
- feature-dependent,
- and exploratory.
