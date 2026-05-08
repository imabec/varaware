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
