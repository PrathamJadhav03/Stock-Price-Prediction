# DECISIONS.md — Architecture Decision Records

> **Purpose**: Log significant technical decisions and their rationale.

---

## Decisions

### [DECISION-001] Strict Package Management via `uv`

**Date**: 2026-11-08  
**Status**: Accepted  

#### Context
The existing prototype notebook lacked dependency pinning and environment reproducibility. Traditional `pip` and manual `venv` management can lead to version drift, slower container builds, and loose dependency trees.

#### Decision
Use `uv` (`pyproject.toml` and `uv.lock`) exclusively for dependency resolution, package locking, test execution (`uv run pytest`), server launches, and multi-stage Docker builds. Direct invocation of `pip` and manual `venv` activation are forbidden.

#### Rationale
`uv` provides deterministic lockfiles, rapid resolution speeds, and hermetic virtual environment orchestration, aligning with enterprise-grade reproducible ML standards.

#### Consequences
All command executions, developer tooling, CI/CD scripts, and Docker images must be prefixed with `uv run` or utilize `uv sync`.

---

### [DECISION-002] Non-Stationarity Mitigation via Log Returns

**Date**: 2026-11-08  
**Status**: Accepted  

#### Context
The original notebook trained a PyTorch GRU directly on raw equity `Close` prices. Raw stock prices are non-stationary random walks with time-varying means, leading to spurious correlation, artificial drift tracking, and degraded out-of-sample generalization.

#### Decision
Model stationary daily Log Returns $r_t = \ln(P_t / P_{t-1})$ as the primary prediction target. Provide a deterministic vector conversion utility to reconstruct price paths: $\hat{P}_t = P_0 \exp(\sum_{\tau=1}^t \hat{r}_\tau)$.

#### Rationale
Log returns stabilize the distribution variance, remove macro price trend non-stationarity, and are additive over multi-step horizons.

#### Consequences
Models predict relative daily returns rather than absolute dollar prices. Downstream consumers and visual charts will use the reconstruction utility to display dollar price paths and confidence intervals.

---

### [DECISION-003] Zero Data Leakage via Serialized Pipelines

**Date**: 2026-11-08  
**Status**: Accepted  

#### Context
Financial ML models often suffer from lookahead bias caused by fitting scalers on the entire dataset prior to splitting, or using backward-looking indicators computed with future information.

#### Decision
Enforce strict temporal train/validation/test splitting. All feature transformations and scalers (`RobustScaler`/`StandardScaler`) must be fit strictly on the training partition and persisted via `joblib` artifacts for inference.

#### Rationale
Guarantees that test evaluations reflect realistic out-of-sample performance without future distribution leakage.

#### Consequences
Inference code must load saved scaler artifacts rather than recalculating distribution statistics on test batches.

---

### [DECISION-004] Uncertainty Quantification via Monte Carlo Dropout

**Date**: 2026-11-08  
**Status**: Accepted  

#### Context
Point forecasts in financial markets give an illusion of false certainty without communicating model confidence or market regime volatility.

#### Decision
Incorporate LayerNorm and Dropout layers in the PyTorch GRU. At inference time, enable Monte Carlo (MC) Dropout (sampling $N=50-100$ stochastic forward passes) to estimate predictive mean and empirical 95% confidence intervals (2.5th and 97.5th percentiles).

#### Rationale
Provides actionable risk envelopes for traders and portfolio managers without requiring heavy Bayesian neural network Markov Chain Monte Carlo training.

#### Consequences
Inference requires multiple forward passes; optimized batching is necessary to keep latency under 100ms.

---

### [DECISION-005] Naive Persistence Benchmark Protocol

**Date**: 2026-11-08  
**Status**: Accepted  

#### Context
Financial time-series models can appear to achieve low RMSE by simply lagging the true price by one timestep ($\hat{P}_t \approx P_{t-1}$), effectively mimicking a random walk without learning predictive alpha.

#### Decision
Implement a formal `NaivePersistenceBaseline` ($\hat{P}_t = P_{t-1}$ or $\hat{r}_t = 0$) and evaluate directional accuracy (Sign Hit Rate %) in addition to regression metrics. Any model must empirically beat this baseline to be considered viable.

#### Rationale
Prevents deploying models whose apparent accuracy is merely an artifact of high autocorrelation in financial time series.

---

*Last updated: 2026-11-08*
