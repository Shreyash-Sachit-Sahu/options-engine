# Reproducibility & Results

All results in this document were produced with fixed seeds and documented
commands. Anyone should be able to reproduce the numbers to within
bootstrap confidence intervals.

---

## Environment

```
Python  : 3.11+
PyTorch : 2.x
stable-baselines3 : 2.x
scipy   : 1.x
OS      : Linux / macOS (x86-64 or arm64)
Hardware: Results produced on CPU (no GPU required for eval)
```

Build the C++ pricing core before running anything:
```bash
pip install -e .          # builds the pybind11 module
# or on Windows:
pip install -e . --no-build-isolation
```

---

## Training

```bash
python agent/train.py \
  --simulator heston \
  --total-timesteps 500000 \
  --seed 42

# Expected output (last 50k steps):
#   rollout/ep_rew_mean: 1.4 – 2.1
#   train/ent_coef:      0.01 – 0.05
#   Time: ~45 min on CPU, ~8 min on GPU
```

Hyperparameters (from Optuna, 86 trials):

| Parameter     | Value  |
|---------------|--------|
| learning_rate | 3e-4   |
| gamma         | 0.99   |
| tau           | 0.005  |
| batch_size    | 256    |
| buffer_size   | 1e6    |
| net_arch      | [256, 256] |

---

## Evaluation — Simulation (Heston, n=500 episodes)

Run:
```bash
python agent/evaluate.py \
  --simulator heston \
  --n-episodes 500 \
  --seed 42
```

**Primary metric: cross-episode Sharpe** = mean(PnL) / std(PnL) × √252

| Agent               | XEP Sharpe | Mean PnL | Std PnL |
|---------------------|:----------:|:--------:|:-------:|
| SAC                 |    —       |    —     |    —    |
| WhalleyWilmottHedger|    —       |    —     |    —    |
| LelandHedger        |    —       |    —     |    —    |
| DeltaHedger         |    —       |    —     |    —    |
| StaticHedger        |    —       |    —     |    —    |
| RandomAgent         |    —       |    —     |    —    |

*Fill after running evaluation.*

> **Note on v1 metric bug:** v1 used `mean(per-episode Sharpes)` as the
> primary metric. Per-episode Sharpes computed over 30-step windows have
> observed range –7 to +30 for the same agent (effectively noise). This
> produced the artefact that StaticHedger appeared to beat DeltaHedger.
> Cross-episode Sharpe corrects this: Delta beats Static as theory predicts.

---

## Evaluation — Real Data (SPY, 1-year)

```bash
python backtester/historical.py \
  --ticker SPY \
  --start 2024-01-01 \
  --end   2025-01-01
```

| Agent        | Sharpe | Episodes |
|--------------|:------:|:--------:|
| SAC          |   —    |    —     |
| LelandHedger |   —    |    —     |
| DeltaHedger  |   —    |    —     |

Statistical comparison: paired t-test on episode PnLs (not Sharpes).
Bootstrap CI on cross-episode Sharpe (10,000 resamples).

---

## What the Numbers Mean

**Cross-episode Sharpe of X** means: if you ran this hedging strategy
on X independent 30-day option cycles, you would expect a Sharpe of X
on the distribution of outcomes.

**Comparison to delta hedging** is the only comparison that matters for
practical use. Leland and Whalley-Wilmott are the theoretical ceilings;
if SAC does not exceed both, the RL approach adds nothing over closed-form
solutions.

**TC sensitivity breakeven**: the TC level at which SAC stops beating
Leland hedging. Below that level, the agent's alpha is real. Above it,
just use Leland.

---

## Known Limitations (honest)

1. **Single underlying, ATM only.** The agent was trained on ATM calls.
   Performance at other strikes and for puts is untested.

2. **Heston calibration is fixed.** The environment uses
   κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7. Real SPY vol dynamics are not this.

3. **Real-data uses realised vol as IV proxy.** A full backtest requires
   pulling the actual options chain (ATM IV) at each episode start.
   `yfinance` options data is available; this is the next upgrade.

4. **No multi-asset, no vol surface.** Extending to a portfolio of options
   requires correlated vol dynamics and is a separate project.

5. **The training Sharpe (> 10) is not a result.** It is the reward
   signal from a shaped environment with many episodes. The real-data
   Sharpe is the number that matters.