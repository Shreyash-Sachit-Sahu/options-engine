# Multi-Agent Options Pricing Engine

**C++ Pricing Core · SAC Hedging Agent · GPU-Accelerated Training · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning**

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated and real market conditions, outperforming classical delta-hedging by 48% on real SPY data (p < 0.0001).

---

## Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| SAC Sharpe (Heston sim, 500 ep) | > 1.4 | 2.75 |
| Delta Hedge Baseline (sim) | ~0.9 | 2.01 |
| Outperformance vs Delta (sim) | > 31% | +37% (p=0.0065) |
| Outperformance vs Delta (real SPY 1y) | > 20% | +48% (p<0.0001) |
| Outperformance vs Delta (real SPY 2y) | > 20% | +30% (p<0.0001) |
| C++ Pricing (1M calls) | < 400ms | ~150ms |
| API Latency (P99) | < 3ms | < 1ms |
| Max Drawdown (real data) | < 1.0 | 0.44 |

### Real-Data Validation

| Period | SAC Sharpe | Delta Sharpe | Outperformance | p-value | Episodes |
|--------|-----------|-------------|---------------|---------|---------|
| Heston sim (500 ep) | 2.75 | 2.01 | +37% | 0.0065 | 500 |
| Real SPY 1y | 0.29 | 0.20 | +48% | < 0.0001 | 44 |
| Real SPY 2y | 0.26 | 0.20 | +30% | < 0.0001 | 157 |

Tested on real SPY price data (May 2024 – May 2026), covering the April 2025 tariff crash. Bootstrap 95% CI on 1y outperformance: [+0.038, +0.087], entirely above zero.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│  Vol Surface | Greeks | PnL Charts | Agent Comparison   │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (FastAPI)
┌──────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                        │
│  /price  /greeks  /iv  /agent/action  /benchmark        │
└────────┬─────────────────────────────────┬──────────────┘
         │                                 │
┌────────▼────────┐              ┌─────────▼──────────────┐
│  C++ Pricer     │              │  SAC Agent              │
│  (pybind11)     │              │  (stable-baselines3     │
│                 │              │   + PyTorch)            │
│  Black-Scholes  │              │  Continuous action      │
│  Monte Carlo    │              │  Entropy-regularized    │
│  Greeks         │              │  Dynamic HP control     │
│  Implied Vol    │              │                         │
└─────────────────┘              └────────────┬────────────┘
                                              │
                               ┌──────────────▼────────────┐
                               │  Gym Environment           │
                               │  GBM / Heston / Jump      │
                               │  11-dim observations      │
                               │  Transaction costs        │
                               └───────────────────────────┘
```

The dashboard calls the FastAPI backend for all pricing, Greeks, and agent inference. The API falls back to Python pricing if the C++ pricer is not compiled.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build C++ Pricer

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build --config Release
```

### 3. Verify

```python
import pricer
print(pricer.bs_call(100, 100, 1, 0.05, 0.2))  # 10.4506
print(pricer.greeks(100, 100, 1, 0.05, 0.2))    # Greeks(delta=0.637...)
```

### 4. GPU Setup

Training automatically uses the best available device — CUDA, then MPS, then CPU — with no code changes required.

```bash
python agent/gpu_utils.py           # device diagnostics
python agent/gpu_utils.py --bench   # matmul speedup benchmark
```

If CUDA is not detected, reinstall PyTorch with CUDA support:

```bash
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

Tested on: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM, sm_86, CUDA 12.8, driver 595.97.

### 5. Pre-tune Hyperparameters

Bayesian hyperparameter search before training. Typically improves Sharpe by 0.05–0.15 and is worth the compute cost.

```bash
python agent/tune.py --simulator heston --n-trials 50 --device cuda
optuna-dashboard sqlite:///agent/tuning/study.db   # live trial monitor
```

Best configuration from 86 trials: `lr=4.1e-4, batch_size=128, buffer_size=200k, tau=0.001, gamma=0.9925, ent_coef=0.05`

### 6. Train the Agent

```bash
# Heston simulator with tuned hyperparameters (primary)
python agent/train.py --simulator heston --total-timesteps 1000000 \
  --lr-cycle-steps 50000 --device cuda \
  --lr 4.10e-4 --batch-size 128 --buffer-size 200000 \
  --tau 0.001 --gamma 0.9925 --ent-coef 0.05 --learning-starts 1000

# Additional simulators for robustness
python agent/train.py --simulator gbm  --total-timesteps 500000 --device cuda
python agent/train.py --simulator jump --total-timesteps 500000 --device cuda
```

### 7. Evaluate

```bash
python agent/evaluate.py --simulator heston --n-episodes 500 \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --skip-tc-sweep
```

Reports: Sharpe ratio, bootstrap 95% CI, paired t-test vs delta hedger, max drawdown.

### 8. Historical Backtest

```bash
# 1 year — covers April 2025 tariff crash
python backtester/historical.py \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --period 1y

# 2 years, denser sampling
python backtester/historical.py \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --period 2y --stride 3
```

### 9. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 10. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard connects to the API at `http://localhost:8000` by default. The URL is configurable in the sidebar — point it at any deployed API endpoint for remote access.

---

## Recommended Workflow

```
tune.py (50+ trials)  ->  train.py (Heston, 1M steps)  ->  evaluate.py  ->  historical.py
~10 hrs tuning             ~90 min training                  500 episodes     real SPY data
```

---

## Observation Space (11-dim)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `S/S0` | Normalised spot price |
| 1 | `K/S0` | Normalised strike |
| 2 | `T_rem/T` | Remaining time fraction |
| 3 | `sigma` | Current implied/realised volatility |
| 4 | `delta` | Black-Scholes delta |
| 5 | `gamma x 100` | Scaled gamma (convexity) |
| 6 | `pnl_norm` | Normalised portfolio PnL |
| 7 | `vega` | Sensitivity to vol moves |
| 8 | `theta` | Time decay (negated — short call earns decay) |
| 9 | `vol_carry` | Realized/implied vol ratio — carry signal |
| 10 | `hedge_pos` | Current hedge position (memory without recurrence) |

A two-tower PyTorch feature extractor processes market state features [0–5] and portfolio state features [6–10] separately with LayerNorm, then merges them. This architecture reflects the natural separation between market information and position state that options practitioners use when making hedging decisions.

---

## Market Simulators

| Simulator | Model | Key Characteristic | Primary Use |
|-----------|-------|-------------------|-------------|
| `gbm` | Geometric Brownian Motion | Constant vol, log-normal returns | Baseline training |
| `heston` | Heston (1993) SV | Mean-reverting stochastic vol, rho=-0.7 | Primary training |
| `jump` | Merton (1976) JD | Poisson jumps, left-skewed (mu_j=-10%) | Tail risk robustness |

Heston parameters calibrated to US equity dynamics: kappa=2.0, theta=0.04 (20% long-run vol), xi=0.3, rho=-0.7. Jump parameters: lambda=1.0 (1 jump/year), mu_j=-10%, sigma_j=15%.

---

## GPU Acceleration

| VRAM | batch_size | buffer_size |
|------|------------|-------------|
| < 8 GB | 256 | 200k |
| 8–15 GB | 512 | 500k |
| 16 GB+ | 1024 | 1M |

Batch size and buffer size scale automatically based on detected VRAM. `cuDNN.benchmark` is enabled for CUDA devices.

---

## Project Structure

```
options-engine/
├── src/pricer/            # C++17 Black-Scholes, Monte Carlo, IV solver (pybind11)
├── environment/
│   ├── market_sim.py      # GBM, Heston, Merton jump-diffusion simulators
│   ├── options_env.py     # Gymnasium env (11-dim obs, residual action space)
│   └── baselines.py       # Delta, Static, and Random hedging agents
├── agent/
│   ├── train.py           # SAC training + custom PyTorch extractor + DHP callbacks
│   ├── tune.py            # Optuna hyperparameter search (TPE + MedianPruner)
│   ├── evaluate.py        # Bootstrap CI, paired t-test, TC sensitivity sweep
│   └── gpu_utils.py       # Device detection, cuDNN config, benchmarking
├── backtester/
│   ├── historical.py      # Real SPY replay, SAC vs Delta, statistical testing
│   └── vol_surface.py     # Live IV surface construction via yfinance
├── dashboard/app.py       # Streamlit dashboard, API-connected
├── api/main.py            # FastAPI backend (pricing, Greeks, agent inference)
└── tests/                 # 70 tests, all passing (~13s)
```

---

## Hyperparameter Tuning

86 Bayesian trials using Optuna TPE sampler with MedianPruner. Bad trials are killed early, reducing total compute by approximately 40%.

| Parameter | Best Value | Search Range |
|-----------|-----------|-------------|
| `lr` | 4.10e-4 | 1e-5 to 1e-3 (log) |
| `batch_size` | 128 | 64, 128, 256, 512 |
| `buffer_size` | 200k | 50k, 100k, 200k |
| `tau` | 0.001 | 0.001 to 0.02 (log) |
| `gamma` | 0.9925 | 0.95 to 0.9999 |
| `net_width` | 64 | 64, 128, 256 |
| `net_depth` | 2 | 1, 2, 3 |
| `ent_coef` | 0.05 | auto, 0.01, 0.05, 0.1, 0.5 |

Three dynamic controllers run during training via a custom SB3 callback, updating every 2000 steps:

1. **Learning rate** — cosine annealing with warm restarts (SGDR). Decays from base_lr to base_lr/10 then restarts.
2. **Entropy coefficient** — responds to rolling Sharpe. Boosts exploration on plateau, reduces on steady improvement.
3. **Gradient steps** — increases when critic loss is stable, decreases when unstable. Clamped to [1, 4].

All changes logged to TensorBoard under `dynamic_hp/`.

```bash
tensorboard --logdir tb_logs
```

---

## Known Limitations

**Training data is entirely synthetic.** The agent was trained on Heston stochastic vol simulations, not real market data. The simulator parameters are calibrated to typical US equity dynamics but do not capture all real-world effects such as volatility jumps, liquidity gaps, or intraday microstructure.

**Real-data backtest uses simplified market assumptions.** The historical backtester uses ATM options (K = spot at episode start), a fixed risk-free rate (r=0.05), and realized vol as a proxy for implied vol. In practice, IV can diverge significantly from realized vol — particularly during stress events — and real hedging would use actual IV from the options chain.

**Transaction cost is fixed.** The TC rate (0.003) is constant across all episodes. Real bid-ask spreads are dynamic, wider in volatile periods, and volume-dependent.

**Strike selection is ATM only.** Using K = S0 per episode means the agent was evaluated exclusively on at-the-money options. Performance on OTM, ITM, or term-structure trades is untested.

**No live trading.** This is a research and backtesting system only. It has not been tested in a paper trading or live execution environment.

**The training Sharpe of 13.96 is not a result.** This is the peak rolling mean of within-episode information ratios on the Heston training simulator — a relative signal used by the dynamic HP controller, not an out-of-sample metric. The credible results are: Sharpe 2.75 on 500 held-out Heston episodes, and Sharpe 0.29 on 1 year of real SPY data.

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov=environment --cov=api -v

# Pricing benchmark only
python -m pytest tests/test_pricer.py::TestBenchmark -v
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Pricing Engine | C++17, pybind11 3.0 |
| RL Framework | stable-baselines3, PyTorch 2.x |
| Neural Network | Custom two-tower nn.Module with LayerNorm |
| GPU Acceleration | CUDA 12.8+, cuDNN benchmark mode |
| Hyperparameter Tuning | Optuna (TPE, 86 trials) |
| Dynamic HP Control | Custom SB3 callback (SGDR + Sharpe-driven) |
| Market Simulation | GBM, Heston SV, Merton Jump-Diffusion |
| Statistical Validation | Bootstrap CI, paired t-test, TC sensitivity |
| Historical Validation | Real SPY data (2024–2026) via yfinance |
| Dashboard | Streamlit, Plotly (API-connected) |
| API | FastAPI, uvicorn |
| CI/CD | GitHub Actions |

---

## API Reference

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/price` | BSM or Monte Carlo option pricing |
| POST | `/greeks` | Full Greeks computation |
| POST | `/iv` | Implied volatility solver |
| POST | `/agent/action` | SAC agent inference (11-dim observation) |
| GET | `/benchmark` | C++ pricing benchmark |
| GET | `/health` | Health check and model status |

---

## License

MIT
