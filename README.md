# ⚡ Multi-Agent Options Pricing Engine

> C++ Pricing Core · SAC Hedging Agent · GPU-Accelerated Training · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated and real market conditions — **outperforming classical delta-hedging by 48% on real SPY data (p < 0.0001)**.

## 🎯 Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| SAC Sharpe (Heston sim, 500 ep) | > 1.4 | ✅ 2.75 |
| Delta Hedge Baseline (sim) | ~0.9 | ✅ 2.01 |
| Outperformance vs Delta (sim) | > 31% | ✅ +37% (p=0.0065) |
| Outperformance vs Delta (real SPY 1y) | > 20% | ✅ +48% (p<0.0001) |
| Outperformance vs Delta (real SPY 2y) | > 20% | ✅ +30% (p<0.0001) |
| C++ Pricing (1M calls) | < 400ms | ✅ ~150ms |
| API Latency (P99) | < 3ms | ✅ < 1ms |
| Max Drawdown (real data) | < 1.0 | ✅ 0.44 |

### Real-Data Validation Summary

| Period | SAC Sharpe | Delta Sharpe | Outperformance | p-value | Episodes |
|--------|-----------|-------------|---------------|---------|---------|
| Heston sim (500 ep) | 2.75 | 2.01 | +37% | 0.0065 | 500 |
| Real SPY 1y | 0.29 | 0.20 | **+48%** | **< 0.0001** | 44 |
| Real SPY 2y | 0.26 | 0.20 | +30% | < 0.0001 | 157 |

Tested on real SPY price data (May 2024 – May 2026), covering the April 2025 tariff crash — a genuine tail risk stress test. Bootstrap 95% CI on 1y outperformance: [+0.038, +0.087], entirely above zero.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│  Vol Surface │ Greeks │ PnL Charts │ Agent Comparison   │
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
│  • Black-Scholes│              │                         │
│  • Monte Carlo  │              │  • Continuous           │
│  • Greeks       │              │    action space         │
│  • Implied Vol  │              │  • Entropy-regularized  │
│                 │              │  • Dynamic HP control   │
└─────────────────┘              └────────────┬────────────┘
                                              │
                               ┌──────────────▼────────────┐
                               │  Gym Environment           │
                               │  • GBM / Heston / Jump    │
                               │  • 11-dim observations    │
                               │  • Transaction costs      │
                               └───────────────────────────┘
```

The dashboard calls the FastAPI backend for all pricing, Greeks, and agent inference. The API falls back to Python pricing if the C++ pricer is not compiled.

## 🚀 Quick Start

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

### 4. GPU Setup (Recommended)

Training automatically uses the best available device — **CUDA → MPS → CPU** — with no code changes needed. Verify detection:

```bash
python agent/gpu_utils.py
python agent/gpu_utils.py --bench   # matmul speedup vs CPU
```

If CUDA is not detected, reinstall PyTorch with CUDA support:

```bash
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

> **Tested on:** NVIDIA GeForce RTX 3050 Laptop GPU · 4 GB VRAM · sm_86 · CUDA 12.8 / driver 595.97

### 5. Pre-tune Hyperparameters (Recommended)

```bash
python agent/tune.py --simulator heston --n-trials 50 --device cuda
optuna-dashboard sqlite:///agent/tuning/study.db   # monitor live
```

Best params from 86 trials: `lr=4.1e-4, batch_size=128, buffer_size=200k, tau=0.001, gamma=0.9925, ent_coef=0.05`

### 6. Train the Agent

```bash
# With tuned hyperparameters (recommended)
python agent/train.py --simulator heston --total-timesteps 1000000 \
  --lr-cycle-steps 50000 --device cuda \
  --lr 4.10e-4 --batch-size 128 --buffer-size 200000 \
  --tau 0.001 --gamma 0.9925 --ent-coef 0.05 --learning-starts 1000

# GBM and Jump simulators for robustness
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

### 8. Historical Backtest Against Real SPY Data
```bash
# 1 year (includes April 2025 tariff crash)
python backtester/historical.py \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --period 1y

# 2 years, more episodes
python backtester/historical.py \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --period 2y --stride 3
```

### 9. Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

### 10. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard connects to the API at `http://localhost:8000` by default. The URL is configurable in the sidebar — change it to any deployed API endpoint for remote access. All pricing, Greeks, and agent inference go through the API with automatic local fallback if offline.

## 📋 Recommended Full Workflow

```
tune.py (50+ trials)  →  train.py (Heston, 1M steps)  →  evaluate.py  →  historical.py
~10 hrs tuning            ~90 min training                 500 episodes    real SPY data
```

## 🧠 Observation Space (11-dim)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `S/S0` | Normalised spot price |
| 1 | `K/S0` | Normalised strike |
| 2 | `T_rem/T` | Remaining time fraction |
| 3 | `sigma` | Current implied/realised volatility |
| 4 | `delta` | Black-Scholes delta |
| 5 | `gamma × 100` | Scaled gamma (convexity) |
| 6 | `pnl_norm` | Normalised portfolio PnL |
| 7 | `vega` | Sensitivity to vol moves |
| 8 | `theta` | Time decay (negated — short call earns decay) |
| 9 | `vol_carry` | Realized/implied vol ratio — carry signal |
| 10 | `hedge_pos` | Current hedge position (memory without recurrence) |

**Two-tower feature extractor:** market state [0–5] and portfolio state [6–10] are processed separately with LayerNorm then merged — reflects how options traders mentally separate market vs position information.

## 📈 Market Simulators

| Simulator | Model | Key Characteristic | Use Case |
|-----------|-------|-------------------|----------|
| `gbm` | Geometric Brownian Motion | Constant vol, log-normal | Baseline |
| `heston` | Heston (1993) SV | Mean-reverting stochastic vol, ρ=−0.7 | Primary training |
| `jump` | Merton (1976) JD | Poisson jumps, left-skewed (μ_j=−10%) | Tail risk |

## ⚡ GPU Acceleration

| VRAM | `batch_size` | `buffer_size` |
|------|-------------|---------------|
| < 8 GB | 256 | 200k |
| 8–15 GB | 512 | 500k |
| 16 GB+ | 1024 | 1M |

## 📁 Project Structure

```
options-engine/
├── src/pricer/            # C++17 Black-Scholes, Monte Carlo, IV solver
├── environment/
│   ├── market_sim.py      # GBM + Heston + Merton jump-diffusion
│   ├── options_env.py     # Gymnasium env (11-dim obs, residual action)
│   └── baselines.py       # Delta / Static / Random agents
├── agent/
│   ├── train.py           # SAC + custom PyTorch extractor + DHP callbacks
│   ├── tune.py            # Optuna (TPE + MedianPruner, 86 trials run)
│   ├── evaluate.py        # Bootstrap CI + paired t-test + TC sweep
│   └── gpu_utils.py       # Device detection, cuDNN config, benchmarking
├── backtester/
│   ├── historical.py      # Real SPY replay — SAC vs Delta + significance tests
│   └── vol_surface.py     # Live IV surface from yfinance
├── dashboard/app.py       # Streamlit dashboard — calls FastAPI backend
├── api/main.py            # FastAPI backend (pricing + Greeks + agent inference)
└── tests/                 # 70 tests, all passing
```

## 🔬 Hyperparameter Tuning

86 Bayesian trials (TPE + MedianPruner). Best found configuration:

| Parameter | Best Value | Search Range |
|-----------|-----------|-------------|
| `lr` | 4.10e-4 | 1e-5 → 1e-3 (log) |
| `batch_size` | 128 | 64 / 128 / 256 / 512 |
| `buffer_size` | 200k | 50k / 100k / 200k |
| `tau` | 0.001 | 0.001 → 0.02 (log) |
| `gamma` | 0.9925 | 0.95 → 0.9999 |
| `net_width` | 64 | 64 / 128 / 256 |
| `net_depth` | 2 | 1–3 |
| `ent_coef` | 0.05 | auto / 0.01 / 0.05 / 0.1 / 0.5 |

Dynamic HP control during training: cosine annealing LR (SGDR), Sharpe-driven entropy, critic-stability-driven gradient steps. All logged to TensorBoard under `dynamic_hp/`.

## ⚠️ Known Limitations

These are honest constraints worth understanding before drawing conclusions from the results.

**Training data is entirely synthetic.** The agent was trained on Heston stochastic vol simulations, not real market data. The Heston parameters (κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7) are calibrated to typical US equity dynamics but do not capture all real-world microstructure effects such as volatility jumps, liquidity gaps, or intraday patterns.

**Real-data backtest uses simplified market assumptions.** The historical backtester uses ATM options (K = spot at episode start), a fixed risk-free rate (r=0.05), and realized vol as a proxy for implied vol. In practice, IV can diverge significantly from realized vol — especially during stress events — and real hedging would use actual IV from the options chain. The April 2025 tariff crash episodes are included but the agent was never trained on any real price data.

**Transaction cost is fixed.** The TC rate (0.003) is constant across all episodes. Real bid-ask spreads are dynamic, wider in volatile periods, and volume-dependent. The TC sensitivity sweep shows the agent beats delta hedging across all tested TC levels (0 to 0.005), but this does not account for market impact.

**Strike selection is ATM only.** Using K = S0 per episode avoids the deep-OTM premium explosion problem but means the agent was only evaluated on at-the-money options. Performance on skewed or term-structure trades is untested.

**No live trading.** This is a research and backtesting system. It has not been tested in a paper trading or live execution environment. Past backtest performance does not guarantee future results.

**Simulated Sharpe of 13.96 during training is not a result.** This is the peak rolling mean of within-episode information ratios computed over 30-step training episodes on the Heston simulator — a relative signal used by the dynamic HP controller, not an out-of-sample metric. The credible results are the out-of-sample numbers: Sharpe 2.75 on 500 held-out Heston episodes and Sharpe 0.29 on 1 year of real SPY data.

## 🧪 Testing

```bash
python -m pytest tests/ -v          # 70 tests, ~13s
python -m pytest tests/ --cov=src --cov=environment --cov=api -v
```

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Pricing Engine | C++17, pybind11 3.0 |
| RL Framework | stable-baselines3, PyTorch 2.x |
| Neural Network | Custom two-tower `nn.Module` with LayerNorm |
| GPU Acceleration | CUDA 12.8+, cuDNN benchmark mode |
| Hyperparameter Tuning | Optuna (TPE, 86 trials) |
| Dynamic HP Control | Custom SB3 callback (SGDR + Sharpe-driven) |
| Market Simulation | GBM · Heston SV · Merton Jump-Diffusion |
| Statistical Validation | Bootstrap CI · Paired t-test · TC sensitivity |
| Historical Validation | Real SPY data (2024–2026) via yfinance |
| Dashboard | Streamlit, Plotly (API-connected) |
| API | FastAPI, uvicorn |
| CI/CD | GitHub Actions |

## 📊 API Documentation

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/price` | BSM or MC option pricing |
| `POST` | `/greeks` | Full Greeks computation |
| `POST` | `/iv` | Implied volatility solver |
| `POST` | `/agent/action` | SAC agent inference (11-dim obs) |
| `GET` | `/benchmark` | Run C++ pricing benchmark |
| `GET` | `/health` | Health check + model status |

## 📜 License

MIT