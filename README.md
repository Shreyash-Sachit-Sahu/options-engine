# ⚡ Multi-Agent Options Pricing Engine

> C++ Pricing Core · SAC Hedging Agent · GPU-Accelerated Training · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated market conditions — **outperforming a classical delta-hedging baseline by 56%**.

## 🎯 Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| SAC Agent Sharpe | > 1.4 | ✅ 1.42 |
| Delta Hedge Baseline | ~0.9 | ✅ 0.91 |
| Outperformance vs Delta | > 31% | ✅ 56% |
| C++ Pricing (1M calls) | < 400ms | ✅ ~150ms |
| API Latency (P99) | < 3ms | ✅ < 1ms |
| Max Drawdown | < 15% | ✅ 12% |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│  Vol Surface │ Greeks │ PnL Charts │ Agent Comparison   │
└──────────────────────────┬──────────────────────────────┘
                           │
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
                               │  • GBM / Heston sim       │
                               │  • 7-dim observations     │
                               │  • Transaction costs      │
                               └───────────────────────────┘
```

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
```

If CUDA is not detected, reinstall PyTorch with CUDA support:

```bash
# CUDA 12.8 (driver 525+, recommended)
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8 (older driver)
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu118
```

Run a matmul benchmark to confirm the speedup:

```bash
python agent/gpu_utils.py --bench
```

Monitor GPU utilisation while training:

```bash
nvidia-smi dmon -s u
```

> **Tested on:** NVIDIA GeForce RTX 3050 Laptop GPU · 4 GB VRAM · sm_86 · CUDA 12.8 / driver 595.97

### 5. Pre-tune Hyperparameters (Recommended)

Run Bayesian hyperparameter search **before** training to find the best configuration. This typically pushes Sharpe 0.05–0.15 higher than default settings and is worth the 2–3 hour compute cost.

```bash
# GBM simulator — good starting point, faster trials
python agent/tune.py --simulator gbm --n-trials 50

# Heston simulator — more realistic vol dynamics, run more trials
python agent/tune.py --simulator heston --n-trials 100

# Tune then immediately train with best params (recommended)
python agent/tune.py --simulator gbm --n-trials 50 --train-after

# Monitor trials live in browser
optuna-dashboard sqlite:///agent/tuning/study.db
```

Results are saved to `agent/tuning/best_params_<study>.json`. A ready-to-paste `train.py` command is printed on completion.

### 6. Train the Agent

Train on GBM first, then Heston for robustness. GPU is auto-detected.

```bash
# GBM — standard log-normal price dynamics
python agent/train.py --total-timesteps 500000 --simulator gbm

# Heston — stochastic volatility, more realistic, harder to learn
# Use faster LR cycles to help navigate the more complex landscape
python agent/train.py --total-timesteps 500000 --simulator heston --lr-cycle-steps 50000

# With pre-tuned params (copy the command printed by tune.py)
python agent/train.py --lr 3.2e-4 --batch-size 256 --gamma 0.9971 ...

# Force a specific device
python agent/train.py --device cuda
python agent/train.py --device cpu

# Enable dynamic entropy control
python agent/train.py --ent-coef 0.1
```

> **Why train on both simulators?** GBM assumes constant volatility — fast to train and a good baseline. Heston models volatility as a mean-reverting stochastic process, which is closer to real market behaviour. An agent trained only on GBM can be brittle when volatility clusters or spikes.

### 7. Evaluate
```bash
python agent/evaluate.py --n-episodes 1000
```

### 8. Historical Backtest Against Real SPY Data

After training on simulated data, validate the agent on **real SPY price history**. This is the most credible test — it checks whether the agent generalises to actual market dynamics rather than just simulation artefacts.

```bash
python backtester/historical.py
```

The backtester fetches 6 months of SPY closing prices via yfinance, replays historical price paths through the option lifecycle, and compares the SAC agent against delta hedging on real data. Key metrics reported: PnL distribution, Sharpe ratio, max drawdown, and hedge ratio stability.

```bash
# Longer history for more robust statistics
python -c "
from backtester.historical import fetch_spy_data, HistoricalBacktester
import numpy as np

prices = fetch_spy_data(period='1y')['Close'].values
bt = HistoricalBacktester(prices, K=prices[-1], T_days=30)
# load your trained agent and run bt.run_episode(agent, start_idx=i)
"
```

### 9. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard connects directly to Python modules — **no API server needed**. Tick "Use live SPY data" on the Vol Surface panel to fetch a real IV surface from Yahoo Finance.

### 10. Start API (Optional)

The FastAPI server provides HTTP access to the pricer and agent for external tools. The Streamlit dashboard does **not** use it — they are independent.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

## 📋 Recommended Full Workflow

```
tune.py          →   train.py (GBM)   →   train.py (Heston)   →   historical.py   →   evaluate.py
find best HPs        500k steps            500k steps               real SPY data       final metrics
~2–3 hrs             ~1–2 hrs              ~2–3 hrs                 ~5 min              1000 episodes
```

Each stage builds on the last. Skip stages if time is short — GBM training alone is sufficient for the baseline metrics.

## ⚡ GPU Acceleration

All three agent scripts share the same device detection logic via `agent/gpu_utils.py`.

### Device Priority

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | `cuda:0` | NVIDIA GPU + CUDA-enabled PyTorch |
| 2 | `mps` | Apple Silicon (M1/M2/M3) |
| 3 | `cpu` | Fallback |

### `gpu_utils.py` Reference

| Function | Description |
|----------|-------------|
| `get_device(prefer='auto')` | Returns device string e.g. `'cuda:0'`, `'mps'`, `'cpu'` |
| `patch_sb3_device(device)` | Enables `cuDNN.benchmark` for CUDA; sets MPS memory flags |
| `device_banner()` | Prints full diagnostics: PyTorch build, VRAM, cuDNN version |
| `run_benchmark(device)` | Matmul speedup comparison vs CPU |
| `recommended_batch_size(device)` | Scales batch size by available VRAM |
| `recommended_buffer_size(device)` | Scales replay buffer by available VRAM |

### VRAM Auto-scaling

Batch size and replay buffer scale automatically when still at their CLI defaults:

| VRAM | `batch_size` | `buffer_size` |
|------|-------------|---------------|
| < 8 GB | 256 | 200k |
| 8–15 GB | 512 | 500k |
| 16 GB+ | 1024 | 1M |

### `--device` Flag

All three scripts accept `--device auto|cuda|mps|cpu`:

```bash
python agent/train.py    --device cuda
python agent/tune.py     --device cuda --n-trials 50
python agent/evaluate.py --device cuda --n-episodes 1000
```

## 📁 Project Structure

```
options-engine/
├── src/
│   ├── pricer/
│   │   ├── bs_pricer.hpp      # Black-Scholes + Greeks + IV solver (Brent's)
│   │   ├── mc_pricer.hpp      # Monte Carlo with antithetic variates
│   │   ├── pricer_py.py       # Pure-Python fallback (scipy)
│   │   └── __init__.py        # Auto C++/Python detection
│   └── bindings.cpp           # pybind11 module definition
├── environment/
│   ├── market_sim.py          # GBM + Heston + Regime-switching simulators
│   ├── options_env.py         # Gymnasium environment (7-dim obs, continuous action)
│   └── baselines.py           # Delta / Static / Random agents
├── agent/
│   ├── train.py               # SAC training pipeline + dynamic HP control
│   ├── tune.py                # Pre-training hyperparameter search (Optuna)
│   ├── evaluate.py            # Full evaluation + metrics comparison
│   ├── gpu_utils.py           # Device detection, cuDNN config, benchmarking
│   └── models/                # Saved checkpoints + VecNormalize stats
├── backtester/
│   ├── historical.py          # yfinance SPY replay — real market validation
│   └── vol_surface.py         # IV surface construction (live + synthetic)
├── dashboard/
│   └── app.py                 # Streamlit dashboard (6 panels, direct imports)
├── api/
│   └── main.py                # FastAPI backend (optional, external HTTP access)
├── tests/
│   ├── test_pricer.py         # Put-call parity, Greeks bounds, benchmarks
│   ├── test_environment.py    # Gym compliance, simulator tests, baselines
│   └── test_api.py            # API endpoint correctness + latency
├── CMakeLists.txt
├── requirements.txt
└── .github/workflows/ci.yml   # GitHub Actions CI
```

## 🔬 Hyperparameter Tuning

Two complementary systems cover the full tuning lifecycle.

### Pre-training Search (`tune.py`)

Finds good hyperparameters **before** training using [Optuna](https://optuna.org) Bayesian optimisation (TPE) with MedianPruner. Bad trials are killed early, cutting total compute by ~40%. All trials run on the selected device.

```bash
pip install optuna optuna-dashboard

# Run 50 Bayesian trials — safe to interrupt and resume
python agent/tune.py --simulator gbm --n-trials 50

# Heston simulator, more trials
python agent/tune.py --simulator heston --n-trials 100

# Resume a previous study (nothing is lost)
python agent/tune.py --study-name sac_hedger_gbm

# Monitor live in browser while tuning runs
optuna-dashboard sqlite:///agent/tuning/study.db
```

Parameters searched:

| Parameter | Range | Why |
|-----------|-------|-----|
| `lr` | `1e-5 → 1e-3` log | Most sensitive SAC parameter |
| `ent_coef` | `auto, 0.01, 0.05, 0.1, 0.5` | "auto" doesn't always win |
| `gamma` | `0.95 → 0.9999` | Short 30-step episodes may prefer < 0.99 |
| `batch_size` | `64, 128, 256, 512` | Gradient quality vs. speed |
| `net_width` + `net_depth` | `64–256`, `1–3 layers` | Architecture for 7-dim input |
| `tau` | `0.001 → 0.02` log | Target network update speed |
| `buffer_size` | `50k, 100k, 200k` | Replay buffer capacity |
| `learning_starts` | `500, 1000, 2000` | Exploration before first gradient |

---

### Dynamic Control During Training (`DynamicHPController`)

Adapts hyperparameters **mid-run** based on live performance signals via a custom SB3 callback. Three independent controllers run every 2000 steps:

**1. Learning Rate — Cosine Annealing with Warm Restarts (SGDR)**

```
lr(t) = lr_min + 0.5 * (lr_base - lr_min) * (1 + cos(π * t / cycle))
```

LR decays from `base_lr → base_lr/10` then restarts. Warm restarts help escape flat loss regions mid-training. Cycle length defaults to 100k steps (configurable via `--lr-cycle-steps`).

**2. Entropy Coefficient — Sharpe-driven adaptation** *(when `--ent-coef` is a float)*

| Signal | Action |
|--------|--------|
| Sharpe plateau for 200 episodes | `ent_coef × 1.5` — explore more |
| Sharpe improving steadily | `ent_coef × 0.85` — exploit more |

When `--ent-coef auto` (default), SAC's own dual gradient descent manages entropy and this controller stays out of the way.

**3. Gradient Steps — Critic stability tracking**

| Signal | Action |
|--------|--------|
| Critic loss CV > 0.5 (unstable) | `gradient_steps − 1` |
| Critic loss CV < 0.2 + Sharpe > 0.5 | `gradient_steps + 1` |

Range clamped to `[1, 4]`.

All three controllers log to TensorBoard under `dynamic_hp/`:

```bash
tensorboard --logdir tb_logs
# Tracks: learning_rate, ent_coef, gradient_steps,
#         rolling_sharpe, critic_loss_cv, steps_since_improvement
```

**CLI flags for dynamic control:**

```bash
# LR annealing + dynamic grad steps (always on)
python agent/train.py

# Also enable Sharpe-driven entropy control
python agent/train.py --ent-coef 0.1

# Faster LR cycles (recommended for Heston)
python agent/train.py --simulator heston --lr-cycle-steps 50000
```

## 🧪 Testing

```bash
# All tests (70 tests, ~13s)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov=environment --cov=api -v

# Benchmark only
python -m pytest tests/test_pricer.py::TestBenchmark -v
```

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Pricing Engine | C++17, pybind11 3.0 |
| RL Framework | stable-baselines3, PyTorch 2.x |
| GPU Acceleration | CUDA 12.8+, cuDNN (auto-detected via `agent/gpu_utils.py`) |
| Pre-training Tuning | Optuna (TPE + MedianPruner) |
| Dynamic HP Control | Custom SB3 callback (SGDR + Sharpe-driven) |
| Market Simulation | NumPy, SciPy (GBM + Heston + Regime-switching) |
| Historical Validation | yfinance SPY replay (`backtester/historical.py`) |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, uvicorn |
| Data | yfinance |
| CI/CD | GitHub Actions |

## 📊 API Documentation

The FastAPI server is **optional** — the Streamlit dashboard works without it via direct Python imports. Start it only if you need external HTTP access to the pricer or agent.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/price` | BSM or MC option pricing |
| `POST` | `/greeks` | Full Greeks computation |
| `POST` | `/iv` | Implied volatility solver |
| `POST` | `/agent/action` | SAC agent inference |
| `GET` | `/benchmark` | Run pricing benchmark |
| `GET` | `/health` | Health check |

## 📜 License

MIT