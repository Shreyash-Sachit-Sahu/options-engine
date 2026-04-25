# ⚡ Multi-Agent Options Pricing Engine

> C++ Pricing Core · SAC Hedging Agent · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning

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

### 4. (Optional) Pre-tune Hyperparameters
```bash
# Find best hyperparameters before training (Bayesian search, 50 trials)
python agent/tune.py --simulator gbm --n-trials 50

# Tune then immediately train with best params
python agent/tune.py --simulator gbm --n-trials 50 --train-after
```

### 5. Train the Agent
```bash
# Default — dynamic HP control enabled automatically
python agent/train.py --total-timesteps 500000 --simulator gbm

# Also enable dynamic entropy control (instead of SAC auto mode)
python agent/train.py --ent-coef 0.1

# With pre-tuned params (copy the command printed by tune.py)
python agent/train.py --lr 3.2e-4 --batch-size 256 --gamma 0.9971 ...
```

### 6. Evaluate
```bash
python agent/evaluate.py --n-episodes 1000
```

### 7. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 8. Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
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
│   └── models/                # Saved checkpoints + VecNormalize stats
├── backtester/
│   ├── historical.py          # yfinance SPY replay environment
│   └── vol_surface.py         # IV surface construction (live + synthetic)
├── dashboard/
│   └── app.py                 # Streamlit dashboard (6 panels)
├── api/
│   └── main.py                # FastAPI backend
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

Finds good hyperparameters **before** training using [Optuna](https://optuna.org) Bayesian optimisation (TPE) with MedianPruner. Bad trials are killed early, cutting total compute by ~40%.

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

Results are saved to `agent/tuning/best_params_<study>.json` and a ready-to-paste `train.py` command is printed on completion.

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

---

### Recommended Workflow

```
tune.py (50 trials)  →  train.py --train-after  →  evaluate.py
   find best HPs           dynamic control              full metrics
   ~2–3 hrs                during 500k steps            1000 episodes
```

## 🧪 Testing

```bash
# All tests
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
| RL Framework | stable-baselines3, PyTorch |
| Pre-training Tuning | Optuna (TPE + MedianPruner) |
| Dynamic HP Control | Custom SB3 callback (SGDR + Sharpe-driven) |
| Market Simulation | NumPy, SciPy |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, uvicorn |
| Data | yfinance |
| CI/CD | GitHub Actions |

## 📊 API Documentation

Start the server and visit `http://localhost:8000/docs` for interactive Swagger UI.

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