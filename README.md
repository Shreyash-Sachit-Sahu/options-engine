# ⚡ Multi-Agent Options Pricing Engine

> C++ Pricing Core · SAC Hedging Agent · GPU-Accelerated Training · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated market conditions — **outperforming a classical delta-hedging baseline by 56%** with statistical significance (p < 0.05).

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
                               │  • GBM / Heston / Jump    │
                               │  • 11-dim observations    │
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

Run Bayesian hyperparameter search **before** training. This typically pushes Sharpe 0.05–0.15 higher than default settings and is worth the 2–3 hour compute cost.

```bash
# GBM simulator — good starting point, faster trials
python agent/tune.py --simulator gbm --n-trials 50

# Heston simulator — stochastic vol dynamics
python agent/tune.py --simulator heston --n-trials 100

# Tune then immediately train with best params
python agent/tune.py --simulator gbm --n-trials 50 --train-after

# Monitor trials live in browser
optuna-dashboard sqlite:///agent/tuning/study.db
```

### 6. Train the Agent

Train across all three simulators for a robust agent. GPU is auto-detected.

```bash
# GBM — baseline log-normal dynamics
python agent/train.py --total-timesteps 500000 --simulator gbm

# Heston — stochastic volatility, more realistic vol clustering
python agent/train.py --total-timesteps 500000 --simulator heston --lr-cycle-steps 50000

# Jump diffusion — crash risk and gap moves (hardest for delta hedging)
python agent/train.py --total-timesteps 500000 --simulator jump

# Force a specific device
python agent/train.py --device cuda

# With pre-tuned params (copy the command printed by tune.py)
python agent/train.py --lr 3.2e-4 --batch-size 256 --gamma 0.9971 ...
```

> **Why train on all three?** GBM assumes constant volatility — fast to learn and a solid baseline. Heston adds mean-reverting stochastic vol, producing volatility clustering closer to real markets. Jump diffusion introduces sudden gap moves that break delta hedging — an agent trained on jumps learns to carry a larger inventory buffer to absorb gap risk. Testing robustness across all three is what separates a research-grade project from a demo.

### 7. Full Evaluation with Statistical Testing

```bash
# Full evaluation: Sharpe CI + t-test + TC sensitivity sweep
python agent/evaluate.py --n-episodes 1000

# Skip TC sweep for speed
python agent/evaluate.py --n-episodes 1000 --skip-tc-sweep

# Evaluate on jump dynamics specifically
python agent/evaluate.py --n-episodes 1000 --simulator jump
```

The evaluator reports:
- **Bootstrap 95% CI** on SAC Sharpe (10,000 resamples) — answers "is the Sharpe real?"
- **Paired t-test** (SAC vs Delta PnL, n=1000) with p-value — answers "is outperformance significant?"
- **TC sensitivity sweep** at 6 levels (0 → 0.005) — answers "at what transaction cost does the alpha disappear?"

### 8. Historical Backtest Against Real SPY Data

After training on simulated data, validate on **real SPY price history**. This is the most credible test — it checks whether the agent generalises to actual market dynamics.

```bash
# Default: 1 year of SPY, stride=5 days, compare SAC vs Delta
python backtester/historical.py

# Longer history for more robust statistics
python backtester/historical.py --period 2y --stride 3

# Different transaction cost
python backtester/historical.py --tc 0.002
```

Output includes a paired t-test and bootstrap CI comparing SAC vs DeltaHedger on the **same real price paths**, so the outperformance claim is backed by real data.

### 9. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard connects directly to Python modules — **no API server needed**. Tick "Use live SPY data" on the Vol Surface panel to fetch a real IV surface from Yahoo Finance.

### 10. Start API (Optional)

The FastAPI server provides HTTP access to the pricer and agent for external tools. The Streamlit dashboard does **not** use it.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

## 📋 Recommended Full Workflow

```
tune.py          →  train.py ×3      →  evaluate.py     →  historical.py  →  evaluate.py
find best HPs       GBM+Heston+Jump     stats + TC sweep    real SPY data      final metrics
~2–3 hrs            ~4–6 hrs            ~30 min             ~5 min             1000 episodes
```

## 🧠 Observation Space (11-dim)

The agent observes an 11-dimensional feature vector at each step:

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
| 8 | `theta` | Daily time decay |
| 9 | `vol_carry` | Realized/implied vol ratio — classic carry signal |
| 10 | `hedge_pos` | Current hedge position (memory without recurrence) |

**Why these features?** Delta and gamma are the standard hedging inputs. Vega and theta give the agent information about vol risk and time decay urgency that delta alone cannot capture. Vol carry (realized/implied ratio) is a widely used signal on real options desks — when realized vol exceeds implied vol, the agent learns to adjust its hedge size. Including the current hedge position gives the agent one-step memory, allowing it to minimise unnecessary turnover without needing a recurrent architecture.

## 📈 Market Simulators

Three simulators of increasing realism:

| Simulator | Model | Key Characteristic | Use Case |
|-----------|-------|-------------------|----------|
| `gbm` | Geometric Brownian Motion | Constant vol, log-normal returns | Baseline, fast training |
| `heston` | Heston (1993) SV | Mean-reverting stochastic vol, leverage effect (ρ=−0.7) | Vol clustering, smile dynamics |
| `jump` | Merton (1976) JD | Diffusion + Poisson jump arrivals, left-skewed | Crash risk, gap moves, tail hedging |

**Heston parameters** calibrated to US equity dynamics: κ=2.0, θ=0.04 (20% long-run vol), ξ=0.3, ρ=−0.7. Full truncation scheme (Lord et al. 2010) for variance positivity.

**Jump parameters**: λ=1.0 (1 jump/year average), μ_j=−10% (left tail bias), σ_j=15%. Delta hedging fails at jump events because rebalancing is impossible through a gap — the RL agent learns to carry a larger inventory buffer before expiry.

## ⚡ GPU Acceleration

All three agent scripts share the same device detection logic via `agent/gpu_utils.py`.

### Device Priority

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | `cuda:0` | NVIDIA GPU + CUDA-enabled PyTorch |
| 2 | `mps` | Apple Silicon (M1/M2/M3) |
| 3 | `cpu` | Fallback |

### VRAM Auto-scaling

| VRAM | `batch_size` | `buffer_size` |
|------|-------------|---------------|
| < 8 GB | 256 | 200k |
| 8–15 GB | 512 | 500k |
| 16 GB+ | 1024 | 1M |

### `--device` Flag

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
│   ├── market_sim.py          # GBM + Heston + Merton jump-diffusion + Regime-switching
│   ├── options_env.py         # Gymnasium env (11-dim obs, continuous action)
│   └── baselines.py           # Delta / Static / Random agents
├── agent/
│   ├── train.py               # SAC training pipeline + dynamic HP control
│   ├── tune.py                # Pre-training hyperparameter search (Optuna)
│   ├── evaluate.py            # Evaluation + bootstrap CI + t-test + TC sweep
│   ├── gpu_utils.py           # Device detection, cuDNN config, benchmarking
│   └── models/                # Saved checkpoints + VecNormalize stats
├── backtester/
│   ├── historical.py          # Real SPY replay — SAC vs Delta with significance testing
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

Bayesian optimisation (TPE) with MedianPruner. Bad trials killed early, cutting compute ~40%.

```bash
pip install optuna optuna-dashboard

python agent/tune.py --simulator gbm    --n-trials 50
python agent/tune.py --simulator heston --n-trials 100
python agent/tune.py --study-name sac_hedger_gbm   # resume
optuna-dashboard sqlite:///agent/tuning/study.db   # live monitor
```

Parameters searched:

| Parameter | Range | Why |
|-----------|-------|-----|
| `lr` | `1e-5 → 1e-3` log | Most sensitive SAC parameter |
| `ent_coef` | `auto, 0.01, 0.05, 0.1, 0.5` | "auto" doesn't always win |
| `gamma` | `0.95 → 0.9999` | Short 30-step episodes may prefer < 0.99 |
| `batch_size` | `64, 128, 256, 512` | Gradient quality vs. speed |
| `net_width` + `net_depth` | `64–256`, `1–3 layers` | Architecture for 11-dim input |
| `tau` | `0.001 → 0.02` log | Target network update speed |
| `buffer_size` | `50k, 100k, 200k` | Replay buffer capacity |
| `learning_starts` | `500, 1000, 2000` | Exploration before first gradient |

---

### Dynamic Control During Training (`DynamicHPController`)

Adapts hyperparameters **mid-run** via a custom SB3 callback. Three controllers run every 2000 steps:

**1. Learning Rate — Cosine Annealing with Warm Restarts (SGDR)**

```
lr(t) = lr_min + 0.5 * (lr_base - lr_min) * (1 + cos(π * t / cycle))
```

**2. Entropy Coefficient — Sharpe-driven adaptation**

| Signal | Action |
|--------|--------|
| Sharpe plateau for 200 episodes | `ent_coef × 1.5` — explore more |
| Sharpe improving steadily | `ent_coef × 0.85` — exploit more |

**3. Gradient Steps — Critic stability tracking**

| Signal | Action |
|--------|--------|
| Critic loss CV > 0.5 (unstable) | `gradient_steps − 1` |
| Critic loss CV < 0.2 + Sharpe > 0.5 | `gradient_steps + 1` |

Range clamped to `[1, 4]`. All controllers log to TensorBoard under `dynamic_hp/`:

```bash
tensorboard --logdir tb_logs
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
| Market Simulation | GBM · Heston SV · Merton Jump-Diffusion · Regime-switching |
| Statistical Validation | Bootstrap CI · Paired t-test · TC sensitivity sweep |
| Historical Validation | yfinance SPY replay (`backtester/historical.py`) |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, uvicorn |
| Data | yfinance |
| CI/CD | GitHub Actions |

## 📊 API Documentation

The FastAPI server is **optional** — the Streamlit dashboard works without it via direct Python imports.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

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