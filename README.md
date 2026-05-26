# ⚡ Options Pricing Engine

**C++ Pricing Core · SAC Hedging Agent · GPU-Accelerated Training · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning**

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated and real market conditions, outperforming all classical baselines including the theoretically-optimal Whalley-Wilmott hedger by **65%** on simulation and **+56%** vs delta hedging on real SPY data (p < 0.0001).

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Project](#running-the-project)
5. [Key Metrics](#key-metrics)
6. [Project Structure](#project-structure)
7. [How It Works](#how-it-works)
8. [Known Limitations](#known-limitations)
9. [Technology Stack](#technology-stack)
10. [API Reference](#api-reference)
11. [License](#license)

---

## What This Project Does

This project builds a system that automatically learns to hedge options portfolios using Reinforcement Learning — specifically the Soft Actor-Critic (SAC) algorithm.

In plain terms: when you sell an options contract, you take on risk if the underlying stock price moves. Traditionally, traders manage this risk using "delta hedging" — a mathematical formula that tells you how much of the underlying to hold. This project trains an AI agent to do the same job, but better. It learns to minimise hedging variance from experience, achieving **half the PnL variance** of classical delta hedging while maintaining comparable returns — giving a dramatically higher Sharpe ratio.

The system includes:
- A **C++ pricing engine** for fast option pricing (sub-millisecond per call)
- A **trained RL agent** that beats all classical baselines including theoretically-optimal hedgers
- A **live dashboard** showing the vol surface, Greeks, IV calculator, and agent performance
- A **REST API** for pricing, Greeks, IV, and agent inference
- **Real SPY data validation** covering the April 2025 tariff crash

---

## Prerequisites

### Required

**Python 3.10 or higher**
```bash
python --version   # Should show 3.10+
```

**CMake 3.15 or higher** (for building the C++ pricer)

```bash
# Windows
winget install Kitware.CMake
# macOS
brew install cmake
# Linux
sudo apt install cmake
```

**A C++ compiler**
- Windows: Visual Studio Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
- macOS: `xcode-select --install`
- Linux: `sudo apt install build-essential`

### Optional but Recommended

**NVIDIA GPU with CUDA** — speeds up training from ~4 hours to ~45 minutes on RTX 3050.

```bash
nvidia-smi   # Check GPU availability
```

---

## Installation

### Step 1: Clone and set up environment

```bash
git clone https://github.com/yourusername/options-engine.git
cd options-engine

python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: PyTorch with GPU support (skip if no GPU)

```bash
# CUDA 12.8 (driver 525+)
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python agent/gpu_utils.py
```

### Step 3: Build the C++ Pricing Engine

**Windows (PowerShell):**
```powershell
$pybind11Dir = python -c "import pybind11; print(pybind11.get_cmake_dir())"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "-Dpybind11_DIR=$pybind11Dir"
cmake --build build --config Release
```

**macOS/Linux:**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build --config Release
```

Verify:
```bash
python -c "import pricer; print(pricer.bs_call(100, 100, 1, 0.05, 0.2))"
# Should print: 10.450583...
```

Falls back to Python pricer automatically if C++ build is unavailable.

### Step 4: Configure environment

```bash
cp .env.example .env
```

Edit `.env` to set model paths and device:
```
MODEL_PATH=agent/models/best_heston/best_model
VNORM_PATH=agent/models/vec_normalize_heston.pkl
DEVICE=auto
```

### Step 5: Verify installation

```bash
python -m pytest tests/ -v
```

All tests should pass in ~60 seconds.

---

## Running the Project

### Option A: One command (recommended)

**Windows:**
```powershell
.\start.ps1
```

**macOS/Linux:**
```bash
bash start.sh
```

### Option B: Docker

```bash
docker-compose up
```

### Option C: Manual

```bash
# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

Then open:
- **Dashboard**: http://localhost:8501
- **API docs**: http://localhost:8000/docs

---

### Training Your Own Agent

#### Step 1: Train

```bash
python agent/train.py --simulator heston --total-timesteps 2000000
```

~2 hours on RTX 3050, ~4 hours on CPU.

Monitor in TensorBoard:
```bash
tensorboard --logdir tb_logs
```

Key metrics to watch:
- `eval/mean_reward` — should trend upward
- `train/critic_loss` — should drop and flatten
- `train/actor_loss` — should be negative and stable

#### Step 2: Evaluate

```bash
python agent/evaluate.py \
  --simulator heston \
  --n-episodes 500 \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl
```

#### Step 3: Historical backtest

```bash
python backtester/historical.py \
  --period 2y \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl
```

---

## Key Metrics

### Simulation Results (Heston, n=5000 episodes)

| Agent | XEP Sharpe | Std PnL | Max Drawdown |
|-------|:----------:|:-------:|:------------:|
| **SAC** | **9.92** | **0.50** | **0.39** |
| Whalley-Wilmott | 5.99 | 0.92 | 0.79 |
| Leland | 5.54 | 1.01 | 0.86 |
| Delta | 5.50 | 1.02 | 0.87 |
| Static | 5.32 | 1.12 | 1.01 |
| Random | 3.45 | 1.19 | 1.12 |

Bootstrap CI: [9.39, 10.48] — entirely above all baselines. All comparisons p < 0.01.

SAC achieves similar mean PnL to baselines with **50% lower variance** and **50% lower drawdown** — the variance minimisation objective working as designed.

TC sweep: SAC Sharpe 13.0 vs Delta 7.2 at all transaction cost levels — the agent is TC-aware and trades less when spreads are wide.

### Real Data Validation (SPY)

| Period | SAC Sharpe | Delta Sharpe | Outperformance | p-value | Episodes |
|--------|:----------:|:------------:|:--------------:|:-------:|:--------:|
| Real SPY 1y | 0.301 | 0.193 | **+56%** | **< 0.0001** | 44 |

95% Bootstrap CI on PnL difference: [+0.048, +0.109] — entirely above zero. Tested on May 2025–May 2026 SPY data including the April 2025 tariff crash.

### C++ Pricing Performance

| Benchmark | Result | Target |
|-----------|:------:|:------:|
| BSM 100k calls | < 400ms | < 400ms ✓ |
| Per-call latency | < 5μs | < 10μs ✓ |
| Monte Carlo 50k paths | < 50ms | < 100ms ✓ |
| API P99 latency | < 1ms | < 3ms ✓ |

---

## Project Structure

```
options-engine/
├── src/pricer/              # C++17 pricing core (pybind11)
│   ├── bs_pricer.hpp        # Black-Scholes pricing + Greeks
│   ├── mc_pricer.hpp        # Monte Carlo with antithetic variates
│   └── iv_solver.hpp        # Brent's method IV solver (1e-8 precision)
├── environment/
│   ├── options_env.py       # Gymnasium env (12-dim obs, residual actions)
│   ├── baselines.py         # Delta, Leland, Whalley-Wilmott, Static, Random
│   └── market_sim.py        # GBM, Heston, Jump, Regime simulators
├── agent/
│   ├── train.py             # SAC + two-tower extractor + dynamic HP control
│   ├── tune.py              # Optuna hyperparameter search
│   ├── evaluate.py          # Full evaluation vs all baselines
│   └── gpu_utils.py         # Device detection + cuDNN config
├── backtester/
│   ├── historical.py        # Real SPY backtest with VIX as IV proxy
│   └── vol_surface.py       # Live IV surface construction via yfinance
├── api/
│   └── main.py              # FastAPI backend (startup model loading, .env config)
├── dashboard/
│   └── app.py               # Streamlit dashboard (all endpoints connected)
├── tests/                   # 86 tests — pricer, env, baselines, API
├── Dockerfile               # Multi-stage build (C++ + runtime)
├── docker-compose.yml       # One-command startup
├── start.ps1                # Windows one-command startup
├── start.sh                 # Linux/macOS one-command startup
├── .env.example             # Environment variable template
└── RESULTS.md               # Reproducibility table with exact commands
```

---

## How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│  Vol Surface | Greeks | IV Calc | PnL | Risk | Bench    │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (FastAPI)
┌──────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                        │
│  /price  /greeks  /iv  /agent/action  /benchmark        │
│  Startup model loading · .env config · health check     │
└────────┬─────────────────────────────────┬──────────────┘
         │                                 │
┌────────▼────────┐              ┌─────────▼──────────────┐
│  C++ Pricer     │              │  SAC Agent              │
│  (pybind11)     │              │  Two-tower network      │
│                 │              │  Fixed ent_coef=0.15    │
│  Black-Scholes  │              │  Residual action space  │
│  Monte Carlo    │              │  Variance-min reward    │
│  Greeks         │              │  Dynamic HP control     │
│  Implied Vol    │              │                         │
└─────────────────┘              └────────────┬────────────┘
                                              │
                               ┌──────────────▼────────────┐
                               │  Gymnasium Environment     │
                               │  Heston stochastic vol    │
                               │  12-dim observations      │
                               │  Execution delay=0        │
                               │  Transaction cost=0.003   │
                               └───────────────────────────┘
```

### Observation Space (12-dim)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `S/S0` | Normalised spot price |
| 1 | `K/S0` | Normalised strike |
| 2 | `T_rem/T` | Remaining time fraction |
| 3 | `sigma` | Current realised volatility |
| 4 | `delta` | Black-Scholes delta |
| 5 | `gamma × 100` | Scaled gamma |
| 6 | `pnl_norm` | Normalised portfolio PnL |
| 7 | `vega` | Sensitivity to vol moves |
| 8 | `theta` | Time decay (clipped ≥ 0) |
| 9 | `vol_carry` | Realised / implied vol ratio |
| 10 | `hedge_pos` | Current executed hedge position |
| 11 | `vol_regime` | 0.0=low / 0.5=medium / 1.0=high |

### Neural Network Architecture

```
obs[:6]  → Market Tower  : Linear(6,256) → LayerNorm → ReLU → 256-dim
obs[6:]  → Portfolio Tower: Linear(6,128) → LayerNorm → ReLU → 128-dim
         → Merge: cat(256,128) → Linear(384,512) → ReLU → 512-dim features
         → Actor:  [512, 256] → action ∈ [-1, 1]
         → Critic: [512, 256] → Q-value
```

The two-tower structure separates market information (price, vol, Greeks) from portfolio state (PnL, hedge position, regime), reflecting how options traders mentally partition market vs position risk.

### Action Space

```
action ∈ [-1, 1]  (residual correction to BS delta)
target_hedge = clip(delta + 0.3 × action, -1, 1)
```

Action = 0 means pure delta hedge. The agent learns when and by how much to deviate.

### Reward Function

```
reward = -0.5 × PnL² - TC - 0.005 × |Δposition| + 0.3
```

The `-0.5 × PnL²` variance penalty directly aligns the objective with Sharpe maximisation. The `+0.3` constant offset makes Q-values positive (fixes actor loss) without affecting the optimisation gradient.

### Why SAC Beats Leland and Whalley-Wilmott

Leland (1985) and Whalley-Wilmott (1997) are analytically derived optimal solutions under BSM assumptions. SAC must discover something they cannot capture:

- **Whalley-Wilmott** minimises TC cost for a CARA investor — optimal under constant vol and TC
- **Leland** corrects BS delta for discrete hedging costs — optimal under BSM with TC

SAC wins by learning a globally conservative strategy: it generates slightly less mean PnL but with dramatically less variance, giving a higher Sharpe ratio. The key insight is that minimising `PnL²` (the reward objective) is more directly aligned with Sharpe than the TC-focused analytical solutions.

### Market Simulators

| Simulator | Model | Key Characteristic |
|-----------|-------|-------------------|
| `gbm` | Geometric Brownian Motion | Constant vol, log-normal |
| `heston` | Heston (1993) | Mean-reverting stochastic vol |
| `jump` | Merton (1976) | Poisson jumps, left-skewed |
| `regime` | Markov chain vol | Switches between low/high vol states |

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 1e-4 | Cosine annealing, cycle=T/2 |
| `buffer_size` | 500k | Replay buffer |
| `batch_size` | 256 | Per gradient update |
| `ent_coef` | 0.15 | Fixed — prevents entropy collapse |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.01 | Target network soft update |
| `learning_starts` | 50000 | Steps before first gradient update |
| `gradient_steps` | 1–2 | Dynamic, based on critic stability |

---

## Troubleshooting

**`import pricer` fails after building**
Run `python -c "import pricer"` from the project root. If it fails, check that CMake and a C++ compiler are installed. The Python fallback works automatically.

**Training is slow**
On CPU, 2M steps takes ~4 hours. Use `--total-timesteps 500000` for a quicker test. GPU is strongly recommended.

**Dashboard shows 0.00 for all Sharpe values**
Run `evaluate.py` first to generate `agent/evaluation_results.json`. The dashboard reads from this file.

**`yfinance` errors during historical backtest**
Yahoo Finance rate-limits occasionally. Wait a few minutes and retry.

**SAC model loads without VecNormalize**
Check that `vec_normalize_heston.pkl` exists in `agent/models/`. It is saved automatically at the end of `train.py`.

**Actor loss is positive during training**
Check that the reward in `options_env.py` includes the `+0.3` constant offset. Without it, all Q-values are negative and actor loss is positive.

---

## Known Limitations

**Trained on synthetic data only.** The agent was trained exclusively on Heston stochastic vol simulations. Real market dynamics include microstructure, liquidity gaps, and vol jumps not captured by Heston.

**Real-data Sharpe is modest (0.30) in absolute terms.** The meaningful claim is relative outperformance vs delta hedging (+56%, p<0.0001), not the absolute Sharpe vs risk-free rate.

**IV proxy.** The historical backtester uses VIX as a proxy for SPY 30-day implied vol. This is accurate on average but diverges during stress events.

**ATM only.** The agent was evaluated exclusively on at-the-money calls. OTM, ITM, puts, and term structure are untested.

**44 real-data episodes.** One year of daily SPY data with stride=5 gives 44 non-overlapping episodes — statistically significant but thin.

**The training Sharpe (> 10) is not a result.** It is the cross-episode Sharpe on the Heston training distribution — the same distribution the agent was trained on. The credible result is the 5000-episode held-out Heston evaluation (9.92) and the real SPY backtest (0.30 vs 0.19).

---

## Testing

```bash
# Full suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov=environment --cov=api -v

# Pricer only
python -m pytest tests/test_pricer.py -v

# Environment + baselines only
python -m pytest tests/test_environment.py -v
```

86 tests covering: pricer correctness (put-call parity, Greeks identities, IV round-trip, MC convergence), environment Gymnasium compliance, execution delay behaviour, variable TC, vol regime flag, baseline ordering (Leland ≥ Delta, Delta ≥ Random), Whalley-Wilmott band logic, and the v1 metric-bug regression test.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Pricing Engine | C++17, pybind11 3.0 |
| RL Framework | stable-baselines3 2.x, PyTorch 2.x |
| Neural Network | Custom two-tower nn.Module, LayerNorm |
| GPU | CUDA 12.8+, cuDNN |
| Hyperparameter Tuning | Optuna (TPE sampler, MedianPruner) |
| Dynamic HP Control | Custom SB3 callback (SGDR + Sharpe-driven) |
| Market Simulation | GBM, Heston SV, Merton Jump, Regime-switching |
| Statistical Validation | Bootstrap CI, paired t-test, TC sensitivity sweep |
| Historical Validation | Real SPY data (2024–2026) via yfinance, VIX as IV proxy |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, uvicorn, python-dotenv |
| Containerisation | Docker, docker-compose |

---

## API Reference

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, model info, uptime |
| POST | `/price` | BSM or Monte Carlo pricing |
| POST | `/greeks` | Δ, Γ, ν, Θ, ρ |
| POST | `/iv` | Implied vol via Brent's method |
| POST | `/agent/action` | SAC hedge ratio (12-dim observation) |
| GET | `/benchmark` | C++ pricer throughput test |

The agent endpoint accepts a 12-dimensional observation vector:
```json
{
  "observation": [S/S0, K/S0, T_rem/T, sigma, delta, gamma*100,
                  pnl_norm, vega, theta, vol_carry, hedge_pos, vol_regime]
}
```

Returns `action ∈ [-1, 1]`. Target hedge = `clip(delta + 0.3 × action, -1, 1)`.

---

## License

MIT