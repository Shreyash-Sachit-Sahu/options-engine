# Multi-Agent Options Pricing Engine

**C++ Pricing Core · SAC Hedging Agent · GPU-Accelerated Training · Live Options Dashboard · Automated & Dynamic Hyperparameter Tuning**

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated and real market conditions, outperforming classical delta-hedging by 48% on real SPY data (p < 0.0001).

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

In plain terms: when you sell an options contract, you take on risk if the underlying stock price moves. Traditionally, traders manage this risk using "delta hedging" — a mathematical formula that tells you how much of the underlying to hold. This project trains an AI agent to do the same job, but better — it learns from experience rather than following a fixed formula, and outperforms the classical approach by 48% on real market data.

The system includes:
- A **C++ pricing engine** for fast option pricing (150ms for 1 million options)
- A **trained RL agent** that outperforms classical delta hedging
- A **live dashboard** showing the vol surface, Greeks, and agent performance
- A **REST API** for pricing and agent inference
- **Real SPY data validation** covering the April 2025 tariff crash

---

## Prerequisites

Before you start, make sure you have the following installed.

### Required

**Python 3.10 or higher**
```bash
python --version   # Should show 3.10+
```
Download from https://python.org if needed.

**Git**
```bash
git --version
```
Download from https://git-scm.com if needed.

**CMake 3.15 or higher** (for building the C++ pricer)

On Windows:
```bash
winget install Kitware.CMake
```
On macOS:
```bash
brew install cmake
```
On Linux:
```bash
sudo apt install cmake
```

**A C++ compiler**
- Windows: Install Visual Studio Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
- macOS: Run `xcode-select --install`
- Linux: `sudo apt install build-essential`

### Optional but Recommended

**NVIDIA GPU with CUDA support** — speeds up training from ~6 hours to ~90 minutes. If you do not have a GPU, training still works on CPU, just slower.

Check if you have a compatible GPU:
```bash
nvidia-smi
```
If this command works and shows your GPU, you have CUDA support.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/options-engine.git
cd options-engine
```

### Step 2: Create a Virtual Environment

This keeps the project dependencies isolated from your system Python.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will take a few minutes. It installs PyTorch, stable-baselines3, FastAPI, Streamlit, and all other dependencies.

### Step 4: Install PyTorch with GPU Support (Skip if no GPU)

If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch. First check your driver version:

```bash
nvidia-smi
```

Look for "CUDA Version" in the top right. Then install the matching PyTorch build:

```bash
# CUDA 12.8 (driver 525 or higher — most modern GPUs)
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8 (older drivers)
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu118
```

Verify the GPU is detected:
```bash
python agent/gpu_utils.py
```
You should see your GPU name and "CUDA available: YES".

### Step 5: Build the C++ Pricing Engine

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build --config Release
```

On Windows, the second command may need `--config Release` explicitly:
```bash
cmake --build build --config Release
```

Verify it worked:
```bash
python -c "import pricer; print(pricer.bs_call(100, 100, 1, 0.05, 0.2))"
# Should print: 10.450583...
```

If you see an ImportError, the C++ build failed. The project will still work using the Python fallback — you just lose the speed advantage. Check that CMake and a C++ compiler are both installed correctly.

### Step 6: Verify the Full Installation

```bash
python -m pytest tests/ -v
```

All 70 tests should pass in about 15 seconds.

---

## Running the Project

You have two options: run the pre-trained demo immediately, or train your own agent from scratch.

### Option A: Run the Demo (Recommended First Step)

Start the API server in one terminal:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal, start the dashboard:
```bash
streamlit run dashboard/app.py
```

Open your browser to `http://localhost:8501`. You will see:
- A live 3D implied volatility surface
- Option Greeks updating in real-time as you move the sliders
- Agent performance charts (if a trained model exists in `agent/models/`)
- A benchmark tab to test C++ pricing speed

The API documentation is at `http://localhost:8000/docs`.

---

### Option B: Train Your Own Agent

This is a 5-step process. Steps 1–2 are one-time setup. Steps 3–5 are the actual training pipeline.

#### Step 1: Run Hyperparameter Tuning (Optional but Recommended)

This finds the best training configuration before the main training run. Takes 2–10 hours depending on hardware.

```bash
python agent/tune.py --simulator heston --n-trials 50 --device cuda
```

Remove `--device cuda` if you do not have a GPU.

To watch the trials in real-time, open a second terminal and run:
```bash
optuna-dashboard sqlite:///agent/tuning/study.db
```
Then open `http://localhost:8080`.

When finished, the best hyperparameters are printed. Copy the retrain command it outputs — it will look like:
```bash
python agent/train.py --device cuda --simulator heston --lr 4.10e-04 ...
```

#### Step 2: Train the Agent

Using the command from tuning (replace with your own values):
```bash
python agent/train.py --simulator heston --total-timesteps 1000000 \
  --lr-cycle-steps 50000 --device cuda \
  --lr 4.10e-4 --batch-size 128 --buffer-size 200000 \
  --tau 0.001 --gamma 0.9925 --ent-coef 0.05 --learning-starts 1000
```

This takes approximately 90 minutes on an RTX 3050 or similar GPU, and 4–6 hours on CPU.

While training, monitor progress in TensorBoard:
```bash
tensorboard --logdir tb_logs
```
Then open `http://localhost:6006`.

Training is complete when you see:
```
[DONE] Training complete in XXXX.Xs
[SAVE] Model: agent/models/sac_hedger_heston_final.zip
```

#### Step 3: Evaluate the Agent

```bash
python agent/evaluate.py --simulator heston --n-episodes 500 \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --skip-tc-sweep
```

This runs 500 test episodes and reports Sharpe ratio, max drawdown, and a statistical comparison against the delta hedging baseline.

#### Step 4: Historical Backtest on Real SPY Data

```bash
python backtester/historical.py \
  --model-path agent/models/best_heston/best_model \
  --vnorm-path agent/models/vec_normalize_heston.pkl \
  --period 1y
```

This downloads real SPY price data via Yahoo Finance and tests the agent on actual market conditions.

#### Step 5: Start the Dashboard

```bash
# Terminal 1
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
streamlit run dashboard/app.py
```

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

## How It Works

### Architecture

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

### Observation Space (11-dim)

At each step, the agent observes an 11-dimensional vector:

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
| 8 | `theta` | Time decay (negated) |
| 9 | `vol_carry` | Realized/implied vol ratio |
| 10 | `hedge_pos` | Current hedge position |

### Market Simulators

| Simulator | Model | Key Characteristic |
|-----------|-------|-------------------|
| `gbm` | Geometric Brownian Motion | Constant vol, log-normal returns |
| `heston` | Heston (1993) SV | Mean-reverting stochastic vol, rho=-0.7 |
| `jump` | Merton (1976) JD | Poisson jumps, left-skewed (mu_j=-10%) |

### GPU Acceleration

| VRAM | batch_size | buffer_size |
|------|------------|-------------|
| < 8 GB | 256 | 200k |
| 8–15 GB | 512 | 500k |
| 16 GB+ | 1024 | 1M |

---

## Troubleshooting

**`import pricer` fails after building**
The `.so` file needs to be in your working directory or on `PYTHONPATH`. Run `python -c "import pricer"` from the project root. If it still fails, check that the build succeeded with no errors.

**`nvidia-smi` not found on Windows**
Your NVIDIA driver may not be installed. Download from https://www.nvidia.com/Download/index.aspx. After installing, restart and try again.

**`python agent/gpu_utils.py` shows CPU despite having a GPU**
Your PyTorch is the CPU-only build. Run the CUDA reinstall command in Step 4.

**Training is very slow**
Without a GPU, 1M steps takes 4–6 hours. You can reduce `--total-timesteps` to 300000 for a quicker test run, though results will be worse.

**`yfinance` errors during historical backtest**
Yahoo Finance rate-limits requests occasionally. Wait a few minutes and retry. The backtest requires an internet connection.

**Dashboard shows no agent data**
You need to run `evaluate.py` first to generate `agent/evaluation_results.json`. The dashboard reads from this file.

---

## Known Limitations

**Training data is entirely synthetic.** The agent was trained on Heston stochastic vol simulations, not real market data. The simulator parameters are calibrated to typical US equity dynamics but do not capture all real-world effects such as volatility jumps, liquidity gaps, or intraday microstructure.

**Real-data backtest uses simplified market assumptions.** The historical backtester uses ATM options (K = spot at episode start), a fixed risk-free rate (r=0.05), and realized vol as a proxy for implied vol. In practice, IV can diverge significantly from realized vol, particularly during stress events.

**Transaction cost is fixed.** The TC rate (0.003) is constant. Real bid-ask spreads are dynamic, wider in volatile periods, and volume-dependent.

**Strike selection is ATM only.** The agent was evaluated exclusively on at-the-money options. Performance on OTM, ITM, or term-structure trades is untested.

**No live trading.** This is a research and backtesting system only. It has not been tested in a paper trading or live execution environment.

**The training Sharpe of 13.96 is not a result.** This is the peak rolling mean of within-episode information ratios on the training simulator — a relative signal used by the dynamic HP controller, not an out-of-sample metric. The credible results are: Sharpe 2.75 on 500 held-out Heston episodes, and Sharpe 0.29 on 1 year of real SPY data.

---

## Hyperparameter Tuning Details

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

Three dynamic controllers run during training, updating every 2000 steps:

1. **Learning rate** — cosine annealing with warm restarts (SGDR).
2. **Entropy coefficient** — responds to rolling Sharpe plateau or improvement.
3. **Gradient steps** — increases when critic loss is stable, decreases when unstable.

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# With coverage report
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