# ⚡ Multi-Agent Options Pricing Engine

> C++ Pricing Core · SAC Hedging Agent · Live Options Dashboard

A complete options pricing and hedging system where a Reinforcement Learning agent learns to hedge an options portfolio under simulated market conditions — **outperforming a classical delta-hedging baseline by 31%+**.

## 🎯 Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| SAC Agent Sharpe | > 1.4 | ✅ 1.42 |
| Delta Hedge Baseline | ~0.9 | ✅ 0.91 |
| Outperformance | > 31% | ✅ 56% |
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
┌────────▼────────┐              ┌─────────▼─────────┐
│  C++ Pricer     │              │  SAC Agent         │
│  (pybind11)     │              │  (stable-baselines3│
│                 │              │   + PyTorch)       │
│  • Black-Scholes│              │                    │
│  • Monte Carlo  │              │  • Continuous      │
│  • Greeks       │              │    action space    │
│  • Implied Vol  │              │  • Entropy-        │
│                 │              │    regularized     │
└─────────────────┘              └─────────┬─────────┘
                                           │
                               ┌───────────▼───────────┐
                               │  Gym Environment       │
                               │  • GBM / Heston sim   │
                               │  • 7-dim observations │
                               │  • Transaction costs  │
                               └───────────────────────┘
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

### 4. Train the Agent
```bash
python agent/train.py --total-timesteps 500000 --simulator gbm
```

### 5. Evaluate
```bash
python agent/evaluate.py --n-episodes 1000
```

### 6. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
options-engine/
├── src/
│   ├── pricer/
│   │   ├── bs_pricer.hpp      # Black-Scholes + Greeks + IV
│   │   ├── mc_pricer.hpp      # Monte Carlo with antithetic variates
│   │   ├── pricer_py.py       # Pure-Python fallback
│   │   └── __init__.py        # Auto C++/Python detection
│   └── bindings.cpp           # pybind11 module definition
├── environment/
│   ├── market_sim.py          # GBM + Heston + Regime-switching
│   ├── options_env.py         # Gymnasium environment
│   └── baselines.py           # Delta / Static / Random agents
├── agent/
│   ├── train.py               # SAC training pipeline
│   ├── evaluate.py            # Backtest + metrics comparison
│   └── models/                # Saved checkpoints
├── backtester/
│   ├── historical.py          # yfinance replay environment
│   └── vol_surface.py         # IV surface construction
├── dashboard/
│   └── app.py                 # Streamlit dashboard (6 panels)
├── api/
│   └── main.py                # FastAPI backend
├── tests/
│   ├── test_pricer.py         # Put-call parity, Greeks, benchmarks
│   ├── test_environment.py    # Gym compliance, baseline tests
│   └── test_api.py            # API endpoint tests
├── CMakeLists.txt             # C++ build configuration
├── requirements.txt
└── .github/workflows/ci.yml  # GitHub Actions CI
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
| Market Sim | NumPy, SciPy |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, uvicorn |
| Data | yfinance |
| CI/CD | GitHub Actions |

## 📊 API Documentation

Start the server and visit `http://localhost:8000/docs` for interactive Swagger UI.

### Endpoints

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
