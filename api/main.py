"""
FastAPI Backend — Options Pricing API

Stateless REST API serving pricing requests, Greeks computation,
implied volatility, and SAC agent actions. All pricing calls use
the C++ compiled module directly for sub-millisecond latency.

Endpoints:
  POST /price   — BSM or MC pricing
  POST /greeks  — full Greeks computation
  POST /iv      — implied volatility solver
  POST /agent/action — SAC agent inference
  GET  /health  — health check
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── Import pricer (C++ first, Python fallback) ─────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import pricer as cpp_pricer
    PRICER_BACKEND = "C++"
    print("✅ Using C++ pricing backend")
except ImportError:
    from src.pricer import pricer_py as cpp_pricer
    PRICER_BACKEND = "Python"
    print("⚠️  Using Python pricing backend (C++ module not found)")


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Options Pricing Engine API",
    description="High-performance options pricing with C++ core, "
                "RL hedging agent, and Greeks computation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class PriceRequest(BaseModel):
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to maturity (years)")
    r: float = Field(0.05, description="Risk-free rate")
    sigma: float = Field(..., gt=0, description="Volatility")
    method: Literal["bs", "mc"] = Field("bs", description="Pricing method")
    n_paths: int = Field(50000, gt=0, description="MC paths (if method='mc')")
    flag: Literal["call", "put"] = Field("call", description="Option type")

class PriceResponse(BaseModel):
    price: float
    method: str
    backend: str
    std_error: Optional[float] = None
    n_paths: Optional[int] = None
    latency_ms: float

class GreeksRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float = Field(0.05)
    sigma: float = Field(..., gt=0)
    flag: Literal["call", "put"] = Field("call")

class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    backend: str
    latency_ms: float

class IVRequest(BaseModel):
    market_price: float = Field(..., gt=0)
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float = Field(0.05)
    flag: Literal["call", "put"] = Field("call")

class IVResponse(BaseModel):
    implied_vol: float
    backend: str
    latency_ms: float

class AgentActionRequest(BaseModel):
    observation: list = Field(
        ..., min_length=7, max_length=7,
        description="7-dim observation: [S/S0, K/S0, T_rem/T, sigma, delta, gamma, pnl]"
    )

class AgentActionResponse(BaseModel):
    action: float
    agent_type: str
    latency_ms: float


# ─── Agent Loading (Lazy) ────────────────────────────────────────────────────

_sac_model = None

def get_sac_model():
    global _sac_model
    if _sac_model is None:
        model_path = Path(__file__).parent.parent / "agent" / "models" / "sac_hedger_final.zip"
        if model_path.exists():
            from stable_baselines3 import SAC
            _sac_model = SAC.load(str(model_path))
            print(f"✅ Loaded SAC model from {model_path}")
        else:
            print(f"⚠️  No SAC model found at {model_path}")
            return None
    return _sac_model


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pricing_backend": PRICER_BACKEND,
        "sac_model_loaded": _sac_model is not None,
    }


@app.post("/price", response_model=PriceResponse)
async def price_option(req: PriceRequest):
    """
    Price a European option using Black-Scholes or Monte Carlo.
    Target latency: < 3ms for BSM, < 50ms for MC (50k paths).
    """
    start = time.perf_counter()

    try:
        if req.method == "bs":
            if req.flag == "call":
                p = cpp_pricer.bs_call(req.S, req.K, req.T, req.r, req.sigma)
            else:
                p = cpp_pricer.bs_put(req.S, req.K, req.T, req.r, req.sigma)

            latency = (time.perf_counter() - start) * 1000
            return PriceResponse(
                price=p, method="black-scholes", backend=PRICER_BACKEND,
                latency_ms=latency
            )

        else:  # mc
            result = cpp_pricer.mc_price(
                req.S, req.K, req.T, req.r, req.sigma,
                n_paths=req.n_paths, flag=req.flag
            )
            latency = (time.perf_counter() - start) * 1000
            return PriceResponse(
                price=result.price, method="monte-carlo",
                backend=PRICER_BACKEND, std_error=result.std_error,
                n_paths=result.n_paths, latency_ms=latency
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/greeks", response_model=GreeksResponse)
async def compute_greeks(req: GreeksRequest):
    """Compute all Greeks for a European option."""
    start = time.perf_counter()

    try:
        g = cpp_pricer.greeks(req.S, req.K, req.T, req.r, req.sigma, req.flag)
        latency = (time.perf_counter() - start) * 1000

        return GreeksResponse(
            delta=g.delta, gamma=g.gamma, vega=g.vega,
            theta=g.theta, rho=g.rho, backend=PRICER_BACKEND,
            latency_ms=latency
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/iv", response_model=IVResponse)
async def compute_iv(req: IVRequest):
    """Compute implied volatility via Brent's method."""
    start = time.perf_counter()

    try:
        iv = cpp_pricer.implied_vol(
            req.market_price, req.S, req.K, req.T, req.r, req.flag
        )
        latency = (time.perf_counter() - start) * 1000

        return IVResponse(
            implied_vol=iv, backend=PRICER_BACKEND, latency_ms=latency
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/agent/action", response_model=AgentActionResponse)
async def agent_action(req: AgentActionRequest):
    """Get SAC agent's hedge action for a given observation."""
    start = time.perf_counter()

    model = get_sac_model()

    if model is None:
        # Fallback to delta-hedging if no model loaded
        obs = np.array(req.observation, dtype=np.float32)
        action = float(obs[4])  # Use delta as action
        latency = (time.perf_counter() - start) * 1000
        return AgentActionResponse(
            action=action, agent_type="delta_fallback", latency_ms=latency
        )

    try:
        obs = np.array(req.observation, dtype=np.float32).reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        latency = (time.perf_counter() - start) * 1000

        return AgentActionResponse(
            action=float(action[0][0]), agent_type="sac",
            latency_ms=latency
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/benchmark")
async def benchmark():
    """Run a quick benchmark of the pricing engine."""
    import time

    # BSM benchmark
    n_calls = 100_000
    start = time.perf_counter()
    for _ in range(n_calls):
        cpp_pricer.bs_call(100, 100, 1, 0.05, 0.2)
    bsm_time = time.perf_counter() - start

    # MC benchmark
    start = time.perf_counter()
    cpp_pricer.mc_price(100, 100, 1, 0.05, 0.2, n_paths=50000)
    mc_time = time.perf_counter() - start

    return {
        "bsm_100k_calls_ms": round(bsm_time * 1000, 2),
        "bsm_per_call_us": round(bsm_time / n_calls * 1e6, 3),
        "mc_50k_paths_ms": round(mc_time * 1000, 2),
        "backend": PRICER_BACKEND,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
