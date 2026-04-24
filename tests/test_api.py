"""
FastAPI Endpoint Tests

Tests API response correctness and latency targets.
"""

import os
import sys
import pytest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "pricing_backend" in data


class TestPriceEndpoint:
    def test_bs_call(self):
        resp = client.post("/price", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "method": "bs", "flag": "call"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["price"] - 10.4506) < 0.01
        assert data["method"] == "black-scholes"

    def test_bs_put(self):
        resp = client.post("/price", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "method": "bs", "flag": "put"
        })
        assert resp.status_code == 200
        assert resp.json()["price"] > 0

    def test_mc_call(self):
        resp = client.post("/price", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "method": "mc", "n_paths": 10000, "flag": "call"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "monte-carlo"
        assert abs(data["price"] - 10.4506) < 1.0  # Wider tolerance for MC

    def test_bsm_latency(self):
        """BSM pricing should be < 5ms."""
        start = time.perf_counter()
        resp = client.post("/price", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "method": "bs", "flag": "call"
        })
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        # Note: latency includes HTTP overhead in test client
        # The actual compute time is in response
        assert resp.json()["latency_ms"] < 5.0

    def test_invalid_params(self):
        resp = client.post("/price", json={
            "S": -100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "method": "bs", "flag": "call"
        })
        assert resp.status_code == 422  # Validation error


class TestGreeksEndpoint:
    def test_greeks_call(self):
        resp = client.post("/greeks", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "flag": "call"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0 < data["delta"] < 1
        assert data["gamma"] > 0
        assert data["vega"] > 0
        assert data["theta"] < 0

    def test_greeks_put(self):
        resp = client.post("/greeks", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.2, "flag": "put"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert -1 < data["delta"] < 0


class TestIVEndpoint:
    def test_iv_round_trip(self):
        # First get a BS price
        price_resp = client.post("/price", json={
            "S": 100, "K": 100, "T": 1.0, "r": 0.05,
            "sigma": 0.25, "method": "bs", "flag": "call"
        })
        bs_price = price_resp.json()["price"]

        # Now solve for IV
        iv_resp = client.post("/iv", json={
            "market_price": bs_price,
            "S": 100, "K": 100, "T": 1.0, "r": 0.05, "flag": "call"
        })
        assert iv_resp.status_code == 200
        assert abs(iv_resp.json()["implied_vol"] - 0.25) < 1e-4


class TestAgentEndpoint:
    def test_agent_action(self):
        resp = client.post("/agent/action", json={
            "observation": [1.0, 1.0, 1.0, 0.2, 0.55, 0.02, 0.0]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert -1.0 <= data["action"] <= 1.0
        assert "agent_type" in data


class TestBenchmarkEndpoint:
    def test_benchmark(self):
        resp = client.get("/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert "bsm_100k_calls_ms" in data
        assert "backend" in data
