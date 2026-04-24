"""
Environment Unit Tests

Tests for the options hedging Gymnasium environment:
- Reset/step cycle
- Observation space bounds
- Reward computation
- Episode termination
- Baseline agent behavior
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.options_env import OptionsHedgingEnv
from environment.market_sim import GBMSimulator, HestonSimulator, VolRegimeSimulator
from environment.baselines import DeltaHedger, StaticHedger, RandomAgent, run_baseline_episode


class TestMarketSimulators:
    """Market simulator tests."""

    def test_gbm_reset(self):
        sim = GBMSimulator(S0=100, seed=42)
        price = sim.reset()
        assert price == 100.0

    def test_gbm_step(self):
        sim = GBMSimulator(S0=100, seed=42)
        sim.reset()
        price, vol = sim.step()
        assert price > 0  # Price should be positive
        assert vol > 0    # Vol should be positive

    def test_gbm_deterministic(self):
        sim = GBMSimulator(S0=100, seed=42)
        sim.reset()
        p1, _ = sim.step()
        sim.reset(seed=42)
        p2, _ = sim.step()
        assert abs(p1 - p2) < 1e-12

    def test_gbm_path_length(self):
        sim = GBMSimulator(S0=100, seed=42)
        path = sim.generate_path(252)
        assert len(path) == 253  # 252 steps + initial

    def test_heston_reset(self):
        sim = HestonSimulator(S0=100, seed=42)
        price = sim.reset()
        assert price == 100.0

    def test_heston_step(self):
        sim = HestonSimulator(S0=100, seed=42)
        sim.reset()
        price, vol = sim.step()
        assert price > 0
        assert vol >= 0  # Vol can be very small but not negative

    def test_heston_path(self):
        sim = HestonSimulator(S0=100, seed=42)
        prices, vols = sim.generate_path(30)
        assert len(prices) == 31
        assert len(vols) == 31
        assert all(p > 0 for p in prices)

    def test_regime_simulator(self):
        sim = VolRegimeSimulator(S0=100, seed=42)
        sim.reset()
        for _ in range(100):
            price, vol = sim.step()
            assert price > 0
            assert vol > 0


class TestOptionsEnv:
    """Gymnasium environment tests."""

    def test_env_creation(self):
        env = OptionsHedgingEnv(seed=42)
        assert env.observation_space.shape == (7,)
        assert env.action_space.shape == (1,)

    def test_env_reset(self):
        env = OptionsHedgingEnv(seed=42)
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert isinstance(info, dict)

    def test_observation_in_bounds(self):
        env = OptionsHedgingEnv(seed=42)
        obs, _ = env.reset()

        # Check observation space bounds (with some tolerance)
        assert env.observation_space.contains(obs), \
            f"Obs out of bounds: {obs}"

    def test_step_returns(self):
        env = OptionsHedgingEnv(seed=42)
        obs, _ = env.reset()
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_full_episode(self):
        """Run a complete 30-step episode."""
        env = OptionsHedgingEnv(n_steps=30, seed=42)
        obs, _ = env.reset()

        done = False
        step_count = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        assert step_count <= 30  # Should terminate at or before 30 steps

    def test_episode_deterministic(self):
        """Same seed should give same trajectory."""
        env = OptionsHedgingEnv(seed=42)

        obs1, _ = env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs1_next, r1, _, _, _ = env.step(action)

        obs2, _ = env.reset(seed=42)
        obs2_next, r2, _, _, _ = env.step(action)

        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_action_clipping(self):
        """Actions outside [-1, 1] should be clipped."""
        env = OptionsHedgingEnv(seed=42)
        env.reset()

        # Large action should be clipped
        obs, _, _, _, _ = env.step(np.array([5.0], dtype=np.float32))
        assert obs.shape == (7,)  # Should not crash

    def test_heston_env(self):
        """Environment works with Heston simulator."""
        env = OptionsHedgingEnv(simulator_type="heston", seed=42)
        obs, _ = env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break


class TestBaselines:
    """Baseline agent tests."""

    def test_delta_hedger(self):
        """DeltaHedger should return delta as action."""
        agent = DeltaHedger()
        obs = np.array([1.0, 1.0, 1.0, 0.2, 0.55, 0.02, 0.0], dtype=np.float32)
        action, _ = agent.predict(obs)

        assert action.shape == (1,)
        assert abs(action[0] - 0.55) < 0.01  # Should match delta

    def test_static_hedger(self):
        """StaticHedger should lock initial delta."""
        agent = StaticHedger()

        obs1 = np.array([1.0, 1.0, 1.0, 0.2, 0.55, 0.02, 0.0], dtype=np.float32)
        a1, _ = agent.predict(obs1)

        obs2 = np.array([1.0, 1.0, 0.5, 0.3, 0.70, 0.01, 0.1], dtype=np.float32)
        a2, _ = agent.predict(obs2)

        assert abs(a1[0] - a2[0]) < 1e-6  # Should be same action

    def test_random_agent(self):
        """RandomAgent returns actions in [-1, 1]."""
        agent = RandomAgent(seed=42)
        obs = np.zeros(7, dtype=np.float32)

        for _ in range(100):
            action, _ = agent.predict(obs)
            assert -1.0 <= action[0] <= 1.0

    def test_baseline_episode_runs(self):
        """Full episode with each baseline should complete."""
        env = OptionsHedgingEnv(seed=42)

        for agent in [DeltaHedger(), StaticHedger(), RandomAgent(42)]:
            result = run_baseline_episode(env, agent, seed=42)
            assert "total_reward" in result
            assert "portfolio_value" in result

    def test_delta_hedger_beats_random(self):
        """DeltaHedger should outperform RandomAgent on average."""
        env = OptionsHedgingEnv(seed=42)

        delta_rewards = []
        random_rewards = []

        for seed in range(50):
            dr = run_baseline_episode(env, DeltaHedger(), seed=seed)
            rr = run_baseline_episode(env, RandomAgent(seed), seed=seed)
            delta_rewards.append(dr["total_reward"])
            random_rewards.append(rr["total_reward"])

        assert np.mean(delta_rewards) > np.mean(random_rewards), \
            f"Delta {np.mean(delta_rewards):.4f} should beat Random {np.mean(random_rewards):.4f}"


class TestEnvChecker:
    """Use Gymnasium's built-in env checker."""

    def test_gymnasium_check_env(self):
        """Environment passes Gymnasium's strict compliance check."""
        from gymnasium.utils.env_checker import check_env
        env = OptionsHedgingEnv(seed=42)
        # This will raise if env doesn't comply
        check_env(env, skip_render_check=True)
