"""
SAC Agent Training Pipeline

Trains a Soft Actor-Critic agent for options hedging using
stable-baselines3. SAC is ideal for this task because:

1. Continuous action space (hedge ratio in [-1, 1])
2. Off-policy (sample efficient — reuses past experiences)
3. Entropy regularization (explores naturally, avoids local optima)
4. Handles stochastic environments well

Reference: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy
Maximum Entropy Deep RL with a Stochastic Actor"
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.logger import configure

from environment.options_env import OptionsHedgingEnv


class SharpeLogCallback(BaseCallback):
    """
    Custom callback that logs episode Sharpe ratios during training.
    Also tracks running statistics for monitoring convergence.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_sharpes = []

    def _on_step(self) -> bool:
        # Check for episode completion in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_sharpe" in info:
                self.episode_sharpes.append(info["episode_sharpe"])

            # Log to tensorboard
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)

                if len(self.episode_sharpes) >= 10:
                    recent = self.episode_sharpes[-100:]
                    self.logger.record("custom/mean_sharpe", np.mean(recent))
                    self.logger.record("custom/std_sharpe", np.std(recent))
                    self.logger.record("custom/episode_count",
                                       len(self.episode_sharpes))

        return True


def make_env(env_config: dict, seed: int = 0):
    """Factory function for creating environments."""
    def _init():
        env = OptionsHedgingEnv(**env_config, seed=seed)
        return env
    return _init


def train(args):
    """Main training loop."""
    print("=" * 70)
    print("  SAC Agent Training - Options Hedging Engine")
    print("=" * 70)

    # ─── Environment Configuration ────────────────────────────────────────
    env_config = {
        "simulator_type": args.simulator,
        "S0": 100.0,
        "K": 100.0,
        "T": 30 / 252,
        "r": 0.05,
        "sigma": 0.2,
        "mu": 0.05,
        "n_steps": 30,
        "transaction_cost": 0.001,
    }

    if args.simulator == "heston":
        env_config.update({
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
        })

    print(f"\n[ENV] Environment: {args.simulator.upper()}")
    print(f"   Steps per episode: {env_config['n_steps']}")
    print(f"   Total timesteps: {args.total_timesteps:,}")

    # ─── Create Vectorized Environment ────────────────────────────────────
    n_envs = args.n_envs
    train_envs = DummyVecEnv([make_env(env_config, seed=i) for i in range(n_envs)])
    train_envs = VecNormalize(
        train_envs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # Evaluation environment (separate, unnormalized rewards for true metrics)
    eval_env = DummyVecEnv([make_env(env_config, seed=9999)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize eval rewards
        clip_obs=10.0,
        training=False,
    )

    # ─── Model Configuration ─────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAC] Configuration:")
    print(f"   Learning rate: {args.lr}")
    print(f"   Buffer size: {args.buffer_size:,}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Tau: {args.tau}")
    print(f"   Gamma: {args.gamma}")
    print(f"   Entropy: {args.ent_coef}")
    print(f"   N envs: {n_envs}")

    # Custom policy network
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),  # Actor & critic networks
    )

    model = SAC(
        "MlpPolicy",
        train_envs,
        verbose=1,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        learning_starts=args.learning_starts,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
    )

    # ─── Callbacks ────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.total_timesteps // 50, 1000),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.total_timesteps // 10, 5000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="sac_hedger",
        verbose=1,
    )

    sharpe_callback = SharpeLogCallback()

    # ─── Training ─────────────────────────────────────────────────────────
    print(f"\n[START] Starting training...")
    start_time = time.time()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback, sharpe_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ─── Save Final Model ─────────────────────────────────────────────────
    final_path = str(model_dir / "sac_hedger_final")
    model.save(final_path)
    train_envs.save(str(model_dir / "vec_normalize.pkl"))

    print(f"\n[SAVE] Model saved to: {final_path}.zip")
    print(f"   VecNormalize saved to: {model_dir}/vec_normalize.pkl")

    # ─── Quick Evaluation ─────────────────────────────────────────────────
    print(f"\n[EVAL] Quick evaluation (100 episodes)...")
    eval_rewards = []
    eval_pnls = []

    test_env = DummyVecEnv([make_env(env_config, seed=42000)])
    test_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), test_env)
    test_env.training = False
    test_env.norm_reward = False

    for ep in range(100):
        obs = test_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = test_env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]

        eval_rewards.append(ep_reward)
        if "total_pnl" in infos[0]:
            eval_pnls.append(infos[0]["total_pnl"])

    mean_rew = np.mean(eval_rewards)
    std_rew = np.std(eval_rewards)
    print(f"   Mean reward: {mean_rew:.4f} ± {std_rew:.4f}")

    if eval_pnls:
        pnl_arr = np.array(eval_pnls)
        sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-10) * np.sqrt(252)
        print(f"   Sharpe ratio: {sharpe:.4f}")

    print("\n" + "=" * 70)
    print("  Training pipeline complete. Run evaluate.py for full analysis.")
    print("=" * 70)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC hedging agent")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--simulator", type=str, default="gbm",
                        choices=["gbm", "heston"],
                        help="Market simulator type")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=200_000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="Entropy coefficient")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Steps before training begins")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model-dir", type=str,
                        default="agent/models",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str,
                        default="tb_logs",
                        help="TensorBoard log directory")

    args = parser.parse_args()
    train(args)
