"""
SAC Agent Training Pipeline — with Dynamic Hyperparameter Control

Trains a Soft Actor-Critic agent for options hedging using
stable-baselines3. SAC is ideal for this task because:

1. Continuous action space (hedge ratio in [-1, 1])
2. Off-policy (sample efficient — reuses past experiences)
3. Entropy regularization (explores naturally, avoids local optima)
4. Handles stochastic environments well

Dynamic hyperparameter control adapts the following during training:
  - Learning rate      : cosine annealing with warm restarts
  - Entropy coefficient: backs off when Sharpe plateaus, boosts on regression
  - Gradient steps     : increases when critic loss is stable

Reference: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy
Maximum Entropy Deep RL with a Stochastic Actor"
"""

import os
import sys
import argparse
import numpy as np
import time
import math
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.gpu_utils import get_device, device_banner, patch_sb3_device, \
    recommended_batch_size, recommended_buffer_size

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.options_env import OptionsHedgingEnv
from typing import cast
import glob
import re

# ─────────────────────────────────────────────────────────────────────────────
# Custom PyTorch feature extractor (12-dim obs space)
# ─────────────────────────────────────────────────────────────────────────────

class OptionsFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom neural network feature extractor for the 12-dim options obs space.

    Why a custom extractor instead of SB3's default MLP?
    The 12 features have very different scales and semantics:
      - [0-1] price ratios        : near 1.0, tight range
      - [2]   time fraction       : [0, 1] linear decay
      - [3]   volatility          : [0.05, 0.5]
      - [4]   delta               : [0, 1] sigmoid-shaped
      - [5]   gamma × 100         : peaked near ATM, near zero OTM/ITM
      - [6]   pnl                 : unbounded, clipped at ±10
      - [7]   vega                : positive, peaked near ATM
      - [8]   theta               : negative, grows in magnitude near expiry
      - [9]   vol_carry           : ratio near 1.0 most of the time
      - [10]  hedge_position      : [-1, 1] current state
      - [11]  vol_regime          : 0.0=low / 0.5=medium / 1.0=high

    LayerNorm after the first layer handles the scale differences without
    relying entirely on VecNormalize. The two-tower structure separates
    market state features (0-5) from portfolio state features (6-11),
    then merges them — reflecting how options traders mentally partition
    market vs position information.

    Architecture:
        Market tower   : [0-5]  → Linear(6, 64) → LayerNorm → ReLU
        Portfolio tower: [6-11] → Linear(6, 32) → LayerNorm → ReLU
        Merge          : cat(64, 32) → Linear(96, features_dim) → ReLU
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Market state: price ratios, time, vol, delta, gamma
        self.market_tower = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Portfolio state: pnl, vega, theta, vol_carry, hedge_pos, vol_regime
        self.portfolio_tower = nn.Sequential(
            nn.Linear(6, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # Merge
        self.merge = nn.Sequential(
            nn.Linear(96, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        market    = self.market_tower(obs[:, :6])
        portfolio = self.portfolio_tower(obs[:, 6:])
        return self.merge(torch.cat([market, portfolio], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Gradient norm monitoring callback
# ─────────────────────────────────────────────────────────────────────────────

class GradientMonitorCallback(BaseCallback):
    """
    Logs gradient norms for actor and critic networks to TensorBoard.

    Why this matters: exploding or vanishing gradients are the most common
    cause of SAC instability. Monitoring grad norm lets you catch this early
    without waiting for the Sharpe to collapse.

    Logs every 2000 steps under:
        debug/actor_grad_norm
        debug/critic_grad_norm
        debug/total_grad_norm
    """

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        policy = self.model.policy
        actor_norm  = self._grad_norm(policy.actor.parameters())
        critic_norm = self._grad_norm(policy.critic.parameters())
        total_norm  = (actor_norm**2 + critic_norm**2) ** 0.5

        self.logger.record("debug/actor_grad_norm",  actor_norm)
        self.logger.record("debug/critic_grad_norm", critic_norm)
        self.logger.record("debug/total_grad_norm",  total_norm)
        return True

    @staticmethod
    def _grad_norm(params) -> float:
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Logging callback
# ─────────────────────────────────────────────────────────────────────────────

class SharpeLogCallback(BaseCallback):
    """Logs episode Sharpe ratios to TensorBoard during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_sharpes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_sharpe" in info:
                self.episode_sharpes.append(info["episode_sharpe"])
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                if len(self.episode_sharpes) >= 10:
                    recent = self.episode_sharpes[-100:]
                    self.logger.record("custom/mean_sharpe", np.mean(recent))
                    self.logger.record("custom/std_sharpe", np.std(recent))
                    self.logger.record("custom/episode_count",
                                       len(self.episode_sharpes))
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Hyperparameter Controller
# ─────────────────────────────────────────────────────────────────────────────

class DynamicHPController(BaseCallback):
    """
    Adapts SAC hyperparameters live during training based on rolling
    performance signals. Three independent controllers run in parallel:

    1. Learning Rate — cosine annealing with warm restarts (SGDR).
    2. Entropy Coefficient — responds to Sharpe trend.
    3. Gradient Steps — responds to critic loss stability.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        min_lr: float | None = None,
        lr_cycle_steps: int = 500_000,
        window: int = 50,
        sharpe_patience: int = 200,
        ent_coef_auto: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.base_lr = base_lr
        self.min_lr = min_lr if min_lr is not None else base_lr / 10
        self.lr_cycle_steps = lr_cycle_steps

        self.ent_coef_auto = ent_coef_auto
        self.sharpe_patience = sharpe_patience
        self._ent_coef_current = 0.1
        self._ent_coef_min = 0.02
        self._ent_coef_max = 0.5
        self._ent_boost_factor = 1.3
        self._ent_decay_factor = 0.90

        self._grad_steps_current = 1
        self._grad_steps_min = 1
        self._grad_steps_max = 2

        self.window = window
        self._sharpe_history = deque(maxlen=500)
        self._critic_loss_history = deque(maxlen=200)
        self._best_rolling_sharpe = -np.inf
        self._steps_since_improvement = 0
        self._last_hp_step = 0
        self._hp_update_freq = 5000

    def _cosine_lr(self) -> float:
        t = self.num_timesteps % self.lr_cycle_steps
        cos_val = math.cos(math.pi * t / self.lr_cycle_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cos_val)

    def _rolling_sharpe(self) -> float:
        if len(self._sharpe_history) < 10:
            return 0.0
        recent = list(self._sharpe_history)[-self.window:]
        return float(np.mean(recent))

    def _critic_loss_cv(self) -> float:
        if len(self._critic_loss_history) < 10:
            return 0.0
        arr = np.array(self._critic_loss_history)
        mean = np.mean(arr)
        return float(np.std(arr) / mean) if mean > 1e-10 else 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            sharpe = None
            if "episode_sharpe" in info:
                sharpe = info["episode_sharpe"]
            elif "final_info" in info and info["final_info"] is not None:
                fi = info["final_info"]
                if isinstance(fi, dict) and "episode_sharpe" in fi:
                    sharpe = fi["episode_sharpe"]
            if sharpe is None and "total_pnl" in info:
                sharpe=float(info["total_pnl"])
            if sharpe is not None:
                self._sharpe_history.append(float(sharpe))

        if hasattr(self.model, 'logger'):
            name_map = self.model.logger.name_to_value
            if "train/critic_loss" in name_map:
                self._critic_loss_history.append(name_map["train/critic_loss"])

        if (self.num_timesteps - self._last_hp_step) < self._hp_update_freq:
            return True

        self._last_hp_step = self.num_timesteps
        self._update_lr()
        self._update_entropy()
        self._update_gradient_steps()
        return True

    def _update_lr(self):
        new_lr = self._cosine_lr()
        self.model.learning_rate = new_lr

        policy = self.model.policy
        optimizers = []
        if hasattr(policy, 'actor') and hasattr(policy.actor, 'optimizer'):
            optimizers.append(policy.actor.optimizer)
        if hasattr(policy, 'critic') and hasattr(policy.critic, 'optimizer'):
            optimizers.append(policy.critic.optimizer)
        if hasattr(policy, 'ent_coef_optimizer'):
            optimizers.append(policy.ent_coef_optimizer)

        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = new_lr

        self.logger.record("dynamic_hp/learning_rate", new_lr)

        if self.verbose >= 1:
            cycle_pos = self.num_timesteps % self.lr_cycle_steps
            if cycle_pos < self._hp_update_freq:
                print(f"\n[DHP] LR warm restart at step {self.num_timesteps:,} "
                      f"→ {new_lr:.2e}")

    def _update_entropy(self):
        if self.ent_coef_auto:
            return

        rolling = self._rolling_sharpe()
        if rolling <= 0 or len(self._sharpe_history) < self.window:
            return

        if rolling > self._best_rolling_sharpe + 0.02:
            self._best_rolling_sharpe = rolling
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        old_ent = self._ent_coef_current

        if self._steps_since_improvement >= self.sharpe_patience:
            self._ent_coef_current = min(
                self._ent_coef_current * self._ent_boost_factor,
                self._ent_coef_max
            )
            self._steps_since_improvement = 0
            if self.verbose >= 1:
                print(f"\n[DHP] Sharpe plateau ({rolling:.3f}) → "
                      f"ent_coef {old_ent:.4f} → {self._ent_coef_current:.4f}")
        elif rolling > self._best_rolling_sharpe * 0.95:
            self._ent_coef_current = max(
                self._ent_coef_current * self._ent_decay_factor,
                self._ent_coef_min
            )

        self.model.ent_coef_tensor = torch.tensor(  # type: ignore[attr-defined]
            self._ent_coef_current, device=self.model.device
        )

        self.logger.record("dynamic_hp/ent_coef", self._ent_coef_current)
        self.logger.record("dynamic_hp/rolling_sharpe", rolling)
        self.logger.record("dynamic_hp/steps_since_improvement",
                           self._steps_since_improvement)

    def _update_gradient_steps(self):
        cv = self._critic_loss_cv()
        rolling = self._rolling_sharpe()
        if(rolling>self._best_rolling_sharpe):
            self._best_rolling_sharpe=rolling
        old_gs = self._grad_steps_current

        if cv > 0.5 and self._grad_steps_current > self._grad_steps_min:
            self._grad_steps_current = max(
                self._grad_steps_current - 1, self._grad_steps_min
            )
        elif (cv < 0.2
              and rolling > 0.5
              and self.num_timesteps > 100_000
              and self._grad_steps_current < self._grad_steps_max):
            self._grad_steps_current = min(
                self._grad_steps_current + 1, self._grad_steps_max
            )

        if self._grad_steps_current != old_gs:
            self.model.gradient_steps = self._grad_steps_current  # type: ignore[attr-defined]
            if self.verbose >= 1:
                print(f"\n[DHP] gradient_steps {old_gs} → "
                      f"{self._grad_steps_current} "
                      f"(critic_cv={cv:.3f}, rolling_sharpe={rolling:.3f})")

        self.logger.record("dynamic_hp/gradient_steps", self._grad_steps_current)
        self.logger.record("dynamic_hp/critic_loss_cv", cv)


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_config: dict, seed: int = 0):
    def _init():
        env = OptionsHedgingEnv(**env_config, seed=seed)
        env= Monitor(env)  # Wrap with Monitor for episode logging
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 70)
    print("  SAC Agent Training — Dynamic HP Control")
    print("=" * 70)

    device = get_device(prefer=args.device)
    patch_sb3_device(device)

    if device != "cpu":
        if args.batch_size == 256:
            args.batch_size = recommended_batch_size(device, base=args.batch_size)
            print(f"[GPU] Auto batch_size  → {args.batch_size}")
        if args.buffer_size == 200_000:
            args.buffer_size = recommended_buffer_size(device, base=args.buffer_size)
            print(f"[GPU] Auto buffer_size → {args.buffer_size:,}")

    env_config = {
        "simulator_type": args.simulator,
        "S0": 100.0,
        "K": 100.0,
        "T": 30 / 252,
        "r": 0.05,
        "sigma": 0.2,
        "mu": 0.05,
        "n_steps": 30,
        "transaction_cost": 0.003,
        "execution_delay": 0,
        "variable_tc": False,
    }
    if args.simulator == "heston":
        env_config.update({
            "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
        })

    print(f"\n[ENV] Simulator     : {args.simulator.upper()}")
    print(f"   Steps/episode   : {env_config['n_steps']}")
    print(f"   Total timesteps : {args.total_timesteps:,}")

    n_envs = args.n_envs
    train_envs = DummyVecEnv(
        [make_env(env_config, seed=i) for i in range(n_envs)]
    )
    train_envs = VecNormalize(
        train_envs, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=args.gamma,
    )

    eval_env = DummyVecEnv([make_env(env_config, seed=9999)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=OptionsFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], qf=[128, 64]),
    )
    ent_coef_auto = (args.ent_coef == "auto")
    if hasattr(args, 'policy_kwargs') and args.policy_kwargs:
        policy_kwargs = args.policy_kwargs

    model = SAC(
        "MlpPolicy",
        train_envs,
        verbose=0,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy if ent_coef_auto else "auto",
        learning_starts=args.learning_starts,
        train_freq=(1, "step"),
        gradient_steps=args.gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        device=device,
        seed=args.seed,
    )

    print(f"\n[SAC] Initial configuration:")
    print(f"   device         : {device}")
    print(f"   lr             : {args.lr}")
    print(f"   buffer_size    : {args.buffer_size:,}")
    print(f"   batch_size     : {args.batch_size}")
    print(f"   tau            : {args.tau}")
    print(f"   gamma          : {args.gamma}")
    print(f"   ent_coef       : {args.ent_coef}")
    print(f"   target_entropy : {args.target_entropy}")
    print(f"   n_envs         : {n_envs}")

    print(f"\n[DHP] Dynamic controllers:")
    print(f"   LR     → cosine annealing [{args.lr:.2e} → {args.lr/10:.2e}], "
          f"cycle={args.lr_cycle_steps:,} steps")
    print(f"   Entropy→ {'disabled (SAC auto mode)' if ent_coef_auto else 'enabled — responds to Sharpe plateau'}")
    print(f"   GradSteps → dynamic [1–2] based on critic loss stability")

    sim_tag_early = args.simulator.lower()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / f"best_{sim_tag_early}"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.total_timesteps // 50, 1000),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.total_timesteps // (10*args.n_envs), 1000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="sac_hedger",
        verbose=1,
    )

    sharpe_callback = SharpeLogCallback()

    dhp_callback = DynamicHPController(
        base_lr=args.lr,
        min_lr=args.lr / 10,
        lr_cycle_steps=args.total_timesteps // 2,
        window=50,
        sharpe_patience=200,
        ent_coef_auto=True,
        verbose=1,
    )
    checkpoints= sorted(glob.glob(str(model_dir / "checkpoints"/"*.zip")))
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        print(f"\n[RESUME] Found checkpoint {latest}, resuming training...")
        model = SAC.load(latest, env=train_envs, device=device)
        print(f"[RESUME]Loaded model from {latest}, resuming training...")

    print(f"\n[START] Starting training...\n")
    start_time = time.time()

    gradient_monitor = GradientMonitorCallback(log_freq=2000)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[
            eval_callback,
            checkpoint_callback,
            sharpe_callback,
            dhp_callback,
            gradient_monitor,
        ],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    sim_tag    = args.simulator.lower()
    final_name = f"sac_hedger_{sim_tag}_final"
    final_path = str(model_dir / final_name)
    vnorm_name = f"vec_normalize_{sim_tag}.pkl"
    vnorm_path = str(model_dir / vnorm_name)

    model.save(final_path)
    train_envs.save(vnorm_path)
    model.save(str(model_dir / "sac_hedger_final"))
    train_envs.save(str(model_dir / "vec_normalize.pkl"))

    print(f"\n[SAVE] Model      : {final_path}.zip")
    print(f"       Best model  : {model_dir}/best_{sim_tag}/best_model.zip")
    print(f"       VecNormalize: {vnorm_path}")

    print(f"\n[EVAL] Quick evaluation (100 episodes)...")
    eval_rewards = []
    eval_pnls = []

    test_env = DummyVecEnv([make_env(env_config, seed=42000)])
    test_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), test_env)
    test_env.training = False
    test_env.norm_reward = False

    for ep in range(100):
        test_env.seed(42000 + ep)
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
        pnl_std = np.std(pnl_arr)
        if pnl_std > 1e-6:
            sharpe = np.mean(pnl_arr) / pnl_std * np.sqrt(252)
            print(f"   Sharpe ratio: {sharpe:.4f}")
        else:
            print("   [WARN] PnL std near zero — check episode seeding")

    print(f"\n[DHP] Final state:")
    print(f"   LR at end          : {dhp_callback._cosine_lr():.2e}")
    print(f"   Final grad_steps   : {dhp_callback._grad_steps_current}")
    if not ent_coef_auto:
        print(f"   Final ent_coef     : {dhp_callback._ent_coef_current:.4f}")
    print(f"   Best rolling Sharpe: {dhp_callback._best_rolling_sharpe:.4f}")

    print("\n" + "=" * 70)
    print("  Training complete. Run evaluate.py for full analysis.")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC hedging agent with dynamic HP control"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--total-timesteps", type=int, default=2000000)      # was 1000000
    parser.add_argument("--simulator", type=str, default="heston",             # was heston
                        choices=["gbm", "heston", "jump"])
    parser.add_argument("--lr", type=float, default=1e-4)                    # was 3e-4
    parser.add_argument("--lr-cycle-steps", type=int, default=500_000)       # was 100_000
    parser.add_argument("--buffer-size", type=int, default=500_000)          # was 200_000
    parser.add_argument("--batch-size", type=int, default=256)               # was 256
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)                # was 0.99
    parser.add_argument("--ent-coef", type=str, default="0.15")              # was auto
    parser.add_argument("--learning-starts", type=int, default=10_000)         # was 1000
    parser.add_argument("--n-envs", type=int, default=6)                     # was 4
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="agent/models")
    parser.add_argument("--log-dir", type=str, default="tb_logs")
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-entropy", type=float, default=-0.5)
    args = parser.parse_args()
    train(args)