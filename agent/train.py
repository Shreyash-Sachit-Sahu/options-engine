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

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)

from environment.options_env import OptionsHedgingEnv  # ← was missing


# ─────────────────────────────────────────────────────────────────────────────
# Logging callback (unchanged from original)
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
       Gradually decays lr from base → min, then restarts. Helps escape
       local optima and fine-tunes in later training.

    2. Entropy Coefficient — responds to Sharpe trend.
       Plateau/regression → boost ent_coef to explore more.
       Steady improvement → reduce ent_coef to exploit.
       Only fires when ent_coef is NOT "auto" (auto manages itself).

    3. Gradient Steps — responds to critic loss stability.
       High critic loss variance → reduce gradient_steps (stabilise).
       Low critic loss variance + decent Sharpe → increase gradient_steps
       (extract more signal per batch of experience).

    All changes are logged to TensorBoard under "dynamic_hp/".

    Args:
        base_lr          : Initial learning rate (from model config)
        min_lr           : Floor for cosine annealing (default: base_lr / 10)
        lr_cycle_steps   : Steps per cosine cycle
        window           : Episode window for rolling Sharpe
        sharpe_patience  : Episodes without improvement before entropy boost
        ent_coef_auto    : Whether ent_coef="auto" (skip manual ent control)
        verbose          : Print HP changes to stdout
    """

    def __init__(
        self,
        base_lr: float = 3e-4,
        min_lr: float = None,
        lr_cycle_steps: int = 100_000,
        window: int = 50,
        sharpe_patience: int = 200,
        ent_coef_auto: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        # ── LR schedule ───────────────────────────────────────────────────
        self.base_lr = base_lr
        self.min_lr = min_lr if min_lr is not None else base_lr / 10
        self.lr_cycle_steps = lr_cycle_steps

        # ── Entropy controller ────────────────────────────────────────────
        self.ent_coef_auto = ent_coef_auto
        self.sharpe_patience = sharpe_patience
        self._ent_coef_current = 0.1
        self._ent_coef_min = 0.001
        self._ent_coef_max = 0.8
        self._ent_boost_factor = 1.5
        self._ent_decay_factor = 0.85

        # ── Gradient steps controller ─────────────────────────────────────
        self._grad_steps_current = 1
        self._grad_steps_min = 1
        self._grad_steps_max = 4

        # ── Rolling metrics ───────────────────────────────────────────────
        self.window = window
        self._sharpe_history = deque(maxlen=500)
        self._critic_loss_history = deque(maxlen=200)
        self._best_rolling_sharpe = -np.inf
        self._steps_since_improvement = 0
        self._last_hp_step = 0
        self._hp_update_freq = 2000     # evaluate every N timesteps

    # ── Helpers ───────────────────────────────────────────────────────────

    def _cosine_lr(self) -> float:
        """Cosine annealing with warm restarts (SGDR)."""
        t = self.num_timesteps % self.lr_cycle_steps
        cos_val = math.cos(math.pi * t / self.lr_cycle_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cos_val)

    def _rolling_sharpe(self) -> float:
        if len(self._sharpe_history) < 10:
            return 0.0
        recent = list(self._sharpe_history)[-self.window:]
        return float(np.mean(recent))

    def _critic_loss_cv(self) -> float:
        """Coefficient of variation of recent critic losses (std / mean)."""
        if len(self._critic_loss_history) < 10:
            return 0.0
        arr = np.array(self._critic_loss_history)
        mean = np.mean(arr)
        return float(np.std(arr) / mean) if mean > 1e-10 else 0.0

    # ── SB3 hook ──────────────────────────────────────────────────────────

    def _on_step(self) -> bool:

        # Collect episode Sharpe
        for info in self.locals.get("infos", []):
            if "episode_sharpe" in info:
                self._sharpe_history.append(info["episode_sharpe"])

        # Collect critic loss from SB3 logger
        if hasattr(self.model, 'logger'):
            name_map = self.model.logger.name_to_value
            if "train/critic_loss" in name_map:
                self._critic_loss_history.append(
                    name_map["train/critic_loss"]
                )

        # Only run HP updates every _hp_update_freq steps
        if (self.num_timesteps - self._last_hp_step) < self._hp_update_freq:
            return True

        self._last_hp_step = self.num_timesteps
        self._update_lr()
        self._update_entropy()
        self._update_gradient_steps()

        return True

    # ── Controller 1: Learning rate ───────────────────────────────────────

    def _update_lr(self):
        new_lr = self._cosine_lr()

        # Update learning rate in model and optimizer directly
        self.model.learning_rate = new_lr
        # Update actor optimizer
        for param_group in self.model.actor.optimizer.param_groups:
            param_group["lr"] = new_lr

        # Update critic optimizer
        for param_group in self.model.critic.optimizer.param_groups:
            param_group["lr"] = new_lr

        # Update entropy optimizer (if using auto)
        if hasattr(self.model, "ent_coef_optimizer") and self.model.ent_coef_optimizer is not None:
            for param_group in self.model.ent_coef_optimizer.param_groups:
                param_group["lr"] = new_lr
        self.logger.record("dynamic_hp/learning_rate", new_lr)

        # Announce warm restarts
        if self.verbose >= 1:
            cycle_pos = self.num_timesteps % self.lr_cycle_steps
            if cycle_pos < self._hp_update_freq:
                print(f"\n[DHP] LR warm restart at step {self.num_timesteps:,} "
                      f"→ {new_lr:.2e}")

    # ── Controller 2: Entropy coefficient ────────────────────────────────

    def _update_entropy(self):
        # SAC's auto mode manages ent_coef via its own dual gradient descent
        # — don't interfere
        if self.ent_coef_auto:
            return

        rolling = self._rolling_sharpe()
        if rolling <= 0 or len(self._sharpe_history) < self.window:
            return

        # Track improvement
        if rolling > self._best_rolling_sharpe + 0.02:
            self._best_rolling_sharpe = rolling
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        old_ent = self._ent_coef_current

        if self._steps_since_improvement >= self.sharpe_patience:
            # Plateau — explore more
            self._ent_coef_current = min(
                self._ent_coef_current * self._ent_boost_factor,
                self._ent_coef_max
            )
            self._steps_since_improvement = 0
            if self.verbose >= 1:
                print(f"\n[DHP] Sharpe plateau ({rolling:.3f}) → "
                      f"ent_coef {old_ent:.4f} → {self._ent_coef_current:.4f}")

        elif rolling > self._best_rolling_sharpe * 0.95:
            # Improving — exploit more
            self._ent_coef_current = max(
                self._ent_coef_current * self._ent_decay_factor,
                self._ent_coef_min
            )

        # Apply to model tensor
        import torch
        self.model.ent_coef_tensor = torch.tensor(
            self._ent_coef_current, device=self.model.device
        )

        self.logger.record("dynamic_hp/ent_coef", self._ent_coef_current)
        self.logger.record("dynamic_hp/rolling_sharpe", rolling)
        self.logger.record("dynamic_hp/steps_since_improvement",
                           self._steps_since_improvement)

    # ── Controller 3: Gradient steps ─────────────────────────────────────

    def _update_gradient_steps(self):
        cv = self._critic_loss_cv()
        rolling = self._rolling_sharpe()
        old_gs = self._grad_steps_current

        if cv > 0.5 and self._grad_steps_current > self._grad_steps_min:
            # Critic unstable — reduce gradient pressure
            self._grad_steps_current = max(
                self._grad_steps_current - 1, self._grad_steps_min
            )
        elif (cv < 0.2
              and rolling > 0.5
              and self._grad_steps_current < self._grad_steps_max):
            # Critic stable + decent performance — extract more signal
            self._grad_steps_current = min(
                self._grad_steps_current + 1, self._grad_steps_max
            )

        if self._grad_steps_current != old_gs:
            self.model.gradient_steps = self._grad_steps_current
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
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 70)
    print("  SAC Agent Training — Dynamic HP Control")
    print("=" * 70)

    # ─── Environment config ───────────────────────────────────────────────
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
            "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
        })

    print(f"\n[ENV] Simulator     : {args.simulator.upper()}")
    print(f"   Steps/episode   : {env_config['n_steps']}")
    print(f"   Total timesteps : {args.total_timesteps:,}")

    # ─── Vectorized envs ──────────────────────────────────────────────────
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

    # ─── Directories ─────────────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ─── Model ────────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    ent_coef_auto = (args.ent_coef == "auto")

    model = SAC(
        "MlpPolicy",
        train_envs,
        verbose=0,                  # DHP callback handles printing
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        learning_starts=args.learning_starts,
        train_freq=(1, "step"),
        gradient_steps=1,           # DHP adjusts this dynamically
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
    )

    print(f"\n[SAC] Initial configuration:")
    print(f"   lr             : {args.lr}")
    print(f"   buffer_size    : {args.buffer_size:,}")
    print(f"   batch_size     : {args.batch_size}")
    print(f"   tau            : {args.tau}")
    print(f"   gamma          : {args.gamma}")
    print(f"   ent_coef       : {args.ent_coef}")
    print(f"   n_envs         : {n_envs}")

    print(f"\n[DHP] Dynamic controllers:")
    print(f"   LR     → cosine annealing [{args.lr:.2e} → {args.lr/10:.2e}], "
          f"cycle={args.lr_cycle_steps:,} steps")
    print(f"   Entropy→ {'disabled (SAC auto mode)' if ent_coef_auto else 'enabled — responds to Sharpe plateau'}")
    print(f"   GradSteps → dynamic [1–4] based on critic loss stability")

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

    dhp_callback = DynamicHPController(
        base_lr=args.lr,
        min_lr=args.lr / 10,
        lr_cycle_steps=args.lr_cycle_steps,
        window=50,
        sharpe_patience=200,
        ent_coef_auto=ent_coef_auto,
        verbose=1,
    )

    # ─── Train ────────────────────────────────────────────────────────────
    print(f"\n[START] Starting training...\n")
    start_time = time.time()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[
            eval_callback,
            checkpoint_callback,
            sharpe_callback,
            dhp_callback,
        ],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ─── Save ─────────────────────────────────────────────────────────────
    final_path = str(model_dir / "sac_hedger_final")
    model.save(final_path)
    train_envs.save(str(model_dir / "vec_normalize.pkl"))
    print(f"\n[SAVE] Model      : {final_path}.zip")
    print(f"       VecNormalize: {model_dir}/vec_normalize.pkl")

    # ─── Quick Evaluation (fixed seeding) ─────────────────────────────────
    print(f"\n[EVAL] Quick evaluation (100 episodes)...")
    eval_rewards = []
    eval_pnls = []

    test_env = DummyVecEnv([make_env(env_config, seed=42000)])
    test_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), test_env)
    test_env.training = False
    test_env.norm_reward = False

    for ep in range(100):
        obs = test_env.reset(seed=42000 + ep)   # unique seed per episode
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

    # ─── DHP summary ──────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC hedging agent with dynamic HP control"
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--simulator", type=str, default="gbm",
                        choices=["gbm", "heston"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-cycle-steps", type=int, default=100_000,
                        help="Steps per cosine annealing LR cycle")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="'auto' lets SAC manage entropy; a float "
                             "enables the dynamic entropy controller")
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="agent/models")
    parser.add_argument("--log-dir", type=str, default="tb_logs")

    args = parser.parse_args()
    train(args)