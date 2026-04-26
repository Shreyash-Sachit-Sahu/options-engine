"""
Hyperparameter Tuning — Optuna + SAC

Automated search over SAC hyperparameters using Bayesian optimisation
(TPE sampler). Each trial trains for a short budget and reports mean
Sharpe as the objective. Bad trials are pruned early via MedianPruner.

Results are persisted to SQLite so you can resume interrupted studies,
inspect with optuna-dashboard, and re-run best params directly.

Usage
-----
# Basic run (50 trials, GBM)
python agent/tune.py

# Heston simulator, more trials
python agent/tune.py --simulator heston --n-trials 100

# Resume a previous study
python agent/tune.py --study-name sac_hedger_gbm

# After tuning, auto-launch full training with best params
python agent/tune.py --train-after

# Inspect results live
optuna-dashboard sqlite:///agent/tuning/study.db
"""

import os
import sys
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from environment.options_env import OptionsHedgingEnv

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Tuning-safe environment
# ─────────────────────────────────────────────────────────────────────────────

class TuningEnv(OptionsHedgingEnv):
    """
    Thin wrapper that disables early termination during tuning.

    Without this, an untrained model at the start of a trial crashes
    the portfolio in 1 step → episode_sharpe / total_pnl never set
    in info → evaluate_trial sees empty pnls list → returns -999.0.

    During tuning we want full 30-step episodes always so the proxy
    Sharpe signal is meaningful.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Allow natural expiry termination but disable catastrophic-loss
        # early exit — untrained agents would always trigger this
        if terminated and self.current_step < self.n_steps:
            # Undo the early termination (keep going)
            terminated = False
            reward = max(reward, -2.0)  # still penalise but don't abort

        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Pruning callback
# ─────────────────────────────────────────────────────────────────────────────

class PruningCallback(BaseCallback):
    """
    Reports rolling Sharpe to Optuna at regular intervals.
    If the trial is clearly worse than the median, Optuna prunes it early.
    """

    def __init__(self, trial: optuna.Trial, report_freq: int = 5_000):
        super().__init__()
        self.trial = trial
        self.report_freq = report_freq
        self._pnls = deque(maxlen=200)
        self._last_report = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Collect total_pnl at episode end
            if "total_pnl" in info and "episode" in info:
                self._pnls.append(info["total_pnl"])

        if (self.num_timesteps - self._last_report >= self.report_freq
                and len(self._pnls) >= 10):

            arr = np.array(self._pnls)
            std = np.std(arr)
            sharpe = float(np.mean(arr) / (std + 1e-10) * np.sqrt(252))

            self.trial.report(sharpe, step=self.num_timesteps)
            self._last_report = self.num_timesteps

            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_config: dict, seed: int = 0, tuning: bool = False):
    """
    Factory for creating environments.
    Uses TuningEnv (no early termination) when tuning=True.
    """
    def _init():
        cls = TuningEnv if tuning else OptionsHedgingEnv
        return cls(**env_config, seed=seed)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Post-trial evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_trial(model: SAC, train_env: VecNormalize,
                   env_config: dict, n_episodes: int = 40) -> float:
    """
    Run deterministic episodes and return mean Sharpe.
    Uses TuningEnv so every episode runs the full 30 steps.
    Syncs VecNormalize obs_rms from training env.
    """
    eval_env = DummyVecEnv(
        [make_env(env_config, seed=90000 + i, tuning=True)
         for i in range(1)]
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False
    )
    # Sync normalisation stats — critical for fair comparison across trials
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    pnls = []
    for ep in range(n_episodes):
        # VecNormalize doesn't accept seed in reset() — set it separately
        eval_env.seed(90000 + ep)
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, infos = eval_env.step(action)
            done = done_arr[0]

        if "total_pnl" in infos[0]:
            pnls.append(infos[0]["total_pnl"])

    eval_env.close()

    if len(pnls) < 5:
        # Still got < 5 episodes with data — something is wrong; return bad score
        print(f"   [WARN] Only {len(pnls)} episodes returned total_pnl")
        return -50.0    # bad but not -999, so pruner can compare

    arr = np.array(pnls)
    std = np.std(arr)
    return float(np.mean(arr) / (std + 1e-10) * np.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(base_env_config: dict, n_train_steps: int, n_eval_episodes: int):

    def objective(trial: optuna.Trial) -> float:

        # ── Search space ─────────────────────────────────────────────────
        lr            = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        buffer_size   = trial.suggest_categorical("buffer_size",
                                                   [50_000, 100_000, 200_000])
        tau           = trial.suggest_float("tau", 0.001, 0.02, log=True)
        gamma         = trial.suggest_float("gamma", 0.95, 0.9999)
        net_width     = trial.suggest_categorical("net_width", [64, 128, 256])
        net_depth     = trial.suggest_int("net_depth", 1, 3)
        ent_coef      = trial.suggest_categorical("ent_coef",
                                                   ["auto", 0.01, 0.05, 0.1, 0.5])
        learning_starts = trial.suggest_categorical("learning_starts",
                                                     [500, 1000, 2000])

        arch = [net_width] * net_depth
        policy_kwargs = dict(net_arch=dict(pi=arch, qf=arch))

        # ── Training envs (TuningEnv — no early termination) ─────────────
        train_env = DummyVecEnv(
            [make_env(base_env_config, seed=i, tuning=True) for i in range(2)]
        )
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0, gamma=gamma
        )

        # ── Model ─────────────────────────────────────────────────────────
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_starts=learning_starts,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=trial.number,
        )

        # ── Train with pruning callback ───────────────────────────────────
        pruning_cb = PruningCallback(
            trial, report_freq=max(n_train_steps // 20, 2_000)
        )

        try:
            model.learn(
                total_timesteps=n_train_steps,
                callback=pruning_cb,
                progress_bar=False,
            )
        except optuna.TrialPruned:
            train_env.close()
            raise

        # ── Final evaluation ──────────────────────────────────────────────
        sharpe = evaluate_trial(
            model, train_env, base_env_config, n_episodes=n_eval_episodes
        )

        train_env.close()
        return sharpe

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_tuning(args):
    print("=" * 70)
    print("  SAC Hyperparameter Tuning — Optuna (TPE + MedianPruner)")
    print("=" * 70)

    base_env_config = {
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
        base_env_config.update({
            "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7
        })

    tuning_dir = Path(args.tuning_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    db_path = tuning_dir / "study.db"
    storage = f"sqlite:///{db_path}"
    study_name = args.study_name or f"sac_hedger_{args.simulator}"

    print(f"\n[CONFIG]")
    print(f"  Simulator       : {args.simulator.upper()}")
    print(f"  Trials          : {args.n_trials}")
    print(f"  Steps/trial     : {args.n_steps_per_trial:,}  "
          f"(recommended >= 100,000)")
    print(f"  Eval episodes   : {args.n_eval_episodes}")
    print(f"  Early term      : DISABLED during tuning (TuningEnv)")
    print(f"  Study name      : {study_name}")
    print(f"  Storage         : {db_path}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,
            multivariate=True,
        ),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1,
        ),
    )

    n_existing = len(study.trials)
    if n_existing:
        print(f"\n[RESUME] Found {n_existing} existing trials. Continuing...")

    objective = make_objective(
        base_env_config,
        n_train_steps=args.n_steps_per_trial,
        n_eval_episodes=args.n_eval_episodes,
    )

    print(f"\n[START] Running {args.n_trials} trials...\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,
        catch=(Exception,),
    )

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TUNING RESULTS")
    print("=" * 70)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n  Completed trials : {len(completed)}")
    print(f"  Pruned trials    : {len(pruned)}")

    if not completed:
        print("\n[WARN] No completed trials. Try increasing --n-steps-per-trial.")
        return

    best = study.best_trial
    print(f"\n  Best trial #{best.number}")
    print(f"  Sharpe (proxy)   : {best.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<20} {v}")

    # Top-5
    print(f"\n  Top 5 trials:")
    print(f"  {'#':<5} {'Sharpe':>8}  {'lr':>10}  {'batch':>6}  "
          f"{'buf':>8}  {'arch':>10}  {'gamma':>8}  {'ent_coef':>10}")
    print("  " + "-" * 70)
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    for t in top5:
        p = t.params
        arch_str = f"{p['net_width']}x{p['net_depth']}"
        print(f"  {t.number:<5} {t.value:>8.4f}  {p['lr']:>10.2e}  "
              f"{p['batch_size']:>6}  {p['buffer_size']:>8,}  "
              f"{arch_str:>10}  {p['gamma']:>8.4f}  {str(p['ent_coef']):>10}")

    # Retrain command
    bp = best.params
    arch = [bp['net_width']] * bp['net_depth']
    print(f"\n  ─── Retrain command ──────────────────────────────────────────")
    print(f"  python agent/train.py \\")
    print(f"    --simulator {args.simulator} \\")
    print(f"    --lr {bp['lr']:.2e} \\")
    print(f"    --batch-size {bp['batch_size']} \\")
    print(f"    --buffer-size {bp['buffer_size']} \\")
    print(f"    --tau {bp['tau']:.4f} \\")
    print(f"    --gamma {bp['gamma']:.6f} \\")
    print(f"    --ent-coef {bp['ent_coef']} \\")
    print(f"    --learning-starts {bp['learning_starts']} \\")
    print(f"    --total-timesteps 500000")
    print(f"  # Network: pi={arch}, qf={arch}  (edit train.py policy_kwargs)")

    # Save JSON
    import json
    results_path = tuning_dir / f"best_params_{study_name}.json"
    with open(results_path, "w") as f:
        json.dump({
            "study_name": study_name,
            "best_trial": best.number,
            "best_sharpe_proxy": best.value,
            "params": best.params,
            "arch": arch,
        }, f, indent=2)
    print(f"\n  Best params saved to: {results_path}")

    if args.train_after:
        print(f"\n[AUTO-TRAIN] Launching full train with best params...")
        _launch_full_train(args, best.params)

    print("\n" + "=" * 70)
    return study


def _launch_full_train(args, params):
    import importlib.util, types

    train_path = Path(__file__).parent / "train.py"
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    train_args = types.SimpleNamespace(
        total_timesteps=500_000,
        simulator=args.simulator,
        lr=params["lr"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        tau=params["tau"],
        gamma=params["gamma"],
        ent_coef=str(params["ent_coef"]),
        learning_starts=params["learning_starts"],
        n_envs=4,
        seed=42,
        model_dir="agent/models",
        log_dir="tb_logs",
        lr_cycle_steps=100_000,
    )

    arch = [params["net_width"]] * params["net_depth"]
    _orig_sac = train_module.SAC

    class PatchedSAC(_orig_sac):
        def __init__(self, policy, env, **kwargs):
            kwargs["policy_kwargs"] = dict(net_arch=dict(pi=arch, qf=arch))
            super().__init__(policy, env, **kwargs)

    train_module.SAC = PatchedSAC
    train_module.train(train_args)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for SAC hedging agent (Optuna)"
    )
    parser.add_argument("--simulator", type=str, default="gbm",
                        choices=["gbm", "heston"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--n-steps-per-trial", type=int, default=150_000,
        help="Training steps per trial. "
             "Must be >> learning_starts (min 2000). "
             "Recommended: 100k-150k for reliable Sharpe signal."
    )
    parser.add_argument("--n-eval-episodes", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--tuning-dir", type=str, default="agent/tuning")
    parser.add_argument("--train-after", action="store_true")

    args = parser.parse_args()

    # Warn if trial budget is too low
    if args.n_steps_per_trial < 50_000:
        print(f"\n[WARN] --n-steps-per-trial={args.n_steps_per_trial:,} is very low.")
        print(f"       Untrained models will produce garbage Sharpe values.")
        print(f"       Recommended minimum: 100,000\n")

    run_tuning(args)