"""
Hyperparameter Tuning — Optuna + SAC

Automated search over SAC hyperparameters using Bayesian optimisation
(TPE sampler). Each trial trains for a short budget (50k steps) and
reports mean Sharpe as the objective. Bad trials are pruned early via
MedianPruner to avoid wasting time.

Results are persisted to SQLite so you can resume interrupted studies,
inspect with optuna-dashboard, and re-run best params directly.

Usage
-----
# Basic run (50 trials, GBM)
python agent/tune.py

# Heston simulator, more trials, parallel
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from environment.options_env import OptionsHedgingEnv


# ── Suppress noisy SB3 output during tuning ──────────────────────────────────
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate reporting callback (enables Optuna pruning)
# ─────────────────────────────────────────────────────────────────────────────

class PruningCallback(BaseCallback):
    """
    Reports rolling Sharpe to Optuna at regular intervals.
    If the trial is clearly worse than the median, Optuna prunes it early.
    """

    def __init__(self, trial: optuna.Trial, report_freq: int = 5000):
        super().__init__()
        self.trial = trial
        self.report_freq = report_freq
        self.episode_pnls = []
        self._last_report_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "total_pnl" in info and "episode" in info:
                self.episode_pnls.append(info["total_pnl"])

        if (self.num_timesteps - self._last_report_step >= self.report_freq
                and len(self.episode_pnls) >= 10):

            recent = np.array(self.episode_pnls[-50:])
            sharpe = float(
                np.mean(recent) / (np.std(recent) + 1e-10) * np.sqrt(252)
            )

            self.trial.report(sharpe, step=self.num_timesteps)
            self._last_report_step = self.num_timesteps

            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_config: dict, seed: int = 0):
    def _init():
        return OptionsHedgingEnv(**env_config, seed=seed)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Post-trial evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_trial(model: SAC, train_env: VecNormalize,
                   env_config: dict, n_episodes: int = 40) -> float:
    """
    Run deterministic episodes and return mean Sharpe.

    Syncs VecNormalize obs_rms from training env so observations
    are on the same scale as what the model was trained on.
    """
    eval_env = DummyVecEnv([make_env(env_config, seed=99999)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False
    )
    # ── Critical: sync normalisation stats ───────────────────────────────
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    pnls = []
    for ep in range(n_episodes):
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
        return -999.0  # degenerate trial

    pnls = np.array(pnls)
    std = np.std(pnls)

    if std < 1e-5:
        return -999.0  # reject degenerate policies

    sharpe = np.mean(pnls) / std * np.sqrt(252)
    if abs(sharpe) > 5:
        return -999.0
    print("mean:", np.mean(pnls), "std:", std, "sharpe:", sharpe)

    # Clip to avoid explosion
    return float(np.clip(sharpe, -10, 10))


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(base_env_config: dict, n_train_steps: int, n_eval_episodes: int):
    """
    Returns the objective function closed over env config and budgets.
    Separated so the same objective can be used with n_jobs > 1.
    """

    def objective(trial: optuna.Trial) -> float:

        # ── Search space ─────────────────────────────────────────────────
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        buffer_size = trial.suggest_categorical(
            "buffer_size", [50_000, 100_000, 200_000]
        )
        tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.9999)
        net_width = trial.suggest_categorical("net_width", [64, 128, 256])
        net_depth = trial.suggest_int("net_depth", 1, 3)
        ent_coef = trial.suggest_categorical(
            "ent_coef", ["auto", 0.01, 0.05, 0.1, 0.5]
        )
        learning_starts = trial.suggest_categorical(
            "learning_starts", [500, 1000, 2000]
        )

        arch = [net_width] * net_depth
        policy_kwargs = dict(net_arch=dict(pi=arch, qf=arch))

        # ── Environments ─────────────────────────────────────────────────
        # 2 parallel envs keeps trial wall-time reasonable
        train_env = DummyVecEnv([make_env(base_env_config, seed=i) for i in range(2)])
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
            seed=trial.number,  # different seed per trial
        )

        # ── Train with pruning callback ───────────────────────────────────
        pruning_cb = PruningCallback(
            trial, report_freq=max(n_train_steps // 20, 1000)
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

    # ── Base environment config (mirrors train.py exactly) ────────────────
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

    # ── Storage ───────────────────────────────────────────────────────────
    tuning_dir = Path(args.tuning_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    db_path = tuning_dir / "study.db"
    storage = f"sqlite:///{db_path}"

    study_name = args.study_name or f"sac_hedger_{args.simulator}"

    print(f"\n[CONFIG]")
    print(f"  Simulator  : {args.simulator.upper()}")
    print(f"  Trials     : {args.n_trials}")
    print(f"  Steps/trial: {args.n_steps_per_trial:,}  (proxy budget)")
    print(f"  Eval eps   : {args.n_eval_episodes}")
    print(f"  Study name : {study_name}")
    print(f"  Storage    : {db_path}")

    # ── Create / resume study ────────────────────────────────────────────
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",          # maximise Sharpe ratio
        load_if_exists=True,           # resume if study already exists
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,       # random exploration before TPE kicks in
            multivariate=True,         # model param correlations
        ),
        pruner=MedianPruner(
            n_startup_trials=5,        # don't prune until 5 trials complete
            n_warmup_steps=10,         # don't prune in first 10 reports
            interval_steps=1,
        ),
    )

    n_existing = len(study.trials)
    if n_existing:
        print(f"\n[RESUME] Found {n_existing} existing trials. Continuing...")

    # ── Run optimisation ─────────────────────────────────────────────────
    objective = make_objective(
        base_env_config,
        n_train_steps=args.n_steps_per_trial,
        n_eval_episodes=args.n_eval_episodes,
    )

    print(f"\n[START] Running {args.n_trials} trials...\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,           # set > 1 only with SubprocVecEnv or if your env is thread-safe
        show_progress_bar=True,
        gc_after_trial=True,          # free memory between trials
        catch=(Exception,),           # log crashes without stopping the study
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

    # ── Top-5 table ───────────────────────────────────────────────────────
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

    # ── CLI snippet to retrain ────────────────────────────────────────────
    bp = best.params
    arch = [bp['net_width']] * bp['net_depth']
    ent  = bp['ent_coef']

    print(f"\n  ─── Retrain command ─────────────────────────────────────────")
    print(f"  python agent/train.py \\")
    print(f"    --simulator {args.simulator} \\")
    print(f"    --lr {bp['lr']:.2e} \\")
    print(f"    --batch-size {bp['batch_size']} \\")
    print(f"    --buffer-size {bp['buffer_size']} \\")
    print(f"    --tau {bp['tau']:.4f} \\")
    print(f"    --gamma {bp['gamma']:.6f} \\")
    print(f"    --ent-coef {ent} \\")
    print(f"    --learning-starts {bp['learning_starts']} \\")
    print(f"    --total-timesteps 500000")
    print(f"  # Network: pi={arch}, qf={arch}  (edit train.py policy_kwargs)")

    # ── Save best params as JSON ──────────────────────────────────────────
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

    # ── Optionally kick off full training immediately ─────────────────────
    if args.train_after:
        print(f"\n[AUTO-TRAIN] Launching full train with best params...")
        _launch_full_train(args, best.params)

    print("\n" + "=" * 70)
    return study


def _launch_full_train(args, params):
    """
    Kick off train.py programmatically with best params.
    Modifies policy_kwargs in-place before calling train().
    """
    import importlib.util, types

    # Dynamically load train.py
    train_path = Path(__file__).parent / "train.py"
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    # Build args namespace
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
    )

    # Monkey-patch policy_kwargs with tuned arch
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
    parser.add_argument(
        "--simulator", type=str, default="gbm",
        choices=["gbm", "heston"],
        help="Market simulator type"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--n-steps-per-trial", type=int, default=50_000,
        help="Training steps per trial (proxy budget; shorter = faster but noisier)"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=40,
        help="Episodes for final evaluation of each trial"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel trials (1 = sequential; >1 requires thread-safe envs)"
    )
    parser.add_argument(
        "--study-name", type=str, default=None,
        help="Study name for resuming (default: sac_hedger_<simulator>)"
    )
    parser.add_argument(
        "--tuning-dir", type=str, default="agent/tuning",
        help="Directory for SQLite DB and result JSON"
    )
    parser.add_argument(
        "--train-after", action="store_true",
        help="Auto-launch full 500k-step training with best params after tuning"
    )

    args = parser.parse_args()
    run_tuning(args)