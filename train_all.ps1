# ============================================================
#  Options Engine - Full Training Pipeline
#  Run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#       .\train_all.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$LogDir = "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Step { param($n, $msg)
    Write-Host ""
    Write-Host "[STEP $n] $msg" -ForegroundColor Cyan
}
function Write-OK   { param($msg) Write-Host "  [OK]   $msg" -ForegroundColor Green  }
function Write-Warn { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  [FAIL] $msg" -ForegroundColor Red    }

Write-Host "============================================================"
Write-Host "  Options Engine - Full Training Pipeline"
Write-Host "  Started: $(Get-Date)"
Write-Host "============================================================"

# Step 1: Delete stale 7-dim models
Write-Step 1 "Removing stale 7-dim models"
foreach ($f in @("agent\models\sac_hedger_final.zip", "agent\models\vec_normalize.pkl")) {
    if (Test-Path $f) { Remove-Item $f; Write-OK "Deleted $f" }
}

# Step 2: Tune
Write-Step 2 "Hyperparameter tuning - Heston 50 trials"
Write-Host "  Log: $LogDir\tune_heston.log"
python agent/tune.py --simulator heston --n-trials 50 --device cuda *> "$LogDir\tune_heston.log"
if ($LASTEXITCODE -ne 0) { Write-Warn "Tuning failed - continuing with default HPs" }
else { Write-OK "Tuning complete" }

# Step 3: Train GBM
Write-Step 3 "Training GBM - 500k steps"
Write-Host "  Log: $LogDir\train_gbm.log"
python agent/train.py --simulator gbm --total-timesteps 500000 --device cuda *> "$LogDir\train_gbm.log"
if ($LASTEXITCODE -ne 0) { Write-Fail "GBM training failed - check $LogDir\train_gbm.log"; exit 1 }
Write-OK "GBM done -> agent\models\sac_hedger_gbm_final.zip"

# Step 4: Train Heston
Write-Step 4 "Training Heston - 500k steps"
Write-Host "  Log: $LogDir\train_heston.log"
python agent/train.py --simulator heston --total-timesteps 500000 --lr-cycle-steps 50000 --device cuda *> "$LogDir\train_heston.log"
if ($LASTEXITCODE -ne 0) { Write-Fail "Heston training failed - check $LogDir\train_heston.log"; exit 1 }
Write-OK "Heston done -> agent\models\sac_hedger_heston_final.zip"

# Step 5: Train Jump
Write-Step 5 "Training Jump-Diffusion - 500k steps"
Write-Host "  Log: $LogDir\train_jump.log"
python agent/train.py --simulator jump --total-timesteps 500000 --device cuda *> "$LogDir\train_jump.log"
if ($LASTEXITCODE -ne 0) { Write-Fail "Jump training failed - check $LogDir\train_jump.log"; exit 1 }
Write-OK "Jump done -> agent\models\sac_hedger_jump_final.zip"

# Step 6: Evaluate Heston
Write-Step 6 "Full evaluation - Heston model 1000 episodes"
Write-Host "  Log: $LogDir\evaluate.log"
python agent/evaluate.py `
    --simulator heston `
    --model-path agent/models/sac_hedger_heston_final `
    --vnorm-path agent/models/vec_normalize_heston.pkl `
    --n-episodes 1000 `
    --device cuda *> "$LogDir\evaluate.log"
if ($LASTEXITCODE -ne 0) { Write-Warn "Evaluation failed - check $LogDir\evaluate.log" }
else { Write-OK "Evaluation done -> agent\evaluation_results.json" }

# Step 7a: Historical backtest 1y
Write-Step "7a" "Historical SPY backtest - 1 year"
Write-Host "  Log: $LogDir\historical_1y.log"
python backtester/historical.py `
    --model-path agent/models/sac_hedger_heston_final `
    --vnorm-path agent/models/vec_normalize_heston.pkl `
    --period 1y *> "$LogDir\historical_1y.log"
if ($LASTEXITCODE -ne 0) { Write-Warn "1y backtest failed - check $LogDir\historical_1y.log" }
else { Write-OK "1y backtest done" }

# Step 7b: Historical backtest 2y
Write-Step "7b" "Historical SPY backtest - 2 years stride 3"
Write-Host "  Log: $LogDir\historical_2y.log"
python backtester/historical.py `
    --model-path agent/models/sac_hedger_heston_final `
    --vnorm-path agent/models/vec_normalize_heston.pkl `
    --period 2y `
    --stride 3 *> "$LogDir\historical_2y.log"
if ($LASTEXITCODE -ne 0) { Write-Warn "2y backtest failed - check $LogDir\historical_2y.log" }
else { Write-OK "2y backtest done" }

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Pipeline complete: $(Get-Date)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Models saved:"
Write-Host "    agent\models\sac_hedger_gbm_final.zip"
Write-Host "    agent\models\sac_hedger_heston_final.zip"
Write-Host "    agent\models\sac_hedger_jump_final.zip"
Write-Host ""
Write-Host "  Results:"
Write-Host "    agent\evaluation_results.json"
Write-Host "    agent\historical_results.json"
Write-Host ""
Write-Host "  Logs:"
Get-ChildItem "$LogDir\*.log" | ForEach-Object { Write-Host "    $($_.FullName)" }
Write-Host ""
Write-Host "  TensorBoard:  tensorboard --logdir tb_logs"