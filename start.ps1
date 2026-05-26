# start.ps1 — One-command startup for Windows (no Docker)
# Usage: .\start.ps1

Write-Host "⚡ Options Pricing Engine" -ForegroundColor Cyan
Write-Host "Starting API and Dashboard..." -ForegroundColor Gray

# Start API in background
$api = Start-Process python -ArgumentList "-m uvicorn api.main:app --host 0.0.0.0 --port 8000" `
    -PassThru -NoNewWindow

Write-Host "✅ API starting on http://localhost:8000" -ForegroundColor Green
Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor Gray

Start-Sleep -Seconds 3

# Start dashboard
Write-Host "✅ Dashboard starting on http://localhost:8501" -ForegroundColor Green
streamlit run dashboard/app.py --server.port 8501

# Cleanup on exit
Stop-Process -Id $api.Id -ErrorAction SilentlyContinue