# Run AI Gymnastics Judge web app (use from Windows PowerShell, not Git Bash)
Set-Location $PSScriptRoot
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
    python -m src.gymnastics_judge.web_app
} else {
    Write-Host "No .venv found. Run: uv sync" -ForegroundColor Yellow
    uv sync
    .\.venv\Scripts\Activate.ps1
    python -m src.gymnastics_judge.web_app
}
