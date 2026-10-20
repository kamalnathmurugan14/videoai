# Updates VideoAI to the latest version on GitHub (Windows PowerShell).
# Run from the project folder (the one containing backend\ and frontend\) with your Python virtualenv activated:
#   .\update-windows.ps1
# It discards local changes to tracked files (e.g. a package-lock.json modified by npm), which is what usually blocks
# `git pull`. Your .env, database and uploaded/rendered media are untracked and are kept.
$ErrorActionPreference = "Stop"
git fetch origin main
git reset --hard origin/main
Push-Location backend
python -m pip install -r requirements.txt
Pop-Location
Push-Location frontend
npm install
Pop-Location
Write-Host ""
Write-Host "Updated. Restart the backend (uvicorn) and the frontend (npm run dev), then open http://localhost:5173/system to confirm the version." -ForegroundColor Green
