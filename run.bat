@echo off
cd /d "%~dp0"
if not exist .env (
  echo [warn] .env not found. Copy .env.example to .env and add your GROQ_API_KEY.
)
".venv\Scripts\python.exe" -m uvicorn app:app --reload --port 8000
pause
