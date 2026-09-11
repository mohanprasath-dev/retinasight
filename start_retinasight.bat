@echo off
title RetinaSight — SIH 2026 Screening Pipeline (PS 26038)
color 0B

echo ======================================================================
echo   RetinaSight: Explainable AI Diabetic Retinopathy Screening Pipeline
echo   Smart India Hackathon 2026 - PS 26038 (MathWorks) - Team OnFocus
echo ======================================================================
echo.

echo [1/3] Starting Python FastAPI Backend on http://127.0.0.1:8000 ...
start "RetinaSight Backend" cmd /k ".\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [2/3] Waiting for Backend Warmup ...
timeout /t 3 /nobreak >nul

echo [3/3] Starting Vite React Frontend Dashboard on http://localhost:5173 ...
cd frontend
start "RetinaSight Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ======================================================================
echo   RetinaSight services launched successfully!
echo   - Web Dashboard: http://localhost:5173
echo   - Backend API:   http://127.0.0.1:8000
echo   - API Docs:      http://127.0.0.1:8000/docs
echo ======================================================================
pause
