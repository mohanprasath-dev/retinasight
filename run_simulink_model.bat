@echo off
title RetinaSight - Simulink District Capacity Model Launcher
echo ===================================================================
echo   RetinaSight: Simulink District Rollout & Capacity Model (PS 26038)
echo ===================================================================
echo.

set MATLAB_EXE=C:\Program Files\MATLAB\R2026a\bin\matlab.exe

if not exist "%MATLAB_EXE%" (
    where matlab >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set MATLAB_EXE=matlab
    ) else (
        echo [ERROR] MATLAB executable not found at "%MATLAB_EXE%".
        echo Please ensure MATLAB R2026a is installed.
        pause
        exit /b 1
    )
)

echo [INFO] Opening Simulink District Screening Model...
echo Model: %~dp0matlab\retinasight_capacity_model.slx
echo.

start "" "%MATLAB_EXE%" -nosplash -r "cd('%~dp0matlab'); open_system('retinasight_capacity_model');"

echo Simulink is opening in the background.
echo You can run the simulation and inspect Scopes directly in Simulink!
echo.
pause
