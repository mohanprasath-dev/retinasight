@echo off
title RetinaSight - MATLAB Diagnostic Pipeline Launcher
echo ===================================================================
echo   RetinaSight: MathWorks MATLAB Diagnostic Suite (PS 26038)
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

echo [INFO] Launching MATLAB with RetinaSight Pipeline...
echo Working Directory: %~dp0matlab
echo.

start "" "%MATLAB_EXE%" -nosplash -r "cd('%~dp0matlab'); edit('retinasight_pipeline.m'); results = retinasight_pipeline();"

echo MATLAB is opening in the background.
echo The 3-panel Diagnostic Figure and Pipeline Output will appear shortly!
echo.
pause
