@echo off
setlocal
echo ======================================================================
echo  RetinaSight -- Automated Test Suite Runner (SIH 2026, PS ID 26038)
echo ======================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [RUNNING] Executing automated tests via: %PYTHON_EXE%
"%PYTHON_EXE%" tests/test_preprocessing.py
set "TEST_EXIT_CODE=%ERRORLEVEL%"

echo.
if %TEST_EXIT_CODE% equ 0 (
    echo ======================================================================
    echo  [SUCCESS] All RetinaSight automated tests completed successfully!
    echo  Visual verification artifacts saved to: outputs/
    echo ======================================================================
) else (
    echo ======================================================================
    echo  [FAILURE] Tests failed with exit code %TEST_EXIT_CODE%.
    echo ======================================================================
)

echo.
pause
