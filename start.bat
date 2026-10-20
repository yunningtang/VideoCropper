@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the Python environment ^(a few minutes^)...
    python -m venv .venv || (echo Python 3.12 is needed: https://www.python.org/downloads/ & pause & exit /b 1)
)
".venv\Scripts\python.exe" -c "import uvicorn, av, cv2, fastapi, rapidocr_onnxruntime" >nul 2>&1 || (
    echo Installing packages ^(a few minutes^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed. Run start.bat again to retry. & pause & exit /b 1)
)
".venv\Scripts\python.exe" run.py %*
pause
