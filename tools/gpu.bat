@echo off
rem Run the tile model on the graphics card (Windows, any DirectX 12 GPU, no CUDA needed).
rem Replaces the CPU onnxruntime with the DirectML build. Run again after any
rem "pip install -r requirements.txt", which puts the CPU build back.
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime onnxruntime-directml
rmdir /s /q ".venv\Lib\site-packages\onnxruntime" 2>nul
".venv\Scripts\python.exe" -m pip install "onnxruntime-directml==1.24.4" || (echo Install failed. & pause & exit /b 1)
".venv\Scripts\python.exe" -c "import onnxruntime as o; print('providers:', o.get_available_providers())"
echo Done. Restart start.bat.
pause
