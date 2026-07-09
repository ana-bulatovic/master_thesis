@echo off
REM Ollama Model Downloader
echo Ollama Model Management
echo.
echo Available commands:
echo   pull llama2     - Download Llama 2 model
echo   pull mistral    - Download Mistral model
echo   pull tinyllama  - Download TinyLLaMa model
echo   list           - List downloaded models
echo.
echo Usage: start_ollama_models.bat pull tinyllama
echo.

if "%1"=="" (
    echo No command specified. Use: start_ollama_models.bat pull llama2
    pause
    exit /b 1
)

echo Running: "C:\Users\ANA\AppData\Local\Programs\Ollama\ollama.exe" %*
"C:\Users\ANA\AppData\Local\Programs\Ollama\ollama.exe" %*
pause