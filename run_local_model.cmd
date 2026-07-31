@echo off
setlocal

if "%LOCAL_LLM_MODEL%"=="" set "LOCAL_LLM_MODEL=llama3.2:latest"
if "%LOCAL_LLM_URL%"=="" set "LOCAL_LLM_URL=http://localhost:11434"

set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%BUNDLED_PYTHON%" (
    "%BUNDLED_PYTHON%" "%~dp0test_local_model.py" %*
) else (
    python "%~dp0test_local_model.py" %*
)

exit /b %ERRORLEVEL%
