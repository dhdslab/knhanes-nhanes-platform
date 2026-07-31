@echo off
setlocal

if "%LOCAL_LLM_MODEL%"=="" set "LOCAL_LLM_MODEL=llama3.2:latest"
if "%LOCAL_LLM_URL%"=="" set "LOCAL_LLM_URL=http://localhost:11434"
if "%R_LIBS_USER%"=="" set "R_LIBS_USER=%~dp0.Rlibs"

python -m streamlit run "%~dp0factory_app.py" --server.address 0.0.0.0 --server.port 8501
