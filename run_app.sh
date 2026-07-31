#!/usr/bin/env bash
set -euo pipefail

export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-llama3.2:latest}"
export LOCAL_LLM_URL="${LOCAL_LLM_URL:-http://localhost:11434}"
export PORT="${PORT:-8501}"
export R_LIBS_USER="${R_LIBS_USER:-$(cd "$(dirname "$0")" && pwd)/.Rlibs}"

python -m streamlit run "$(dirname "$0")/factory_app.py" --server.address 0.0.0.0 --server.port "$PORT"
