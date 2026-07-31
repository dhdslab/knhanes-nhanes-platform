# Local model setup

The app now defaults to `llama3.2:latest` instead of `llama3.3:70b`.
`llama3.3:70b` is usually too large for a normal local Windows laptop, while
`llama3.2:latest` is a practical smoke-test/default model for local report text
and is already present in this environment.

## Requirements

- Ollama must be installed and running.
- Pull the model once:

```powershell
ollama pull llama3.2:latest
ollama serve
```

If you want another model, set it before launching the app:

```powershell
$env:LOCAL_LLM_MODEL = "llama3.1:8b"
$env:LOCAL_LLM_URL = "http://localhost:11434"
```

## Model-only smoke test

This uses the bundled Codex Python and does not require Streamlit or requests:

```powershell
.\run_local_model.ps1
```

If PowerShell script execution is blocked on Windows, use:

```cmd
run_local_model.cmd
```

Or with an explicit model:

```powershell
.\run_local_model.ps1 --model llama3.1:8b
```

## App defaults

`factory_app.py` reads:

- `LOCAL_LLM_MODEL`, default `llama3.2:latest`
- `LOCAL_LLM_URL`, default `http://localhost:11434`

The report builders still fall back to deterministic template text if local LLM
generation fails.

## Full app quick start

```powershell
pip install -r requirements.txt
Rscript install_r_packages.R
python preflight.py
run_app.cmd
```

On a Linux or Jupyter-like terminal:

```bash
pip install -r requirements.txt
Rscript install_r_packages.R
python preflight.py
bash run_app.sh
```
