# Local model setup

The manuscript's agent layer used **Llama 3.3 70B** served locally through Ollama
(`llama3.3:70b`), on one workstation with four NVIDIA RTX PRO 6000 Blackwell GPUs. That is
the default model everywhere in this repository:

| Where | Setting | Default |
|---|---|---|
| Agent layer, `pipeline.py`, `evidence_rag.py` | `--model` | `llama3.3:70b` |
| Corpus report writer, `suppl_generator.py` | `SUPPL_LLM_MODEL` | `llama3.3:70b` |
| Web UI, `factory_app.py` | sidebar "Model" | `llama3.3:70b` |
| `local_llm.py`, `preflight.py`, `run_*.cmd`/`.sh`/`.ps1`, `rag/` | `LOCAL_LLM_MODEL` | `llama3.3:70b` |

No statistic depends on the model. Without Ollama every agent falls back to a deterministic
rule or template, and the report builders write their template text.

## Requirements

- Ollama installed and running
- The model pulled once:

```powershell
ollama pull llama3.3:70b
ollama serve
```

## A smaller model for testing

A 70-billion-parameter model needs roughly 40 GB of GPU memory. On an ordinary laptop, use a
smaller model to check the plumbing, for example:

```powershell
ollama pull llama3.2
$env:LOCAL_LLM_MODEL = "llama3.2"
$env:LOCAL_LLM_URL = "http://localhost:11434"
```

Prose written by a smaller model is not what the manuscript reports.

## Model-only smoke test

This needs neither Streamlit nor `requests`:

```powershell
.\run_local_model.ps1
```

If PowerShell script execution is blocked on Windows, use:

```cmd
run_local_model.cmd
```

Or name a model explicitly:

```powershell
.\run_local_model.ps1 --model llama3.2
```

## Full app quick start

```powershell
pip install -r requirements.txt
Rscript install_r_packages.R
python preflight.py
run_app.cmd
```

On Linux or macOS:

```bash
pip install -r requirements.txt
Rscript install_r_packages.R
python preflight.py
bash run_app.sh
```
