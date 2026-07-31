$ErrorActionPreference = "Stop"

if (-not $env:LOCAL_LLM_MODEL) {
    $env:LOCAL_LLM_MODEL = "llama3.2:latest"
}

if (-not $env:LOCAL_LLM_URL) {
    $env:LOCAL_LLM_URL = "http://localhost:11434"
}

$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $bundledPython) {
    & $bundledPython "$PSScriptRoot\test_local_model.py" @args
} else {
    & python "$PSScriptRoot\test_local_model.py" @args
}
