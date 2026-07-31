# -*- coding: utf-8 -*-
"""Check whether the app folder is ready to run."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import local_llm


PYTHON_PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "pyreadstat": "pyreadstat",
    "docx": "python-docx",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "shap": "shap",
    "xgboost": "xgboost",
}

REQUIRED_APP_FILES = [
    "factory_app.py",
    "factory_core.py",
    "epi_report.py",
    "trend_report.py",
    "ml_report.py",
    "local_llm.py",
]

REQUIRED_R_FILES = [
    "engine.R",
    "epi.R",
    "epi_adv.R",
    "trend.R",
]

DATA_REQUIREMENTS = {
    "KNHANES": ("data/KNHANES", ["hn08_all.sas7bdat", "hn08_dxa.sas7bdat"]),
    "NHANES": ("data/NHANES", ["demo_j.sas7bdat", "bmx_j.sas7bdat", "lux_j.sas7bdat"]),
}


def find_rscript() -> str | None:
    found = shutil.which("Rscript")
    if found:
        return found

    candidates = []
    for base in [Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")]:
        candidates.extend(base.glob("R-*/bin/Rscript.exe"))
        candidates.extend(base.glob("R-*/bin/x64/Rscript.exe"))

    candidates = [p for p in candidates if p.exists()]
    return str(sorted(candidates)[-1]) if candidates else None


def ok(label: str) -> None:
    print(f"[OK] {label}")


def warn(label: str) -> None:
    print(f"[WARN] {label}")


def fail(label: str) -> None:
    print(f"[MISSING] {label}")


def main() -> int:
    root = Path(__file__).resolve().parent
    exit_code = 0

    print(f"Folder: {root}")
    print(f"Python: {sys.executable}")

    for filename in REQUIRED_APP_FILES:
        if (root / filename).exists():
            ok(filename)
        else:
            fail(filename)
            exit_code = 1

    for filename in REQUIRED_R_FILES:
        if (root / filename).exists():
            ok(filename)
        else:
            warn(f"{filename} not found. Analysis buttons that call this R script will fail.")

    for dataset, (folder, examples) in DATA_REQUIREMENTS.items():
        data_dir = root / folder
        found = list(data_dir.rglob("*.sas7bdat")) if data_dir.exists() else []
        if found:
            ok(f"{dataset} data files: {len(found)} sas7bdat file(s) under {folder}")
        else:
            warn(
                f"{dataset} data not found under {folder}. "
                f"Expected examples: {', '.join(examples)}"
            )

    for module, package in PYTHON_PACKAGES.items():
        if importlib.util.find_spec(module):
            ok(f"Python package: {package}")
        else:
            fail(f"Python package: {package}. Install with `pip install -r requirements.txt`.")
            exit_code = 1

    rscript = find_rscript()
    if rscript:
        ok(f"Rscript: {rscript}")
    else:
        warn("Rscript not found in PATH. Survey analyses will fail until R is installed/configured.")

    try:
        models = local_llm.list_models(timeout=5)
        ok("Ollama server is reachable")
        print("     Models: " + (", ".join(models) if models else "(none installed)"))
        if local_llm.DEFAULT_MODEL in models:
            ok(f"Default model: {local_llm.DEFAULT_MODEL}")
        else:
            warn(
                f"Default model {local_llm.DEFAULT_MODEL} was not listed. "
                "Set LOCAL_LLM_MODEL or pull that model."
            )
    except Exception as exc:
        warn(f"Ollama check failed: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
