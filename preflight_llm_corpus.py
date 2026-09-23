# -*- coding: utf-8 -*-
"""Preflight for the LLM-authored corpus run.

Checks everything the full `python suppl_generator.py` run with SUPPL_USE_LLM=1
needs, then generates ONE association abstract end-to-end so the report-writer
path and the numeric guard are proven before committing to the whole corpus.

    python preflight_llm_corpus.py                  # uses SUPPL_LLM_MODEL or llama3.3:70b
    SUPPL_LLM_MODEL=llama3.1:latest python preflight_llm_corpus.py   # plumbing test
"""
import os, sys, time, json
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ)

MODEL = os.environ.get("SUPPL_LLM_MODEL", "llama3.3:70b")
URL   = os.environ.get("SUPPL_LLM_URL", "http://localhost:11434")
ok = True


def check(name, good, detail=""):
    global ok
    print(f"[{'OK ' if good else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not good: ok = False
    return good


# 1. Ollama reachable and the requested model present
try:
    import requests
    tags = requests.get(URL + "/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    check("ollama reachable", True, URL)
    check(f"model {MODEL} pulled", MODEL in names,
          "available: " + ", ".join(names) if MODEL not in names else "")
except Exception as e:
    check("ollama reachable", False, str(e))
    names = []

# 2. R + survey
try:
    import factory_core as fc
    rs = fc._rscript()
    check("Rscript found", bool(rs), rs or "")
except Exception as e:
    check("Rscript found", False, str(e))

# 3. Raw data
for ds, sub in [("KNHANES", "data/KNHANES"), ("NHANES", "data/NHANES")]:
    n = len(os.listdir(sub)) if os.path.isdir(sub) else 0
    check(f"{ds} raw files", n > 0, f"{n} files in {sub}")

# 4. GPU
try:
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    gpus = [g for g in out.splitlines() if g.strip()]
    print(f"[INFO] GPUs: {len(gpus)}")
    for g in gpus: print("        " + g)
except Exception as e:
    print(f"[INFO] nvidia-smi unavailable: {e}")

# 5. End-to-end report-writer + numeric guard on one real fact string
if ok:
    facts = ("Survey: Korea National Health and Nutrition Examination Survey, survey-weighted, "
             "cross-sectional. Exposure: Alanine aminotransferase. Outcome: Metabolic syndrome. "
             "Measure: odds ratio per 1-SD 1.88 (95% CI 1.77 to 2.01). False discovery rate q <0.001. "
             "Analytic sample 37636 adults aged 20 years or older. Weighted outcome prevalence 28.3 percent. "
             "Direction: positively. Adjustment set: age, sex, smoking, alcohol.")
    t0 = time.time()
    gen = fc.llm_prose("Write the single-paragraph structured abstract of this analysis. "
                       "End by stating that the association is cross-sectional and "
                       "hypothesis-generating.", facts, MODEL, URL)
    dt = time.time() - t0
    print(f"\n[INFO] one abstract took {dt:.1f}s with {MODEL}")
    print(f"[INFO] guard tally {fc.LLM_PROSE_STATS}")
    if gen:
        print("[OK ] report-writer produced guarded prose:\n")
        print("      " + gen.replace("\n", "\n      ")[:900])
        n_reports = 7584
        print(f"\n[INFO] extrapolated prose time for {n_reports} reports: "
              f"{n_reports*dt/3600:.1f} h (prose only, excludes R/sklearn compute)")
    else:
        print("[WARN] prose rejected or empty — the deterministic template would be used.")
        print("       Inspect fc.LLM_PROSE_STATS above: 'violation' means the model altered a number.")

print("\n" + ("PREFLIGHT PASSED — safe to run the full corpus." if ok else
             "PREFLIGHT FAILED — fix the items marked FAIL first."))
sys.exit(0 if ok else 1)
