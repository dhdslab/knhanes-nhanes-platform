# Reproducibility

## Environment of the study

- Python 3.12.13 with the versions pinned in `requirements.txt`
- R 4.6.1 with `survey`, `MASS`, `jsonlite`, `splines` and `survival`
- Ollama serving `llama3.3:70b` for the agent layer, on one workstation with four NVIDIA
  RTX PRO 6000 Blackwell GPUs. No statistic depends on the model.

## Determinism

Every statistic is a deterministic survey-weighted estimate from R `survey` or a
fixed-seed scikit-learn/XGBoost model. The seeds are:

- the 50,000-record association sample: seed 0
- the 6,000-participant prediction sample, the 70/30 split and the cross-validation folds:
  seed 0
- the trend bootstrap: seed 1

Re-running the same code on the same raw files gives the same numbers. With the template
prose path, the text is the same as well.

## What produced the released corpus

| Step | Script | Effect on `suppl/` |
|---|---|---|
| 1 | `suppl_generator.py` | Association, trend and prediction reports for both surveys, and the two manifests |
| 2 | Withdrawal of unstable odds ratios (as in `run_ds_suppl.py`) | Association pairs with an odds ratio outside 1/50 to 50 removed, q recomputed over the remaining pairs |
| 3 | `regen_ml_cvselect.py` | All 54 prediction reports regenerated with model selection by training-partition cross-validation |
| 4 | `regen_high_tg.py` | NHANES high-triglyceride report regenerated under the identity guard (`factory_core._ML_IDENTITY_EXCL`) |

Settings (constants in `suppl_generator.py`):

- adults aged 20 years or older
- KNHANES cycles 07–24; NHANES cycles D–J and L
- `ASSOC_SAMPLE_N = 50000`, a fixed random sample of pooled records per survey for the
  association screen; both surveys are larger than this
- `ML_MODELS = ["LogisticRegression", "HistGradientBoosting", "XGBoost"]`
- `ML_SAMPLE_N = 6000`
- `SUPPL_USE_LLM` unset, so the report prose is template text

`suppl/_SUMMARY.txt` is the summary written by step 1. It gives the counts before step 2
(3,592 KNHANES and 4,027 NHANES association files). The released manifests hold 3,522 and
3,956 pairs.

## Check the code against the corpus

```bash
python check_reproduction.py
```

This needs the raw data and R. It re-runs a sample of the corpus in a temporary directory
and compares it with `suppl/`:

- the NHANES hypertension and KNHANES diabetes trend reports, every table cell
- the NHANES high-triglyceride model: selected model, held-out AUROC and AUPRC, feature count
- all NHANES hypertension association rows: estimate, 95% CI, P, N and adjustment set

On the study data every compared value is identical.

## Regenerate the whole corpus

Write to a new directory so that the released corpus is left alone:

```bash
export SUPPL_DIR=suppl_regen          # PowerShell: $env:SUPPL_DIR = "suppl_regen"
python run_ds_suppl.py KNHANES        # steps 1 and 2 for one survey
python run_ds_suppl.py NHANES
python regen_ml_cvselect.py           # step 3
python regen_high_tg.py               # step 4
```

Then compare `suppl_regen/_manifest_association_*.csv` with `suppl/`. Steps 3 and 4 also
rewrite `_ml_regen_summary.csv` in the repository root. To write the prose with the report
writer instead, set `SUPPL_USE_LLM=1` (model `SUPPL_LLM_MODEL`, default `llama3.3:70b`) and
run `python preflight_llm_corpus.py` first. This changes the prose only; the numbers stay
the same.

## Other runs

```bash
# agent layer and a Word report, no data, R or GPU:
python pipeline.py --demo --no-llm

# one survey end to end (data + R, ideally Ollama):
python pipeline.py --survey KNHANES --data-dir data/KNHANES --cycles 08 09 10 11 --out out/
```

Each `pipeline.py` run writes machine-readable results (`results_*.csv`), a per-outcome
status table, a quality-control report (`qc_report.json`) and the decision ledger
(`decision_log.json`). Its association matrix (`fc.run_matrix`) adjusts for age and sex
only. It is a quick screen, not the corpus.

## Data

Raw microdata are **not** distributed, for privacy and licensing reasons. Obtain them from
the KDCA (KNHANES `.sas7bdat`) and NCHS (NHANES `.XPT`) and place them under `data/`, which
git ignores. File names are listed in the main [README](../README.md#data).

## Known limitations

- Steatosis uses different methods by design (HSI and CAP). Its cross-national contrast
  reflects the method as much as biology, and the two prevalences are never pooled. NHANES
  CAP covers 2017–2018 only.
- The primary analyses use the examination weight. NHANES outcomes built on fasting
  analytes were re-estimated with the fasting-subsample weight `WTSAF2YR` as a sensitivity
  analysis (`analysis/fasting_weight_*.py`).
- Trend models use cycle-level summaries and are descriptive.
- Prediction models are unweighted, fitted on 6,000-participant samples, and validated
  within survey only.
- All analyses are cross-sectional and hypothesis-generating.
