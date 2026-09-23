# KNHANES + NHANES evidence platform

Reference implementation and released report corpus for the manuscript
*"Medical evidence at machine scale: an auditable architecture for reusable health data"*
(Kim, Park *et al.*, under review).

The platform harmonizes the **Korea National Health and Nutrition Examination Survey
(KNHANES)** and the **US National Health and Nutrition Examination Survey (NHANES)** under
one registry of operational definitions. It tests every eligible exposure against every
outcome with complex-survey regression, fits cycle-level trend models and leakage-guarded
prediction models, and writes each result as a self-contained Word report.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.6-276DC3?logo=r&logoColor=white)
![Reports](https://img.shields.io/badge/reports-7%2C584-2A9D8F)

---

## The one rule

> **Every number comes from deterministic code.** Survey statistics are computed in R
> `survey`, trend models in base R and `MASS`, and prediction models in scikit-learn and
> XGBoost. Language-model agents audit definitions, coordinate the pipeline and draft prose.
> They never estimate, select or edit a number. Without a language model every agent falls
> back to a deterministic rule or template, so the pipeline runs either way.

Python and R communicate through files. Python writes `*_analytic.csv` and `*_config.json`,
runs `Rscript`, and reads back `*_results.csv`, so every intermediate file stays on disk as
part of the audit trail.

---

## What is in this repository

| Path | Contents |
|---|---|
| `factory_core.py` | Loaders, variable derivation, the definition registry (27 outcomes), the leakage and circularity guard, covariate sets and the R interface |
| `engine.R` | Survey-weighted association models (`svyglm`) and Table 1 |
| `trend_report.py` + `trend.R` | Per-cycle prevalence, age–sex standardization, change-point APC, negative-binomial forecast |
| `ml_report.py` | 16-model prediction registry, tuning, metrics, calibration, SHAP and partial dependence |
| `epi_report.py` + `epi.R`, `epi_adv.R`, `epi_surv.R` | Single-exposure causal-inference report (IPTW, PSM, AIPW, G-computation, TMLE, E-value, RCS, survival) |
| `full_report.py` | Combined single-file report |
| `suppl_generator.py` | Batch generator of the whole corpus (association, trend, ML) |
| `run_ds_suppl.py` | Regenerates one survey and withdraws unstable odds ratios (see below) |
| `regen_ml_cvselect.py`, `regen_high_tg.py` | The two regenerations applied to the prediction reports after the first run |
| `agents.py`, `pipeline.py` | Agent layer (definition auditor, guideline verifier, harmonizer, batch QC, report writer, orchestrator, decision log) and the command-line pipeline |
| `evidence_rag.py`, `rag/` | Question answering over the corpus with cited sources (see [Evidence search](#evidence-search)) |
| `factory_app.py` | Streamlit web UI with seven tabs: interactive analysis, trajectory, epidemiologic report, trend report, ML report, auto-manuscript over all pairs, combined report |
| `suppl/` | **The released corpus**: 7,584 reports and the two association manifests |
| `_ml_regen_summary.csv` | One row per released prediction model: selected model, held-out AUROC and AUPRC |
| `analysis/` | Code behind the Supplementary Appendix sections and figures ([analysis/README.md](analysis/README.md)) |
| `check_reproduction.py` | Re-runs a sample of the corpus from the raw data and compares it with `suppl/` |
| `docs/` | [Operational definitions](docs/OPERATIONAL_DEFINITIONS.md), [agents](docs/AGENTS.md), [evidence RAG](docs/EVIDENCE_RAG.md), [architecture](docs/ARCHITECTURE.md), [reproducibility](docs/REPRODUCIBILITY.md) |
| `tests/` | Smoke tests that need no data, R or GPU |

---

## Data

Raw microdata are **not** distributed here. Download them from the Korea Disease Control and
Prevention Agency (KNHANES) and the US National Center for Health Statistics (NHANES), and
place them under `data/`, which git ignores:

```
data/KNHANES/  hn07_all.sas7bdat ... hn24_all.sas7bdat   (18 annual cycles, 2007-2024)
               hn08_dxa.sas7bdat ... hn11_dxa.sas7bdat   (DXA, merged within year)
data/NHANES/   <MODULE>_<cycle>.XPT for cycles D, E, F, G, H, I, J and L
               (2005-2006 through August 2021-August 2023)
               modules: DEMO BMX BIOPRO GLU GHB INS VID TRIGLY TCHOL HDL CBC LUX BPX BPXO
                        BPQ DIQ ALB_CR SMQ ALQ DXX DXXFEM DXXSPN
```

NHANES elastography (`LUX`) is included for 2017–2018 only (`LUX_J`). The August
2021–August 2023 elastography file (`LUX_L`) was not incorporated.

---

## Install

**Python 3.12** with the pinned environment of the study:

```bash
pip install -r requirements.txt
```

**R 4.6** with `survey`, `MASS`, `jsonlite` and `survival` (`splines` ships with R):

```bash
Rscript install_r_packages.R
```

`Rscript` is found automatically (`factory_core._rscript`); on Windows the platform looks in
`C:\Program Files\R\R-4.6.x\bin`.

**Ollama (optional).** The manuscript's agent layer used Llama 3.3 with 70 billion
parameters, served locally through Ollama on one workstation with four NVIDIA RTX PRO 6000
Blackwell GPUs. That model is the default everywhere in this repository:

```bash
ollama pull llama3.3:70b
ollama serve
```

A smaller model such as `llama3.2` can be selected for a quick plumbing test (the `--model`
flag, the app sidebar, `LOCAL_LLM_MODEL` or `SUPPL_LLM_MODEL`). Output from a smaller model
is not what the manuscript reports. Without Ollama the agents use their deterministic
fallbacks. No statistic depends on the model either way.

Check the set-up:

```bash
python preflight.py
```

---

## Quick start

No data, no R, no GPU:

```bash
python pipeline.py --demo --no-llm      # agent layer on synthetic inputs; writes out/
python evidence_rag.py --demo --no-llm  # evidence search on a synthetic corpus
python -m pytest tests/
```

With the data and R in place:

```bash
streamlit run factory_app.py            # web UI (or run_app.cmd / run_app.sh)
python pipeline.py --survey KNHANES --data-dir data/KNHANES --cycles 08 09 10 11 \
    --outcomes dm htn mets --out out/
```

---

## The released corpus and how it was produced

`suppl/` holds the 7,584 reports and the two manifests behind every estimate in the
manuscript. The same reports are bundled as Supplementary Data S1 to S6.

| | KNHANES | NHANES |
|---|---:|---:|
| Association reports (`suppl/<survey>/association/`) | 3,522 | 3,956 |
| Trend reports (`suppl/<survey>/trend/`) | 27 | 25 |
| Prediction reports (`suppl/<survey>/ml/`) | 27 | 27 |

`_manifest_association_<survey>.csv` lists every association: exposure, outcome, measure,
estimate, 95% CI, P, FDR q, contributing N and adjustment set.

The corpus was produced in four steps, all with the code in this repository:

1. **`suppl_generator.py`** generated association, trend and prediction reports for both
   surveys.
2. **Withdrawal of unstable odds ratios.** Association pairs with an odds ratio outside
   1/50 to 50 (quasi-separation) were withdrawn and the Benjamini–Hochberg q values
   recomputed over the pairs that remain. `run_ds_suppl.py` implements this step. The
   released manifests contain no odds ratio outside that range, and their q values equal a
   fresh Benjamini–Hochberg computation over the released rows.
3. **`regen_ml_cvselect.py`** regenerated all 54 prediction reports, so that
   hyperparameters and the choice between models use only cross-validation on the
   training partition, and the held-out partition is scored once.
4. **`regen_high_tg.py`** regenerated the NHANES high-triglyceride model under the identity
   guard (see [Leakage and circularity](#leakage-and-circularity)).

Runtime configuration of the released corpus (the constants in `suppl_generator.py`):

| Setting | Value |
|---|---|
| Population | Adults aged 20 years or older |
| Cycles | KNHANES 07–24; NHANES D, E, F, G, H, I, J, L |
| Association screen | A fixed random sample of 50,000 pooled records per survey (`ASSOC_SAMPLE_N`, seed 0). Both surveys exceed that size (140,372 and 82,123 released records), so the contributing N of an association is at most 37,636 (KNHANES) and 26,892 (NHANES). A single pair can be re-run on the full data in the app. |
| Prediction models | `LogisticRegression`, `HistGradientBoosting`, `XGBoost` (`ML_MODELS`), on a random sample of 6,000 participants per outcome (`ML_SAMPLE_N`, seed 0) |
| Report prose | Deterministic templates. `SUPPL_USE_LLM` was not set, so the prose of every released report is template text (see [Report writer](#report-writer)). |

`check_reproduction.py` re-runs a sample of the corpus from the raw data with this code and
compares it with `suppl/`. On the study data every compared value is identical: the
NHANES hypertension and KNHANES diabetes trend reports (all table cells), the 63 NHANES
hypertension association rows (estimate, CI, P, N and adjustment set), and the NHANES
high-triglyceride model (selected model, AUROC, AUPRC, feature count).

To regenerate the whole corpus, point `SUPPL_DIR` somewhere new so that the released
`suppl/` is left untouched, then compare:

```bash
SUPPL_DIR=suppl_regen python run_ds_suppl.py KNHANES
SUPPL_DIR=suppl_regen python run_ds_suppl.py NHANES
SUPPL_DIR=suppl_regen python regen_ml_cvselect.py
SUPPL_DIR=suppl_regen python regen_high_tg.py
```

`regen_ml_cvselect.py` and `regen_high_tg.py` also rewrite `_ml_regen_summary.csv` in the
repository root. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Methods, as implemented

### Operational definitions

27 harmonized binary clinical outcomes, defined identically in both surveys where
measurement allows. Examples:

| Outcome | Definition |
|---|---|
| Diabetes | Fasting plasma glucose ≥ 126 mg/dL, HbA1c ≥ 6.5%, or current glucose-lowering medication |
| Hypertension | SBP ≥ 140 or DBP ≥ 90 mmHg, or current antihypertensive treatment |
| Metabolic syndrome | Harmonized NCEP ATP III, at least 3 of 5 components, population-specific waist thresholds |
| Dyslipidaemia | Total cholesterol ≥ 240, HDL < 40 (men) / < 50 (women), triglycerides ≥ 200 or LDL ≥ 160 mg/dL, or current lipid-lowering medication |
| Kidney phenotype | eGFR < 60 mL/min/1.73 m² (2021 race-free CKD-EPI) or urine ACR ≥ 30 mg/g. Single examination, so chronicity is not established |
| Hepatic steatosis | KNHANES: Hepatic Steatosis Index > 36. NHANES: controlled attenuation parameter ≥ 248 dB/m, 2017–2018 only |
| MASLD | Steatosis, at least one cardiometabolic risk factor, and alcohol below the MetALD threshold (2023 multisociety nomenclature) |
| Obesity / overweight | KNHANES BMI ≥ 25 / ≥ 23 kg/m² (KSSO); NHANES ≥ 30 / ≥ 25 kg/m² (WHO) |

Three asymmetries are intentional and declared: obesity and overweight thresholds, and
steatosis ascertainment, which MASLD inherits. Steatosis prevalences are shown side by side
and never pooled. Only measured biomarkers and current medication are used. Lifetime
self-reported diagnoses are not used, and the KNHANES examination-day medication items
(`HE_DMdr`, `HE_HPdr`) are not used as current treatment. Fasting analytes require at least
8 hours of fasting. The complete catalogue is in Supplementary Appendix S2 and
[docs/OPERATIONAL_DEFINITIONS.md](docs/OPERATIONAL_DEFINITIONS.md).

### Missing data

Every clinical binary outcome is built by `factory_core._mk(pos, obs)`, which returns a
missing value, never zero, for a participant who does not meet the outcome's observation
rule. The rule is specific to each outcome:

- **Conjunctive definitions** such as metabolic syndrome: a non-case requires every
  component to be observed.
- **Disjunctive definitions**, including diabetes, hypertension, dyslipidaemia, the kidney
  phenotype and osteoporosis: a participant is classified once any defining component is
  observed. A participant whose observed components are negative is a non-case even if
  another component is missing.

Participants without defining data are therefore dropped rather than counted as controls,
although missing components can still understate the prevalence of the disjunctive
composites.

### Leakage and circularity

A pair is not tested when:

- the two variables share a measurement family;
- one is a defining component of the other, evaluated over a transitively closed component
  graph (age enters CKD-EPI, so age is not tested against the kidney phenotype); or
- a derived index shares a direct input with the outcome (HSI contains BMI, so HSI is not
  tested against obesity).

Age and sex are confounders, not sources of circularity. Outcomes with fewer than 50 events
are blocked per survey. HOMA-IR and fasting insulin are natural-log transformed before
standardization. For prediction only, one more rule removes an algebraic identity. NHANES
releases LDL cholesterol as the Friedewald calculated value, so total cholesterol, HDL and
LDL together recover triglycerides exactly. The lipid panel is therefore withheld from the
NHANES high-triglyceride model (`factory_core._ML_IDENTITY_EXCL`). KNHANES measures LDL
directly, so the identity does not arise there.

### Association models

`svydesign(ids = ~PSU, strata = ~stratum, weights = ~pooled weight, nest = TRUE)` with
`survey.lonely.psu = "adjust"`. Strata are made unique within cycle, and the pooled weight
is the examination weight divided by the number of pooled cycles. Each pair is fitted with
`svyglm`: quasibinomial with logit link for binary outcomes (odds ratio per SD) and
Gaussian for continuous outcomes (β per SD). Each outcome has a prespecified covariate set,
from which the exposure's own measurement family is removed. Multiplicity is controlled by
the Benjamini–Hochberg procedure across the exposure-by-outcome matrix within each survey.
The auto-manuscript tab of the app (`fc.run_matrix`) adjusts for age and sex only and is a
separate, quicker screen.

### Temporal trends (`trend_report.py`, `trend.R`)

- Survey-weighted prevalence per cycle, crude and directly age–sex standardized. The
  standard is five age groups (< 40, 40–49, 50–59, 60–69, ≥ 70) by sex, weighted to the
  pooled population of the included cycles.
- **Annual percentage change**, (exp[slope] − 1) × 100, from log-linear regression of the
  log standardized prevalence on survey year.
  - With **at least five cycles**, one change point is located by minimizing the combined
    residual sum of squares of two log-linear segments. Each segment's APC gets a
    percentile 95% CI from 2,000 bootstrap resamples of the cycle-level points (seed 1).
  - With fewer than five cycles, one slope is fitted.
  - The reports label this "joinpoint regression". It is a single-change-point search, not
    the permutation-tested model selection of the NCI Joinpoint program.
- **Negative-binomial regression** of cycle event counts on year, with the log of the
  number of participants with the outcome observed as offset (Poisson if the
  negative-binomial fit fails). This gives a forecast with a 95% confidence interval for
  the fitted rate.
- A linear projection of the standardized rate, with a 95% prediction interval.
- Outcomes with data in fewer than two cycles are not trended. This applies to NHANES
  steatosis and MASLD.
- These models use cycle-level summaries and do not carry the complex-survey variance into
  the trend coefficient, so they are descriptive.

### Prediction models (`ml_report.py`)

- A median-imputation → standardization → classifier pipeline.
- Tuned by `RandomizedSearchCV` (up to 10 candidates) with stratified 3-fold
  cross-validation on a stratified 70% training partition, scored by AUROC and average
  precision.
- The final model is the one with the highest mean cross-validated AUROC and AUPRC. It is
  refitted on the training partition and scored once on the untouched 30% held-out
  partition. The test partition plays no part in selection.
- Features are the outcome's leakage-free set (`factory_core.ml_features`). Features
  observed in 30% or fewer participants are dropped.
- Reports give the full metric set, a 10-bin quantile calibration curve, a threshold
  table, SHAP and partial dependence.
- Models are fitted without survey weights.
- The registry holds 16 classifiers, 14 of them installed with `requirements.txt`. LightGBM
  and CatBoost are optional. The released corpus used the three-model subset above.

### Fasting subsample weight

The primary analyses use the examination weight. NHANES outcomes defined with fasting
analytes were re-estimated in the fasting subsample with `WTSAF2YR` as a prespecified
sensitivity analysis. The scripts are `analysis/fasting_weight_sensitivity.py` for
prevalence and `analysis/fasting_weight_associations.py` for associations.

### Report writer

`fc.llm_prose` composes a report's prose from the fact string that the deterministic core
emits. A numeric guard accepts generated text only if every number in it is an exact token
of that fact string; otherwise the deterministic template is written. The per-run tally of
accepted text, guard violations and errors is written by `suppl_generator.py` to
`_SUMMARY.txt` in the corpus directory. The path is switched on with `SUPPL_USE_LLM=1` (model `SUPPL_LLM_MODEL`, default `llama3.3:70b`), and
`preflight_llm_corpus.py` checks it end to end on one report first. The released corpus was
generated with the template path.

---

## Evidence search

Two front ends read the released corpus and quote its numbers verbatim, with the source
report of each. Neither computes a number.

- `rag/`: a search tool for non-specialists, with a browser UI (`streamlit run rag/app.py`)
  and a command line (`python rag/ask.py "..."`). Retrieval uses TF-IDF with no external
  service, or `bge-m3` embeddings through Ollama. Answers are written by the local model,
  `llama3.3:70b` by default, or shown as quoted excerpts. See [rag/README.md](rag/README.md).
- `evidence_rag.py`: the agent-layer version, with BM25 retrieval over the manifests and
  reports and optional Ollama embeddings. See [docs/EVIDENCE_RAG.md](docs/EVIDENCE_RAG.md).

---

## Limitations

- **Hepatic steatosis is measured differently in the two surveys** (KNHANES HSI, NHANES
  CAP). This is intended: cross-national steatosis and MASLD contrasts reflect the method as
  much as biology. NHANES CAP covers 2017–2018 only.
- **Adjustment.** Each association is adjusted for its outcome's prespecified covariate
  set, not for the other exposures. The corpus is a screening map, not a set of mutually
  adjusted models.
- **The association screen uses a 50,000-record sample per survey.** Trend and prediction
  models are fitted separately (see above).
- **Weights.** The primary analyses use the examination weight. The fasting-subsample
  weight is examined as a sensitivity analysis, not applied throughout.
- **Trend models are descriptive** and do not propagate the survey variance.
- **Prediction models are unweighted**, fitted on 6,000-participant samples and validated
  within survey only.
- **The audit agents were developed within the pipeline** and have not been evaluated on an
  independent, blinded benchmark.
- **All results are cross-sectional and hypothesis-generating.**

---

## Citation

Please cite the manuscript above. Citation details will be added on publication.
