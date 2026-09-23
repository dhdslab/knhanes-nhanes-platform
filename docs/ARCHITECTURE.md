# Architecture

The platform keeps three things apart: deterministic computation, language-model assistance,
and human scientific review.

## 1. Deterministic statistical core

All computation lives here. `factory_core.py` loads the raw surveys, derives and harmonizes
the variables, builds the analytic sample of each model, and calls R for the survey
statistics. **No language model computes or alters a number.**

- **Association.** `engine.R` fits survey-weighted generalized linear models (`svyglm`):
  quasibinomial with logit link for binary outcomes (odds ratios), Gaussian for continuous
  outcomes (β per SD). Multiplicity is controlled with the Benjamini–Hochberg false
  discovery rate across the exposure-by-outcome matrix of each survey.
- **Trend.** `trend_report.py` computes survey-weighted prevalence per cycle, crude and
  directly age–sex standardized. `trend.R` then fits two models:
  - a log-linear annual percentage change. With at least five cycles it searches for one
    change point and gives each of the two segments a percentile bootstrap CI (2,000
    resamples); with fewer cycles it fits one slope.
  - a negative-binomial count model (`MASS::glm.nb`, Poisson fallback) for the forecast.
- **Machine learning.** `ml_report.py` holds a 16-classifier registry, each classifier in an
  impute → scale → classify pipeline. Tuning uses randomized search over stratified
  cross-validation on the training partition, and so does the choice between models. The
  held-out partition is scored once. Reports include SHAP and partial-dependence
  explanations.
- **Causal inference** for a single exposure: `epi_report.py` with `epi.R`, `epi_adv.R` and
  `epi_surv.R` (IPTW, PSM, AIPW, G-computation, TMLE, E-value, Love plot, VIF, restricted
  cubic splines, survival).

## 2. Language-model agent layer

`agents.py` implements the judgement and prose tasks on one self-hosted model, Llama 3.3
70B (`llama3.3:70b`) served through Ollama. See [AGENTS.md](AGENTS.md).

## 3. Human in the loop

Scientific choices go to a human and are recorded in the decision ledger (`DecisionLog`,
written to `decision_log.json`). Whether to harmonize obesity to one threshold or keep
population-specific thresholds is one example. Accepted changes to a definition or a
configuration trigger deterministic recomputation.

## Inter-process communication is file-based

Python writes an analytic table (`engine_analytic.csv`) and a JSON configuration
(`engine_config.json`), runs `Rscript engine.R` with `subprocess`, and reads back a results
table (`engine_results.csv`). There is no in-process bridge such as rpy2. Nothing fails
silently, the platform runs on Windows without extra set-up, and every intermediate file
persists on disk as an audit trail.

## One run, end to end

`pipeline.py` runs one survey:

```
load_raw -> harmonized definitions -> agent audit (guideline + definition)
         -> run_matrix (R survey)  -> batch QC (deterministic detection + agent summary)
         -> report writer          -> manuscript .docx + results CSV + decision ledger
```

The released two-survey corpus was produced by `suppl_generator.py` and the follow-up
scripts described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
