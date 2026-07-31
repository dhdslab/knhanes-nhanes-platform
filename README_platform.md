# KNHANES + NHANES Research Automation Platform (final)

Select variables and definitions in the web UI; survey-weighted analysis runs internally and
**epidemiology, trend, and ML reports are auto-generated as Word documents**.

## Files (all in the same folder)
- `factory_app.py` -- Streamlit web app (entry point)
- `factory_core.py` -- logic: loaders, preprocessing definitions, unified variables, auto-covariates, survey-weighted analysis, Trajectory, LLM, HTML
- `engine.R` -- Table 1 + associations (continuous = linear beta / binary = logistic OR)
- `epi_report.py` + `epi.R` + `epi_adv.R` -- epidemiology report (Tables 1-6, causal inference, Love, VIF, RCS, E-value)
- `trend_report.py` + `trend.R` -- trend report (standardized rate, projection, APC, NB forecast)
- `ml_report.py` -- ML report (tuning, full metrics, ROC/PR, threshold, calibration, SHAP, PDP)

## Install
```bash
pip install streamlit pandas numpy pyreadstat python-docx requests matplotlib scikit-learn shap xgboost
sudo apt-get install -y r-base r-cran-survey r-cran-jsonlite r-cran-rms r-cran-mass r-cran-sandwich
ollama pull llama3.2:latest && ollama serve  # optional. Without it, prose falls back to deterministic text
```

## Data
Put `hn08_all.sas7bdat` / `hn08_dxa.sas7bdat` (per year) in `data/KNHANES/`, and `demo_j.sas7bdat` etc. in `data/NHANES/`.
The folder path can be changed in the app sidebar.

## Run
```bash
streamlit run factory_app.py
```
On a remote server with blocked ports (e.g., a hospital network), open an SSH tunnel (`ssh -L 8501:localhost:8501 ...`) and browse to `http://localhost:8501`.

## Tabs
1. **Interactive analysis** -- Exposure/Outcome checkboxes (all variables), automatic adjustment, preprocessing definitions -> Table 1 + associations + manuscript Word
2. **Trajectory** -- age trajectory of outcome determinants
3. **Epidemiology report** -- outcome and main exposure -> Table 1-6 (SMD, crude/adj OR, subgroups+P-int, IPTW/AIPW/G-comp, Love, VIF, RCS, E-value) Word
4. **Trend report** -- outcome and years -> standardized rate, projection, APC, NB forecast Word (2 or more cycles)
5. **ML report** -- outcome and features -> model tuning, full metrics, ROC/PR, threshold, calibration, SHAP, PDP Word

## Operational definitions (journal standard)
- Diabetes: FPG>=126 or HbA1c>=6.5% or medication or diagnosis
- Hypertension: SBP>=140 or DBP>=90 or antihypertensive medication
- Metabolic syndrome: harmonized NCEP ATP III (3 of 5)
- Dyslipidemia: TC>=240 or HDL<40 or TG>=200
- Hepatic steatosis: NAFLD-LFS>-0.640 (default) or HSI>36 / NHANES uses CAP>=248
- Smoking/alcohol: never vs past/current
Covariates automatically exclude the exposure, outcome, definitional components, and the anthropometry cluster (to prevent overadjustment).

## Principles
All statistics are R survey design-weighted (deterministic). The LLM (default llama3.2:latest) writes only variable mappings and report prose; it never recomputes numbers.
All findings are hypothesis-generating and require external validation.
