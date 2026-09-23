# Operational definitions

All definitions live in `factory_core.py`. They were audited against the raw codebook labels,
the official survey-derived variables and current clinical guidance. The complete catalogue,
with thresholds, medication variables, fasting requirements and cross-national flags, is
Supplementary Appendix S2 of the manuscript.

## Principles

- **Missing is not absence.** Every clinical binary outcome is built with `_mk(pos, obs)`. A
  participant who does not meet the outcome's observation rule is missing and excluded,
  never coded zero. The rule depends on the outcome:
  - For **conjunctive definitions** such as metabolic syndrome, a non-case requires every
    component to be observed.
  - For **disjunctive definitions**, including diabetes, hypertension, dyslipidaemia, the
    kidney phenotype and osteoporosis, a participant is classified once any defining
    component is observed. A participant whose observed components are negative is a
    non-case even if another component is missing.

  Partial component missingness can therefore still understate the prevalence of those
  composites.
- **Measured biomarkers and current medication only.** Lifetime self-reported diagnoses are
  not used as exposures or outcomes. The KNHANES examination-day medication items
  (`HE_DMdr`, `HE_HPdr`) are not used as current treatment.
- **Population-appropriate thresholds.** Obesity and overweight use Korean Society for the
  Study of Obesity cut-points in KNHANES and WHO cut-points in NHANES. The asymmetry is
  intentional and documented.
- **Leakage and circularity are blocked.** A pair is refused in three cases:
  - the variables share a measurement family;
  - one is a component of the other over a transitively closed component graph (`_DET`); or
  - a derived index shares a direct input with the outcome. For example, the Hepatic
    Steatosis Index contains BMI, so it is not tested against obesity.

  Age and sex are confounders, not sources of circularity. For prediction only, the
  NHANES lipid panel is withheld from the high-triglyceride model, because Friedewald LDL
  makes triglycerides an exact function of total cholesterol, HDL and LDL
  (`_ML_IDENTITY_EXCL`).
- **Event floor 50.** Outcomes with fewer than `MIN_EVENTS` events are blocked per survey.
- **Skew control.** HOMA-IR and fasting insulin are natural-log transformed before
  standardization (`LOG_EXPOSURES`). Their odds ratio is per SD of the log.
- **Fasting.** Fasting analytes require at least 8 hours of fasting.

## Representative definitions

| Outcome | Definition | Guideline / check |
|---|---|---|
| Diabetes | FPG ≥ 126 mg/dL, or HbA1c ≥ 6.5%, or current glucose-lowering medication | ADA; compared with `HE_DM_HbA1c` |
| Hypertension | SBP ≥ 140 or DBP ≥ 90 mmHg, or current antihypertensive | 2023 ESH; compared with `HE_HP` |
| Metabolic syndrome | Harmonized NCEP ATP III, 3 of 5, population-specific waist; all five components measured | Harmonized NCEP |
| Dyslipidaemia | TC ≥ 240, HDL < 40 (men) / < 50 (women), TG ≥ 200 or LDL ≥ 160 mg/dL, or current lipid-lowering medication | KSoLA composite |
| Kidney phenotype | eGFR < 60 (2021 race-free CKD-EPI creatinine) or ACR ≥ 30 mg/g; single examination, chronicity not established | KDIGO thresholds; ACR compared with `URDACT` (r = 1.0) |
| MASLD | Steatosis, at least one cardiometabolic risk factor, alcohol below the MetALD threshold | 2023 multisociety nomenclature |
| Obesity / overweight | KNHANES BMI ≥ 25 / ≥ 23 (KSSO); NHANES ≥ 30 / ≥ 25 (WHO) | Intentional asymmetry |
| Hepatic steatosis | KNHANES HSI > 36; NHANES CAP ≥ 248 dB/m (2017–2018 only; the 2021–2023 elastography file is not incorporated) | Intentional asymmetry |

LDL is measured directly in KNHANES (`HE_LDL_drct`). NHANES releases the Friedewald value
(`LBDLDL`), which is void when TG > 400 mg/dL.

## Error classes the audit detected and corrected

From the manuscript. Prevalences are before and after correction.

| Error class | Example | Before → after | External check |
|---|---|---|---|
| Missing component treated as negative | Metabolic syndrome components | NHANES 24.9% → 37.3% | Literature about 34.7% |
| Exam-day drug proxy used as current treatment | KNHANES hypertension / diabetes | HTN 15.4 → 27.4%, DM 9.95 → 11.6% | `HE_HP` 25.2%, `HE_DM_HbA1c` 13.5% |
| Threshold not aligned with Korean clinical guidance | KNHANES obesity (BMI ≥ 30 vs ≥ 25) | 5.6 → 34.6% | KSSO-threshold national estimate about 35% |
| Incomplete cross-sectional kidney phenotype | eGFR only vs eGFR or ACR | KN 1.9 → 7.9%, NH 5.7 → 13.1% | ACR vs `URDACT` r = 1.0 |
| Silent cycle-variable rename | NHANES cycle L lipids and drugs | cycle-L HTN 17.9 → 35.1% | Adjacent cycles |
| Definitional leakage / circularity | Non-HDL cholesterol → dyslipidaemia; HSI → obesity | Pairs blocked before fitting | Transitive-closure and derived-index guards |
| Scale-induced extreme odds ratio | HOMA-IR → central obesity | NHANES OR 1,013 → 4.7 | Natural-log transform before standardization |
| Assay calibration break pooled across cycles | Serum creatinine (IDMS break) | Early cycles dropped from eGFR-derived phenotype | Removes a spurious 2007 spike |
| Reference population / age-range mismatch | Osteoporosis | ISCD reference applied, age ≥ 50 | Densitometric definition disclosed |
