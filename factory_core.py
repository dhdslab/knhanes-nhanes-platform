# -*- coding: utf-8 -*-
"""Evidence Factory core v4 (KNHANES + NHANES, harmonized)

Design principles enforced here (see README):
1. Smoking/alcohol are BINARY never vs past/current (ever). Groups are survey-weighted.
2. Any exposure x any outcome (all-pairs) is supported for prediction/association.
3. Common outcome set = only outcomes definable IDENTICALLY in BOTH surveys. Country-only
   outcomes (COPD in KNHANES, MDD/PHQ-9 in NHANES) are kept but flagged per dataset.
4. MDD, MASLD, COPD, CKD, advanced liver fibrosis, DM are all available as outcomes.
5. Only cross-sectional (prevalent) outcomes here. Survival/incidence lives in epi_report.
6. No lifetime self-report disease history as exposure/outcome. Clinical outcomes use MEASURED
   biomarkers + CURRENT medication only (not "ever diagnosed" self-report).
7. Vague "other disease" self-reports are excluded (never entered into the registry).
8. Outcomes with too few events (< MIN_EVENTS) in the analytic sample are blocked at run time.
9. Continuous variables have IDENTICAL categorical binnings across both surveys (see CATEG).

All statistics remain deterministic survey-weighted estimates (R survey). The LLM only writes
report prose in English and never recomputes numbers. Findings are hypothesis-generating.
"""
import pyreadstat, pandas as pd, numpy as np, glob, json, subprocess, os, re

MIN_EVENTS = 50   # req #8: block an outcome if fewer than this many events in analytic sample

# Continuous exposures so heavily right-skewed (skewness >10 in pooled data) that a per-1-SD odds
# ratio on the raw scale is uninterpretable — empirically the only exposures that produced OR>50
# (HOMA-IR reached OR>1000). These are natural-log transformed BEFORE z-standardization, so their OR
# is per 1-SD of the log. Every other exposure stays on its raw scale (unchanged).
LOG_EXPOSURES = {"homa_ir", "insulin"}

# ── Unified variable registry (type: c=continuous, b=binary) ─────────────────────────
VARS = {
 # continuous predictors / continuous outcomes (linear beta)
 "age":("Age, years","c"), "bmi":("Body mass index, kg/m2","c"), "wc":("Waist circumference, cm","c"),
 "sbp":("Systolic blood pressure, mmHg","c"), "dbp":("Diastolic blood pressure, mmHg","c"),
 "glucose":("Fasting glucose, mg/dL","c"), "hba1c":("HbA1c, %","c"), "tg":("Triglycerides, mg/dL","c"),
 "tchol":("Total cholesterol, mg/dL","c"), "hdl":("HDL cholesterol, mg/dL","c"),
 "ldl":("LDL cholesterol, mg/dL","c"), "alt":("Alanine aminotransferase, IU/L","c"),
 "ast":("Aspartate aminotransferase, IU/L","c"), "ggt":("Gamma-glutamyl transferase, IU/L","c"),
 "insulin":("Fasting insulin, uIU/mL","c"), "creatinine":("Serum creatinine, mg/dL","c"),
 "egfr":("eGFR (CKD-EPI 2021), mL/min/1.73m2","c"), "hemoglobin":("Hemoglobin, g/dL","c"),
 "platelet":("Platelet count, 10^3/uL","c"), "uric":("Serum uric acid, mg/dL","c"),
 "fib4":("FIB-4 index","c"), "hsi":("Hepatic steatosis index (HSI)","c"),
 "cap":("Controlled attenuation parameter, dB/m","c"), "lsm":("Liver stiffness, kPa","c"),
 "fat_kg":("Total fat mass, kg","c"), "lean_kg":("Total lean mass, kg","c"),
 "asmm_kg":("Appendicular skeletal muscle mass, kg","c"), "bodyfat_pct":("Body fat, %","c"),
 "asm_pct":("Appendicular skeletal muscle, %","c"),
 "height":("Height, cm","c"), "weight":("Weight, kg","c"),
 "whtr":("Waist-to-height ratio","c"), "asmi":("ASM / height^2, kg/m2","c"),
 "nonhdl":("Non-HDL cholesterol, mg/dL","c"), "homa_ir":("HOMA-IR","c"),
 "vitd":("Serum 25(OH)D, ng/mL","c"), "wbc":("White blood cell count, 10^3/uL","c"),
 "fn_bmd":("Femoral neck BMD, g/cm2","c"), "ls_bmd":("Lumbar spine BMD, g/cm2","c"),
 # binary exposures
 "men":("Male sex","b"), "smoking":("Ever smoking (past/current)","b"),
 "current_smoking":("Current smoking","b"), "alcohol":("Ever alcohol (past/current)","b"),
 # binary clinical OUTCOMES (cross-sectional, measured + current medication)
 "dm":("Diabetes mellitus","b"), "prediabetes":("Prediabetes","b"), "htn":("Hypertension","b"),
 "obesity":("Obesity (KSSO >=25 / WHO >=30, country-specific)","b"), "overweight":("Overweight (KSSO >=23 / WHO >=25, country-specific)","b"),
 "abdominal_obesity":("Abdominal obesity","b"), "mets":("Metabolic syndrome","b"),
 "dyslipidemia":("Dyslipidemia (KSoLA composite)","b"), "low_hdl":("Low HDL cholesterol","b"),
 "high_tg":("Hypertriglyceridemia (NCEP TG>=150)","b"), "ckd":("Chronic kidney disease (KDIGO: eGFR<60 or ACR>=30)","b"),
 "adv_fibrosis":("Advanced liver fibrosis (FIB-4>2.67)","b"), "anemia":("Anemia (WHO)","b"),
 "steatosis":("Hepatic steatosis","b"), "masld":("MASLD","b"),
 # additional harmonized binary outcomes (defined identically in both surveys)
 "high_bodyfat":("High body fat (M>=25/F>=35%)","b"), "central_obesity":("Central obesity (WHtR>=0.5)","b"),
 "sarcopenia":("Low muscle mass (low ASMI, AWGS cut)","b"),
 "acr":("Urine albumin-to-creatinine ratio, mg/g","c"),
 "high_alt":("Elevated ALT (>40 IU/L)","b"), "high_ast":("Elevated AST (>40 IU/L)","b"),
 "high_ldl":("High LDL cholesterol (>=160)","b"), "high_nonhdl":("High non-HDL cholesterol (>=190)","b"),
 "atherogenic_dyslipidemia":("Atherogenic dyslipidemia (high TG + low HDL)","b"),
 "insulin_resistance":("Insulin resistance (HOMA-IR>=2.5)","b"),
 "vitd_deficiency":("Vitamin D deficiency (<20 ng/mL)","b"),
 "osteoporosis":("Osteoporosis (T-score<=-2.5, age>=50)","b"),
 "leukocytosis":("Leukocytosis (WBC>11)","b"),
}
def lab(c): return VARS[c][0] if c in VARS else c
def typ(c): return VARS[c][1] if c in VARS else "c"

# ── Ordered binary outcome list (clinical outcomes offered in outcome menus) ──────────
OUTCOMES = ["dm","prediabetes","htn","obesity","overweight","abdominal_obesity","central_obesity",
            "high_bodyfat","sarcopenia","mets","dyslipidemia","low_hdl","high_tg",
            "high_ldl","high_nonhdl","atherogenic_dyslipidemia","insulin_resistance","ckd","high_alt",
            "high_ast","adv_fibrosis","steatosis","masld","anemia","leukocytosis","vitd_deficiency",
            "osteoporosis"]   # sarcopenic_obesity removed (near-empty/artifactual: ASM/ht2 anti-correlates with fat)

# ── Per-dataset derivable variables (with current data + downloaded modules) ──────────
# Every outcome below is defined identically in BOTH surveys (steatosis/MASLD use the best
# per-country measure: KNHANES HSI vs NHANES CAP).
_COMMON_OUT = list(OUTCOMES)
_COMMON_CONT = ["age","bmi","wc","height","weight","whtr","sbp","dbp","glucose","hba1c","homa_ir","tg",
 "tchol","hdl","ldl","nonhdl","alt","ast","creatinine","egfr","hemoglobin","platelet","wbc","fib4",
 "vitd","fat_kg","lean_kg","asmm_kg","asmi","bodyfat_pct","asm_pct","fn_bmd","ls_bmd"]
_KN = _COMMON_CONT + ["insulin","hsi","men","smoking","current_smoking","alcohol"] + _COMMON_OUT
_NH = _COMMON_CONT + ["insulin","ggt","uric","cap","lsm","men","smoking","current_smoking","alcohol"] + _COMMON_OUT
AVAIL = {"KNHANES": _KN, "NHANES": _NH}
DATASETS = ["KNHANES","NHANES"]
STEATOSIS_METHOD = {"KNHANES":"HSI > 36 (surrogate index)", "NHANES":"CAP >= 248 dB/m (transient elastography)"}

# ── Outcome definitional components (leakage / circularity guard) ─────────────────────
_DET = {
 "dm":["glucose","hba1c"], "prediabetes":["glucose","hba1c"], "htn":["sbp","dbp"],
 "obesity":["bmi"], "overweight":["bmi"], "abdominal_obesity":["wc"],
 "mets":["wc","glucose","sbp","dbp","tg","hdl"],
 "dyslipidemia":["tchol","hdl","tg","ldl"], "low_hdl":["hdl"], "high_tg":["tg"],
 "ckd":["creatinine","egfr","acr"], "adv_fibrosis":["ast","alt","platelet","fib4"],
 "anemia":["hemoglobin","men"],
 "masld":["steatosis","cap","hsi","lsm","alt","ast","wc","glucose","sbp","dbp","tg","hdl","bmi"],
 "high_bodyfat":["bodyfat_pct","fat_kg"], "central_obesity":["wc","whtr","height"],
 "sarcopenia":["asmi","asmm_kg","asm_pct","lean_kg","height"],
 "high_alt":["alt"], "high_ast":["ast"], "high_ldl":["ldl"],
 "high_nonhdl":["nonhdl","tchol","hdl"], "atherogenic_dyslipidemia":["tg","hdl"],
 "insulin_resistance":["homa_ir","insulin","glucose"], "vitd_deficiency":["vitd"],
 "osteoporosis":["fn_bmd","ls_bmd"], "leukocytosis":["wbc"],
 # derived continuous variables keyed so the transitive closure blocks their raw components too
 "egfr":["creatinine","age","men"], "fib4":["age","ast","alt","platelet"],
 "hsi":["alt","ast","bmi","men","dm"], "nonhdl":["tchol","hdl"], "whtr":["wc","height"],
 "asmi":["asmm_kg","height"], "homa_ir":["glucose","insulin"], "bmi":["weight","height"],
 "bodyfat_pct":["fat_kg","lean_kg"], "asm_pct":["asmm_kg","fat_kg","lean_kg"], "acr":["ualb","ucrea"],
}
def _det_raw(outcome):
    if outcome == "steatosis":
        # KNHANES HSI = 8*(ALT/AST) + BMI + sex + DM, so those are definitionally inside HSI-steatosis;
        # NHANES steatosis is CAP-based. MASLD contains steatosis by definition on both sides.
        return ["cap","lsm","masld"], ["hsi","alt","ast","bmi","men","dm","masld"]  # (nhanes, knhanes)
    return _DET.get(outcome,[])
def determinants(dataset, outcome):
    """Transitive closure of an outcome's definitional components (so A<-B<-C is blocked, and derived
    indices like eGFR/FIB-4/HSI/non-HDL drag in their raw inputs). Filtered to variables in this dataset."""
    seen=set();
    start=_det_raw(outcome)
    stack=list(start[0] if (outcome=="steatosis" and dataset=="NHANES") else
               (start[1] if outcome=="steatosis" else start))
    while stack:
        v=stack.pop()
        if v in seen: continue
        seen.add(v); stack.extend(_DET.get(v,[]))
    return [v for v in seen if v in AVAIL.get(dataset,[])]
def _direct_det(dataset, v):
    """Direct definitional inputs of v (NO transitive closure) — for the shared-component circularity
    test, so a distant transitive link (e.g. steatosis inheriting MASLD's cardiometabolic inputs) does
    not spuriously block legitimate pairs like insulin-resistance vs steatosis."""
    if v=="steatosis":
        return set(["cap","lsm","masld"] if dataset=="NHANES" else ["hsi","alt","ast","bmi","men","dm","masld"])
    return set(_DET.get(v,[]))

# Variables that measure the SAME underlying construct with correlated metrics. Cross-testing within a
# family is near-tautological (quasi-complete separation, e.g., current vs ever smoking, BMI-obesity vs
# WHtR-obesity, LDL vs non-HDL) so such pairs are not tested.
_FAMILIES = [
 {"smoking","current_smoking"},
 {"bmi","wc","whtr","weight","height","fat_kg","bodyfat_pct","lean_kg","asmm_kg","asmi","asm_pct",
  "obesity","overweight","abdominal_obesity","central_obesity","high_bodyfat","sarcopenia"},
 {"tchol","ldl","nonhdl","hdl","high_ldl","high_nonhdl","low_hdl","dyslipidemia"},
 {"alt","ast","high_alt","high_ast","fib4","platelet","hsi","adv_fibrosis"},
 {"glucose","hba1c","insulin","homa_ir","dm","prediabetes","insulin_resistance"},
 {"creatinine","egfr","acr","ckd"},
]
def _same_family(a,b):
    return any(a in F and b in F for F in _FAMILIES)

def related(dataset, exposure, outcome):
    """True if an exposure-outcome pair is definitionally circular / near-tautological and must not be
    tested: same variable, one is a defining component of the other, both clinical outcomes sharing a
    defining component, or both belong to the same measurement family (see _FAMILIES). Legitimate
    comorbidity pairs (diabetes vs dyslipidemia, obesity vs anemia) are kept."""
    if exposure==outcome: return True
    if _same_family(exposure,outcome): return True
    if exposure in determinants(dataset,outcome): return True
    if outcome in determinants(dataset,exposure): return True
    # (a) two clinical outcomes sharing a defining component (transitive) — unchanged original behavior.
    if exposure in OUTCOMES and (set(determinants(dataset,exposure)) & set(determinants(dataset,outcome))):
        return True
    # (b) a derived continuous index (HSI, FIB-4, non-HDL, WHtR, HOMA-IR...) as exposure is circular with
    # an outcome sharing one of its DIRECT substantive inputs — HSI embeds BMI, so HSI vs a BMI-threshold
    # outcome is tautological. Direct inputs only (no transitive reach, so insulin-resistance vs steatosis
    # is kept), and age/sex demographics ignored (eGFR vs anemia kept).
    if exposure in _DET and exposure not in OUTCOMES:
        if (_direct_det(dataset,exposure) & _direct_det(dataset,outcome)) - {"age","men"}:
            return True
    return False

# Roles are symmetric: any variable can be an exposure OR an outcome. Binary outcome -> logistic OR,
# continuous outcome -> linear beta. related() still blocks definitionally circular pairs either way.
def exposure_candidates(dataset):
    return list(AVAIL.get(dataset,[]))
def outcome_candidates(dataset):
    av=AVAIL.get(dataset,[])
    clin=[v for v in OUTCOMES if v in av]                       # clinical binary outcomes first
    # sex ("men") is excluded as an outcome: "exposure -> male sex" is not a meaningful target and
    # sex dimorphism produces quasi-separation OR (e.g. muscle mass -> male sex, OR>200). Sex remains
    # available as an EXPOSURE. ever-smoking / ever-alcohol are kept as behavioral outcomes.
    other_b=[v for v in av if typ(v)=="b" and v not in clin and v!="men"]
    cont=[v for v in av if typ(v)=="c"]                         # any continuous var as a linear-beta outcome
    return clin+other_b+cont

# ═══ Per-outcome, literature-informed specification (association adjustment + prediction) ══════════
# For each clinical outcome: cf = confounder set adjusted for in association models (minimal sufficient
# adjustment from standard epidemiology); extra_excl = variables never used as an exposure/ML feature
# beyond the automatic definitional/family exclusions (mediators or over-influential proxies);
# rat = short methods rationale printed in every report.
_L="age, sex, BMI, smoking and alcohol"
OUTCOME_SPEC = {
 "dm":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":["mets"],
       "rat":f"Adjusted for {_L}. Glycemic and insulin-resistance measures are excluded as definitional or mediating."},
 "prediabetes":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":["mets"],
       "rat":f"Adjusted for {_L}. Glycemic and insulin-resistance measures are excluded as definitional or mediating."},
 "insulin_resistance":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":["mets"],
       "rat":f"Adjusted for {_L}. Glycemic measures are excluded as definitional or mediating."},
 "htn":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],
       "rat":f"Adjusted for {_L}."},
 "obesity":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],
       "rat":"Adjusted for age, sex, smoking and alcohol. Adiposity measures are the exposure family and are not used for adjustment."},
 "overweight":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],
       "rat":"Adjusted for age, sex, smoking and alcohol. Adiposity measures are not used for adjustment."},
 "abdominal_obesity":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],
       "rat":"Adjusted for age, sex, smoking and alcohol. Adiposity measures are not used for adjustment."},
 "central_obesity":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],
       "rat":"Adjusted for age, sex, smoking and alcohol. Adiposity measures are not used for adjustment."},
 "high_bodyfat":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],
       "rat":"Adjusted for age, sex, smoking and alcohol. Adiposity measures are not used for adjustment."},
 "sarcopenia":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],
       "rat":f"Adjusted for {_L}. Appendicular muscle measures are the outcome family and are not used for adjustment."},
 "mets":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],
       "rat":"Adjusted for age, sex, smoking and alcohol. The five metabolic-syndrome components are excluded as definitional."},
 "dyslipidemia":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "low_hdl":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "high_tg":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "high_ldl":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "high_nonhdl":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "atherogenic_dyslipidemia":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "ckd":{"cf":["age","men","bmi","smoking","alcohol","dm","htn"],"extra_excl":[],
       "rat":"Adjusted for age, sex, BMI, smoking, alcohol, diabetes and hypertension (major kidney-disease risk factors)."},
 "high_alt":{"cf":["age","men","bmi","alcohol","smoking"],"extra_excl":[],
       "rat":"Adjusted for age, sex, BMI, alcohol and smoking (alcohol is a key hepatic confounder)."},
 "high_ast":{"cf":["age","men","bmi","alcohol","smoking"],"extra_excl":[],
       "rat":"Adjusted for age, sex, BMI, alcohol and smoking (alcohol is a key hepatic confounder)."},
 "adv_fibrosis":{"cf":["age","men","bmi","alcohol","smoking","dm"],"extra_excl":[],
       "rat":"Adjusted for age, sex, BMI, alcohol, smoking and diabetes."},
 "steatosis":{"cf":["age","men","alcohol","smoking"],"extra_excl":[],
       "rat":"Adjusted for age, sex, alcohol and smoking. BMI is part of the steatosis construct and is not used for adjustment."},
 "masld":{"cf":["age","men","alcohol","smoking"],"extra_excl":[],
       "rat":"Adjusted for age, sex, alcohol and smoking. Cardiometabolic components are part of the MASLD construct and are not used for adjustment."},
 "anemia":{"cf":["age","men","smoking","alcohol"],"extra_excl":[],"rat":"Adjusted for age, sex, smoking and alcohol."},
 "leukocytosis":{"cf":["age","men","smoking","alcohol","bmi"],"extra_excl":[],
       "rat":f"Adjusted for {_L} (smoking raises the white-cell count)."},
 "vitd_deficiency":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
 "osteoporosis":{"cf":["age","men","bmi","smoking","alcohol"],"extra_excl":[],"rat":f"Adjusted for {_L}."},
}
def outcome_rationale(outcome):
    sp=OUTCOME_SPEC.get(outcome)
    return sp["rat"] if sp else "Adjusted for age, sex, smoking, alcohol and (where not collinear) BMI."

def confounders_for(dataset, outcome):
    """Curated confounder whitelist available in this dataset. Confounding is not leakage: we adjust for
    these even when they are correlated with the outcome's definition (e.g. age for CKD, whose eGFR
    formula contains age). Only the outcome itself is dropped. Exposure-side circularity is handled
    separately by related()/determinants(); per-exposure over-adjustment by adjustment_for()."""
    sp=OUTCOME_SPEC.get(outcome)
    cf=sp["cf"] if sp else (["age","men","smoking","alcohol"]+(["bmi"] if outcome not in ADIPOSITY else []))
    return [c for c in cf if c in AVAIL.get(dataset,[]) and c!=outcome]

def _extra_excl(outcome):
    return set(OUTCOME_SPEC.get(outcome,{}).get("extra_excl",[]))

def association_exposures(dataset, outcome):
    """Every candidate exposure for this outcome (drops only definitional/family variables and
    over-influential proxies). Confounders are included and each is adjusted for the others."""
    ex=_extra_excl(outcome)
    return [v for v in exposure_candidates(dataset)
            if not related(dataset,v,outcome) and v not in ex]

def adjustment_for(dataset, outcome, exposure):
    """Covariates for one exposure: the outcome confounder set minus the exposure itself and any
    confounder in the SAME family as the exposure (prevents self- and over-adjustment)."""
    return [c for c in confounders_for(dataset,outcome)
            if c!=exposure and not _same_family(c,exposure)]

# Survey-specific algebraic identities among the released variables.
#
# NHANES publishes LBDLDL as the Friedewald CALCULATED LDL, so TC - HDL - LDL = TG/5
# holds exactly: over the 19,899 adult rows carrying all four lipids the residual has
# median 0.000 and SD 0.28 mg/dL and 100% of rows fall within 1 mg/dL, in every cycle.
# Reconstructing TG as 5*(TC - HDL - LDL) correlates 0.9998 with the measured value and
# reproduces the TG >= 150 label for 99.2% of participants. A classifier handed TC, HDL
# and LDL together therefore COMPUTES a triglyceride threshold rather than predicting
# one, and the symptom is unmistakable: penalized logistic regression beat both
# gradient-boosted ensembles at auROC 0.997, because the relation it has to find is a
# linear identity. Non-HDL closes the same algebra with LDL, so it goes too.
#
# KNHANES measures LDL directly (HE_LDL_drct); the same residual there has median -7.0
# and SD 13.1 and only 5.6% of rows fall within 1 mg/dL, so no identity exists and its
# high-triglyceride model is left alone.
#
# This is an ML-only guard. The association engine admits one exposure at a time against
# a fixed covariate set, and the identity needs three lipids simultaneously, so it cannot
# close there; the association corpus is therefore unchanged.
_ML_IDENTITY_EXCL = {
    ("NHANES", "high_tg"): {"tchol", "hdl", "ldl", "nonhdl"},
}


def ml_features(dataset, outcome):
    """Predictors for an ML model of this outcome: everything except the outcome, its definitional/
    family variables, over-influential proxies, and any variable that closes an algebraic identity
    with the outcome in this survey (leakage prevention). Confounders are kept."""
    ex=_extra_excl(outcome)|_ML_IDENTITY_EXCL.get((dataset,outcome),set())
    return [v for v in AVAIL.get(dataset,[])
            if v!=outcome and not related(dataset,v,outcome) and v not in ex]

# body-size / composition cluster (avoid mutual over-adjustment)
ADIPOSITY = {"bmi","wc","fat_kg","lean_kg","asmm_kg","bodyfat_pct","asm_pct"}

def auto_covariates(dataset, exposures, outcomes):
    cov = ["age","men"]
    for c in ["smoking","alcohol"]:
        if c in AVAIL[dataset]: cov.append(c)
    if "bmi" in AVAIL[dataset] and not any(e in ADIPOSITY for e in exposures):
        cov.append("bmi")
    excl = set(exposures) | set(outcomes)
    for o in outcomes: excl |= set(determinants(dataset,o))
    if any(e in ADIPOSITY for e in exposures): excl |= ADIPOSITY
    return [c for c in cov if c in AVAIL[dataset] and c not in excl]

DEFAULT_DEFS = {
 "dm":{"FPG >=126 mg/dL":True,"HbA1c >=6.5%":True,"Current glucose-lowering medication":True},
 "htn":{"SBP >=140 or DBP >=90 mmHg":True,"Current antihypertensive":True},
 "mets":{"Harmonized NCEP ATP III (3 of 5)":True},
 "dld":{"TC >=240 mg/dL":True,"HDL low (M<40/F<50)":True,"TG >=200 mg/dL":True,"LDL >=160 mg/dL":True},
 "steatosis":{"KNHANES HSI>36 / NHANES CAP>=248":True},
 "mdd":{"PHQ-9 >=10":True},
 "pop":{"age_min":20}}

# ═══ Categorical binnings — IDENTICAL cutpoints applied to BOTH surveys (req #9) ═══════
CATEG = {
 "agegrp":  ("age",   [0,30,40,50,60,70,200], ["<30","30-39","40-49","50-59","60-69","70+"]),
 "bmicat":  ("bmi",   [0,18.5,23,25,30,100],  ["Underweight","Normal","Overweight (Asian)","Pre-obese","Obese"]),
 "bpcat":   ("sbp",   [0,120,130,140,300],    ["Normal","Elevated","Stage 1","Stage 2"]),  # by SBP (ACC/AHA-like)
 "glucat":  ("glucose",[0,100,126,1000],      ["Normoglycemia","Prediabetes","Diabetes range"]),
 "a1ccat":  ("hba1c", [0,5.7,6.5,30],         ["<5.7","5.7-6.4",">=6.5"]),
 "tgcat":   ("tg",    [0,150,200,10000],      ["Normal","Borderline","High"]),
}
CAT_LABELS = {
 "agegrp":"Age group","bmicat":"BMI category","bpcat":"Blood pressure category",
 "glucat":"Glycemic category","a1ccat":"HbA1c category","tgcat":"Triglyceride category",
 "smk3":"Smoking status","edu3":"Education","inc3":"Income (relative)","alc2":"Alcohol use",
}
def categorize(d):
    """Add IDENTICAL categorical columns to a derived dataframe (both surveys)."""
    for name,(src,bins,labels) in CATEG.items():
        if src in d.columns:
            d[name] = pd.cut(pd.to_numeric(d[src],errors="coerce"), bins=bins, labels=labels, right=False)
    # smoking status 3-level: never / former / current (identical concept)
    if "smoking" in d.columns and "current_smoking" in d.columns:
        sm=pd.to_numeric(d["smoking"],errors="coerce"); cur=pd.to_numeric(d["current_smoking"],errors="coerce")
        smk=np.select([sm==0,(sm==1)&(cur==1),(sm==1)&(cur==0)],["Never","Current","Former"],default="")
        d["smk3"]=pd.Series(smk,index=d.index).replace("",np.nan)
    if "alcohol" in d.columns:
        al=pd.to_numeric(d["alcohol"],errors="coerce")
        d["alc2"]=pd.Series(np.select([al==1,al==0],["Ever","Never"],default=""),index=d.index).replace("",np.nan)
    return d
def lab_cat(c): return CAT_LABELS.get(c, c)

# ── Loaders ───────────────────────────────────────────────────────────────────────────
def _find(dd, stem):
    for ext in (".sas7bdat",".xpt",".XPT",".Xpt"):
        h = glob.glob(os.path.join(dd,"**",stem+ext), recursive=True)
        if h: return h[0]
    return None
def _read(path, usecols=None):
    """Read SAS/XPT; retry alternate encodings on decode failure. usecols (when given) reads only those
    columns, which is far faster for the wide KNHANES all-files."""
    kw={} if usecols is None else {"usecols":usecols}
    if path.lower().endswith(".xpt"):
        for enc in (None,"latin1","cp1252","iso-8859-1"):
            try:
                return pyreadstat.read_xport(path,**kw) if enc is None else pyreadstat.read_xport(path,encoding=enc,**kw)
            except (UnicodeDecodeError, pyreadstat.ReadstatError):
                continue
        return pyreadstat.read_xport(path,encoding="latin1",**kw)
    errors = []
    for encoding in (None,"cp949","euc-kr","latin1"):
        try:
            if encoding is None: return pyreadstat.read_sas7bdat(path,**kw)
            return pyreadstat.read_sas7bdat(path, encoding=encoding,**kw)
        except pyreadstat.ReadstatError as exc:
            errors.append(f"{encoding or 'auto'}: {exc}")
    raise pyreadstat.ReadstatError(f"SAS file could not be decoded: {path}\n"+"\n".join(errors))

def _meta_cols(path):
    """Column names only (fast), for computing usecols before a full read."""
    for enc in ("latin1","cp949",None):
        try:
            kw={} if enc is None else {"encoding":enc}
            _,m=pyreadstat.read_sas7bdat(path,metadataonly=True,**kw); return list(m.column_names)
        except Exception: continue
    return None

def load_raw(dataset, data_dir, cycles):
    return _load_knhanes(data_dir,cycles) if dataset=="KNHANES" else _load_nhanes(data_dir,cycles)

def _load_knhanes(data_dir, years):
    ka=["id","kstrata","psu","wt_itvex","wt_tot","wt_ex1","age","sex","HE_BMI","HE_wc","HE_ht","HE_wt",
        "HE_glu","HE_HbA1c","HE_TG","HE_chol","HE_HDL_st2","HE_LDL_drct","HE_ast","HE_alt","HE_insulin",
        "HE_sbp","HE_dbp","HE_Bplt","HE_crea","HE_HB","HE_WBC","HE_vitD","HE_DMdr","HE_HPdr",
        "DE1_31","DE1_32","DI1_2","DI2_2","HE_Ualb","HE_Ucrea","HE_UCREA","HE_fst","BD1_11","BD2_1",
        "BS1_1","BS3_1","sm_presnt","BD1","edu","incm5","HE_prg","N_PRG"]
    kx=["id","DW_WBT_FT","DW_WBT_LN","DW_WBT_MS","DW_Llg_LN","DW_Rlg_LN","DW_Lrm_LN","DW_Rrm_LN",
        "DX_NK_BMD","DX_LSP_BMD"]
    frames=[]
    for ci,y in enumerate(years):
        fa,fx=_find(data_dir,f"hn{y}_all"),_find(data_dir,f"hn{y}_dxa")
        if not fa: continue
        mc=_meta_cols(fa) or []
        usea=[c for c in mc if c in ka or c.lower()=="id"]   # only needed columns (fast read)
        a,_=_read(fa, usecols=usea or None)
        a=a.rename(columns={c:"id" for c in a.columns if c.lower()=="id"})
        a=a[[c for c in a.columns if c in ka or c=="id"]].copy()
        if fx:   # merge DXA WITHIN the year (KNHANES id repeats across years, so never merge pooled)
            mcx=_meta_cols(fx) or []
            usex=[c for c in mcx if c in kx or c.lower()=="id"]
            x,_=_read(fx, usecols=usex or None); x=x.rename(columns={c:"id" for c in x.columns if c.lower()=="id"})
            x=x[[c for c in x.columns if c in kx or c=="id"]]
            if "id" in a.columns and "id" in x.columns:
                a=a.merge(x.drop_duplicates(subset="id"),on="id",how="left")
        a["_cyc"]=ci; a["_cyclab"]=str(y)   # _cyclab = survey-year label, load-position-independent
        frames.append(a)
    DF=pd.concat(frames,ignore_index=True); DF.attrs["dataset"]="KNHANES"; DF.attrs["n_cycles"]=len(frames); return DF

def _load_nhanes(data_dir, cycles):
    # module -> wanted columns. New modules (HDL/GHB/BPQ/DIQ/DPQ/ALB_CR) were downloaded to unlock
    # metabolic syndrome, low-HDL, HbA1c-based DM, and MDD (PHQ-9).
    spec={"demo":["SEQN","RIDAGEYR","RIAGENDR","SDMVSTRA","SDMVPSU","WTMEC2YR","DMDEDUC2","INDFMPIR","RIDEXPRG"],
          "bmx":["SEQN","BMXBMI","BMXWAIST","BMXHT","BMXWT"],
          "biopro":["SEQN","LBXSASSI","LBXSATSI","LBXSGTSI","LBXSCR","LBXSUA","LBXSCH"],
          "glu":["SEQN","LBXGLU","LBXIN"],"ghb":["SEQN","LBXGH"],"GHB":["SEQN","LBXGH"],
          "ins":["SEQN","LBXIN"],"INS":["SEQN","LBXIN"],
          "vid":["SEQN","LBXVIDMS","LBDVIDMS"],"VID":["SEQN","LBXVIDMS","LBDVIDMS"],
          "trigly":["SEQN","LBXTR","LBXTLG","LBDLDL"],"tchol":["SEQN","LBXTC"],
          "hdl":["SEQN","LBDHDD"],"HDL":["SEQN","LBDHDD"],
          "cbc":["SEQN","LBXHGB","LBXPLTSI","LBXWBCSI"],
          "lux":["SEQN","LUXCAPM","LUXSMED"],"bpx":["SEQN","BPXSY1","BPXDI1"],
          "bpxo":["SEQN","BPXOSY1","BPXODI1"],"BPXO":["SEQN","BPXOSY1","BPXODI1"],  # 2021+ oscillometric BP
          "bpq":["SEQN","BPQ040A","BPQ150","BPQ090D","BPQ100D","BPQ101D"],
          "BPQ":["SEQN","BPQ040A","BPQ150","BPQ090D","BPQ100D","BPQ101D"],
          "diq":["SEQN","DIQ070","DID070","DIQ050"],"DIQ":["SEQN","DIQ070","DID070","DIQ050"],
          "alb_cr":["SEQN","URDACT","URXUMA","URXUCR"],"ALB_CR":["SEQN","URDACT","URXUMA","URXUCR"],
          "smq":["SEQN","SMQ020","SMQ040"],
          "alq":["SEQN","ALQ111","ALQ101","ALQ130","ALQ120Q","ALQ120U","ALQ121"],
          "dxx":["SEQN","DXDTOFAT","DXDTOLI","DXDTOTOT","DXXLALI","DXXRALI","DXXLLLI","DXXRLLI","DXXTRFAT"],
          "dxxfem":["SEQN","DXXNKBMD","DXXOFBMD"],"DXXFEM":["SEQN","DXXNKBMD","DXXOFBMD"],
          "dxxspn":["SEQN","DXXOSBMD"],"DXXSPN":["SEQN","DXXOSBMD"]}
    frames=[]
    for ci,cyc in enumerate(cycles):
        base=None; seen=set()
        for mod,want in spec.items():
            f=_find(data_dir,f"{mod}_{cyc}")
            if not f: continue
            key=os.path.basename(f).lower()
            if key in seen: continue
            seen.add(key)
            df,_=_read(f); df=df[[c for c in want if c in df.columns]]
            # some files (e.g., 1999-2006 whole-body DXA) ship 5 multiply-imputed rows per SEQN;
            # keep one row per person so the merge does not explode
            if "SEQN" in df.columns: df=df.drop_duplicates(subset="SEQN")
            base=df if base is None else base.merge(df,on="SEQN",how="left")
        if base is not None:
            base["_cyc"]=ci; base["_cyclab"]=str(cyc); frames.append(base)   # cycle index + label
    DF=pd.concat(frames,ignore_index=True); DF.attrs["dataset"]="NHANES"; DF.attrs["n_cycles"]=len(frames); return DF

# ── Derived clinical formulas (identical across surveys) ──────────────────────────────
def _egfr(scr, age, female):
    scr=pd.to_numeric(scr,errors="coerce"); k=np.where(female,0.7,0.9); a=np.where(female,-0.241,-0.302)
    r=scr/k
    return 142*np.minimum(r,1)**a*np.maximum(r,1)**(-1.200)*(0.9938**age)*np.where(female,1.012,1.0)
def _fib4(age, ast, alt, plt):
    ast=pd.to_numeric(ast,errors="coerce"); alt=pd.to_numeric(alt,errors="coerce"); plt=pd.to_numeric(plt,errors="coerce")
    return (age*ast)/(plt*np.sqrt(alt.clip(lower=0.01)))
def _hsi(alt, ast, bmi, female, dm):
    alt=pd.to_numeric(alt,errors="coerce"); ast=pd.to_numeric(ast,errors="coerce"); bmi=pd.to_numeric(bmi,errors="coerce")
    return 8*(alt/ast)+bmi+np.where(female,2,0)+np.where(dm==1,2,0)
def _acr(ualb, ucrea):
    """Urine albumin-to-creatinine ratio (mg/g) = albumin(mg/L) / creatinine(mg/dL) * 100.
    Validated against NHANES URDACT (KNHANES has no official derived ACR)."""
    a=pd.to_numeric(ualb,errors="coerce"); c=pd.to_numeric(ucrea,errors="coerce")
    return a/c.replace(0,np.nan)*100.0

# Alcohol grams/day for the MASLD alcohol strata (MetALD floor: F>=20, M>=30 g/day).
# Ethanol per standard drink is country-specific and physically defined: Korea ~8 g, US ~14 g.
# Category midpoints are the KNHANES survey conventions; occasions are annualized then divided by 365.
_KN_FREQ2OCCDAY={1:0.0, 2:6/365, 3:12/365, 4:36/365, 5:130/365, 6:260/365}   # BD1_11 (1yr drinking freq)
_KN_AMT2DRINKS ={1:1.5, 2:3.5, 3:5.5, 4:8.0, 5:12.0}                          # BD2_1 (drinks per occasion)
def _alc_gday_kn(freq, amt):
    f=pd.to_numeric(freq,errors="coerce").map(_KN_FREQ2OCCDAY)   # 8(N/A)->NaN, 9(unknown)->NaN
    a=pd.to_numeric(amt,errors="coerce").map(_KN_AMT2DRINKS)
    f=f.where(pd.to_numeric(freq,errors="coerce")!=8, 0.0)       # 8 = never/non-applicable -> 0 occasions
    a=a.where(pd.to_numeric(amt,errors="coerce")!=8, 0.0)
    return f*a*8.0
_NH_ALQ121_OCCYR={0:0, 1:365, 2:330, 3:182, 4:104, 5:52, 6:30, 7:12, 8:9, 9:4.5, 10:1.5}
def _alc_gday_nh(alq130, alq120q, alq120u, alq121):
    d=pd.to_numeric(alq130,errors="coerce"); d=d.where(d<400)     # 777/999 = refused/unknown
    # occasions/year: old cycles ALQ120Q x unit(1=wk,2=mo,3=yr); new cycle ALQ121 category
    q=pd.to_numeric(alq120q,errors="coerce"); q=q.where(q<400)
    u=pd.to_numeric(alq120u,errors="coerce").map({1:52.0,2:12.0,3:1.0})
    occ_old=q*u
    occ_new=pd.to_numeric(alq121,errors="coerce").map(_NH_ALQ121_OCCYR)
    occ=occ_old.fillna(occ_new)
    return occ*d.fillna(0)*14.0/365.0

def _num(d, name):
    """Numeric Series for a column, or an all-NaN Series aligned to d if the column is absent
    (so single-cycle loads that lack a variable never crash coalescing logic)."""
    s = d[name] if name in d.columns else pd.Series(np.nan, index=d.index)
    return pd.to_numeric(s, errors="coerce")

def _mk(pos, obs):
    """Build a nullable 0/1 outcome. obs=True where the defining data exist (0 if not positive),
    pos=True for cases. Rows without defining data stay <NA> so they are dropped, not counted as
    controls. This prevents missing measurements from being miscoded as negatives."""
    obs=obs.reindex(pos.index).fillna(False).astype(bool)
    pos=pos.fillna(False).astype(bool)
    out=pd.Series(pd.NA, index=pos.index, dtype="Int64")
    out[obs]=0; out[pos & obs]=1   # pos must also be observed; pos&~obs stays <NA> (no defining data)
    return out

# ── Preprocessing: raw -> standard variables + outcomes (identical definitions) ───────
def apply_definitions(DF, defs):
    ds=DF.attrs.get("dataset"); nc=DF.attrs.get("n_cycles",1)
    d = _derive_knhanes(DF,defs,nc) if ds=="KNHANES" else _derive_nhanes(DF,defs,nc)
    _coverage_warn(d, ds)
    return categorize(d)

def _coverage_warn(d, ds):
    """Warn (do not raise) when a required standardized variable is entirely missing in some cycle -
    the failure mode behind the silent cycle-L variable renames. Legitimate gaps (e.g. vitd) still warn."""
    import sys
    core=["glucose","hba1c","tg","tchol","hdl","sbp","dbp","creatinine","hemoglobin","bmi","wc","alt","ast"]
    if "_cyc" not in d.columns: return
    for v in core:
        if v not in d.columns: sys.stderr.write(f"[coverage] {ds}: standardized var '{v}' ABSENT entirely\n"); continue
        cov=d.groupby("_cyc")[v].apply(lambda s: pd.to_numeric(s,errors="coerce").notna().sum())
        zero=[int(c) for c,n in cov.items() if n==0]
        if zero: sys.stderr.write(f"[coverage] {ds}: '{v}' has ZERO non-missing in cycle index {zero}\n")

def _cardiometabolic_masld(d, female):
    """MASLD cardiometabolic criterion (>=1 of 5), 2023 multisociety nomenclature:
    overweight/WC, dysglycemia, high BP (or Rx), high TG (or Rx), low HDL (or Rx). Ethnicity-specific
    cut-offs: BMI >=23 (Asian/KNHANES) vs >=25 (NHANES); WC KR 90/85 vs US 102/88."""
    kn = d.attrs.get("dataset")=="KNHANES"
    wc_m,wc_f = (90,85) if kn else (102,88)
    bmi_cut = 23 if kn else 25
    cm = (d.bmi>=bmi_cut)
    cm = cm | (((~female)&(d.wc>=wc_m))|((female)&(d.wc>=wc_f)))
    cm = cm | (d.glucose>=100) | (pd.to_numeric(d.get("hba1c"),errors="coerce")>=5.7) | (d.get("dm",0)==1)
    cm = cm | (d.sbp>=130) | (d.dbp>=85) | (pd.to_numeric(d.get("hp_med_flag"),errors="coerce")==1)
    cm = cm | (d.tg>=150) | (pd.to_numeric(d.get("lip_med_flag"),errors="coerce")==1)
    if "hdl" in d: cm = cm | (((~female)&(d.hdl<40))|((female)&(d.hdl<50)))
    return cm.fillna(False)

def _derive_extra(d, female):
    """Derived indices + additional harmonized outcomes computed from the standardized columns
    (identical formulas in both surveys). Requires height, weight, insulin, vitd, wbc, fn_bmd,
    ls_bmd, and the core anthropometry/lipids/glucose columns to be set first."""
    ht=pd.to_numeric(d.get("height"),errors="coerce")
    d["whtr"]=pd.to_numeric(d.get("wc"),errors="coerce")/ht
    d["asmi"]=pd.to_numeric(d.get("asmm_kg"),errors="coerce")/((ht/100.0)**2)
    d["nonhdl"]=pd.to_numeric(d.get("tchol"),errors="coerce")-pd.to_numeric(d.get("hdl"),errors="coerce")
    d["homa_ir"]=pd.to_numeric(d.get("glucose"),errors="coerce")*pd.to_numeric(d.get("insulin"),errors="coerce")/405.0
    bf=pd.to_numeric(d.get("bodyfat_pct"),errors="coerce"); asmi=d["asmi"]
    hdl=pd.to_numeric(d.get("hdl"),errors="coerce"); tg=pd.to_numeric(d.get("tg"),errors="coerce")
    low_hdl=((~female)&(hdl<40))|((female)&(hdl<50))
    # body composition
    d["high_bodyfat"]=_mk(((~female)&(bf>=25))|((female)&(bf>=35)), bf.notna())
    d["central_obesity"]=_mk(d["whtr"]>=0.5, d["whtr"].notna())
    # low muscle mass (labelled "sarcopenia" key kept for compatibility): AWGS ASMI cut, mass-only.
    # NOTE: AWGS/EWGSOP2 sarcopenia also require grip/gait, absent in both surveys -> reported as low muscle mass.
    d["sarcopenia"]=_mk(((~female)&(asmi<7.0))|((female)&(asmi<5.4)), asmi.notna())
    # sarcopenic_obesity removed: ASM/height^2 anti-correlates with fat -> near-empty/artifactual cell
    # liver enzymes
    d["high_alt"]=_mk(pd.to_numeric(d.get("alt"),errors="coerce")>40, pd.to_numeric(d.get("alt"),errors="coerce").notna())
    d["high_ast"]=_mk(pd.to_numeric(d.get("ast"),errors="coerce")>40, pd.to_numeric(d.get("ast"),errors="coerce").notna())
    # lipids
    ldl=pd.to_numeric(d.get("ldl"),errors="coerce")
    d["high_ldl"]=_mk(ldl>=160, ldl.notna())
    d["high_nonhdl"]=_mk(d["nonhdl"]>=190, d["nonhdl"].notna())
    d["atherogenic_dyslipidemia"]=_mk((tg>=150)&low_hdl, tg.notna()&hdl.notna())
    # insulin resistance
    d["insulin_resistance"]=_mk(d["homa_ir"]>=2.5, d["homa_ir"].notna())
    # vitamin D deficiency (<20 ng/mL)
    vd=pd.to_numeric(d.get("vitd"),errors="coerce"); d["vitd_deficiency"]=_mk(vd<20, vd.notna())
    # osteoporosis: femoral neck or lumbar spine T-score <= -2.5, ISCD-standard NHANES III white-female
    # young-adult reference (both sexes, both surveys). ISCD restricts the T-score diagnosis to age >=50.
    fn=pd.to_numeric(d.get("fn_bmd"),errors="coerce"); ls=pd.to_numeric(d.get("ls_bmd"),errors="coerce")
    fn_t=(fn-0.858)/0.120; ls_t=(ls-1.047)/0.110
    _age50=pd.to_numeric(d.get("age"),errors="coerce")>=50
    d["osteoporosis"]=_mk((fn_t<=-2.5)|(ls_t<=-2.5), (fn.notna()|ls.notna())&_age50)
    # leukocytosis (WBC > 11 x10^9/L, clinical reference-range convention)
    wb=pd.to_numeric(d.get("wbc"),errors="coerce"); d["leukocytosis"]=_mk(wb>11, wb.notna())
    return d

def _derive_knhanes(DF, de, nc):
    d=DF.copy(); d.attrs["dataset"]="KNHANES"
    female=(d.sex==2)
    d["age"]=pd.to_numeric(d.age,errors="coerce"); d["men"]=(d.sex==1).astype(int)
    d["bmi"]=d.HE_BMI; d["wc"]=d.HE_wc; d["sbp"]=d.HE_sbp; d["dbp"]=d.HE_dbp; d["glucose"]=d.HE_glu
    d["hba1c"]=d.HE_HbA1c; d["tg"]=d.HE_TG; d["tchol"]=d.HE_chol; d["hdl"]=d.HE_HDL_st2
    d["ldl"]=d.get("HE_LDL_drct"); d["alt"]=d.HE_alt; d["ast"]=d.HE_ast; d["insulin"]=d.get("HE_insulin")
    # fasting (>=8h) required for FPG/TG/LDL (harmonize with NHANES fasting subsample); drop KNOWN non-fasters
    _notfast=pd.to_numeric(d.get("HE_fst"),errors="coerce").lt(8)
    for _c in ["glucose","tg","ldl"]: d[_c]=pd.to_numeric(d[_c],errors="coerce").mask(_notfast)
    _clab=d.get("_cyclab").astype(str) if "_cyclab" in d.columns else pd.Series("",index=d.index)
    # IDMS creatinine calibration break in hn07-08: drop from eGFR/CKD (no official correction factor)
    d["creatinine"]=pd.to_numeric(d.HE_crea,errors="coerce").mask(_clab.isin(["07","08"]))
    d["hemoglobin"]=d.HE_HB; d["platelet"]=d.get("HE_Bplt")
    d["acr"]=_acr(_num(d,"HE_Ualb"), _num(d,"HE_Ucrea").fillna(_num(d,"HE_UCREA")))
    d["height"]=d.get("HE_ht"); d["weight"]=d.get("HE_wt"); d["vitd"]=d.get("HE_vitD")
    d["wbc"]=d.get("HE_WBC"); d["fn_bmd"]=d.get("DX_NK_BMD"); d["ls_bmd"]=d.get("DX_LSP_BMD")
    # body composition (DXA, grams -> kg)
    limb=d[["DW_Llg_LN","DW_Rlg_LN","DW_Lrm_LN","DW_Rrm_LN"]].sum(axis=1,min_count=4) if "DW_Llg_LN" in d else np.nan
    d["fat_kg"]=d.get("DW_WBT_FT",np.nan)/1000; d["lean_kg"]=d.get("DW_WBT_LN",np.nan)/1000; d["asmm_kg"]=limb/1000
    d["bodyfat_pct"]=d.get("DW_WBT_FT",np.nan)/d.get("DW_WBT_MS",np.nan)*100
    d["asm_pct"]=limb/d.get("DW_WBT_MS",np.nan)*100
    # smoking never(BS1_1==3) vs ever(1,2); current = sm_presnt / BS3_1==1
    d["smoking"]=np.where(d.BS1_1==3,0,np.where(d.BS1_1.isin([1,2]),1,np.nan))
    cur = d["sm_presnt"] if "sm_presnt" in d else (d.BS3_1==1).astype(float)
    d["current_smoking"]=np.where(pd.to_numeric(cur,errors="coerce")==1,1,
                          np.where(d["smoking"].notna(),0,np.nan))
    # alcohol never(BD1==1) vs ever(BD1==2)
    d["alcohol"]=np.where(d.BD1==1,0,np.where(d.BD1==2,1,np.nan))
    # derived indices
    d["egfr"]=_egfr(d.creatinine,d.age,female)
    d["fib4"]=_fib4(d.age,d.ast,d.alt,d.platelet)
    # current glucose-lowering treatment: insulin (DE1_31) or oral agent (DE1_32); ==1 (0=no,1=yes,8=N/A).
    # HE_DMdr was "medication ON exam day" (fasting-prep question), not current treatment -> not used.
    dm_med=(_num(d,"DE1_31")==1)|(_num(d,"DE1_32")==1)
    # DM: FPG>=126 or HbA1c>=6.5 or current medication (measured + current tx; no lifetime dx)
    dm_pos=pd.Series(False,index=d.index)
    if de["dm"].get("FPG >=126 mg/dL"): dm_pos|=(d.glucose>=126)
    if de["dm"].get("HbA1c >=6.5%"): dm_pos|=(d.hba1c>=6.5)
    if de["dm"].get("Current glucose-lowering medication"): dm_pos|=dm_med
    dm_obs=d.glucose.notna()|d.hba1c.notna()|dm_med
    d["dm"]=_mk(dm_pos,dm_obs)
    pre=_mk(((d.glucose>=100)&(d.glucose<126))|((d.hba1c>=5.7)&(d.hba1c<6.5)), d.glucose.notna()|d.hba1c.notna())
    pre[d["dm"]==1]=0; d["prediabetes"]=pre
    d["hsi"]=_hsi(d.alt,d.ast,d.bmi,female,dm_pos)
    # HTN: BP or current antihypertensive. DI1_2 (혈압조절제 복용) 1-4=taking / 5=no / 8=N/A.
    # HE_HPdr was "medication ON exam day", not current treatment -> not used.
    hp_med=_num(d,"DI1_2").isin([1,2,3,4])
    lip_med=_num(d,"DI2_2").isin([1,2,3,4])   # 이상지질혈증 약복용 (current lipid-lowering treatment)
    htn_pos=pd.Series(False,index=d.index)
    if de["htn"].get("SBP >=140 or DBP >=90 mmHg"): htn_pos|=(d.sbp>=140)|(d.dbp>=90)
    if de["htn"].get("Current antihypertensive"): htn_pos|=hp_med
    d["htn"]=_mk(htn_pos,(d.sbp.notna()&d.dbp.notna())|hp_med)
    # anthropometric outcomes (KSSO 2022 Korean cut-offs: obesity BMI>=25, overweight BMI>=23)
    d["obesity"]=_mk(d.bmi>=25,d.bmi.notna()); d["overweight"]=_mk(d.bmi>=23,d.bmi.notna())
    d["abdominal_obesity"]=_mk(((~female)&(d.wc>=90))|((female)&(d.wc>=85)), d.wc.notna())
    # metabolic syndrome (harmonized NCEP, >=3 of 5, KR waist cut; TG/HDL include "or drug treatment")
    c_wc=((~female)&(d.wc>=90))|((female)&(d.wc>=85)); c_glu=(d.glucose>=100)|dm_med
    c_bp=(d.sbp>=130)|(d.dbp>=85)|hp_med; c_tg=(d.tg>=150)
    c_hdl=((~female)&(d.hdl<40))|((female)&(d.hdl<50))
    # lipid-drug clause omitted from TG/HDL components: surveys can't distinguish TG- vs LDL-lowering
    # agents, so "any lipid med" would double-count one statin user across both components.
    cnt=sum(x.astype("boolean").fillna(False).astype(int) for x in [c_wc,c_glu,c_bp,c_tg,c_hdl])
    d["mets"]=_mk(cnt>=3, d.wc.notna()&d.glucose.notna()&d.sbp.notna()&d.tg.notna()&d.hdl.notna())
    d["hp_med_flag"]=hp_med.astype("boolean").astype("Int64"); d["lip_med_flag"]=lip_med.astype("boolean").astype("Int64")
    # dyslipidemia (KSoLA composite) / low HDL / high TG
    dld=pd.Series(False,index=d.index)
    if de["dld"].get("TC >=240 mg/dL"): dld|=(d.tchol>=240)
    if de["dld"].get("HDL low (M<40/F<50)"): dld|=c_hdl
    if de["dld"].get("TG >=200 mg/dL"): dld|=(d.tg>=200)
    if de["dld"].get("LDL >=160 mg/dL"): dld|=(d.ldl>=160)
    dld|=lip_med   # KSoLA: current lipid-lowering treatment counts as dyslipidemia
    d["dyslipidemia"]=_mk(dld, (d.tchol.notna()|d.hdl.notna()|d.tg.notna()|d.ldl.notna())|lip_med)
    d["low_hdl"]=_mk(c_hdl,d.hdl.notna()); d["high_tg"]=_mk(d.tg>=150,d.tg.notna())
    # CKD (KDIGO: eGFR<60 or ACR>=30). Restrict to cycles where urine albumin was measured so the
    # definition is homogeneous (KNHANES ACR only in hn11-14,19-24); eGFR-only cycles would otherwise
    # count albuminuria-unmeasured people as non-CKD and dilute/heterogenize the estimate.
    _acr_cyc=d.groupby("_cyclab")["acr"].transform(lambda s: s.notna().any())
    d["ckd"]=_mk((d.egfr<60)|(d.acr>=30), (d.egfr.notna()|d.acr.notna())&_acr_cyc)
    d["adv_fibrosis"]=_mk(d.fib4>2.67,d.fib4.notna())
    _preg=((_num(d,"HE_prg")==1)|(_num(d,"N_PRG")==1)).fillna(False)
    d["anemia"]=_mk(((~female)&(d.hemoglobin<13))|(female&~_preg&(d.hemoglobin<12))|(female&_preg&(d.hemoglobin<11)),
                    d.hemoglobin.notna())
    # steatosis (KNHANES = HSI>36) and true MASLD (steatosis + cardiometabolic + alcohol below MetALD floor)
    gday=_alc_gday_kn(_num(d,"BD1_11"), _num(d,"BD2_1"))
    gday=gday.where(pd.to_numeric(d.get("alcohol"),errors="coerce")!=0, 0.0); d["alc_gday"]=gday
    heavy=((~female)&(gday>=30))|((female)&(gday>=20))
    d["steatosis"]=_mk(d.hsi>36,d.hsi.notna())
    d["masld"]=_mk((d.steatosis==1)&_cardiometabolic_masld(d,female)&(~heavy.fillna(False)),
                   d.steatosis.notna()&gday.notna())
    d=_derive_extra(d,female)   # sarcopenia, body-fat/central obesity, liver enzymes, lipids, HOMA-IR, vitD, osteoporosis, WBC
    # education 3-level, income tertile
    d["edu3"]=pd.to_numeric(d.get("edu"),errors="coerce").map({1:"<=Middle",2:"<=Middle",3:"High school",4:"College+"})
    q=pd.to_numeric(d.get("incm5"),errors="coerce")
    d["inc3"]=q.map({1:"Low",2:"Low",3:"Middle",4:"High",5:"High"})
    # survey design — wt_itvex (health-interview + examination linked weight) is the standard
    # integrated weight covering all cycles; wt_ex1 is a small fasting subsample weight (missing in 2010).
    # current pregnancy: HE_prg (임신여부, exam) or N_PRG (임신·수유 여부, ==1 pregnant)
    d["pregnant"]=(pd.to_numeric(d.get("HE_prg"),errors="coerce")==1)|(pd.to_numeric(d.get("N_PRG"),errors="coerce")==1)
    wcol=next((c for c in ["wt_itvex","wt_tot","wt_ex1"] if c in d.columns), None)
    d["wt_pool"]=(pd.to_numeric(d[wcol],errors="coerce")/max(nc,1)) if wcol else np.nan
    # pooled variance strata unique per cycle (kstrata values repeat across survey years)
    cyc=pd.to_numeric(d.get("_cyc"),errors="coerce").fillna(0)
    d["kstrata"]=cyc*100000+pd.to_numeric(d["kstrata"],errors="coerce"); d["psu"]=d.psu
    return d

def _derive_nhanes(DF, de, nc):
    d=DF.copy(); d.attrs["dataset"]="NHANES"
    female=(_num(d,"RIAGENDR")==2)
    d["age"]=_num(d,"RIDAGEYR"); d["men"]=(_num(d,"RIAGENDR")==1).astype(int)
    d["bmi"]=_num(d,"BMXBMI"); d["wc"]=_num(d,"BMXWAIST")
    d["sbp"]=_num(d,"BPXSY1").fillna(_num(d,"BPXOSY1")); d["dbp"]=_num(d,"BPXDI1").fillna(_num(d,"BPXODI1"))
    d["glucose"]=_num(d,"LBXGLU"); d["hba1c"]=_num(d,"LBXGH")
    d["tg"]=_num(d,"LBXTR").fillna(_num(d,"LBXTLG")); d["tchol"]=_num(d,"LBXTC")   # LBXTR->LBXTLG (cycle L rename)
    d["hdl"]=_num(d,"LBDHDD"); d["ldl"]=_num(d,"LBDLDL")
    d["ast"]=_num(d,"LBXSASSI"); d["alt"]=_num(d,"LBXSATSI"); d["ggt"]=_num(d,"LBXSGTSI")
    _clab=d.get("_cyclab").astype(str).str.lower() if "_cyclab" in d.columns else pd.Series("",index=d.index)
    # 2005-06 (cycle D) serum creatinine calibration break: drop from eGFR/CKD (no official correction)
    d["creatinine"]=_num(d,"LBXSCR").mask(_clab=="d"); d["hemoglobin"]=_num(d,"LBXHGB"); d["platelet"]=_num(d,"LBXPLTSI")
    d["acr"]=_num(d,"URDACT").fillna(_acr(_num(d,"URXUMA"), _num(d,"URXUCR")))   # ACR (mg/g), incl. cycles D/E lacking URDACT
    d["uric"]=_num(d,"LBXSUA"); d["cap"]=_num(d,"LUXCAPM"); d["lsm"]=_num(d,"LUXSMED")
    d["height"]=_num(d,"BMXHT"); d["weight"]=_num(d,"BMXWT"); d["insulin"]=_num(d,"LBXIN")
    d["wbc"]=_num(d,"LBXWBCSI"); d["fn_bmd"]=_num(d,"DXXNKBMD"); d["ls_bmd"]=_num(d,"DXXOSBMD")
    # 25(OH)D: LBXVIDMS (2007+) or LBDVIDMS (2005-2006), both standardized nmol/L -> ng/mL (KNHANES scale)
    d["vitd"]=_num(d,"LBXVIDMS").fillna(_num(d,"LBDVIDMS"))/2.496
    fat=_num(d,"DXDTOFAT"); li=_num(d,"DXDTOLI"); tot=fat+li
    if all(c in d.columns for c in ["DXXLALI","DXXRALI","DXXLLLI","DXXRLLI"]):
        limb=d[["DXXLALI","DXXRALI","DXXLLLI","DXXRLLI"]].apply(pd.to_numeric,errors="coerce").sum(axis=1,min_count=4)
    else: limb=pd.Series(np.nan,index=d.index)
    d["fat_kg"]=fat/1000; d["lean_kg"]=li/1000; d["asmm_kg"]=limb/1000
    d["bodyfat_pct"]=fat/tot*100; d["asm_pct"]=limb/tot*100
    # smoking never(SMQ020==2) vs ever(SMQ020==1); current = SMQ040 in (1,2)
    d["smoking"]=np.where(_num(d,"SMQ020")==1,1,np.where(_num(d,"SMQ020")==2,0,np.nan))
    d["current_smoking"]=np.where(_num(d,"SMQ040").isin([1,2]),1,
                          np.where(pd.Series(d["smoking"],index=d.index).notna(),0,np.nan))
    # alcohol ever: ALQ111 (2017+) or ALQ101 (2007-2016, >=12 drinks in any year); 1=yes, 2=no
    _alc=_num(d,"ALQ111").fillna(_num(d,"ALQ101"))
    d["alcohol"]=np.where(_alc==1,1,np.where(_alc==2,0,np.nan))
    d["egfr"]=_egfr(d.creatinine,d.age,female)
    d["fib4"]=_fib4(d.age,d.ast,d.alt,d.platelet)
    dm_med=(_num(d,"DIQ070")==1)|(_num(d,"DID070")==1)|(_num(d,"DIQ050")==1)   # DID070 = DIQ070 in cycles D/E
    dm_pos=pd.Series(False,index=d.index)
    if de["dm"].get("FPG >=126 mg/dL"): dm_pos|=(d.glucose>=126)
    if de["dm"].get("HbA1c >=6.5%"): dm_pos|=(d.hba1c>=6.5)
    if de["dm"].get("Current glucose-lowering medication"): dm_pos|=dm_med
    dm_obs=d.glucose.notna()|d.hba1c.notna()|dm_med
    d["dm"]=_mk(dm_pos,dm_obs)
    pre=_mk(((d.glucose>=100)&(d.glucose<126))|((d.hba1c>=5.7)&(d.hba1c<6.5)), d.glucose.notna()|d.hba1c.notna())
    pre[d["dm"]==1]=0; d["prediabetes"]=pre
    hp_med=(_num(d,"BPQ040A")==1)|(_num(d,"BPQ150")==1)   # BPQ040A->BPQ150 (cycle L rename)
    lip_med=(_num(d,"BPQ100D")==1)|(_num(d,"BPQ101D")==1) # BPQ100D->BPQ101D (cycle L rename)
    htn_pos=pd.Series(False,index=d.index)
    if de["htn"].get("SBP >=140 or DBP >=90 mmHg"): htn_pos|=(d.sbp>=140)|(d.dbp>=90)
    if de["htn"].get("Current antihypertensive"): htn_pos|=hp_med
    d["htn"]=_mk(htn_pos,(d.sbp.notna()&d.dbp.notna())|hp_med)
    d["obesity"]=_mk(d.bmi>=30,d.bmi.notna()); d["overweight"]=_mk(d.bmi>=25,d.bmi.notna())   # WHO cut (US)
    d["abdominal_obesity"]=_mk(((~female)&(d.wc>=102))|((female)&(d.wc>=88)), d.wc.notna())
    c_wc=((~female)&(d.wc>=102))|((female)&(d.wc>=88)); c_glu=(d.glucose>=100)|dm_med
    c_bp=(d.sbp>=130)|(d.dbp>=85)|hp_med; c_tg=(d.tg>=150)
    c_hdl=((~female)&(d.hdl<40))|((female)&(d.hdl<50))
    # lipid-drug clause omitted from TG/HDL components (avoids double-counting one lipid med across both)
    cnt=sum(x.astype("boolean").fillna(False).astype(int) for x in [c_wc,c_glu,c_bp,c_tg,c_hdl])
    d["mets"]=_mk(cnt>=3, d.wc.notna()&d.glucose.notna()&d.sbp.notna()&d.tg.notna()&d.hdl.notna())
    d["hp_med_flag"]=hp_med.astype("boolean").astype("Int64"); d["lip_med_flag"]=lip_med.astype("boolean").astype("Int64")
    dld=pd.Series(False,index=d.index)
    if de["dld"].get("TC >=240 mg/dL"): dld|=(d.tchol>=240)
    if de["dld"].get("HDL low (M<40/F<50)"): dld|=c_hdl
    if de["dld"].get("TG >=200 mg/dL"): dld|=(d.tg>=200)
    if de["dld"].get("LDL >=160 mg/dL"): dld|=(d.ldl>=160)
    dld|=lip_med
    d["dyslipidemia"]=_mk(dld, (d.tchol.notna()|d.hdl.notna()|d.tg.notna()|d.ldl.notna())|lip_med)
    d["low_hdl"]=_mk(c_hdl,d.hdl.notna()); d["high_tg"]=_mk(d.tg>=150,d.tg.notna())
    # CKD (KDIGO harmonized both surveys): eGFR<60 or ACR>=30. NHANES has ACR every cycle so the
    # albuminuria-cycle guard is a no-op here, but kept symmetric with KNHANES.
    _acr_cyc=d.groupby("_cyclab")["acr"].transform(lambda s: s.notna().any())
    d["ckd"]=_mk((d.egfr<60)|(d.acr>=30), (d.egfr.notna()|d.acr.notna())&_acr_cyc)
    d["adv_fibrosis"]=_mk(d.fib4>2.67,d.fib4.notna())
    _preg=(_num(d,"RIDEXPRG")==1).fillna(False)
    d["anemia"]=_mk(((~female)&(d.hemoglobin<13))|(female&~_preg&(d.hemoglobin<12))|(female&_preg&(d.hemoglobin<11)),
                    d.hemoglobin.notna())
    # steatosis (NHANES = CAP>=248) and true MASLD (+ alcohol below MetALD floor)
    gday=_alc_gday_nh(_num(d,"ALQ130"), _num(d,"ALQ120Q"), _num(d,"ALQ120U"), _num(d,"ALQ121"))
    gday=gday.where(pd.to_numeric(d.get("alcohol"),errors="coerce")!=0, 0.0); d["alc_gday"]=gday
    heavy=((~female)&(gday>=30))|((female)&(gday>=20))
    d["steatosis"]=_mk(d.cap>=248,d.cap.notna())
    d["masld"]=_mk((d.steatosis==1)&_cardiometabolic_masld(d,female)&(~heavy.fillna(False)),
                   d.steatosis.notna()&gday.notna())
    d=_derive_extra(d,female)   # sarcopenia, body-fat/central obesity, liver enzymes, lipids, HOMA-IR, vitD, osteoporosis, WBC
    # education 3-level, income tertile (PIR standard cutpoints)
    d["edu3"]=pd.to_numeric(d.get("DMDEDUC2"),errors="coerce").map({1:"<=Middle",2:"<=Middle",3:"High school",4:"College+",5:"College+"})
    pir=pd.to_numeric(d.get("INDFMPIR"),errors="coerce")
    d["inc3"]=pd.cut(pir,[0,1.3,3.5,100],labels=["Low","Middle","High"],right=False)
    # pooled variance strata must be unique per cycle (SDMVSTRA values can repeat across cycles)
    d["pregnant"]=(_num(d,"RIDEXPRG")==1)   # pregnant at exam
    cyc=pd.to_numeric(d.get("_cyc"),errors="coerce").fillna(0)
    d["kstrata"]=cyc*100000+pd.to_numeric(d.SDMVSTRA,errors="coerce")
    d["psu"]=d.SDMVPSU; d["wt_pool"]=pd.to_numeric(d.WTMEC2YR,errors="coerce")/max(nc,1)
    return d

# ── Analytic frame + exposure terms (continuous = z-standardized, binary = 0/1) ───────
def build_analytic(d, exposures, outcomes, covariates, age_min):
    keep=(d.age>=age_min)&(d.wt_pool>0)&d.wt_pool.notna()&d.kstrata.notna()&d.psu.notna()
    ana=d[keep].copy()
    for o in outcomes: ana=ana[ana[o].notna()]
    for e in exposures:
        if typ(e)=="c":
            col=pd.to_numeric(ana[e],errors="coerce")
            if e in LOG_EXPOSURES: col=np.log(col.where(col>0))   # log skewed exposure (positive only)
            ana[f"x_{e}"]=(col-col.mean())/col.std()
        else: ana[f"x_{e}"]=ana[e]
    return ana

def event_counts(ana, outcomes):
    """req #8: number of events for each binary outcome in the analytic sample."""
    out={}
    for o in outcomes:
        if o in ana and typ(o)=="b":
            s=pd.to_numeric(ana[o],errors="coerce"); out[o]=int((s==1).sum())
    return out

def _rscript():
    for p in (r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe", r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"):
        if os.path.isfile(p): return p
    from shutil import which
    return which("Rscript") or "Rscript"

def run_engine(ana, dataset, exposures, outcomes, covariates, workdir=None, skip_table1=False, reuse_analytic=False):
    project_dir=os.path.dirname(os.path.abspath(__file__))
    workdir=os.path.abspath(workdir or project_dir)
    rscript_path=_rscript(); engine_path=os.path.join(workdir,"engine.R")
    if not os.path.isfile(engine_path): raise FileNotFoundError(f"engine.R not found: {engine_path}")
    cont_base=["age","bmi","wc","sbp","glucose","hba1c","tg","tchol","hdl","alt","ast","egfr","fib4",
               "hemoglobin","fat_kg","asmm_kg","bodyfat_pct","asm_pct"]
    cont=[c for c in dict.fromkeys(cont_base+[e for e in exposures if typ(e)=="c"]) if c in ana.columns]
    primary=outcomes[0]; group=primary if typ(primary)=="b" else ""
    binv=[b for b in ["men","smoking","alcohol","dm","htn","mets","dyslipidemia"]
          if b!=group and b in ana.columns and ana[b].notna().any()]
    cov=[c for c in covariates if c in ana.columns]
    pairs=[{"y":o,"x":f"x_{e}","otype":typ(o)} for e in exposures for o in outcomes
           if not related(dataset,e,o)]
    labels={k:lab(k) for k in cont+binv}
    cfg={"group":group,"cont":cont,"bin":binv,"cov":cov,"labels":labels,"pairs":pairs,"skip_table1":bool(skip_table1)}
    if not reuse_analytic:   # reuse the already-written CSV across cov-groups of the same outcome
        ana.to_csv(os.path.join(workdir,"engine_analytic.csv"),index=False,encoding="utf-8-sig")
    with open(os.path.join(workdir,"engine_config.json"),"w",encoding="utf-8") as f:
        json.dump(cfg,f,ensure_ascii=False)
    r=subprocess.run([rscript_path,engine_path],cwd=workdir,capture_output=True,text=True,errors="replace")
    if r.returncode!=0:
        raise RuntimeError(f"R analysis failed\nRscript: {rscript_path}\nSTDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}")
    t1p=os.path.join(workdir,"engine_table1.csv"); rp=os.path.join(workdir,"engine_results.csv")
    return pd.read_csv(t1p), pd.read_csv(rp), cfg, pairs

def add_fdr(res):
    if len(res)==0: return res
    p=res.p.values; n=len(p); o=np.argsort(p)
    q=np.empty(n); q[o]=np.minimum.accumulate((p[o]*n/np.arange(1,n+1))[::-1])[::-1]
    res=res.copy(); res["q"]=np.clip(q,0,1); return res

def effect_str(r): return f"{r.est:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})"
def measure_name(m): return "OR" if m=="OR" else "β"

# ── LLM (ollama) + deterministic fallback ─────────────────────────────────────────────
def ollama_chat(prompt, model="llama3.3:70b", url="http://localhost:11434", fmt=None):
    import requests
    body={"model":model,"prompt":prompt,"stream":False}
    if fmt: body["format"]=fmt
    return requests.post(url+"/api/generate",json=body,timeout=300).json()["response"]
STYLE=("Write in the style of a medical research paper, in English. Do not use em dashes, en dashes, "
       "arrows, semicolons, colons, or (i)(ii) enumeration in body text. First/Second/Third is allowed. "
       "Do not change any numbers.")

# ── report-writer invariant ────────────────────────────────────────────────────
# The report-writer agent may compose prose but may never introduce or alter a
# number. The prompt states this; the guard below *enforces* it. Any generated
# sentence containing a numeric token absent from the deterministic fact string
# is discarded and the deterministic template is emitted instead, so no number
# that did not originate in the R survey / scikit-learn core can reach a report.
_NUM_TOKEN=re.compile(r"\d+(?:\.\d+)?")

def _num_tokens(text):
    """Numeric tokens of `text`, with thousands separators removed so that a
    generated '37,636' matches a computed '37636'. Trailing zeros are NOT
    normalised: 1.8 must not be accepted in place of 1.88."""
    return set(_NUM_TOKEN.findall((text or "").replace(",","")))

def numeric_guard(generated, facts):
    """True iff every numeric token in `generated` is an exact token of `facts`.

    Exact-token (not substring) matching is deliberate: substring matching would
    accept a truncated '1.8' for a computed '1.88'. Rejection is safe — the
    caller falls back to the deterministic template."""
    return _num_tokens(generated) <= _num_tokens(facts)

LLM_PROSE_STATS={"llm":0,"fallback":0,"violation":0,"error":0}

def llm_prose(instruction, facts, model, url, min_chars=80):
    """Ask the report-writer agent for prose and return it only if it passes the
    numeric guard. Returns "" to signal the caller to use its deterministic
    template. Every outcome is tallied in LLM_PROSE_STATS for the audit trail."""
    try:
        out=ollama_chat(f"{STYLE}\n{instruction}\nFacts: {facts}",model,url).strip()
    except Exception:
        LLM_PROSE_STATS["error"]+=1; return ""
    if len(out)<min_chars:
        LLM_PROSE_STATS["fallback"]+=1; return ""
    if not numeric_guard(out,facts):
        LLM_PROSE_STATS["violation"]+=1; return ""
    LLM_PROSE_STATS["llm"]+=1; return out

def gen_manuscript(t1,res,dataset,exposures,outcomes,covariates,defs,model,url,use_llm):
    res=add_fdr(res)
    exp_n=", ".join(lab(e) for e in exposures); out_n=", ".join(lab(o) for o in outcomes)
    cov_n=", ".join(lab(c) for c in covariates)
    src={"KNHANES":"Korea National Health and Nutrition Examination Survey",
         "NHANES":"National Health and Nutrition Examination Survey"}[dataset]
    fnd="; ".join(f"{lab(r.exposure[2:])} and {lab(r.outcome)} {measure_name(r.measure)} {r.est:.2f} "
                  f"(95% CI {r.ci_low:.2f}-{r.ci_high:.2f}, FDR q {r.q:.3f})" for _,r in res.iterrows())
    facts=(f"Data: {src}, survey-weighted. Exposures: {exp_n}. Outcomes: {out_n}. Covariates: {cov_n}. "
           f"Smoking and alcohol grouped as never versus past or current. Continuous exposures modeled per 1-SD. "
           f"Binary outcomes used survey logistic regression reporting odds ratios; continuous outcomes used "
           f"survey linear regression reporting beta coefficients. Findings: {fnd}.")
    if use_llm:
        try:
            def sec(nm,ins): return ollama_chat(f"{STYLE}\nWrite the {nm} paragraph of a research paper in English using only the facts below. Do not change any numbers.\nFacts: {facts}\n{ins}",model,url).strip()
            return {"Methods":sec("Methods","Describe data, survey design, variable grouping, and models."),
                    "Results":sec("Results","Report each association with its effect measure, CI, FDR q."),
                    "Discussion":sec("Discussion","Interpret, note hypothesis-generating nature and limitations."),
                    "_source":"llm"}
        except Exception: pass
    meth=(f"We analyzed {src} participants using complex-survey methods accounting for stratification, clustering, "
          f"and sampling weights. Smoking and alcohol were grouped as never versus past or current. {exp_n} served "
          f"as exposures and {out_n} as outcomes, adjusting for {cov_n}. Continuous exposures were modeled per 1-SD. "
          f"Binary outcomes used survey-weighted logistic regression, and continuous outcomes used survey-weighted "
          f"linear regression. Multiplicity was controlled with the Benjamini-Hochberg false discovery rate.")
    rl=[]
    for _,r in res.sort_values("q").iterrows():
        if r.measure=="OR":
            rl.append(f"{lab(r.exposure[2:])} was {'inversely' if r.est<1 else 'positively'} associated with "
                      f"{lab(r.outcome)} (odds ratio {r.est:.2f}, 95% CI {r.ci_low:.2f} to {r.ci_high:.2f}, FDR q {r.q:.3f}).")
        else:
            rl.append(f"{lab(r.exposure[2:])} was associated with a {'higher' if r.est>0 else 'lower'} {lab(r.outcome)} "
                      f"(beta {r.est:.2f}, 95% CI {r.ci_low:.2f} to {r.ci_high:.2f}, FDR q {r.q:.3f}).")
    disc=("These survey-weighted associations are hypothesis-generating and require external validation. "
          "The cross-sectional design precludes causal inference.")
    return {"Methods":meth,"Results":" ".join(rl),"Discussion":disc,"_source":"fallback"}

def build_docx(title, sections, t1, res, group_label=""):
    from docx import Document
    res=add_fdr(res); doc=Document(); doc.add_heading(title,0)
    for nm in ["Methods","Results","Discussion"]:
        doc.add_heading(nm,1); doc.add_paragraph(sections.get(nm,""))
    doc.add_heading("Table 1. Descriptive characteristics",1)
    cols=list(t1.columns); tb=doc.add_table(rows=1,cols=len(cols)); tb.style="Light Grid Accent 1"
    for j,c in enumerate(cols): tb.rows[0].cells[j].text=str(c)
    for _,row in t1.iterrows():
        cc=tb.add_row().cells
        for j,c in enumerate(cols): cc[j].text=str(row[c])
    doc.add_heading("Table 2. Adjusted associations",1)
    rc=["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q"]; tb2=doc.add_table(rows=1,cols=5); tb2.style="Light Grid Accent 1"
    for j,c in enumerate(rc): tb2.rows[0].cells[j].text=c
    for _,r in res.sort_values("q").iterrows():
        cc=tb2.add_row().cells
        cc[0].text=lab(r.exposure[2:]); cc[1].text=lab(r.outcome); cc[2].text=measure_name(r.measure)
        cc[3].text=effect_str(r); cc[4].text=f"{r.q:.3f}"
    import io; buf=io.BytesIO(); doc.save(buf); return buf.getvalue()

# ═══ Full matrix: every exposure × every outcome (per-outcome analytic sample) ════════
def run_matrix(d, exposures, outcomes, dataset, age_min, workdir=None, progress=None):
    """Run every exposure against every outcome. Each outcome uses its OWN analytic sample
    (rows where that outcome is observed) so unrelated missingness never shrinks the others.
    Outcomes below the event floor or that fail to fit are skipped and recorded. Returns
    (associations DataFrame with FDR across the whole matrix, per-outcome info DataFrame,
    representative Table 1 from the first successful outcome)."""
    all_res=[]; info=[]; t1_primary=None
    outs=[o for o in outcomes if o in AVAIL.get(dataset,[])]
    base=[c for c in ["age","men"] if c in AVAIL.get(dataset,[])]   # universal adjustment set (age, sex)
    for i,o in enumerate(outs):
        if progress: progress(i, len(outs), o)
        cov=[c for c in base if c!=o]   # age/sex are confounders, always adjust (not leakage; keep even if in determinants)
        # screen every exposure EXCEPT the adjusters, the outcome, and any definitionally-circular pair
        exps_o=[e for e in exposures if e not in cov and not related(dataset,e,o)]
        if not exps_o:
            info.append({"outcome":lab(o),"n":0,"events":0,"exposures_tested":0,"status":"no eligible exposures"}); continue
        try:
            ana=build_analytic(d, exps_o, [o], cov, age_min)
        except Exception as e:
            info.append({"outcome":lab(o),"n":0,"events":0,"exposures_tested":0,"status":f"build error: {e}"}); continue
        n=len(ana)
        if typ(o)=="b":
            ev=int((pd.to_numeric(ana[o],errors="coerce")==1).sum()) if o in ana else 0
            if ev < MIN_EVENTS:
                info.append({"outcome":lab(o),"n":n,"events":ev,"exposures_tested":0,
                             "status":f"skipped (< {MIN_EVENTS} events)"}); continue
        else:
            ev=n   # continuous outcome (linear beta): the event floor does not apply
        # retry guards against transient Windows file contention on the shared engine_*.csv files
        import time as _time
        res=None; err=None
        for _attempt in range(3):
            try:
                t1,res,cfg,pairs=run_engine(ana, dataset, exps_o, [o], cov, workdir=workdir); err=None; break
            except Exception as e:
                err=e; _time.sleep(0.5)
        if res is None:
            info.append({"outcome":lab(o),"n":n,"events":ev,"exposures_tested":0,"status":f"engine error: {str(err)[:120]}"}); continue
        if t1_primary is None: t1_primary=t1
        if len(res): res=res.copy(); res["n"]=n; res["events"]=ev; all_res.append(res)
        info.append({"outcome":lab(o),"n":n,"events":ev,"exposures_tested":len(res),"status":"ok"})
    RES=pd.concat(all_res,ignore_index=True) if all_res else pd.DataFrame(
        columns=["exposure","outcome","measure","est","ci_low","ci_high","p","n","events"])
    if len(RES): RES=add_fdr(RES)
    return RES, pd.DataFrame(info), t1_primary

def _src_name(dataset):
    return {"KNHANES":"Korea National Health and Nutrition Examination Survey",
            "NHANES":"National Health and Nutrition Examination Survey"}[dataset]

def gen_full_manuscript(RES, info, dataset, exposures, outcomes, model, url, use_llm):
    """llama writes a full manuscript (Abstract..Conclusion) from the computed numbers only.
    Deterministic template fallback is used if the LLM is unavailable."""
    src=_src_name(dataset)
    exp_n=", ".join(lab(e) for e in exposures); out_n=", ".join(lab(o) for o in outcomes)
    npairs=len(RES); nsig=int((RES.q<0.05).sum()) if len(RES) else 0
    nmin=int(info.n.min()) if len(info) else 0; nmax=int(info.n.max()) if len(info) else 0
    top=RES.sort_values("q").head(20) if len(RES) else RES
    fnd="; ".join(f"{lab(r.exposure[2:])} and {lab(r.outcome)} {measure_name(r.measure)} {r.est:.2f} "
                  f"(95% CI {r.ci_low:.2f} to {r.ci_high:.2f}, FDR q {r.q:.3f})" for _,r in top.iterrows())
    facts=(f"Data source {src}, survey-weighted cross-sectional analysis. Exposures {exp_n}. Outcomes {out_n}. "
           f"Smoking and alcohol were grouped as never versus past or current. Continuous exposures were modeled "
           f"per 1-SD and reported as odds ratios for binary outcomes from survey-weighted logistic regression, "
           f"adjusting for age, sex and lifestyle where not collinear with the exposure. Analytic sample sizes "
           f"ranged from {nmin} to {nmax}. {npairs} exposure-outcome associations were estimated and {nsig} were "
           f"significant after Benjamini-Hochberg false discovery rate control. Leading associations: {fnd}.")
    order=["Abstract","Introduction","Methods","Results","Discussion","Conclusion"]
    instr={"Abstract":"Write a structured single-paragraph abstract with background, methods, results and conclusion.",
           "Introduction":"Write a short introduction motivating the study of these exposures and outcomes.",
           "Methods":"Describe the survey design, weighting, variable grouping, models, and multiplicity control.",
           "Results":"Report the leading associations with their effect measures, confidence intervals and FDR q values.",
           "Discussion":"Interpret the pattern of findings, note the cross-sectional and hypothesis-generating nature, and list limitations including differing steatosis measurement across surveys and single survey weight.",
           "Conclusion":"State a brief concluding sentence."}
    if use_llm:
        try:
            secs={}
            for nm in order:
                secs[nm]=ollama_chat(f"{STYLE}\nWrite the {nm} of a research paper in English using only the facts "
                                     f"below. Do not invent numbers and do not change any numbers.\nFacts: {facts}\n{instr[nm]}",
                                     model,url).strip()
            secs["_source"]="llm"; return secs
        except Exception: pass
    # deterministic fallback
    rlines=[]
    for _,r in top.iterrows():
        if r.measure=="OR":
            rlines.append(f"{lab(r.exposure[2:])} was {'inversely' if r.est<1 else 'positively'} associated with "
                          f"{lab(r.outcome)} (odds ratio {r.est:.2f}, 95% CI {r.ci_low:.2f} to {r.ci_high:.2f}, FDR q {r.q:.3f}).")
        else:
            rlines.append(f"{lab(r.exposure[2:])} was associated with {lab(r.outcome)} "
                          f"(beta {r.est:.2f}, 95% CI {r.ci_low:.2f} to {r.ci_high:.2f}, FDR q {r.q:.3f}).")
    return {"_source":"fallback",
        "Abstract":(f"Background. We examined associations of {exp_n} with {out_n} in {src}. Methods. Using complex-survey "
            f"methods we fitted survey-weighted logistic regressions for each binary outcome, modeling continuous exposures "
            f"per 1-SD and adjusting for age, sex and lifestyle where not collinear. Results. Across {npairs} associations, "
            f"{nsig} were significant after false discovery rate control, with analytic samples of {nmin} to {nmax}. "
            f"Conclusion. These survey-weighted cross-sectional associations are hypothesis-generating."),
        "Introduction":(f"Cardiometabolic and related conditions share overlapping determinants. Nationally representative "
            f"health examination surveys allow simultaneous assessment of how {exp_n} relate to {out_n} under a common "
            f"analytic framework. We estimated these associations in {src}."),
        "Methods":(f"We analyzed {src} participants using complex-survey methods that account for stratification, clustering "
            f"and sampling weights. Smoking and alcohol were grouped as never versus past or current. Each outcome was "
            f"analyzed in its own complete-data sample. Continuous exposures were modeled per 1-SD. Binary outcomes used "
            f"survey-weighted logistic regression adjusting for age, sex and lifestyle where not collinear with the exposure "
            f"or a determinant of the outcome. Multiplicity across all exposure-outcome pairs was controlled with the "
            f"Benjamini-Hochberg false discovery rate."),
        "Results":(f"A total of {npairs} exposure-outcome associations were estimated and {nsig} remained significant after "
            f"false discovery rate control. "+" ".join(rlines)),
        "Discussion":("These survey-weighted associations are hypothesis-generating and require external validation. The "
            "cross-sectional design precludes causal inference. Hepatic steatosis was measured by a surrogate index in "
            "KNHANES and by controlled attenuation parameter in NHANES, so steatosis and MASLD are not directly comparable "
            "across surveys. A single survey weight was used across outcomes."),
        "Conclusion":(f"In {src}, {exp_n} showed reproducible survey-weighted associations with multiple cardiometabolic and "
            f"related outcomes that warrant prospective study.")}

def build_full_manuscript_docx(RES, info, sections, dataset, exposures, outcomes, t1_primary=None):
    from docx import Document
    src=_src_name(dataset)
    doc=Document()
    title=(f"Associations of {', '.join(lab(e) for e in exposures)} with cardiometabolic and related outcomes "
           f"in the {dataset}: a survey-weighted cross-sectional study")
    doc.add_heading(title,0)
    doc.add_paragraph(f"Data: {src} · all exposure-outcome pairs · survey-weighted · prose source: {sections.get('_source','fallback')}")
    for nm in ["Abstract","Introduction","Methods","Results","Discussion","Conclusion"]:
        doc.add_heading(nm,1); doc.add_paragraph(sections.get(nm,""))
    def add_tab(title, df, fmt=None):
        doc.add_heading(title,1); cols=list(df.columns)
        tb=doc.add_table(rows=1,cols=len(cols)); tb.style="Light Grid Accent 1"
        for j,c in enumerate(cols): tb.rows[0].cells[j].text=str(c)
        for _,row in df.iterrows():
            cc=tb.add_row().cells
            for j,c in enumerate(cols): cc[j].text=str(row[c])
    if t1_primary is not None:
        add_tab("Table 1. Descriptive characteristics (representative outcome)", t1_primary)
    if len(RES):
        M=RES.sort_values(["outcome","q"]).copy()
        M["Exposure"]=M.exposure.str[2:].map(lab); M["Outcome"]=M.outcome.map(lab)
        M["Measure"]=M.measure.map(measure_name)
        M["Estimate (95% CI)"]=M.apply(lambda x:f"{x.est:.2f} ({x.ci_low:.2f}-{x.ci_high:.2f})",axis=1)
        M["FDR q"]=M.q.map(lambda q:"<0.001" if q<0.001 else f"{q:.3f}"); M["N"]=M.n
        add_tab("Table 2. All exposure-outcome associations (FDR-adjusted)",
                M[["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q","N"]])
    add_tab("Table 3. Per-outcome analytic sample and status", info)
    doc.add_paragraph("Statistics are deterministic survey-weighted estimates (R survey). Text was written from the "
                      "computed values without recomputation. All findings are hypothesis-generating.")
    import io; buf=io.BytesIO(); doc.save(buf); return buf.getvalue()

def nl_to_config(question, dataset, model, url, use_llm):
    av=AVAIL[dataset]
    if use_llm:
        try:
            o=json.loads(ollama_chat(f"Map the question to an analysis config. Available variables (shared exposure/outcome)={av}. "
                f"Return only JSON {{'exposures':[],'outcomes':[]}}.\nQuestion: {question}",model,url,fmt="json"))
            e=[x for x in o.get("exposures",[]) if x in av]; oo=[x for x in o.get("outcomes",[]) if x in av]
            if e and oo: return {"exposures":e,"outcomes":oo,"covariates":auto_covariates(dataset,e,oo)}
        except Exception: pass
    q=question.lower()
    kw={"지방간":"steatosis","masld":"masld","당뇨":"dm","전단계":"prediabetes","고혈압":"htn","대사증후군":"mets",
        "이상지질":"dyslipidemia","콩팥":"ckd","신장":"ckd","섬유화":"adv_fibrosis","빈혈":"anemia",
        "비만":"obesity","복부":"abdominal_obesity","근육":"asm_pct","체지방":"bodyfat_pct",
        "bmi":"bmi","허리":"wc","흡연":"smoking","음주":"alcohol","혈당":"glucose","중성지방":"tg","콜레스테롤":"tchol"}
    hits=[v for k,v in kw.items() if k in q and v in av]
    e=[hits[0]] if hits else (["bodyfat_pct"] if "bodyfat_pct" in av else [av[0]])
    oo=[hits[1]] if len(hits)>1 else (["dm"] if "dm" in av else ["obesity"])
    return {"exposures":e,"outcomes":oo,"covariates":auto_covariates(dataset,e,oo)}

# ── Trajectory ────────────────────────────────────────────────────────────────────────
def trajectory(d, dataset, outcome, split_sex=False, bin_width=5, age_range=(20,80)):
    vs=[v for v in determinants(dataset,outcome) if v in d.columns and d[v].notna().any()]
    dd=d[(d.age>=age_range[0])&(d.age<=age_range[1])&d.wt_pool.notna()&(d.wt_pool>0)].copy()
    dd["agebin"]=(np.floor(dd.age/bin_width)*bin_width).astype(int)
    def wmean(sub,col):
        s=sub[[col,"wt_pool"]].dropna()
        return float(np.average(s[col].astype(float),weights=s.wt_pool)) if len(s) else np.nan
    rows=[]; grouped=dd.groupby(["agebin","men"]) if split_sex else dd.groupby("agebin")
    for kv,sub in grouped:
        if split_sex: ab,mn=kv; sx="Men" if mn==1 else "Women"
        else: ab=kv; sx="All"
        for v in vs: rows.append({"agebin":int(ab),"sex":sx,"variable":lab(v),"value":wmean(sub,v)})
        rows.append({"agebin":int(ab),"sex":sx,"variable":f"{lab(outcome)} prevalence, %","value":wmean(sub,outcome)*100})
    return pd.DataFrame(rows), vs

def plot_trajectory(df, outcome, dataset, split_sex=False):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels=list(dict.fromkeys(df.variable)); n=len(panels); ncol=min(3,n) or 1; nrow=(n+ncol-1)//ncol
    fig,axes=plt.subplots(nrow,ncol,figsize=(4*ncol,3*nrow),squeeze=False)
    for i,pan in enumerate(panels):
        ax=axes[i//ncol][i%ncol]; sub=df[df.variable==pan]
        if split_sex:
            for sx,g in sub.groupby("sex"): ax.plot(g.agebin,g.value,marker="o",label=sx)
            ax.legend(fontsize=7)
        else: ax.plot(sub.agebin,sub.value,marker="o",color="#0F6E56")
        ax.set_title(pan,fontsize=9); ax.set_xlabel("Age, years",fontsize=8); ax.grid(alpha=0.3)
    for j in range(n,nrow*ncol): axes[j//ncol][j%ncol].axis("off")
    fig.suptitle(f"{lab(outcome)} determinants — age trajectory ({dataset}, survey-weighted)",fontsize=11)
    fig.tight_layout(); return fig

def fig_to_b64(fig):
    import io,base64,matplotlib.pyplot as plt
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=90,bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── Static HTML report ────────────────────────────────────────────────────────────────
def build_html_report(dataset, config, t1, res, manuscript, traj=None):
    import html as _h
    res=add_fdr(res); rows=""
    for _,r in res.sort_values("q").iterrows():
        risk=(r.measure=="OR" and r.est>1) or (r.measure=="beta" and r.est>0)
        col="#A32D2D" if risk else "#0F6E56"; sig="background:#EAF3DE;" if r.q<0.05 else ""
        rows+=(f'<tr style="{sig}"><td>{_h.escape(lab(r.exposure[2:]))}</td><td>{_h.escape(lab(r.outcome))}</td>'
               f'<td>{measure_name(r.measure)}</td>'
               f'<td style="color:{col};text-align:right;font-weight:500">{effect_str(r)}</td>'
               f'<td style="text-align:right">{"<0.001" if r.q<0.001 else f"{r.q:.3f}"}</td></tr>')
    t1_html=t1.to_html(index=False,border=0)
    ms_html="".join(f'<h2>{k}</h2><p>{_h.escape(str(manuscript.get(k,"")))}</p>' for k in ["Methods","Results","Discussion"])
    traj_html="".join(f'<div><h3>{_h.escape(t)}</h3><img src="data:image/png;base64,{b}"></div>' for t,b in (traj or []))
    exp_n=", ".join(lab(e) for e in config.get("exp",[])); out_n=", ".join(lab(o) for o in config.get("out",[]))
    cov_n=", ".join(lab(c) for c in config.get("cov",[])); src=manuscript.get("_source","fallback")
    return f"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8><title>Evidence Report — {dataset}</title><style>
body{{{{font-family:system-ui,'Segoe UI',sans-serif;max-width:1000px;margin:20px auto;padding:0 18px;color:#2c2c2a;line-height:1.5}}}}
h1{{{{font-size:22px;margin-bottom:2px}}}} h2{{{{font-size:17px;border-bottom:2px solid #d3d1c7;padding-bottom:4px;margin-top:26px}}}}
.bar{{{{background:#F1EFE8;border-radius:10px;padding:12px 16px;font-size:13px;margin:12px 0}}}}
table{{{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}}} th,td{{{{border-bottom:1px solid #e7e7e2;padding:7px;text-align:left}}}}
th{{{{background:#fafaf7;border-bottom:2px solid #d3d1c7}}}} img{{{{max-width:100%;border:1px solid #eee;border-radius:6px}}}}</style></head><body>
<h1>Evidence Report — {dataset}</h1>
<div style="color:#666;font-size:14px">Survey-weighted. Continuous outcome linear beta, binary outcome logistic OR. Steatosis: {STEATOSIS_METHOD[dataset]}</div>
<div class=bar><b>Exposure</b>: {_h.escape(exp_n)} &nbsp;|&nbsp; <b>Outcome</b>: {_h.escape(out_n)} &nbsp;|&nbsp;
<b>Adjustment</b>: {_h.escape(cov_n)} &nbsp;|&nbsp; text: {src}</div>
{ms_html}
<h2>Table 1. Descriptive characteristics</h2>{t1_html}
<h2>Table 2. Adjusted associations</h2>
<table><thead><tr><th>Exposure</th><th>Outcome</th><th>Measure</th><th style=text-align:right>Estimate (95% CI)</th><th style=text-align:right>FDR q</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Age trajectory of outcome determinants</h2>{traj_html or "<p>(none)</p>"}
<p style="color:#888;font-size:12px;margin-top:18px">Statistics are deterministic survey-weighted estimates (R survey). Text is written from computed values without recomputation. All findings are hypothesis-generating.</p>
</body></html>"""
