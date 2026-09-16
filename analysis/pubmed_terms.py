# -*- coding: utf-8 -*-
"""PubMed query fragments for every variable in the corpus.

Term choice drives every number in the literature-coverage analysis, so the map
lives in its own file and is meant to be read and edited by a domain expert
before the results are used. Each entry is an OR-block combining a MeSH term
where a clean one exists with title/abstract synonyms.

Conventions
  - [Mesh] terms are not exploded manually; PubMed explodes by default.
  - [tiab] synonyms cover the spellings actually used in survey epidemiology.
  - Abbreviations that are ambiguous in isolation (ALT, AST, CAP, LSM, WC) are
    given with a disambiguating phrase rather than the bare acronym.
"""

SURVEY = ('("Nutrition Surveys"[Mesh] OR NHANES[tiab] OR '
          '"National Health and Nutrition Examination Survey"[tiab] OR KNHANES[tiab] OR '
          '"Korea National Health and Nutrition Examination Survey"[tiab] OR '
          '"Korean National Health and Nutrition Examination Survey"[tiab])')

TERMS = {
 # ── adiposity / anthropometry ──────────────────────────────────────────────
 "bmi": '"Body Mass Index"[Mesh] OR "body mass index"[tiab] OR BMI[tiab]',
 "obesity": '"Obesity"[Mesh] OR obesity[tiab] OR obese[tiab]',
 "overweight": '"Overweight"[Mesh] OR overweight[tiab]',
 "wc": '"Waist Circumference"[Mesh] OR "waist circumference"[tiab]',
 "whtr": '"waist-to-height ratio"[tiab] OR "waist to height ratio"[tiab] OR "waist-height ratio"[tiab]',
 "abdominal_obesity": '"Obesity, Abdominal"[Mesh] OR "abdominal obesity"[tiab] OR "central adiposity"[tiab]',
 "central_obesity": '"central obesity"[tiab] OR "waist-to-height ratio"[tiab]',
 "bodyfat_pct": '"Adiposity"[Mesh] OR "body fat percentage"[tiab] OR "percent body fat"[tiab] OR "body fat percent"[tiab]',
 "high_bodyfat": '"high body fat"[tiab] OR "excess adiposity"[tiab] OR "Adiposity"[Mesh]',
 "fat_kg": '"fat mass"[tiab] OR "Adipose Tissue"[Mesh]',
 "lean_kg": '"lean mass"[tiab] OR "lean body mass"[tiab] OR "fat-free mass"[tiab]',
 "weight": '"Body Weight"[Mesh] OR "body weight"[tiab]',
 "height": '"Body Height"[Mesh] OR "body height"[tiab] OR stature[tiab]',

 # ── muscle / bone ──────────────────────────────────────────────────────────
 "asmm_kg": '"appendicular skeletal muscle mass"[tiab] OR "appendicular lean mass"[tiab]',
 "asmi": '"appendicular skeletal muscle mass index"[tiab] OR "skeletal muscle mass index"[tiab] OR "appendicular lean mass index"[tiab]',
 "asm_pct": '"appendicular skeletal muscle"[tiab] OR "skeletal muscle percentage"[tiab]',
 "sarcopenia": '"Sarcopenia"[Mesh] OR sarcopenia[tiab] OR "low muscle mass"[tiab]',
 "osteoporosis": '"Osteoporosis"[Mesh] OR osteoporosis[tiab] OR "low bone mineral density"[tiab]',
 "fn_bmd": '"Bone Density"[Mesh] OR "femoral neck bone mineral density"[tiab] OR "femoral neck BMD"[tiab]',
 "ls_bmd": '"Bone Density"[Mesh] OR "lumbar spine bone mineral density"[tiab] OR "lumbar spine BMD"[tiab]',

 # ── glycaemia ──────────────────────────────────────────────────────────────
 "dm": '"Diabetes Mellitus, Type 2"[Mesh] OR "Diabetes Mellitus"[Mesh] OR diabetes[tiab] OR diabetic[tiab]',
 "prediabetes": '"Prediabetic State"[Mesh] OR prediabetes[tiab] OR "pre-diabetes"[tiab] OR "impaired fasting glucose"[tiab]',
 "glucose": '"Blood Glucose"[Mesh] OR "fasting glucose"[tiab] OR "fasting plasma glucose"[tiab] OR "blood glucose"[tiab]',
 "hba1c": '"Glycated Hemoglobin"[Mesh] OR HbA1c[tiab] OR "glycated hemoglobin"[tiab] OR "glycosylated hemoglobin"[tiab]',
 "insulin": '"Insulin"[Mesh] OR "fasting insulin"[tiab] OR "serum insulin"[tiab]',
 "homa_ir": '"HOMA-IR"[tiab] OR "homeostasis model assessment"[tiab] OR "homeostatic model assessment"[tiab]',
 "insulin_resistance": '"Insulin Resistance"[Mesh] OR "insulin resistance"[tiab] OR "HOMA-IR"[tiab]',

 # ── blood pressure ─────────────────────────────────────────────────────────
 "htn": '"Hypertension"[Mesh] OR hypertension[tiab] OR hypertensive[tiab] OR "high blood pressure"[tiab]',
 "sbp": '"systolic blood pressure"[tiab] OR "Blood Pressure"[Mesh]',
 "dbp": '"diastolic blood pressure"[tiab] OR "Blood Pressure"[Mesh]',

 # ── lipids ─────────────────────────────────────────────────────────────────
 "tchol": '"Cholesterol"[Mesh] OR "total cholesterol"[tiab]',
 "hdl": '"Cholesterol, HDL"[Mesh] OR "HDL cholesterol"[tiab] OR "high-density lipoprotein"[tiab]',
 "ldl": '"Cholesterol, LDL"[Mesh] OR "LDL cholesterol"[tiab] OR "low-density lipoprotein"[tiab]',
 "nonhdl": '"non-HDL cholesterol"[tiab] OR "non high-density lipoprotein cholesterol"[tiab]',
 "tg": '"Triglycerides"[Mesh] OR triglyceride[tiab] OR triglycerides[tiab]',
 "low_hdl": '"low HDL cholesterol"[tiab] OR "low high-density lipoprotein"[tiab] OR "Cholesterol, HDL"[Mesh]',
 "high_ldl": '"high LDL cholesterol"[tiab] OR hypercholesterolemia[tiab] OR "Hypercholesterolemia"[Mesh]',
 "high_nonhdl": '"non-HDL cholesterol"[tiab] OR "Dyslipidemias"[Mesh]',
 "high_tg": '"Hypertriglyceridemia"[Mesh] OR hypertriglyceridemia[tiab] OR "elevated triglycerides"[tiab]',
 "dyslipidemia": '"Dyslipidemias"[Mesh] OR dyslipidemia[tiab] OR dyslipidaemia[tiab] OR hyperlipidemia[tiab]',
 "atherogenic_dyslipidemia": '"atherogenic dyslipidemia"[tiab] OR "atherogenic dyslipidaemia"[tiab] OR "Dyslipidemias"[Mesh]',
 "mets": '"Metabolic Syndrome"[Mesh] OR "metabolic syndrome"[tiab]',

 # ── liver ──────────────────────────────────────────────────────────────────
 "alt": '"Alanine Transaminase"[Mesh] OR "alanine aminotransferase"[tiab] OR "serum ALT"[tiab]',
 "ast": '"Aspartate Aminotransferases"[Mesh] OR "aspartate aminotransferase"[tiab] OR "serum AST"[tiab]',
 "high_alt": '"elevated alanine aminotransferase"[tiab] OR "elevated ALT"[tiab] OR "Alanine Transaminase"[Mesh]',
 "high_ast": '"elevated aspartate aminotransferase"[tiab] OR "elevated AST"[tiab] OR "Aspartate Aminotransferases"[Mesh]',
 "ggt": '"gamma-Glutamyltransferase"[Mesh] OR "gamma-glutamyl transferase"[tiab] OR "gamma glutamyl transferase"[tiab] OR GGT[tiab]',
 "steatosis": '"Fatty Liver"[Mesh] OR "hepatic steatosis"[tiab] OR "fatty liver"[tiab] OR NAFLD[tiab] OR "Non-alcoholic Fatty Liver Disease"[Mesh]',
 "masld": 'MASLD[tiab] OR "metabolic dysfunction-associated steatotic liver disease"[tiab] OR "metabolic dysfunction associated fatty liver"[tiab] OR MAFLD[tiab]',
 "hsi": '"hepatic steatosis index"[tiab]',
 "fib4": '"FIB-4"[tiab] OR "fibrosis-4"[tiab]',
 "adv_fibrosis": '"Liver Cirrhosis"[Mesh] OR "advanced fibrosis"[tiab] OR "liver fibrosis"[tiab]',
 "cap": '"controlled attenuation parameter"[tiab]',
 "lsm": '"liver stiffness"[tiab] OR "transient elastography"[tiab] OR "Elasticity Imaging Techniques"[Mesh]',
 "platelet": '"Platelet Count"[Mesh] OR "platelet count"[tiab]',

 # ── kidney ─────────────────────────────────────────────────────────────────
 "ckd": '"Renal Insufficiency, Chronic"[Mesh] OR "chronic kidney disease"[tiab] OR "chronic renal"[tiab]',
 "egfr": '"Glomerular Filtration Rate"[Mesh] OR "glomerular filtration rate"[tiab] OR eGFR[tiab]',
 "creatinine": '"Creatinine"[Mesh] OR "serum creatinine"[tiab]',
 "uric": '"Uric Acid"[Mesh] OR "uric acid"[tiab] OR hyperuricemia[tiab]',

 # ── haematology / other ────────────────────────────────────────────────────
 "anemia": '"Anemia"[Mesh] OR anemia[tiab] OR anaemia[tiab]',
 "hemoglobin": '"Hemoglobins"[Mesh] OR hemoglobin[tiab] OR haemoglobin[tiab]',
 "wbc": '"Leukocyte Count"[Mesh] OR "white blood cell count"[tiab] OR "leukocyte count"[tiab]',
 "leukocytosis": '"Leukocytosis"[Mesh] OR leukocytosis[tiab] OR "elevated white blood cell"[tiab]',
 "vitd": '"Vitamin D"[Mesh] OR "vitamin D"[tiab] OR "25-hydroxyvitamin D"[tiab] OR "25(OH)D"[tiab]',
 "vitd_deficiency": '"Vitamin D Deficiency"[Mesh] OR "vitamin D deficiency"[tiab] OR "vitamin D insufficiency"[tiab]',

 # ── demographics / behaviour ───────────────────────────────────────────────
 "age": '"Age Factors"[Mesh] OR aging[tiab] OR "older adults"[tiab]',
 "men": '"Sex Factors"[Mesh] OR "sex difference"[tiab] OR "sex differences"[tiab] OR "gender difference"[tiab]',
 "smoking": '"Smoking"[Mesh] OR smoking[tiab] OR smoker[tiab] OR smokers[tiab]',
 "current_smoking": '"Smoking"[Mesh] OR "current smoking"[tiab] OR "current smoker"[tiab]',
 "alcohol": '"Alcohol Drinking"[Mesh] OR "alcohol consumption"[tiab] OR "alcohol intake"[tiab] OR drinking[tiab]',
}


def concept(var):
    """OR-block for one variable, or None if the variable has no usable query."""
    t = TERMS.get(var)
    return f"({t})" if t else None


def pair_query(a, b):
    ca, cb = concept(a), concept(b)
    if not ca or not cb:
        return None
    return f"{SURVEY} AND {ca} AND {cb}"


def single_query(a):
    ca = concept(a)
    return f"{SURVEY} AND {ca}" if ca else None
