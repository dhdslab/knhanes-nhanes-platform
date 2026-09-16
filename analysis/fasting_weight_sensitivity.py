# -*- coding: utf-8 -*-
"""Review section 7: re-estimate the fasting-dependent NHANES outcomes with the
fasting subsample weight (WTSAF2YR) instead of the examination weight (WTMEC2YR).

NHANES releases WTSAF2YR for the morning fasting subsample; the platform used
WTMEC2YR throughout. The reviewer asks that this be done, not deferred to a
limitation. Three columns are reported so the weight effect is separated from
the effect of restricting to the fasting subsample:

  (1) published : WTMEC2YR, every participant whose outcome is defined
  (2) MEC/fast  : WTMEC2YR, restricted to the fasting subsample
  (3) SAF/fast  : WTSAF2YR, restricted to the fasting subsample   <- appropriate

(1) vs (3) is the net change to the paper; (2) vs (3) isolates the weight itself.
"""
import os, sys, glob, re
import numpy as np, pandas as pd, pyreadstat

PROJ = r"c:\Users\IMDL\Desktop\NHANES 부수기\knhanes_platform"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import factory_core as fc

CYCLES = ("d", "e", "f", "g", "h", "i", "j", "l")
NC = len(CYCLES)
AGE = 20
DEFS = {**fc.DEFAULT_DEFS, "pop": {"age_min": AGE}}

# outcomes whose definition uses a fasting analyte (glucose, triglycerides, insulin, LDL)
FASTING_OUTCOMES = ["dm", "prediabetes", "mets", "insulin_resistance", "high_tg",
                    "high_ldl", "atherogenic_dyslipidemia", "dyslipidemia", "masld"]

# ── fasting subsample weight, pooled the same way as the examination weight ──
saf = []
for f in sorted(glob.glob(os.path.join(PROJ, "data", "NHANES", "TRIGLY_*.XPT"))):
    df, _ = pyreadstat.read_xport(f, encoding="latin1")
    if "WTSAF2YR" in df.columns:
        saf.append(df[["SEQN", "WTSAF2YR"]])
SAF = pd.concat(saf, ignore_index=True).drop_duplicates("SEQN")
print(f"WTSAF2YR rows: {len(SAF)} across {len(saf)} cycle files")

raw = fc.load_raw("NHANES", os.path.join(PROJ, "data", "NHANES"), CYCLES)
d = fc.apply_definitions(raw, DEFS)
d = d[d.age >= AGE].copy()
d["SEQN"] = pd.to_numeric(raw.loc[d.index, "SEQN"], errors="coerce")
d = d.merge(SAF, on="SEQN", how="left")

d["w_mec"] = pd.to_numeric(d["wt_pool"], errors="coerce")
d["w_saf"] = pd.to_numeric(d["WTSAF2YR"], errors="coerce") / NC
fastsub = d.w_saf.notna() & (d.w_saf > 0)
print(f"adults 20+: {len(d)}   in fasting subsample (WTSAF2YR>0): {int(fastsub.sum())}"
      f"  ({100*fastsub.mean():.1f}%)\n")


def wprev(col, wcol, mask):
    y = pd.to_numeric(d[col], errors="coerce")
    w = d[wcol]
    m = mask & y.notna() & w.notna() & (w > 0)
    if int(m.sum()) == 0:
        return np.nan, 0
    return 100 * np.average((y[m] == 1).astype(float), weights=w[m]), int(m.sum())


rows = []
allm = pd.Series(True, index=d.index)
print(f"{'outcome':<26} {'(1) published':>15} {'(2) MEC/fast':>15} {'(3) SAF/fast':>15}   {'(3)-(1)':>8}")
for o in FASTING_OUTCOMES:
    if o not in d.columns:
        print(f"{o:<26} not available"); continue
    p1, n1 = wprev(o, "w_mec", allm)
    p2, n2 = wprev(o, "w_mec", fastsub)
    p3, n3 = wprev(o, "w_saf", fastsub)
    rows.append(dict(outcome=o, label=fc.lab(o), published=p1, n_published=n1,
                     mec_fasting=p2, saf_fasting=p3, n_fasting=n3,
                     diff_saf_minus_published=p3 - p1, diff_saf_minus_mec=p3 - p2))
    print(f"{fc.lab(o)[:25]:<26} {p1:>13.1f}% {p2:>13.1f}% {p3:>13.1f}%   {p3-p1:>+7.1f}")

R = pd.DataFrame(rows)
print(f"\nmedian |(3)-(1)| = {R.diff_saf_minus_published.abs().median():.2f} pp,"
      f"  max = {R.diff_saf_minus_published.abs().max():.2f} pp")
print(f"median |(3)-(2)| = {R.diff_saf_minus_mec.abs().median():.2f} pp,"
      f"  max = {R.diff_saf_minus_mec.abs().max():.2f} pp   (pure weight effect)")
out = os.path.join(r"c:\Users\IMDL\Desktop\NHANES 부수기\최종본\_src",
                   "fasting_weight_sensitivity.csv")
R.to_csv(out, index=False, encoding="utf-8-sig")
print("\nwrote", out)
