# -*- coding: utf-8 -*-
"""Review section 7, association side: refit the NHANES associations whose OUTCOME
is fasting-dependent under WTSAF2YR and compare with the published WTMEC2YR fit.

Both arms are restricted to the fasting subsample so the comparison isolates the
weight. Same exposures, same adjustment sets, same engine.
"""
import os, sys, glob
import numpy as np, pandas as pd, pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)                  # repository root; raw files in data/NHANES
sys.path.insert(0, PROJ); os.chdir(PROJ)
import factory_core as fc

CYCLES = ("d","e","f","g","h","i","j","l"); NC = len(CYCLES); AGE = 20
DEFS = {**fc.DEFAULT_DEFS, "pop": {"age_min": AGE}}
OUTS = ["dm", "mets", "insulin_resistance", "high_tg"]

saf = []
for f in sorted(glob.glob(os.path.join(PROJ, "data", "NHANES", "TRIGLY_*.XPT"))):
    df, _ = pyreadstat.read_xport(f, encoding="latin1")
    if "WTSAF2YR" in df.columns: saf.append(df[["SEQN", "WTSAF2YR"]])
SAF = pd.concat(saf, ignore_index=True).drop_duplicates("SEQN")

raw = fc.load_raw("NHANES", os.path.join(PROJ, "data", "NHANES"), CYCLES)
d = fc.apply_definitions(raw, DEFS)
d["SEQN"] = pd.to_numeric(raw["SEQN"], errors="coerce")
d = d.merge(SAF, on="SEQN", how="left")
d["w_saf"] = pd.to_numeric(d["WTSAF2YR"], errors="coerce") / NC
d = d[d.w_saf.notna() & (d.w_saf > 0)].reset_index(drop=True)
print(f"fasting subsample rows: {len(d)}")

def fit(dd, O, tag):
    exps = fc.association_exposures("NHANES", O)
    groups = {}
    cfset = set(fc.confounders_for("NHANES", O))
    for e in exps:
        if e in cfset:
            cov = () if e in ("age", "men") else tuple(c for c in ("age", "men") if c in cfset and c != e)
        else:
            cov = tuple(fc.adjustment_for("NHANES", O, e))
        groups.setdefault(cov, []).append(e)
    ana = fc.build_analytic(dd, exps, [O], [], AGE)
    out = []
    wrote = False
    for adj, es in groups.items():
        _, res, _, _ = fc.run_engine(ana, "NHANES", es, [O], list(adj),
                                     skip_table1=True, reuse_analytic=wrote)
        wrote = True
        for _, r in res.iterrows():
            out.append(dict(exp=r.exposure[2:], out=O, measure=r.measure,
                            **{f"est_{tag}": r.est, f"p_{tag}": r.p}))
    return pd.DataFrame(out)

frames = []
for O in OUTS:
    dm_ = d.copy(); dm_["wt_pool"] = pd.to_numeric(dm_["wt_pool"], errors="coerce")
    ds_ = d.copy(); ds_["wt_pool"] = ds_["w_saf"]
    a = fit(dm_, O, "mec"); b = fit(ds_, O, "saf")
    m = a.merge(b, on=["exp", "out", "measure"])
    print(f"  {O}: {len(m)} pairs refitted under both weights")
    frames.append(m)

M = pd.concat(frames, ignore_index=True)
OR = M[M.measure == "OR"].copy()
OR["lmec"], OR["lsaf"] = np.log(OR.est_mec), np.log(OR.est_saf)
OR["pchg"] = 100 * (OR.est_saf - OR.est_mec) / OR.est_mec
same_dir = (np.sign(OR.lmec) == np.sign(OR.lsaf)).mean()
from scipy.stats import pearsonr
print(f"\nodds-ratio pairs compared           : {len(OR)}")
print(f"correlation of log OR (MEC vs SAF)  : r = {pearsonr(OR.lmec, OR.lsaf)[0]:.4f}")
print(f"same direction                      : {100*same_dir:.1f}%")
print(f"median |% change| in OR             : {OR.pchg.abs().median():.2f}%")
print(f"90th percentile |% change|          : {np.percentile(OR.pchg.abs(),90):.2f}%")
print(f"max |% change|                      : {OR.pchg.abs().max():.2f}%")
agree = ((OR.p_mec < 0.05) == (OR.p_saf < 0.05)).mean()
print(f"nominal significance agreement      : {100*agree:.1f}%")
out = os.path.join(HERE, "fasting_weight_associations.csv")
M.to_csv(out, index=False, encoding="utf-8-sig"); print("\nwrote", out)
