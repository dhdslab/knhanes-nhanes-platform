# -*- coding: utf-8 -*-
"""Regenerate all machine-learning reports after the model-selection fix.

Before: the winning model was chosen by its held-out test-set score (ml_report.py),
        so the 30% partition had entered model selection and was not independent.
After:  hyperparameter search AND between-model selection both use only
        training-partition cross-validation (mean of CV auROC and CV auPRC);
        the selected model is refitted on the full training partition and
        evaluated once on the untouched held-out partition.

Everything else (3-model registry, 6,000-participant subsample, seeds, features,
leakage guard) is unchanged, so the regenerated corpus is comparable to the old one.
"""
import sys, os, time, json, traceback
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ)
import factory_core as fc
import ml_report
import pandas as pd
from suppl_generator import slug, SUPPL, AGE, DEFS, DATA, ML_MODELS, ML_SAMPLE_N

OUT = os.path.join(PROJ, "_ml_regen_summary.csv")
LOGP = os.path.join(PROJ, "_ml_regen.log")
LOG = open(LOGP, "w", encoding="utf-8")


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def main():
    t0 = time.time(); rows = []
    for ds in ["KNHANES", "NHANES"]:
        ddir, cyc = DATA[ds]
        DF = fc.load_raw(ds, ddir, cyc)
        d = fc.apply_definitions(DF, DEFS)
        outs = [o for o in fc.OUTCOMES if o in fc.AVAIL[ds]]
        outdir = os.path.join(SUPPL, ds, "ml"); os.makedirs(outdir, exist_ok=True)
        for oi, O in enumerate(outs):
            ev = int((pd.to_numeric(d[O], errors="coerce") == 1).sum()) if O in d else 0
            if ev < fc.MIN_EVENTS:
                log(f"  [{ds} ml {oi+1}/{len(outs)}] {O}: SKIP {ev} events"); continue
            try:
                t1 = time.time()
                feats = fc.ml_features(ds, O)
                summary = {}
                docx = ml_report.build_ml_report(ds, DF, O, feats, DEFS, "", "", False,
                                                 models=ML_MODELS, sample_n=ML_SAMPLE_N,
                                                 summary=summary)
                path = os.path.join(outdir, f"{slug(O)}__prediction_{ds}.docx")
                open(path, "wb").write(docx)
                s = summary.get("ml", {})
                rows.append(dict(survey=ds, outcome=O, label=fc.lab(O), best=s.get("best"),
                                 auROC=s.get("auROC"), auPRC=s.get("auPRC"),
                                 n=s.get("n"), prev=s.get("prev"), n_feats=s.get("n_feats"),
                                 secs=round(time.time() - t1, 1)))
                log(f"  [{ds} ml {oi+1}/{len(outs)}] {O}: {s.get('best')} "
                    f"auROC={s.get('auROC')} ({time.time()-t1:.1f}s)")
            except Exception as ex:
                log(f"  [{ds} ml] {O}: ERROR {ex}"); LOG.write(traceback.format_exc() + "\n")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    log(f"ML REGEN DONE: {len(rows)} reports in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
