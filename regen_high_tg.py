# -*- coding: utf-8 -*-
"""Re-run the NHANES high-triglyceride prediction model under the identity guard.

NHANES releases LBDLDL as the Friedewald calculated LDL, so TC - HDL - LDL = TG/5 holds
exactly and a model given all three computes the triglyceride threshold instead of
predicting it. factory_core._ML_IDENTITY_EXCL now removes the lipid panel from this one
model in this one survey; see the comment there for the numbers that establish the
identity. Nothing else in the corpus changes: the association engine admits one exposure
at a time, and the identity needs three lipids at once.

Rewrites the NHANES report and its row of _ml_regen_summary.csv, and for information
also fits the KNHANES counterpart without its lipid panel, which is NOT released - the
KNHANES LDL is a direct assay and its model is left as published.
"""
import os
import sys
import time

import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import factory_core as fc                                            # noqa: E402
import ml_report                                                     # noqa: E402
from suppl_generator import slug, SUPPL, DEFS, DATA, ML_MODELS, ML_SAMPLE_N   # noqa: E402

OUTCOME = "high_tg"
SUMMARY = os.path.join(PROJ, "_ml_regen_summary.csv")


def fit(ds, DF, feats, write):
    t0 = time.time()
    summary = {}
    doc = ml_report.build_ml_report(ds, DF, OUTCOME, feats, DEFS, "", "", False,
                                    models=ML_MODELS, sample_n=ML_SAMPLE_N,
                                    summary=summary)
    if write:
        path = os.path.join(SUPPL, ds, "ml", f"{slug(OUTCOME)}__prediction_{ds}.docx")
        open(path, "wb").write(doc)
        print(f"    wrote {os.path.basename(path)}")
    s = summary.get("ml", {})
    s["secs"] = round(time.time() - t0, 1)
    return s


def main():
    old = pd.read_csv(SUMMARY)
    print("before:")
    print(old[old.outcome == OUTCOME][["survey", "best", "auROC", "auPRC", "n_feats"]]
          .to_string(index=False))

    rows = {}
    for ds in ("NHANES", "KNHANES"):
        DF = fc.load_raw(ds, *DATA[ds])
        feats = fc.ml_features(ds, OUTCOME)
        print(f"\n  {ds}: {len(feats)} features, lipid panel "
              f"{'removed' if not {'tchol','hdl','ldl','nonhdl'} & set(feats) else 'retained'}")
        # NHANES is the released re-run; KNHANES is refitted without lipids for
        # comparison only, so its released report is not touched.
        rows[ds] = fit(ds, DF, feats, write=(ds == "NHANES"))
        print(f"    {rows[ds].get('best')}  auROC {rows[ds].get('auROC')}  "
              f"auPRC {rows[ds].get('auPRC')}  ({rows[ds]['secs']}s)")
        if ds == "KNHANES":
            lip = [v for v in fc.ml_features(ds, OUTCOME)]
            bare = fit(ds, DF, [v for v in lip if v not in {"tchol", "hdl", "ldl", "nonhdl"}],
                       write=False)
            print(f"    for information, KNHANES without its lipid panel: "
                  f"{bare.get('best')} auROC {bare.get('auROC')}")
            rows["KNHANES_nolipid"] = bare

    s = rows["NHANES"]
    m = (old.survey == "NHANES") & (old.outcome == OUTCOME)
    for k in ("best", "auROC", "auPRC", "n", "prev", "n_feats", "secs"):
        old.loc[m, k] = s.get(k)
    old.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    print("\nafter:")
    print(old[old.outcome == OUTCOME][["survey", "best", "auROC", "auPRC", "n_feats"]]
          .to_string(index=False))
    print(f"\nhighest auROC in the corpus is now "
          f"{old.auROC.max():.4f} ({old.loc[old.auROC.idxmax(), 'survey']} "
          f"{old.loc[old.auROC.idxmax(), 'outcome']})")


if __name__ == "__main__":
    main()
