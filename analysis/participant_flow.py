# -*- coding: utf-8 -*-
"""Participant flow cascade and design degrees of freedom (STROBE 13a, Reporting Summary 7).

STROBE item 13a asks for the number of individuals at each stage. The platform never
materialises such a cascade because it filters inside `build_analytic`, so the numbers
have to be recovered by walking the same filter in the same order the analysis applies
it:

    released records  ->  aged 20 or over  ->  valid survey design  ->  analysed

`build_analytic` applies exactly one compound condition:

    (age >= age_min) & (wt_pool > 0) & wt_pool.notna() & kstrata.notna() & psu.notna()

so "valid survey design" means a positive, non-missing pooled weight together with a
non-missing stratum and primary sampling unit. For NHANES the released file is the
interviewed sample and the pooled weight is the examination weight, so the design step
is where interviewed-but-not-examined participants leave; for KNHANES the released file
is already the examined sample.

The design degrees of freedom are computed as the number of primary sampling units minus
the number of strata, counting primary sampling units as distinct (stratum, unit) pairs
because the design object is built with nest = TRUE and the NHANES unit code is only
unique within its stratum.

Output: analysis/participant_flow.csv, and the numbers printed for transcription.
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PLAT = os.path.dirname(HERE)                  # repository root; raw files in data/
sys.path.insert(0, PLAT)
os.chdir(PLAT)
import factory_core as fc                                    # noqa: E402
from suppl_generator import DATA, DEFS, AGE                   # noqa: E402

OUT = os.path.join(HERE, "participant_flow.csv")

rows = []
for ds in ["KNHANES", "NHANES"]:
    ddir, cyc = DATA[ds]
    raw = fc.load_raw(ds, ddir, cyc)
    n_released = len(raw)
    n_cycles = raw.attrs.get("n_cycles")

    d = fc.apply_definitions(raw, DEFS)
    age = pd.to_numeric(d.age, errors="coerce")

    # step 1: age
    ok_age = age >= AGE
    n_age = int(ok_age.sum())
    n_under = int((age < AGE).sum())
    n_age_missing = int(age.isna().sum())

    # step 2: valid complex-survey design, among those already age-eligible
    w = pd.to_numeric(d.wt_pool, errors="coerce")
    ok_w = (w > 0) & w.notna()
    ok_s = d.kstrata.notna()
    ok_p = d.psu.notna()
    ok_design = ok_w & ok_s & ok_p
    n_analysed = int((ok_age & ok_design).sum())
    n_drop_design = n_age - n_analysed
    n_drop_w = int((ok_age & ~ok_w).sum())
    n_drop_s = int((ok_age & ok_w & ~ok_s).sum())
    n_drop_p = int((ok_age & ok_w & ok_s & ~ok_p).sum())

    # design degrees of freedom on the analytic sample
    ana = d[ok_age & ok_design]
    n_strata = int(ana.kstrata.nunique())
    n_psu = int(ana.groupby(["kstrata", "psu"]).ngroups)
    ddf = n_psu - n_strata

    rows.append(dict(survey=ds, cycles=n_cycles, released=n_released,
                     under_20=n_under, age_missing=n_age_missing, aged_20_plus=n_age,
                     no_valid_weight=n_drop_w, no_stratum=n_drop_s, no_psu=n_drop_p,
                     excluded_design=n_drop_design, analysed=n_analysed,
                     strata=n_strata, psu=n_psu, design_df=ddf))

    print(f"\n=== {ds} ({n_cycles} cycles) ===")
    print(f"  released records                {n_released:>9,}")
    print(f"  aged under 20                  -{n_under:>9,}"
          + (f"   (age missing: {n_age_missing})" if n_age_missing else ""))
    print(f"  aged 20 or over                 {n_age:>9,}")
    print(f"  no valid pooled weight         -{n_drop_w:>9,}")
    print(f"  no stratum                     -{n_drop_s:>9,}")
    print(f"  no primary sampling unit       -{n_drop_p:>9,}")
    print(f"  ANALYSED                        {n_analysed:>9,}")
    print(f"  strata {n_strata}   PSUs {n_psu}   design df = {n_psu} - {n_strata} = {ddf}")

pd.DataFrame(rows).to_csv(OUT, index=False)
print("\nwrote", OUT)
