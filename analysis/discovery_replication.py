# -*- coding: utf-8 -*-
"""Cross-survey replication, addressed the way review section 9 asks for.

Two problems with the current headline (97.3% directional agreement):
  1. It conditions on "FDR-significant in BOTH surveys", which selects for
     agreement and overstates replication.
  2. The 1,582 pairs are not independent -- they share exposures and outcomes --
     so a Pearson CI computed under independence is too narrow.

This script therefore runs a proper discovery-replication design in both
directions and puts a cluster-bootstrap CI on the correlation.
"""
import numpy as np, pandas as pd, os
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
K = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_KNHANES.csv"))
N = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_NHANES.csv"))

def ors(d, tag):
    d = d[(d.measure == "OR") & (d.est > 0)].copy()
    return d[["exp", "out", "est", "p", "q"]].rename(
        columns={"est": "est" + tag, "p": "p" + tag, "q": "q" + tag})

M = ors(K, "K").merge(ors(N, "N"), on=["exp", "out"]).dropna(
    subset=["estK", "estN", "pK", "pN", "qK", "qN"])
M["lK"], M["lN"] = np.log(M.estK), np.log(M.estN)
print(f"shared exposure-outcome pairs modelled as OR in both surveys: {len(M)}")
print(f"distinct outcomes {M.out.nunique()}, distinct exposures {M.exp.nunique()}\n")

# ── 1. unconditional agreement (no selection on significance) ────────────────
same_all = (np.sign(M.lK) == np.sign(M.lN)).mean()
print("UNCONDITIONAL, all shared pairs")
print(f"  same direction              : {100*same_all:.1f}%  ({int((np.sign(M.lK)==np.sign(M.lN)).sum())}/{len(M)})")
print(f"  Pearson r  (log OR)         : {pearsonr(M.lK, M.lN)[0]:.3f}")
print(f"  Spearman rho                : {spearmanr(M.lK, M.lN)[0]:.3f}\n")

# ── 2. discovery -> replication, both directions ─────────────────────────────
def dr(disc, rep, dq, rq, rp, dl, rl, name):
    D = M[M[dq] < 0.05]
    n = len(D)
    same = (np.sign(D[dl]) == np.sign(D[rl]))
    nom = same & (D[rp] < 0.05)
    fdr = same & (D[rq] < 0.05)
    print(f"{name}")
    print(f"  discovery set (FDR q<0.05 in {disc})            : {n}")
    print(f"  same direction in {rep:<8}                      : {100*same.mean():5.1f}%  ({int(same.sum())}/{n})")
    print(f"  same direction and nominal p<0.05 in {rep:<8}   : {100*nom.mean():5.1f}%  ({int(nom.sum())}/{n})")
    print(f"  same direction and FDR q<0.05 in {rep:<8}       : {100*fdr.mean():5.1f}%  ({int(fdr.sum())}/{n})")
    r = pearsonr(D[dl], D[rl])[0]
    print(f"  Pearson r among discovered pairs               : {r:.3f}\n")
    return dict(direction=f"{disc}->{rep}", n_discovery=n,
                same_direction=100*same.mean(), nominal=100*nom.mean(),
                fdr=100*fdr.mean(), r=r)

rows = [dr("KNHANES", "NHANES", "qK", "qN", "pN", "lK", "lN",
           "DISCOVERY KNHANES -> REPLICATION NHANES"),
        dr("NHANES", "KNHANES", "qN", "qK", "pK", "lN", "lK",
           "DISCOVERY NHANES -> REPLICATION KNHANES")]

# ── 3. the current headline, for comparison ──────────────────────────────────
B = M[(M.qK < 0.05) & (M.qN < 0.05)]
sb = (np.sign(B.lK) == np.sign(B.lN))
print("CURRENT HEADLINE (conditioned on FDR-significant in BOTH)")
print(f"  pairs {len(B)}, same direction {100*sb.mean():.1f}%, "
      f"Pearson r {pearsonr(B.lK, B.lN)[0]:.3f}")
print("  -> this is a selected subset; the discovery-replication rates above are\n"
      "     the honest measure of how often a finding carries to the other survey.\n")

# ── 4. cluster bootstrap CI for the correlation ──────────────────────────────
def cluster_boot(df, key, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in df.groupby(key)}
    keys = np.array(list(groups))
    out = []
    for _ in range(B):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        s = df.loc[idx]
        if len(s) > 10:
            out.append(pearsonr(s.lK, s.lN)[0])
    return np.percentile(out, [2.5, 97.5]), np.mean(out)

print("PEARSON r WITH CLUSTER BOOTSTRAP 95% CI (2,000 resamples)")
r_point = pearsonr(M.lK, M.lN)[0]
print(f"  point estimate                    : {r_point:.3f}")
for key, lab in [("out", "outcome"), ("exp", "exposure")]:
    (lo, hi), mean = cluster_boot(M, key)
    print(f"  resampling {lab:<9} clusters   : 95% CI {lo:.3f} to {hi:.3f}")
naive_lo, naive_hi = np.tanh(np.arctanh(r_point) + np.array([-1, 1]) * 1.96 / np.sqrt(len(M) - 3))
print(f"  naive independent-pair CI          : {naive_lo:.3f} to {naive_hi:.3f}  (too narrow)")

pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "discovery_replication_results.csv"), index=False)
print("\nwrote discovery_replication_results.csv")
