# -*- coding: utf-8 -*-
"""Supplementary figures and tables describing the released corpus and its
relationship to the manuscript.

Everything here is computed from the two association manifests and the released
Word bundles, so each number can be traced to a file a reader has in hand.

Figures (written to Figures/)
  SupplFigureS1  the whole evidence map as a volcano plot
  SupplFigureS2  per-outcome yield: exposures analysed and surviving FDR control
  SupplFigureS3  analytic sample size actually used, by survey and measure
  SupplFigureS4  cross-survey concordance within each measurement family

Tables (written to _src/ as CSV, inserted into the Supplement by add_s11_s13.py)
  S11  contents and integrity of the released Word files
  S12  per-outcome corpus summary
"""
import os, sys, glob, hashlib, collections
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")
sys.path.insert(0, os.path.join(ROOT, "knhanes_platform"))
import factory_core as fc

NAVY, PINK, TEAL, GREY = "#22405F", "#E5A3A3", "#2A9D8F", "#B9BEC4"
K = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_KNHANES.csv"))
N = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_NHANES.csv"))
K["survey"], N["survey"] = "KNHANES", "NHANES"
A = pd.concat([K, N], ignore_index=True)
A["effect"] = np.where(A.measure == "OR", np.log(A.est.clip(lower=1e-12)), A.est)
A["sig"] = A.q < 0.05
A["nlq"] = -np.log10(A.q.clip(lower=1e-300))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#333"); ax.spines["bottom"].set_color("#333")
    ax.grid(axis="y", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)


# ── S1: volcano ──────────────────────────────────────────────────────────────
def fig_s1():
    # A beta is in the units of its own outcome, so betas are not comparable across
    # outcomes and must not share an axis. For continuous outcomes the effect is
    # therefore shown as a partial correlation, r = t / sqrt(t^2 + n), derived from
    # the released estimate and confidence interval, which is unit-free.
    se = (A.hi - A.lo) / (2 * 1.96)
    t = A.est / se.replace(0, np.nan)
    A["partial_r"] = t / np.sqrt(t ** 2 + A.n)
    YCAP = 100                      # q values below 1e-100 are drawn at the cap
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=150)
    panels = [("OR", "effect", "Binary outcomes: log odds ratio per 1 SD"),
              ("beta", "partial_r", "Continuous outcomes: partial correlation")]
    for ax, (meas, xcol, lab) in zip(axes, panels):
        d = A[(A.measure == meas) & A[xcol].notna()]
        for sv, col in [("KNHANES", NAVY), ("NHANES", PINK)]:
            s = d[d.survey == sv]
            ax.scatter(s[xcol], s.nlq.clip(upper=YCAP), s=7, alpha=.45, linewidths=0,
                       color=col, label=f"{sv} (n={len(s):,})", zorder=3)
        ax.axhline(-np.log10(0.05), ls="--", lw=1, color=TEAL, zorder=4)
        ax.annotate("FDR q = 0.05", (ax.get_xlim()[0], -np.log10(0.05)),
                    xytext=(4, 4), textcoords="offset points", fontsize=9, color=TEAL)
        ax.set_ylim(-3, YCAP + 6)
        ax.set_xlabel(lab, fontsize=11)
        ax.set_ylabel(f"-log10 FDR q (capped at {YCAP})", fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        style(ax)
    fig.suptitle("Supplementary Figure S1. The released evidence map, all 7,478 association analyses",
                 fontsize=13, y=.99)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS1.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ── S2: per-outcome yield ────────────────────────────────────────────────────
def fig_s2():
    g = (A.groupby(["survey", "out"])
           .agg(analysed=("est", "size"), sig=("sig", "sum")).reset_index())
    g["label"] = [fc.lab(o) for o in g.out]
    order = (g.groupby("out").analysed.sum().sort_values(ascending=False).index.tolist())
    order = [o for o in order][:34]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9), dpi=150, sharey=True)
    for ax, sv, col in zip(axes, ["KNHANES", "NHANES"], [NAVY, PINK]):
        s = g[g.survey == sv].set_index("out").reindex(order)
        y = np.arange(len(order))
        ax.barh(y, s.analysed.fillna(0), color=GREY, zorder=2, label="exposures analysed")
        ax.barh(y, s.sig.fillna(0), color=col, zorder=3, label="survived FDR q<0.05")
        ax.set_yticks(y)
        ax.set_yticklabels([fc.lab(o)[:38] for o in order], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("exposure-outcome pairs", fontsize=11)
        ax.set_title(sv, fontsize=12); ax.legend(fontsize=9, frameon=False, loc="lower right")
        style(ax)
    fig.suptitle("Supplementary Figure S2. Yield per outcome: pairs analysed and pairs surviving "
                 "false-discovery-rate control", fontsize=13, y=.995)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS2.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ── S3: analytic N actually used ─────────────────────────────────────────────
def fig_s3():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    bins = np.linspace(0, A.n.max(), 60)
    for sv, col in [("KNHANES", NAVY), ("NHANES", PINK)]:
        s = A[A.survey == sv]
        ax.hist(s.n, bins=bins, alpha=.65, color=col, zorder=3,
                label=f"{sv}: median {int(s.n.median()):,}, range {int(s.n.min()):,}-{int(s.n.max()):,}")
    ax.axvline(110239, ls="--", lw=1.2, color=NAVY, zorder=4)
    ax.axvline(47558, ls="--", lw=1.2, color=PINK, zorder=4)
    ax.annotate("KNHANES analytic sample 110,239", (110239, ax.get_ylim()[1]*.95),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=9, color=NAVY)
    ax.annotate("NHANES analytic sample 47,558", (47558, ax.get_ylim()[1]*.80),
                xytext=(6, 0), textcoords="offset points", ha="left", fontsize=9, color=PINK)
    ax.set_xlabel("participants contributing to the fitted model", fontsize=11)
    ax.set_ylabel("exposure-outcome pairs", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    style(ax)
    ax.set_title("Supplementary Figure S3. Sample size contributing to each fitted analysis, with the "
                 "headline analytic samples marked", fontsize=12)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS3.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ── S4: concordance within measurement families ──────────────────────────────
def fig_s4():
    ko = K[(K.measure == "OR") & (K.est > 0)][["exp", "out", "est"]].rename(columns={"est": "eK"})
    no = N[(N.measure == "OR") & (N.est > 0)][["exp", "out", "est"]].rename(columns={"est": "eN"})
    M = ko.merge(no, on=["exp", "out"])
    M["lK"], M["lN"] = np.log(M.eK), np.log(M.eN)
    fam = {}
    for i, F in enumerate(fc._FAMILIES):
        for v in F:
            fam[v] = i
    names = {0: "smoking", 1: "body composition", 2: "lipids", 3: "liver",
             4: "glycaemia", 5: "kidney"}
    M["fam"] = [fam.get(e, -1) for e in M.exp]
    groups = [(-1, "other / unassigned")] + [(i, names.get(i, str(i))) for i in sorted(names)]
    groups = [(i, nm) for i, nm in groups if (M.fam == i).sum() >= 20]
    ncol = 3; nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.9*nrow), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, (i, nm) in zip(axes, groups):
        s = M[M.fam == i]
        r = pearsonr(s.lK, s.lN)[0]
        ax.scatter(s.lK, s.lN, s=9, alpha=.5, color=NAVY, linewidths=0, zorder=3)
        lim = [min(s.lK.min(), s.lN.min()), max(s.lK.max(), s.lN.max())]
        ax.plot(lim, lim, "--", lw=1, color=TEAL, zorder=4)
        ax.set_title(f"{nm}  (n={len(s)}, r={r:.2f})", fontsize=10)
        ax.set_xlabel("KNHANES log OR", fontsize=9); ax.set_ylabel("NHANES log OR", fontsize=9)
        style(ax)
    for ax in axes[len(groups):]:
        ax.axis("off")
    fig.suptitle("Supplementary Figure S4. Cross-survey concordance within each exposure "
                 "measurement family", fontsize=13, y=.995)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS4.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ── S11: released file inventory ─────────────────────────────────────────────
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def table_s11():
    import docx
    spec = [("Supplementary_Appendix_fixed.docx", "both", "Methods appendix S1-S10", None),
            ("Supplementary_Data_S1.docx", "KNHANES", "association reports", 3522),
            ("Supplementary_Data_S2.docx", "NHANES", "association reports", 3956),
            ("Supplementary_Data_S3.docx", "KNHANES", "temporal-trend reports", 27),
            ("Supplementary_Data_S4.docx", "NHANES", "temporal-trend reports", 25),
            ("Supplementary_Data_S5.docx", "KNHANES", "machine-learning reports", 27),
            ("Supplementary_Data_S6.docx", "NHANES", "machine-learning reports", 27)]
    rows = []
    for fn, sv, kind, n in spec:
        p = os.path.join(BASE, "Suppl", fn)
        if not os.path.exists(p):
            continue
        d = docx.Document(p)
        rows.append(dict(file=fn.replace("_fixed", ""), survey=sv, contents=kind,
                         reports=("-" if n is None else f"{n:,}"),
                         tables=f"{len(d.tables):,}", size_mb=f"{os.path.getsize(p)/1e6:.1f}",
                         md5=md5(p)[:12]))
        print(f"  {fn}: {len(d.tables)} tables")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, "S11_file_inventory.csv"), index=False, encoding="utf-8-sig")
    return T


# ── S12: per-outcome corpus summary ──────────────────────────────────────────
def table_s12():
    g = (A.groupby(["survey", "out", "measure"])
           .agg(pairs=("est", "size"), sig=("sig", "sum"),
                med_eff=("effect", lambda s: np.median(np.abs(s))),
                med_n=("n", "median"), min_n=("n", "min"), max_n=("n", "max"))
           .reset_index())
    g["outcome"] = [fc.lab(o) for o in g.out]
    g["pct_sig"] = 100 * g.sig / g.pairs
    g = g.sort_values(["survey", "pairs"], ascending=[True, False])
    out = g[["survey", "outcome", "measure", "pairs", "sig", "pct_sig",
             "med_eff", "med_n", "min_n", "max_n"]]
    out.to_csv(os.path.join(HERE, "S12_outcome_summary.csv"), index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    print("figures:")
    for f in (fig_s1, fig_s2, fig_s3, fig_s4):
        print("  wrote", os.path.basename(f()))
    print("tables:")
    t11 = table_s11(); print(f"  S11: {len(t11)} files")
    t12 = table_s12(); print(f"  S12: {len(t12)} outcome-by-survey rows")
    print(f"\ncorpus: {len(A):,} association analyses, {int(A.sig.sum()):,} FDR-significant "
          f"({100*A.sig.mean():.1f}%)")
