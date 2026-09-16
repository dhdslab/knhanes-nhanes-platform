# -*- coding: utf-8 -*-
"""Compare the released corpus with the published literature on the three things
that can be compared without assuming a common exposure scaling.

A published odds or hazard ratio is expressed per whatever contrast the authors
chose - a category, a quartile, a unit, a doubling - while every corpus estimate
is per one standard deviation. Magnitudes are therefore NOT comparable and are
never compared here. What is comparable:

  1. DIRECTION   is the exposure reported as a risk factor or as protective?
  2. RANK        within one outcome, which exposures are the strongest?
  3. DISCRIMINATION  published AUC for an outcome against the corpus auROC.

Products
  SupplFigureS5  direction concordance, overall and per outcome
  SupplFigureS6  pair-level risk-factor direction map
  SupplFigureS7  prediction performance against the published distribution
  S14_direction_concordance.csv, S15_ml_benchmark.csv
  direction_precision_sample.csv   for manual adjudication
"""
import os, sys, re, collections
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "knhanes_platform"))
import pubmed_terms as PT
import factory_core as fc

NAVY, PINK, TEAL, GREY, AMBER = "#22405F", "#E5A3A3", "#2A9D8F", "#B9BEC4", "#E9A23B"

# ── effect size and direction from an abstract ───────────────────────────────
EFF = re.compile(
    r'\b(aOR|OR|aHR|HR|RR|PR|odds ratio|hazard ratio|risk ratio|prevalence ratio)\b'
    r'\s*(?:of|was|were|=|:|,)?\s*'
    r'(\d\.\d{1,3})'
    r'(?:\s*\(?\s*95\s*%\s*(?:CI|confidence interval)\s*[:,]?\s*'
    r'(\d\.\d{1,3})\s*(?:-|–|—|to|,)\s*(\d\.\d{1,3}))?', re.I)

POS = re.compile(r'\b(positively associated|direct(?:ly)? associated|increased (?:risk|odds|'
                 r'prevalence|likelihood)|higher (?:risk|odds|prevalence|likelihood)|'
                 r'greater (?:risk|odds|likelihood)|independent risk factor|'
                 r'associated with (?:an )?increased|associated with higher)\b', re.I)
NEG = re.compile(r'\b(inversely associated|negatively associated|decreased (?:risk|odds|'
                 r'prevalence|likelihood)|lower (?:risk|odds|prevalence|likelihood)|'
                 r'reduced (?:risk|odds|likelihood)|protective|'
                 r'associated with (?:a )?(?:decreased|reduced|lower))\b', re.I)

SENT = re.compile(r'(?<=[.!?])\s+')

AUC = re.compile(r'\b(AUC|AUROC|area under the (?:receiver[- ]operating[- ]characteristic )?curve'
                 r'|C[- ]statistic|c[- ]index|concordance index)\b'
                 r'[^.;]{0,40}?(0\.\d{2,4})', re.I)

# A phrase naming the low end of a continuous measurement
LOWEND = re.compile(r'\b(low|lower|lowest|reduced|decreased|deficien\w*|depleted|poor)\b', re.I)


def _terms_for(var):
    from literature_studied_pairs import KEYS
    return KEYS.get(var, [])


def abstract_direction(text, var_a=None, var_b=None):
    """(+1 risk factor, -1 protective, 0 undecided), the effect size, and the basis.

    The signal must come from a sentence that names both variables, so a direction
    stated about some other relationship in the same abstract cannot leak in."""
    if not isinstance(text, str) or len(text) < 40:
        return 0, None, "no abstract"
    ta, tb = _terms_for(var_a), _terms_for(var_b)
    if not ta or not tb:
        return 0, None, "no terms"
    sents = [s for s in SENT.split(text) if len(s) > 15]
    low = [s.lower() for s in sents]
    both = [s for s, l in zip(sents, low)
            if any(t in l for t in ta) and any(t in l for t in tb)]
    if not both:
        return 0, None, "no joint sentence"
    joint = " ".join(both)
    ests = [float(m.group(2)) for m in EFF.finditer(joint) if 0.05 < float(m.group(2)) < 30]
    if ests:
        signs = {1 if e > 1 else (-1 if e < 1 else 0) for e in ests}
        if len(signs) == 1 and 0 not in signs:
            s = signs.pop()
            return s, (max(ests) if s > 0 else min(ests)), "effect size"
    p, n = len(POS.findall(joint)), len(NEG.findall(joint))
    if p and not n:
        return 1, None, "wording"
    if n and not p:
        return -1, None, "wording"
    return 0, None, "ambiguous"


def extract_aucs(text, outcome):
    """AUC values reported within 250 characters of a mention of the outcome."""
    if not isinstance(text, str) or len(text) < 40:
        return []
    term = PT.concept(outcome) or ""
    words = [w.strip('"').lower() for w in re.findall(r'"([^"]+)"\[tiab\]|(\w[\w-]{3,})\[tiab\]', term)
             for w in ([w[0] or w[1]])]
    low = text.lower()
    hits = [m.start() for w in words for m in re.finditer(re.escape(w), low)]
    out = []
    for m in AUC.finditer(text):
        v = float(m.group(2))
        if not (0.5 <= v <= 1.0):
            continue
        if not hits or min(abs(m.start() - h) for h in hits) <= 250:
            out.append(v)
    return out


# ── build the comparison tables ──────────────────────────────────────────────
def build():
    A = pd.read_csv(os.path.join(HERE, "abstracts_association.csv"))
    X = pd.read_csv(os.path.join(HERE, "literature_studied.csv"))
    X = X[X.var_a.notna() & X.var_b.notna() & (X.var_a != X.var_b)]
    X["pmid"] = X.pmid.astype(str); A["pmid"] = A.pmid.astype(str)
    M = X.merge(A[["pmid", "abstract"]], on="pmid", how="left")

    rec = []
    for r in M.itertuples():
        s, est, how = abstract_direction(r.abstract, r.var_a, r.var_b)
        # Polarity. A title that says "low skeletal muscle mass is associated with
        # insulin resistance" reports a direction for the LOW end of a continuous
        # measurement, so the sign for that continuous variable is the opposite.
        # Each side that carries a low-end modifier on a continuous variable flips
        # the sign once; two flips cancel.
        flips = []
        for phrase, var in ((r.phrase_a, r.var_a), (r.phrase_b, r.var_b)):
            if isinstance(phrase, str) and LOWEND.search(phrase) and fc.typ(var) == "c":
                flips.append(var)
        if len(flips) % 2 == 1:
            s = -s
        rec.append(dict(pmid=r.pmid, var_a=r.var_a, var_b=r.var_b, title=r.title,
                        phrase_a=r.phrase_a, phrase_b=r.phrase_b,
                        lit_sign=s, lit_est=est, basis=how,
                        polarity_flipped="; ".join(flips)))
    L = pd.DataFrame(rec)
    L.to_csv(os.path.join(HERE, "literature_direction.csv"), index=False, encoding="utf-8-sig")

    # corpus direction for the same unordered pair, either survey
    K = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_KNHANES.csv"))
    N = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_NHANES.csv"))
    K["survey"], N["survey"] = "KNHANES", "NHANES"
    C = pd.concat([K, N], ignore_index=True)
    C["eff"] = np.where(C.measure == "OR", np.log(C.est.clip(lower=1e-12)), C.est)
    csign, cmag, cq = {}, {}, {}
    for r in C.itertuples():
        k = frozenset((r.exp, r.out))
        # keep the estimate with the smallest q as the corpus verdict for the pair
        if k not in cq or r.q < cq[k]:
            cq[k] = r.q; csign[k] = int(np.sign(r.eff)); cmag[k] = abs(r.eff)
    L["pair"] = [frozenset((a, b)) for a, b in zip(L.var_a, L.var_b)]
    L["corpus_sign"] = [csign.get(p, 0) for p in L.pair]
    L["corpus_absmag"] = [cmag.get(p) for p in L.pair]
    L["corpus_q"] = [cq.get(p) for p in L.pair]

    D = L[(L.lit_sign != 0) & (L.corpus_sign != 0)].copy()
    D["agree"] = D.lit_sign == D.corpus_sign
    D.drop(columns=["pair"]).to_csv(os.path.join(HERE, "S14_direction_concordance.csv"),
                                    index=False, encoding="utf-8-sig")

    # ── ML benchmark ─────────────────────────────────────────────────────────
    P = pd.read_csv(os.path.join(HERE, "abstracts_prediction.csv"))
    rows = []
    for r in P.itertuples():
        for v in extract_aucs(r.abstract, r.outcome):
            rows.append(dict(outcome=r.outcome, pmid=str(r.pmid), year=r.year, auc=v,
                             title=r.title))
    AUCS = pd.DataFrame(rows).drop_duplicates(["outcome", "pmid", "auc"])
    R = pd.read_csv(os.path.join(ROOT, "knhanes_platform", "_ml_regen_summary.csv"))
    AUCS.to_csv(os.path.join(HERE, "S15_ml_benchmark.csv"), index=False, encoding="utf-8-sig")
    return D, AUCS, R


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#333"); ax.spines["bottom"].set_color("#333")
    ax.grid(axis="y", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)


# ── S5: direction concordance ────────────────────────────────────────────────
def fig_s5(D):
    fig = plt.figure(figsize=(13, 5.6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=.32)

    ax = fig.add_subplot(gs[0, 0])
    tab = np.zeros((2, 2), int)
    for r in D.itertuples():
        tab[0 if r.lit_sign > 0 else 1, 0 if r.corpus_sign > 0 else 1] += 1
    im = ax.imshow(tab, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i,j]}", ha="center", va="center", fontsize=16,
                    color="white" if tab[i, j] > tab.max()*.55 else NAVY, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["risk factor", "protective"], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["risk factor", "protective"], fontsize=10)
    ax.set_xlabel("corpus", fontsize=11); ax.set_ylabel("published literature", fontsize=11)
    agree = 100 * np.trace(tab) / tab.sum()
    ax.set_title(f"Direction agreement {agree:.1f}%\n({np.trace(tab)} of {tab.sum()} comparisons)",
                 fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1])
    for lab, sub, col in [("effect size in abstract", D[D.basis == "effect size"], NAVY),
                          ("wording only", D[D.basis == "wording"], PINK)]:
        if len(sub):
            ax2.bar(lab.split()[0], 100*sub.agree.mean(), color=col, zorder=3)
            ax2.text(lab.split()[0], 100*sub.agree.mean()+1.5,
                     f"{100*sub.agree.mean():.0f}%\nn={len(sub)}", ha="center", fontsize=9)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(["effect size\nin abstract", "wording\nonly"],
                                                  fontsize=10)
    ax2.set_ylim(0, 108); ax2.set_ylabel("direction agreement (%)", fontsize=11)
    ax2.set_title("Agreement by the basis of the\nliterature direction", fontsize=11)
    style(ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    g = (D.groupby("var_b").agree.agg(["size", "mean"]).reset_index()
           .query("size >= 4").sort_values("mean"))
    if len(g):
        y = np.arange(len(g))
        ax3.barh(y, 100*g["mean"], color=NAVY, zorder=3)
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"{fc.lab(o)[:30]} (n={int(n)})"
                             for o, n in zip(g.var_b, g["size"])], fontsize=8)
        ax3.axvline(100*D.agree.mean(), ls="--", lw=1, color=TEAL, zorder=4)
        ax3.set_xlabel("direction agreement (%)", fontsize=11); ax3.set_xlim(0, 105)
        ax3.set_title("By outcome (outcomes with at least 4 comparisons)", fontsize=11)
        style(ax3)
    fig.suptitle("Supplementary Figure S5. Does the corpus agree with the published literature on "
                 "the direction of association?", fontsize=13, y=.99)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS5.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ── S6: pair-level risk-factor map ───────────────────────────────────────────
def fig_s6(D):
    d = D.copy()
    d["lbl"] = [f"{fc.lab(a)[:26]}  →  {fc.lab(b)[:26]}" for a, b in zip(d.var_a, d.var_b)]
    g = (d.groupby(["lbl", "var_b"])
           .agg(papers=("pmid", "nunique"),
                lit=("lit_sign", "mean"), cor=("corpus_sign", "first"),
                q=("corpus_q", "first"), mag=("corpus_absmag", "first")).reset_index())
    g = g[g.papers >= 2].sort_values(["var_b", "papers"], ascending=[True, False])
    g = g.head(46)
    fig, ax = plt.subplots(figsize=(11.5, max(6, .3*len(g))), dpi=150)
    y = np.arange(len(g))
    for yy, r in zip(y, g.itertuples()):
        lc = NAVY if r.lit > 0 else TEAL
        cc = NAVY if r.cor > 0 else TEAL
        ax.scatter(0, yy, s=115, marker="s", color=lc, zorder=3)
        ax.scatter(1, yy, s=115, marker="o", color=cc, zorder=3)
        ax.plot([0, 1], [yy, yy], color=(GREY if (r.lit > 0) == (r.cor > 0) else AMBER),
                lw=2.4 if (r.lit > 0) != (r.cor > 0) else 1.0, zorder=2)
        ax.text(1.12, yy, f"{r.papers}", va="center", fontsize=8, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(g.lbl, fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["published\nliterature", "corpus"], fontsize=10)
    ax.set_xlim(-.45, 1.45); ax.invert_yaxis()
    ax.text(1.12, -0.8, "papers", fontsize=8, color="#555")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=NAVY,
                              markersize=10, label="risk factor (positive)"),
                       Line2D([0], [0], marker="s", color="none", markerfacecolor=TEAL,
                              markersize=10, label="protective (inverse)"),
                       Line2D([0], [0], color=AMBER, lw=2.4, label="direction disagrees")],
              fontsize=9, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Supplementary Figure S6. Risk-factor direction, pair by pair "
                 "(pairs with at least two published papers)", fontsize=12, pad=32)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS6.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ── S7: discrimination against the published distribution ────────────────────
def fig_s7(AUCS, R):
    outs = [o for o in R.outcome.unique() if (AUCS.outcome == o).sum() >= 3]
    order = sorted(outs, key=lambda o: AUCS[AUCS.outcome == o].auc.median())
    fig, ax = plt.subplots(figsize=(12, max(5.5, .42*len(order))), dpi=150)
    for i, o in enumerate(order):
        a = AUCS[AUCS.outcome == o].auc.values
        ax.scatter(a, np.full(len(a), i) + np.random.default_rng(i).normal(0, .07, len(a)),
                   s=16, color=GREY, alpha=.85, linewidths=0, zorder=2)
        ax.plot([np.median(a)]*2, [i-.28, i+.28], color="#666", lw=2, zorder=3)
        for sv, col, mk in [("KNHANES", NAVY, "D"), ("NHANES", PINK, "D")]:
            r = R[(R.outcome == o) & (R.survey == sv)]
            if len(r):
                ax.scatter(r.auROC.iloc[0], i, s=95, marker=mk, color=col,
                           edgecolors="white", linewidths=.8, zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{fc.lab(o)[:34]}  (n={int((AUCS.outcome==o).sum())})" for o in order],
                       fontsize=9)
    ax.set_xlabel("area under the receiver-operating-characteristic curve", fontsize=11)
    ax.set_xlim(0.5, 1.02)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
                              markersize=8, label="published NHANES/KNHANES models"),
                       Line2D([0], [0], color="#666", lw=2, label="published median"),
                       Line2D([0], [0], marker="D", color="none", markerfacecolor=NAVY,
                              markersize=9, label="this corpus, KNHANES"),
                       Line2D([0], [0], marker="D", color="none", markerfacecolor=PINK,
                              markersize=9, label="this corpus, NHANES")],
              fontsize=9, frameon=False, loc="lower right", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)
    ax.set_title("Supplementary Figure S7. Held-out discrimination of the corpus models "
                 "against\npublished discrimination for the same outcomes",
                 fontsize=12, loc="left")
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS7.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


if __name__ == "__main__":
    D, AUCS, R = build()
    print(f"direction comparisons usable : {len(D)} "
          f"(effect size {int((D.basis=='effect size').sum())}, wording {int((D.basis=='wording').sum())})")
    print(f"  direction agreement        : {100*D.agree.mean():.1f}%")
    print(f"published AUC values extracted: {len(AUCS)} over {AUCS.outcome.nunique()} outcomes")
    for f, a in [(fig_s5, (D,)), (fig_s6, (D,)), (fig_s7, (AUCS, R))]:
        print("  wrote", os.path.basename(f(*a)))
    # sample for manual adjudication of the direction call
    D.drop(columns=["pair"], errors="ignore").sample(min(30, len(D)), random_state=5).to_csv(
        os.path.join(HERE, "direction_precision_sample.csv"), index=False, encoding="utf-8-sig")
    print("wrote direction_precision_sample.csv")
