# -*- coding: utf-8 -*-
"""Rebuild Supplementary Figures S5 and S6 around what the comparison actually found.

Three findings drive the design.

1. A published odds ratio for a continuous biomarker is a category contrast - the
   deficient group, the lowest quartile - so its sign cannot be transferred to a
   per-standard-deviation estimate. Manual adjudication of 34 high-confidence
   comparisons showed this was the dominant error: papers reporting OR 2.63 for
   "serum 25-hydroxyvitamin D and metabolic syndrome" mean the low-vitamin-D
   group, which agrees with the corpus rather than contradicting it. Direction is
   therefore compared only where both variables are binary clinical constructs.

2. Within those pairs, 11 of the 20 disagreements involve ever-versus-never
   alcohol or smoking. The literature grades dose; the platform codes lifetime
   exposure as a binary, which is dominated by sick-quitter selection. Agreement
   is 48% for those exposures and 74% for everything else.

3. Tightening the extraction lowers agreement rather than raising it, which is
   what rules out extraction noise as the main cause.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")
sys.path.insert(0, os.path.join(ROOT, "knhanes_platform"))
import factory_core as fc

NAVY, PINK, TEAL, GREY, AMBER = "#22405F", "#E5A3A3", "#2A9D8F", "#B9BEC4", "#E9A23B"
LIFESTYLE = {"alcohol", "smoking", "current_smoking"}

D = pd.read_csv(os.path.join(HERE, "S14_direction_concordance.csv"))
D["ta"] = [fc.typ(v) for v in D.var_a]
D["tb"] = [fc.typ(v) for v in D.var_b]
D["bb"] = (D.ta == "b") & (D.tb == "b")
D["lifestyle"] = [(a in LIFESTYLE) or (b in LIFESTYLE) for a, b in zip(D.var_a, D.var_b)]
B = D[D.bb].copy()


def wilson(k, n):
    p = k / n
    se = (p * (1 - p) / n) ** .5
    return 100 * p, 100 * max(p - 1.96 * se, 0), 100 * min(p + 1.96 * se, 1)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#333"); ax.spines["bottom"].set_color("#333")
    ax.grid(axis="y", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)


def fig_s5():
    fig = plt.figure(figsize=(16.0, 5.6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[.85, 1.0, 1.45], wspace=.62)

    ax = fig.add_subplot(gs[0, 0])
    tab = np.zeros((2, 2), int)
    for r in B.itertuples():
        tab[0 if r.lit_sign > 0 else 1, 0 if r.corpus_sign > 0 else 1] += 1
    ax.imshow(tab, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i, j]}", ha="center", va="center", fontsize=17,
                    color="white" if tab[i, j] > tab.max() * .55 else NAVY, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["risk\nfactor", "protective"], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["risk\nfactor", "protective"], fontsize=10)
    ax.set_xlabel("corpus", fontsize=11); ax.set_ylabel("published literature", fontsize=11)
    p, lo, hi = wilson(int(B.agree.sum()), len(B))
    ax.set_title(f"a  Binary clinical pairs (n={len(B)})\nstrata are separated in panel b", fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1])
    strata = [("all binary\npairs", B),
              ("excluding\never-vs-never\nalcohol, smoking", B[~B.lifestyle]),
              ("ever-vs-never\nalcohol, smoking\nonly", B[B.lifestyle])]
    xs = np.arange(len(strata))
    for x, (lab, sub) in zip(xs, strata):
        p, lo, hi = wilson(int(sub.agree.sum()), len(sub))
        col = NAVY if "excluding" in lab else (AMBER if "only" in lab else GREY)
        ax2.bar(x, p, color=col, zorder=3, width=.62)
        ax2.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#444",
                     capsize=4, lw=1.2, zorder=4)
        ax2.text(x, hi + 2.5, f"{p:.0f}%\nn={len(sub)}", ha="center", fontsize=9)
    ax2.axhline(50, ls="--", lw=1, color="#999", zorder=2)
    ax2.text(-0.42, 52.5, "chance", fontsize=8, color="#777", ha="left")
    ax2.set_xticks(xs); ax2.set_xticklabels([s[0] for s in strata], fontsize=9)
    ax2.set_ylim(0, 112); ax2.set_ylabel("direction agreement (%)", fontsize=11)
    ax2.set_title("b  Where the disagreement sits\n(95% confidence intervals)", fontsize=11)
    style(ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    dis = B[~B.agree]
    cnt = {}
    for r in dis.itertuples():
        for v in (r.var_a, r.var_b):
            if v in LIFESTYLE or fc.typ(v) == "b":
                cnt[v] = cnt.get(v, 0) + 1
    s = pd.Series(cnt).sort_values().tail(12)
    y = np.arange(len(s))
    ax3.barh(y, s.values, color=[AMBER if i in LIFESTYLE else GREY for i in s.index], zorder=3)
    ax3.set_yticks(y); ax3.set_yticklabels([fc.lab(i).split(" (")[0][:24] for i in s.index], fontsize=8)
    ax3.set_xlabel("appearances among disagreeing pairs", fontsize=10)
    ax3.set_title("c  Which variables carry the disagreement", fontsize=11)
    ax3.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=AMBER,
                               markersize=9, label="ever-vs-never lifestyle exposure"),
                        Line2D([0], [0], marker="s", color="none", markerfacecolor=GREY,
                               markersize=9, label="other")],
               fontsize=8, frameon=False, loc="lower right")
    style(ax3)

    fig.suptitle("Supplementary Figure S5. Direction agreement between the corpus and the published "
                 "literature, and where it fails", fontsize=13, y=.995)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS5.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def fig_s6():
    d = B.copy()
    _nm = lambda v: fc.lab(v).split(" (")[0].strip()
    d["lbl"] = [f"{_nm(a)}  →  {_nm(b)}" for a, b in zip(d.var_a, d.var_b)]
    g = (d.groupby(["lbl", "lifestyle"])
           .agg(papers=("pmid", "nunique"), lit=("lit_sign", "mean"),
                cor=("corpus_sign", "first"), agree=("agree", "mean")).reset_index())
    g = g.sort_values(["lifestyle", "agree", "papers"], ascending=[True, True, False])
    fig, ax = plt.subplots(figsize=(11.8, max(6.5, .295 * len(g))), dpi=150)
    y = np.arange(len(g))
    for yy, r in zip(y, g.itertuples()):
        lc = NAVY if r.lit > 0 else TEAL
        cc = NAVY if r.cor > 0 else TEAL
        same = (r.lit > 0) == (r.cor > 0)
        ax.plot([0, 1], [yy, yy], color=GREY if same else AMBER,
                lw=1.0 if same else 2.6, zorder=2)
        ax.scatter(0, yy, s=112, marker="s", color=lc, zorder=3)
        ax.scatter(1, yy, s=112, marker="o", color=cc, zorder=3)
        ax.text(1.10, yy, f"{r.papers}", va="center", fontsize=8, color="#555")
        if r.lifestyle:
            ax.text(-0.40, yy, "●", va="center", fontsize=9, color=AMBER)
    ax.set_yticks(y); ax.set_yticklabels(g.lbl, fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["published\nliterature", "corpus"], fontsize=10)
    ax.set_xlim(-.5, 1.4); ax.invert_yaxis()
    ax.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=NAVY,
                              markersize=10, label="risk factor"),
                       Line2D([0], [0], marker="s", color="none", markerfacecolor=TEAL,
                              markersize=10, label="protective"),
                       Line2D([0], [0], color=AMBER, lw=2.6, label="direction disagrees"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor=AMBER,
                              markersize=8, label="ever-vs-never lifestyle exposure")],
              fontsize=9, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Supplementary Figure S6. Risk-factor direction pair by pair, corpus against the "
                 "published literature\n(binary clinical pairs only; the rightmost number is the "
                 "count of published papers)", fontsize=11.5, pad=30)
    fig.tight_layout()
    p = os.path.join(FIGS, "SupplFigureS6.png"); fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


if __name__ == "__main__":
    print("binary-binary pairs:", len(B))
    for nm, sub in [("all", B), ("excl lifestyle", B[~B.lifestyle]), ("lifestyle", B[B.lifestyle])]:
        p, lo, hi = wilson(int(sub.agree.sum()), len(sub))
        print(f"  {nm:<16} {int(sub.agree.sum())}/{len(sub)} = {p:.1f}% ({lo:.0f}-{hi:.0f})")
    print("wrote", os.path.basename(fig_s5()))
    print("wrote", os.path.basename(fig_s6()))
