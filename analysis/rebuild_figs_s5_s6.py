# -*- coding: utf-8 -*-
"""Supplementary Figures S5 and S6, built around what the comparison actually found.

Three findings drive the design.

1. A published odds ratio for a continuous biomarker is a category contrast - the
   deficient group, the lowest quartile - so its sign cannot be transferred to a
   per-standard-deviation estimate. Manual adjudication of 34 high-confidence
   comparisons showed this was the dominant error: papers reporting OR 2.63 for
   "serum 25-hydroxyvitamin D and metabolic syndrome" mean the low-vitamin-D
   group, which agrees with the corpus rather than contradicting it. Direction is
   therefore compared only where both variables are binary clinical constructs.

2. Within those pairs, 13 of the 20 disagreements involve ever-versus-never
   alcohol or smoking. The literature grades dose; the platform codes lifetime
   exposure as a binary, which is dominated by sick-quitter selection. Agreement
   is 48% for those exposures and 74% for everything else.

3. Tightening the extraction lowers agreement rather than raising it, which is
   what rules out extraction noise as the main cause.

Colour carries one meaning per figure and it is consistent between the panels:
in S5 navy is the ever-versus-never lifestyle stratum and grey is everything
else; in S6 navy is a risk factor and teal a protective association, and a
disagreement shows as a square and a circle of different colour on the same row,
so no third hue is needed to mark it.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "knhanes_platform"))
import figstyle as fs
import factory_core as fc

LIFESTYLE = {"alcohol", "smoking", "current_smoking"}
NAVY_RAMP = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", fs.NAVY])

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


def band(ax, n, x0=0, x1=1, step=2):
    """Alternating row bands, so a long list of rows stays readable."""
    for i in range(0, n, step):
        ax.add_patch(Rectangle((x0, i - .5), x1 - x0, 1, transform=ax.get_yaxis_transform(),
                               facecolor=fs.GREY_LT, alpha=.55, linewidth=0, zorder=0))


def fig_s5():
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[.80, .95, 1.45], wspace=.58,
                          left=.075, right=.985, top=.80, bottom=.20)

    # ── a: the 2x2 table ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    tab = np.zeros((2, 2), int)
    for r in B.itertuples():
        tab[0 if r.lit_sign > 0 else 1, 0 if r.corpus_sign > 0 else 1] += 1
    ax.imshow(tab, cmap=NAVY_RAMP, vmin=0, vmax=tab.max() * 1.05)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i, j]}", ha="center", va="center", fontsize=12,
                    color="white" if tab[i, j] > tab.max() * .55 else fs.NAVY,
                    fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["risk\nfactor", "protective"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["risk\nfactor", "protective"])
    ax.set_xlabel("corpus"); ax.set_ylabel("published literature")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(fs.GREY); s.set_linewidth(0.7)
    ax.set_title(f"Binary clinical pairs, $n$ = {len(B)}", loc="left", fontsize=8.6,
                 color=fs.INK)
    fs.label(ax, "a", x=-0.42, y=1.10)

    # ── b: agreement by stratum ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    strata = [("all\npairs", B, fs.GREY),
              ("without\nlifestyle", B[~B.lifestyle], fs.GREY),
              ("lifestyle\nonly", B[B.lifestyle], fs.NAVY)]
    for x, (lab, sub, col) in enumerate(strata):
        p, lo, hi = wilson(int(sub.agree.sum()), len(sub))
        ax.bar(x, p, color=col, zorder=3, width=.60, linewidth=0)
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=fs.AXIS,
                    capsize=2.5, lw=0.8, zorder=4)
        ax.text(x, hi + 2, f"{p:.0f}%", ha="center", va="bottom", fontsize=8,
                color=fs.INK, fontweight="bold")
        ax.text(x, 2.5, f"$n$ = {len(sub)}", ha="center", va="bottom", fontsize=7,
                color="white" if col == fs.NAVY else fs.INK2)
    ax.axhline(50, ls=(0, (3, 2)), lw=0.8, color=fs.AXIS, zorder=2)
    ax.annotate("chance", xy=(1.0, 50), xycoords=("axes fraction", "data"),
                xytext=(-1, 2), textcoords="offset points", ha="right", va="bottom",
                fontsize=7, color=fs.INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([s[0] for s in strata])
    ax.set_xlim(-.62, 2.62); ax.set_ylim(0, 100)
    ax.set_ylabel("direction agreement (%)")
    fs.panel(ax, grid="y")
    ax.tick_params(axis="x", length=0)
    ax.set_title("Where the disagreement sits", loc="left", fontsize=8.6, color=fs.INK)
    fs.label(ax, "b", x=-0.33, y=1.10)

    # ── c: which variables carry it ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    cnt = {}
    for r in B[~B.agree].itertuples():
        for v in (r.var_a, r.var_b):
            if v in LIFESTYLE or fc.typ(v) == "b":
                cnt[v] = cnt.get(v, 0) + 1
    s = pd.Series(cnt).sort_values().tail(12)
    y = np.arange(len(s))
    ax.barh(y, s.values, height=.66, linewidth=0, zorder=3,
            color=[fs.NAVY if i in LIFESTYLE else fs.GREY for i in s.index])
    ax.set_yticks(y)
    ax.set_yticklabels([fs.label_fit(fc.lab(i), 24) for i in s.index],
                       fontsize=7.4)
    ax.set_ylim(-.65, len(s) - .35)
    ax.set_xlabel("appearances among disagreeing pairs")
    ax.set_xlim(0, s.max() * 1.02)
    fs.panel(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    # navy marks the two lifestyle exposures; named in the caption, so no legend box
    ax.set_title("Which variables carry it", loc="left", fontsize=8.6, color=fs.INK)
    fs.label(ax, "c", x=-0.40, y=1.10)

    fs.title(fig, "Direction agreement between the corpus and the published literature")
    return fs.save(fig, os.path.join(FIGS, "SupplFigureS5.png"))


def fig_s6():
    d = B.copy()
    _nm = lambda v: fs.pretty(fc.lab(v).split(" (")[0].strip())
    d["lbl"] = [f"{_nm(a)}  →  {_nm(b)}" for a, b in zip(d.var_a, d.var_b)]
    g = (d.groupby(["lbl", "lifestyle"])
           .agg(papers=("pmid", "nunique"), lit=("lit_sign", "mean"),
                cor=("corpus_sign", "first"), agree=("agree", "mean")).reset_index())
    # the two blocks are kept apart because they behave differently; inside each,
    # the disagreements come first and then the best-supported pairs
    g = g.sort_values(["lifestyle", "agree", "papers"], ascending=[True, True, False])
    g = g.reset_index(drop=True)
    split = int((~g.lifestyle).sum())          # first row of the lifestyle block
    # one blank row between the blocks, which is where the block heading goes
    g["y"] = g.index + g.lifestyle.astype(int)
    nrow = int(g.y.max()) + 1

    XL, XC, XNE, XP = 0.0, 0.34, 0.50, 0.62    # literature, corpus, "differs", papers
    fig, ax = plt.subplots(figsize=(6.1, max(4.4, .158 * nrow + 0.9)))
    for i in range(0, nrow, 2):
        ax.add_patch(Rectangle((0, i - .5), 1, 1, transform=ax.get_yaxis_transform(),
                               facecolor=fs.GREY_LT, alpha=.55, linewidth=0, zorder=0))
    for _, r in g.iterrows():
        yy = r.y
        lc = fs.NAVY if r.lit > 0 else fs.TEAL
        cc = fs.NAVY if r.cor > 0 else fs.TEAL
        ax.plot([XL, XC], [yy, yy], color=fs.GREY, lw=0.8, zorder=2,
                solid_capstyle="butt")
        ax.scatter(XL, yy, s=34, marker="s", color=lc, zorder=3,
                   edgecolors="white", linewidths=.5)
        ax.scatter(XC, yy, s=34, marker="o", color=cc, zorder=3,
                   edgecolors="white", linewidths=.5)
        if (r.lit > 0) != (r.cor > 0):
            ax.text(XNE, yy, "≠", va="center", ha="center", fontsize=9,
                    color=fs.INK, fontweight="bold")
        ax.text(XP, yy, f"{int(r.papers)}", va="center", ha="center",
                fontsize=7.2, color=fs.INK2)

    ax.text(XL - .045, split, "ever-versus-never alcohol or smoking",
            va="center", ha="left", fontsize=7.6, color=fs.INK, fontweight="bold",
            zorder=6)

    ax.set_yticks(g.y.tolist())
    ax.set_yticklabels(g.lbl, fontsize=7.4)
    ax.set_xticks([XL, XC, XNE, XP])
    ax.set_xticklabels(["published\nliterature", "corpus", "differs", "papers"],
                       fontsize=7.8)
    ax.set_xlim(XL - .055, XP + .045)
    ax.set_ylim(nrow - .4, -.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="none", markersize=5.4,
                              markerfacecolor=fs.NAVY, markeredgecolor="none",
                              label="risk factor"),
                       Line2D([0], [0], marker="s", ls="none", markersize=5.4,
                              markerfacecolor=fs.TEAL, markeredgecolor="none",
                              label="protective")],
              loc="lower left", bbox_to_anchor=(0, 1.0 + 3.0 / nrow), ncol=2,
              fontsize=7.6, columnspacing=1.4, handletextpad=0.4)
    fs.title(fig, "Risk-factor direction pair by pair")
    return fs.save(fig, os.path.join(FIGS, "SupplFigureS6.png"))


if __name__ == "__main__":
    print("binary-binary pairs:", len(B))
    for nm, sub in [("all", B), ("excl lifestyle", B[~B.lifestyle]), ("lifestyle", B[B.lifestyle])]:
        p, lo, hi = wilson(int(sub.agree.sum()), len(sub))
        print(f"  {nm:<16} {int(sub.agree.sum())}/{len(sub)} = {p:.1f}% ({lo:.0f}-{hi:.0f})")
    print("wrote", os.path.basename(fig_s5()))
    print("wrote", os.path.basename(fig_s6()))
