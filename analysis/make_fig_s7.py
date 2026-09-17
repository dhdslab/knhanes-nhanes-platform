# -*- coding: utf-8 -*-
"""Supplementary Figure S7: the corpus models against the published discrimination
for the same outcomes.

Standalone, so the figure can be redrawn without re-running the PubMed retrieval.
It reads the two files that retrieval already produced:

  _src/S15_ml_benchmark.csv                 every published auROC / C-statistic /
                                            concordance index extracted, one row
                                            per value (outcome, pmid, year, auc)
  knhanes_platform/_ml_regen_summary.csv    the corpus models, one row per
                                            outcome and survey

The comparison is between a published value and the corpus value for the same
OUTCOME, not the same predictor set, so it measures whether the corpus is
competitive, not whether it is better. That is stated in the caption and in
Supplementary Appendix S14.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "knhanes_platform"))
import figstyle as fs
import factory_core as fc

AUCS = pd.read_csv(os.path.join(HERE, "S15_ml_benchmark.csv"))
R = pd.read_csv(os.path.join(ROOT, "knhanes_platform", "_ml_regen_summary.csv"))
MIN_PUB = 3          # an outcome needs at least this many published values to appear


def fig_s7():
    outs = [o for o in R.outcome.unique() if (AUCS.outcome == o).sum() >= MIN_PUB]
    order = sorted(outs, key=lambda o: AUCS[AUCS.outcome == o].auc.median())
    n = len(order)

    fig, ax = plt.subplots(figsize=(7.2, .215 * n + 1.15))
    for i in range(0, n, 2):
        ax.add_patch(Rectangle((0, i - .5), 1, 1, transform=ax.get_yaxis_transform(),
                               facecolor=fs.GREY_LT, alpha=.5, linewidth=0, zorder=0))

    above = 0
    for i, o in enumerate(order):
        a = AUCS[AUCS.outcome == o].auc.values
        jit = np.random.default_rng(i).normal(0, .055, len(a))
        ax.scatter(a, np.full(len(a), i) + jit, s=4.5, color=fs.GREY_DK, alpha=.55,
                   linewidths=0, zorder=2)
        med = float(np.median(a))
        ax.plot([med, med], [i - .30, i + .30], color=fs.INK, lw=1.4, zorder=4,
                solid_capstyle="butt")
        for sv in ("KNHANES", "NHANES"):
            r = R[(R.outcome == o) & (R.survey == sv)]
            if len(r):
                v = float(r.auROC.iloc[0])
                above += v >= med
                ax.scatter(v, i, s=26, marker="D", color=fs.SURVEY_DK[sv],
                           edgecolors="white", linewidths=.55, zorder=5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([fs.label_fit(fc.lab(o), 38) for o in order], fontsize=7.6)
    ax.set_ylim(n - .45, -.55)
    # the count of published values, in its own right-hand column
    for i, o in enumerate(order):
        ax.annotate(f"{int((AUCS.outcome == o).sum())}", xy=(1.0, i),
                    xycoords=("axes fraction", "data"), xytext=(14, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.2, color=fs.INK2, annotation_clip=False)
    ax.annotate("published\nvalues", xy=(1.0, -.55), xycoords=("axes fraction", "data"),
                xytext=(16, 6), textcoords="offset points", ha="right", va="bottom",
                fontsize=7.4, color=fs.INK2, linespacing=1.3, annotation_clip=False)

    ax.set_xlim(0.5, 1.0)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xlabel("area under the receiver-operating-characteristic curve")
    fs.panel(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="none", markersize=3.6,
                              markerfacecolor=fs.GREY_DK, markeredgecolor="none",
                              label="published NHANES or KNHANES model"),
                       Line2D([0], [0], color=fs.INK, lw=1.4, label="published median"),
                       Line2D([0], [0], marker="D", ls="none", markersize=4.8,
                              markerfacecolor=fs.NAVY, markeredgecolor="white",
                              markeredgewidth=.55, label="this corpus, KNHANES"),
                       Line2D([0], [0], marker="D", ls="none", markersize=4.8,
                              markerfacecolor=fs.PINK_DK, markeredgecolor="white",
                              markeredgewidth=.55, label="this corpus, NHANES")],
              loc="lower left", bbox_to_anchor=(0, 1.0 + 2.6 / n), ncol=4,
              fontsize=7.4, columnspacing=1.5, handletextpad=0.45, handlelength=1.5)
    fs.title(fig, "Held-out discrimination of the corpus models against the literature")
    print(f"  corpus at or above the published median in {above} of "
          f"{int(R.outcome.isin(order).sum())} outcome-by-survey comparisons")
    return fs.save(fig, os.path.join(FIGS, "SupplFigureS7.png"))


if __name__ == "__main__":
    print(f"published values {len(AUCS):,} over {AUCS.outcome.nunique()} outcomes")
    print("wrote", os.path.basename(fig_s7()))
