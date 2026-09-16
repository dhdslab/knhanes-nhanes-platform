# -*- coding: utf-8 -*-
"""Supplementary Figure S8: participant flow (STROBE item 13c).

The cascade numbers come from Reporting/participant_flow.csv, which participant_flow.py
computes by walking the same filter `build_analytic` applies. The per-analysis range at
the foot of each column comes from the two released association manifests, so every
number in the figure can be recomputed from a file in the reader's hands.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")

NAVY, GREY, TEAL = "#22405F", "#B9BEC4", "#2A9D8F"

F = pd.read_csv(os.path.join(BASE, "Reporting", "participant_flow.csv")).set_index("survey")
MAN = {}
for ds in ("KNHANES", "NHANES"):
    m = pd.read_csv(os.path.join(ROOT, "suppl", f"_manifest_association_{ds}.csv"))
    MAN[ds] = (len(m), int(m.n.min()), int(m.n.median()), int(m.n.max()))

CYCLES = {"KNHANES": "18 annual cycles, 2007 to 2024",
          "NHANES": "8 two-year cycles,\n2005-2006 through 2021-2023"}
WHY = {"KNHANES": "no positive pooled\ninterview-and-examination weight",
       "NHANES": "no positive pooled examination\nweight (interviewed, not examined)"}
# spine centre, exclusion-lane centre. The two lanes never overlap, so a wide
# exclusion box cannot collide with the other survey's column.
GEOM = {"KNHANES": (0.155, 0.385), "NHANES": (0.650, 0.880)}
W, XW = 0.26, 0.19


def box(ax, x, y, w, h, text, face="white", edge=NAVY, lw=1.3, size=8.4, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color="#1a1a1a",
            zorder=3, linespacing=1.45, fontweight=weight)


def arrow(ax, x, y0, y1, color=NAVY):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=1.2, zorder=2,
                                 shrinkA=0, shrinkB=0))


def column(ax, ds):
    r = F.loc[ds]
    npairs, nmin, nmed, nmax = MAN[ds]
    x0, xe = GEOM[ds]
    # Box centres and heights are chosen so that every gap between two main boxes is
    # taller than the exclusion box that sits in it; nothing overlaps and nothing is
    # clipped by the axes.
    ys = [0.855, 0.645, 0.395, 0.115]

    box(ax, x0, 0.965, W, 0.050, ds,
        face="#EEF2F6", edge=NAVY, lw=1.6, size=11.5, weight="bold")

    box(ax, x0, ys[0], W, 0.115,
        "Released records\n" + CYCLES[ds] + f"\nn = {int(r.released):,}")

    def step(y_from, y_to, halfa, halfb, excl):
        arrow(ax, x0, y_from - halfa, y_to + halfb)
        ym = (y_from - halfa + y_to + halfb) / 2
        if excl is not None:
            txt, h, edge = excl
            ax.plot([x0, xe - XW / 2], [ym, ym], color=GREY, linewidth=1.1, zorder=1)
            box(ax, xe, ym, XW, h, txt, face="#FAFAFA", edge=edge, lw=1.0, size=7.6)

    step(ys[0], ys[1], 0.0575, 0.0725,
         (f"Excluded, aged under 20 years\nn = {int(r.under_20):,}", 0.056, GREY))
    box(ax, x0, ys[1], W, 0.145,
        f"Aged 20 years or older\nn = {int(r.aged_20_plus):,}\n"
        "(the eligible population\nof the platform)")

    step(ys[1], ys[2], 0.0725, 0.0825,
         (f"Excluded, incomplete survey design\nn = {int(r.excluded_design):,}\n" + WHY[ds],
          0.082, GREY))
    box(ax, x0, ys[2], W, 0.165,
        f"Valid complex-survey design\nn = {int(r.analysed):,}\n"
        f"{int(r.strata):,} strata, {int(r.psu):,} primary sampling units\n"
        f"design degrees of freedom = {int(r.design_df):,}", face="#EEF2F6")

    step(ys[2], ys[3], 0.0825, 0.1025,
         ("Pooled association estimates\nscreened on a fixed random\n"
          "subsample, n = 50,000 (Methods)", 0.072, TEAL) if ds == "KNHANES" else None)
    box(ax, x0, ys[3], W, 0.205,
        "Analysed\nEach model is fitted on the participants\n"
        "with complete data for its own variables,\nso there is no single analytic n. Across\n"
        f"the {npairs:,} association analyses of this survey\n"
        f"the contributing n runs from {nmin:,} to\n{nmax:,}, median {nmed:,}.",
        lw=1.6, size=8.0)


fig, ax = plt.subplots(figsize=(13.6, 8.6), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
column(ax, "KNHANES")
column(ax, "NHANES")
ax.plot([0.5015, 0.5015], [0.015, 0.995], color="#E4E7EA", linewidth=1.0, zorder=0)
fig.tight_layout(pad=0.4)
p = os.path.join(FIGS, "SupplFigureS8.png")
fig.savefig(p, facecolor="white"); plt.close(fig)
print("wrote", p)
for ds in ("KNHANES", "NHANES"):
    r = F.loc[ds]
    print(f"  {ds}: {int(r.released):,} -> {int(r.aged_20_plus):,} -> {int(r.analysed):,}"
          f"   design df {int(r.design_df):,}   per-analysis n {MAN[ds][1]:,}-{MAN[ds][3]:,}")
