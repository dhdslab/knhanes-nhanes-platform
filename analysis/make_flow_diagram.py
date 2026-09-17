# -*- coding: utf-8 -*-
"""Supplementary Figure S8: participant flow (STROBE item 13c).

The cascade numbers come from Reporting/participant_flow.csv, which participant_flow.py
computes by walking the same filter `build_analytic` applies. The per-analysis range at
the foot of each column comes from the two released association manifests, so every
number in the figure can be recomputed from a file in the reader's hands.

Drawn in the house style of figstyle.py: hairline rules, one weight of type, the
column heading as plain bold text rather than a filled tab, and exclusion boxes in a
lane that cannot collide with the other survey's column.

The figure is 7.2 inches wide and each survey gets half of it, so a main box holds
about 36 characters a line and an exclusion box about 28. Every string below is
written to that measure; a line that overruns it is a bug, and `check_widths`
asserts it at build time rather than leaving it to be noticed in the PDF.
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
FIGS = os.path.join(BASE, "Figures")
sys.path.insert(0, HERE)
import figstyle as fs

F = pd.read_csv(os.path.join(BASE, "Reporting", "participant_flow.csv")).set_index("survey")
MAN = {}
for ds in ("KNHANES", "NHANES"):
    m = pd.read_csv(os.path.join(ROOT, "suppl", f"_manifest_association_{ds}.csv"))
    MAN[ds] = (len(m), int(m.n.min()), int(m.n.median()), int(m.n.max()))

CYCLES = {"KNHANES": "18 annual cycles, 2007 to 2024",
          "NHANES": "8 two-year cycles,\n2005-2006 to 2021-2023"}
WHY = {"KNHANES": "no pooled interview-and-\nexamination weight",
       "NHANES": "no examination weight\n(interviewed, not examined)"}
# spine centre, exclusion-lane centre. The lanes are laid out so that no box
# overlaps another and neither survey's exclusion box crosses the centre rule.
GEOM = {"KNHANES": (0.142, 0.382), "NHANES": (0.642, 0.882)}
W, XW = 0.275, 0.200
BODY, SMALL = 7.4, 6.9
MAIN_CHARS, EXCL_CHARS = 36, 28


def check(text, limit, where):
    """Assert every line fits the box. `$n$` is one rendered glyph, not five, so the
    mathtext delimiters are dropped before counting."""
    for line in text.split("\n"):
        width = len(line.replace("$", ""))
        assert width <= limit, f"[{where}] {width} > {limit} chars: {line!r}"
    return text


def box(ax, x, y, w, h, text, fill="white", edge=fs.NAVY, lw=0.8, size=BODY,
        color=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.003,rounding_size=0.005",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=3,
            linespacing=1.5, color=color or fs.INK)


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=7,
                                 color=fs.NAVY, linewidth=0.8, zorder=2,
                                 shrinkA=0, shrinkB=0))


FIGW, FIGH = 7.2, 5.6
PAD = 0.013          # vertical breathing room inside a box, in axes units
TOP = 0.920          # top edge of the first box


def height(text, size):
    """Box height that actually holds the text: one line is `size` points at the
    rcParam linespacing of 1.5, expressed as a fraction of the figure height."""
    return len(text.split("\n")) * size * 1.5 / (FIGH * 72) + PAD


def stages_for(ds):
    """Each stage is (text, size, fill, linewidth); each gap carries the exclusion box
    that leaves at it."""
    r = F.loc[ds]
    npairs, nmin, nmed, nmax = MAN[ds]
    stages = [
        ("Released records\n" + CYCLES[ds] + f"\n$n$ = {int(r.released):,}",
         BODY, "white", 0.8),
        (f"Aged 20 years or older\n$n$ = {int(r.aged_20_plus):,}\n"
         "(the eligible population\nof the platform)", BODY, "white", 0.8),
        (f"Valid complex-survey design\n$n$ = {int(r.analysed):,}\n"
         f"{int(r.strata):,} strata, {int(r.psu):,} sampling units\n"
         f"design degrees of freedom = {int(r.design_df):,}", BODY, "#F0F3F6", 0.8),
        ("Analysed\nEach model uses complete cases\n"
         "for its own variables, so there is\nno single analytic $n$. "
         f"Per analysis:\n{nmin:,} to {nmax:,}, median {nmed:,}\n"
         f"over {npairs:,} association analyses", SMALL, "white", 1.1),
    ]
    exclusions = [
        (f"Excluded, aged under 20\n$n$ = {int(r.under_20):,}", fs.GREY),
        (f"Excluded, incomplete design\n$n$ = {int(r.excluded_design):,}\n" + WHY[ds],
         fs.GREY),
        (("Pooled estimates screened\non a fixed random subsample\n"
          "$n$ = 50,000 (Methods)", fs.TEAL) if ds == "KNHANES" else None),
    ]
    for i, (text, size, _, _) in enumerate(stages):
        check(text, MAIN_CHARS, f"{ds} stage {i}")
    for i, ex in enumerate(exclusions):
        if ex:
            check(ex[0], EXCL_CHARS, f"{ds} exclusion {i}")
    return stages, exclusions


# One shared vertical grid for both surveys, taken from the taller of the two at each
# stage, so the rows line up across the columns and the cascade can be read across.
ST = {ds: stages_for(ds) for ds in ("KNHANES", "NHANES")}
HS = [max(height(ST[ds][0][i][0], ST[ds][0][i][1]) for ds in ST)
      for i in range(len(ST["KNHANES"][0]))]
GAPS = [max(max(height(ST[ds][1][i][0], SMALL) for ds in ST if ST[ds][1][i]) + 0.030,
            0.062) for i in range(len(ST["KNHANES"][1]))]
TOPS, _t = [], TOP
for _h, _g in zip(HS, GAPS + [0]):
    TOPS.append(_t); _t -= _h + _g
assert _t > 0.004, f"the cascade overruns the canvas: bottom {_t:.3f}"


def column(ax, ds):
    x0, xe = GEOM[ds]
    stages, exclusions = ST[ds]

    ax.text(x0, 0.975, ds, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=fs.INK)
    ax.plot([x0 - W / 2, x0 + W / 2], [0.950, 0.950], color=fs.NAVY, lw=1.0,
            solid_capstyle="butt")

    for i, (text, size, fill, lw) in enumerate(stages):
        top, h = TOPS[i], HS[i]
        box(ax, x0, top - h / 2, W, h, text, fill=fill, lw=lw, size=size)
        if i == len(stages) - 1:
            break
        bottom, gap = top - h, GAPS[i]
        arrow(ax, x0, bottom, bottom - gap)
        ex = exclusions[i]
        if ex:
            eh = height(ex[0], SMALL)
            ym = bottom - gap / 2
            ax.plot([x0, xe - XW / 2], [ym, ym], color=fs.GREY, lw=0.7, zorder=1)
            box(ax, xe, ym, XW, eh, ex[0], fill="#F7F8F9", edge=ex[1], lw=0.7,
                size=SMALL, color=fs.INK2)


fig, ax = plt.subplots(figsize=(FIGW, FIGH))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
column(ax, "KNHANES")
column(ax, "NHANES")
ax.plot([0.4925, 0.4925], [0.02, 0.985], color="#EDEFF1", lw=0.8, zorder=0)
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
p = fs.save(fig, os.path.join(FIGS, "SupplFigureS8.png"))
print("wrote", p)
for ds in ("KNHANES", "NHANES"):
    r = F.loc[ds]
    print(f"  {ds}: {int(r.released):,} -> {int(r.aged_20_plus):,} -> {int(r.analysed):,}"
          f"   design df {int(r.design_df):,}   per-analysis n {MAN[ds][1]:,}-{MAN[ds][3]:,}")
