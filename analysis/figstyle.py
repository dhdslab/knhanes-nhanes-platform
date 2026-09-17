# -*- coding: utf-8 -*-
"""House style for every supplementary figure.

One module so that the supplement reads as one set of figures rather than eight
separate ones, and so that a change of taste is a change in one file.

The palette is the one the main figures already use, because those are fixed:
navy for KNHANES, salmon for NHANES, teal for a reference line or a second
category, neutral grey for anything recessive.

Two additions were forced by an accessibility check rather than by taste. The
salmon and the neutral grey of the main figures are 8.8 apart in OKLab and 4.9
apart under deuteranopia, which is not a distinguishable pair, and they were
placed together in two figures. So:

  PINK_DK  a darker salmon for small marks, strokes and text (16.8 / 12.8 vs GREY)
  GREY_DK  a darker neutral for grey DATA marks that sit beside salmon ones
           (18.2 / 13.7 vs PINK)

The pale PINK is kept for large fills, where its only neighbour is navy or white.

Figures carry no baked-in title. The supplement prints the full legend directly
beneath each image, so a title inside the image duplicates it; panels are
identified the way a journal does it, with a bold lower-case letter. Set
TITLES = True to get the titles back.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

TITLES = False

# ── palette ──────────────────────────────────────────────────────────────────
NAVY = "#22405F"      # KNHANES, and the primary category everywhere
PINK = "#E5A3A3"      # NHANES, large fills only
PINK_DK = "#CE7B7B"   # NHANES, small marks / strokes / text
TEAL = "#2A9D8F"       # reference lines, and "protective" in S5-S6
GREY = "#B9BEC4"      # recessive fills and banding
GREY_DK = "#7F878F"   # grey data marks that must separate from PINK
GREY_LT = "#E7E9EB"   # row banding, unfilled-bar strokes
GRID = "#EBEDEF"
INK = "#1A1A1A"       # primary text
INK2 = "#55595E"      # secondary text
AXIS = "#3C4043"

SURVEY = {"KNHANES": NAVY, "NHANES": PINK}
SURVEY_DK = {"KNHANES": NAVY, "NHANES": PINK_DK}

# ── global rc ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.7,
    "axes.titlepad": 7,
    "axes.labelpad": 4,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handletextpad": 0.6,
    "legend.labelspacing": 0.45,
    "legend.borderpad": 0.2,
    "figure.dpi": 400,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "text.color": INK,
    "lines.solid_capstyle": "round",
})


def panel(ax, grid="y", spines=("left", "bottom")):
    """Journal frame: two spines, one faint grid direction, grid behind the data."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, pad=2.5)
    return ax


def label(ax, letter, x=-0.085, y=1.045, size=11):
    """The bold lower-case panel letter, in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left", color=INK)


def fig_label(fig, letter, x, y, size=11):
    """Panel letter in figure coordinates, for panels of unequal height."""
    fig.text(x, y, letter, fontsize=size, fontweight="bold", va="bottom",
             ha="left", color=INK)


def thousands(ax, axis="x"):
    f = FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def title(ax_or_fig, text, **kw):
    """A title only when TITLES is on, so the default output has none."""
    if not TITLES:
        return
    if hasattr(ax_or_fig, "suptitle"):
        ax_or_fig.suptitle(text, fontsize=11, color=INK, **kw)
    else:
        ax_or_fig.set_title(text, fontsize=10, color=INK, **kw)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


# ── shared text helpers ──────────────────────────────────────────────────────
_SYM = [(">=", "≥"), ("<=", "≤"), ("10^3", "10³"), ("10^9", "10⁹"),
        ("cm2", "cm²"), ("m2", "m²"), ("->", "→"), ("--", "–"),
        ("uIU", "μIU"), ("uL", "μL"), ("ug", "μg")]


def pretty(s):
    """Typeset a variable label: real inequality and unit glyphs, no ASCII stand-ins."""
    for a, b in _SYM:
        s = s.replace(a, b)
    return s


def wrap(s, width):
    """Break a label onto as few lines as possible without cutting a word."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def shorten(s, width):
    """Trim a label at a word boundary and mark the trim, never mid-word."""
    s = pretty(s)
    if len(s) <= width:
        return s
    cut = s[:width].rsplit(" ", 1)[0]
    return cut + "…"


def label_fit(s, width):
    """Fit a variable label to `width` without ever cutting a definition in half.

    The parenthetical in a platform label is the diagnostic threshold, e.g.
    "Obesity (KSSO >=25 / WHO >=30, country-specific)". Truncating that mid-clause
    reads as an error, so it is dropped whole; the full definition is in the
    operational-definition catalogue. Only if the bare name is still too long is
    the name itself trimmed.
    """
    s = pretty(s)
    if len(s) <= width:
        return s
    bare = s.split(" (")[0].strip()
    return bare if len(bare) <= width else shorten(bare, width)
