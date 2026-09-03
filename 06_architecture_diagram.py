# -*- coding: utf-8 -*-
"""
Step 6 - the pipeline diagram, for the opening slide and the book.

Draws the whole project on one page: what goes in, what each stage does to it,
and what comes out. It reads nothing and depends on nothing, so it can be run
at any time.

Outputs:
    pipeline_diagram.png
    model_diagram.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fly_common as fc

INK = "#1b2430"
BLUE = "#3d6fa8"
RED = "#b8384a"
GREEN = "#2f7a52"
SAND = "#c98a2e"
GREY = "#8a8f96"


def box(ax, xy, w, h, title, lines, edge, face="white", title_size=10.5,
        body_size=8.4, fig_height_in=8.4):
    """Draw a labelled box, and make sure the text actually fits inside it.

    Matplotlib will happily draw text straight through a box's border. Rather
    than hand-tuning every height until nothing overflows, the body text is
    shrunk automatically whenever the lines would not fit in the space left
    below the title.
    """
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=2, edgecolor=edge, facecolor=face,
                                zorder=3))
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=edge, zorder=4)

    top = y + h - 0.105          # where the body text starts
    available = top - y - 0.022  # leave a margin above the bottom border
    linespacing = 1.4
    # one line of `body_size` points, expressed in axes units
    line_units = (body_size / 72.0) * linespacing / fig_height_in
    needed = len(lines) * line_units
    if needed > available and needed > 0:
        body_size *= available / needed

    ax.text(x + w / 2, top, "\n".join(lines), ha="center", va="top",
            fontsize=body_size, color=INK, zorder=4, linespacing=linespacing)


def arrow(ax, start, end, colour=GREY, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=17, linewidth=1.8,
                                 color=colour, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.018,
                label, ha="center", va="bottom", fontsize=8,
                color=colour, style="italic", zorder=4)


def pipeline_diagram(output_dir):
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.975, "Predicting where a fly will be, a fraction of a second from now",
            ha="center", fontsize=15.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.937,
            "Kim & Dickinson (2017) Drosophila tracking data  ->  movement "
            "clustering  ->  a learned next-position predictor",
            ha="center", fontsize=10, color=GREY)

    box(ax, (0.02, 0.70), 0.20, 0.18, "1. RAW DATA",
        ["11 recordings, 326 flies", "up to 1 hour each, 60 Hz",
         "x, y, heading, velocity", "per frame, in millimetres"], BLUE)

    box(ax, (0.27, 0.70), 0.20, 0.18, "2. FIND REAL MOVEMENT",
        ["flies sit still ~70% of", "the time, so keep only",
         "'bouts': >2 mm/s for at", "least a second"], BLUE)

    box(ax, (0.52, 0.70), 0.20, 0.18, "3. BODY-CENTRED FRAME",
        ["rotate each window so the", "fly starts at (0,0) facing 0",
         "-> the model learns HOW it", "moves, not WHERE it is"], BLUE)

    box(ax, (0.77, 0.70), 0.21, 0.18, "4. SLIDING WINDOWS",
        ["10 frames of history in,", "displacement over the next",
         "3 frames (50 ms) out", "one row = one moment"], BLUE)

    arrow(ax, (0.22, 0.79), (0.27, 0.79))
    arrow(ax, (0.47, 0.79), (0.52, 0.79))
    arrow(ax, (0.72, 0.79), (0.77, 0.79))

    # --- the two branches --------------------------------------------------
    box(ax, (0.04, 0.37), 0.26, 0.24, "BRANCH A - CLUSTERING (step 1)",
        ["soft-DTW K-Means over 40-frame", "segments, k swept from 2 upward",
         "silhouette picks the best k", "",
         "answers: how many kinds of", "movement are there?"], SAND)

    box(ax, (0.37, 0.37), 0.26, 0.24, "BRANCH B - TrajLearn (step 2)",
        ["millimetre-scale hex grid", "replaces H3's metre-scale one",
         "transformer over cell sequences", "",
         "answers: which cell does the", "fly move into next?"], SAND)

    box(ax, (0.70, 0.37), 0.28, 0.24, "BRANCH C - LSTM (step 3)  <- MAIN",
        ["2-layer LSTM, 64 hidden units", "predicts a continuous position",
         "in millimetres, not a cell", "",
         "answers: exactly where will the", "fly be in 50 ms?"], RED)

    arrow(ax, (0.30, 0.70), (0.17, 0.61), rad=0.15)
    arrow(ax, (0.55, 0.70), (0.50, 0.61), rad=0.05)
    arrow(ax, (0.87, 0.70), (0.84, 0.61), rad=-0.05)

    # --- honesty layer -----------------------------------------------------
    box(ax, (0.04, 0.11), 0.42, 0.21, "HOW WE KNOW IT WORKS",
        ["split by FLY, never by frame: 70% train / 10% validation / 20% test",
         "so a test fly is one the model has never seen in any form",
         "",
         "validation decides when to stop training;",
         "the test set is touched once, at the very end"], GREEN)

    box(ax, (0.52, 0.11), 0.46, 0.21, "WHAT WE COMPARE AGAINST",
        ["CONSTANT VELOCITY - assume the fly carries straight on.",
         "Free, needs no training, and over a few ms it is hard to beat.",
         "STOPPED - assume no movement at all.",
         "",
         "The model is only worth its complexity if it beats constant velocity."],
        GREEN)

    arrow(ax, (0.17, 0.37), (0.20, 0.32), rad=0.0)
    arrow(ax, (0.84, 0.37), (0.78, 0.32), rad=0.0)

    ax.text(0.5, 0.045,
            "Output: median error in mm, win rate against physics, per-fly "
            "trajectory figures, and the accuracy-vs-horizon curve.",
            ha="center", fontsize=10, color=INK, style="italic")

    fig.tight_layout()
    path = os.path.join(output_dir, "pipeline_diagram.png")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved '{path}'")


def model_diagram(output_dir):
    """A closer look at the predictor itself - the 'how does it work' slide."""
    fig_h = 6.4
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.95, "Inside the predictor", ha="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(0.5, 0.90,
            "Everything is measured in the fly's own frame of reference, so the "
            "model cannot memorise arena positions",
            ha="center", fontsize=9.8, color=GREY)

    box(ax, (0.02, 0.36), 0.235, 0.42, "INPUT  (10 frames)",
        ["for each of the last 10 frames:", "",
         "forward step (mm)", "sideways step (mm)", "speed (mm/s)",
         "turn rate (rad/s)", "heading change (rad)", "",
         "= a 10 x 5 table"], BLUE, fig_height_in=fig_h)

    box(ax, (0.315, 0.42), 0.20, 0.30, "LSTM  x2 layers",
        ["reads the 10 frames", "in order, carrying a",
         "memory of what came", "before", "", "64 hidden units"], RED,
        fig_height_in=fig_h)

    box(ax, (0.575, 0.44), 0.17, 0.26, "SMALL NETWORK",
        ["64 -> 64 -> 2", "", "turns the memory", "into one answer"], RED,
        fig_height_in=fig_h)

    box(ax, (0.80, 0.42), 0.18, 0.30, "OUTPUT",
        ["where the fly will be", "50 ms from now:", "",
         "forward (mm)", "sideways (mm)", "",
         "relative to itself, now"], GREEN, fig_height_in=fig_h)

    arrow(ax, (0.255, 0.57), (0.315, 0.57), colour=INK)
    arrow(ax, (0.515, 0.57), (0.575, 0.57), colour=INK)
    arrow(ax, (0.745, 0.57), (0.80, 0.57), colour=INK)

    ax.text(0.5, 0.28,
            "Trained with SmoothL1 loss (beta = 0.1, matched to the ~0.1 mm "
            "scale of a single step), AdamW at 1e-3,\n"
            "batches of 512, gradient clipping at 1.0, and early stopping once "
            "the validation score stops improving for 5 epochs.",
            ha="center", fontsize=9.5, color=INK, linespacing=1.7)

    ax.text(0.5, 0.15,
            "Why predict a DISPLACEMENT and not a POSITION?",
            ha="center", fontsize=11, fontweight="bold", color=SAND)
    ax.text(0.5, 0.055,
            "A position would let the model learn that flies hang around the "
            "edges of this particular arena - true, but useless.\n"
            "A displacement in the fly's own frame forces it to learn how a fly "
            "turns and accelerates, which transfers to any fly, anywhere.",
            ha="center", fontsize=9.5, color=INK, linespacing=1.7)

    fig.tight_layout()
    path = os.path.join(output_dir, "model_diagram.png")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved '{path}'")


def main():
    os.makedirs(fc.OUTPUT_DIR, exist_ok=True)
    pipeline_diagram(fc.OUTPUT_DIR)
    model_diagram(fc.OUTPUT_DIR)
    print("\nBoth diagrams are drawn from scratch, so they need no data and "
          "can be\nregenerated at any time.")


if __name__ == "__main__":
    main()
