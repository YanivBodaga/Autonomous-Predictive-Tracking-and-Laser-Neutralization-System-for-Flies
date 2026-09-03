# -*- coding: utf-8 -*-
"""
Step 5 - "how far ahead should we predict?", as one presentation slide.

Reads horizon_sweep.csv (written by the horizon sweep cell in the Colab
notebook, which re-runs step 3 at several prediction horizons) and turns it
into a figure that answers a question the project has to answer anyway: the
goal is to hit a moving fly, so how far in advance is it actually worth
predicting?

The answer is not "as far as possible". Two effects pull against each other:

  * predicting further ahead is harder for everyone, so both the model and the
    physics baseline get worse in absolute terms
  * but a straight-line assumption decays much faster than a learned one,
    because over a longer window the fly has time to turn

So the model's ADVANTAGE over physics rises, peaks, and then falls again. The
peak is the horizon where learning buys the most, and that is a result worth a
slide of its own.

Outputs:
    horizon_efficiency.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fly_common as fc

MODEL_NAME = "LSTM (learned)"
BASELINE_NAME = "Constant velocity (baseline)"
STOPPED_NAME = "Stopped (no movement)"


def load_sweep():
    path = os.path.join(fc.OUTPUT_DIR, "horizon_sweep.csv")
    if not os.path.exists(path):
        sys.exit(
            f"'{path}' not found.\n"
            "Run the horizon-sweep cell in the Colab notebook first - it runs\n"
            "03_continuous_model.py at several values of FLY_HORIZON and\n"
            "collects the results into that file."
        )
    df = pd.read_csv(path)
    missing = {"model", "median_error_mm", "horizon_ms"} - set(df.columns)
    if missing:
        sys.exit(f"'{path}' is missing columns: {sorted(missing)}")
    return df.sort_values("horizon_ms")


def plot(df, output_dir):
    model = df[df["model"] == MODEL_NAME].sort_values("horizon_ms")
    base = df[df["model"] == BASELINE_NAME].sort_values("horizon_ms")
    stopped = df[df["model"] == STOPPED_NAME].sort_values("horizon_ms")

    if model.empty or base.empty:
        sys.exit("horizon_sweep.csv has no LSTM or baseline rows.")

    merged = model.merge(base, on="horizon_ms", suffixes=("_model", "_base"))
    merged["improvement_%"] = (
        100 * (merged["median_error_mm_base"] - merged["median_error_mm_model"])
        / merged["median_error_mm_base"])

    best = merged.loc[merged["improvement_%"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- left: raw accuracy, both methods ---------------------------------
    ax = axes[0]
    ax.plot(base["horizon_ms"], base["median_error_mm"], marker="s",
            color="steelblue", linewidth=2.2, markersize=8,
            label="constant velocity (physics)")
    ax.plot(model["horizon_ms"], model["median_error_mm"], marker="o",
            color="crimson", linewidth=2.2, markersize=8, label="LSTM (learned)")
    if not stopped.empty:
        ax.plot(stopped["horizon_ms"], stopped["median_error_mm"], marker="^",
                color="0.6", linewidth=1.6, linestyle=":",
                label="assume the fly is stopped")
    ax.fill_between(merged["horizon_ms"], merged["median_error_mm_model"],
                    merged["median_error_mm_base"],
                    color="mediumseagreen", alpha=0.22,
                    label="what the model saves")
    ax.set_xlabel("how far ahead we predict (ms)")
    ax.set_ylabel("median error (mm)")
    ax.set_title("Predicting further ahead is harder for both methods\n"
                 "(but not equally harder)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # --- right: the advantage, which is the actual message ------------------
    ax = axes[1]
    ax.plot(merged["horizon_ms"], merged["improvement_%"], marker="o",
            color="darkgreen", linewidth=2.4, markersize=9)
    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(merged["horizon_ms"], 0, merged["improvement_%"],
                    color="mediumseagreen", alpha=0.25)

    ax.scatter([best["horizon_ms"]], [best["improvement_%"]], s=260,
               facecolor="none", edgecolor="darkgreen", linewidth=2.5, zorder=6)
    # Parked in the empty upper-right of the axes rather than next to the peak,
    # where it would sit on top of the curve and the percentage labels.
    ax.annotate(f"best trade-off:  {best['horizon_ms']:.0f} ms\n"
                f"{best['improvement_%']:.0f}% more accurate than physics",
                xy=(best["horizon_ms"], best["improvement_%"]),
                xytext=(0.97, 0.93), textcoords="axes fraction",
                ha="right", va="top",
                fontsize=10.5, color="darkgreen", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="darkgreen", linewidth=1.6, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="darkgreen",
                                linewidth=1.6,
                                connectionstyle="arc3,rad=-0.2"))

    for _, r in merged.iterrows():
        # keep the peak's own label clear of the ring drawn around it
        dy = -20 if r["horizon_ms"] == best["horizon_ms"] else 13
        ax.annotate(f"{r['improvement_%']:.0f}%",
                    xy=(r["horizon_ms"], r["improvement_%"]),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=9, color="darkgreen")

    ax.set_xlabel("how far ahead we predict (ms)")
    ax.set_ylabel("model's advantage over physics (%)")
    ax.set_title("The advantage peaks, then fades\n"
                 "there is a right amount of time to look ahead", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(min(0, merged["improvement_%"].min() * 1.25),
                merged["improvement_%"].max() * 1.45)

    fig.suptitle(
        "Choosing the prediction horizon. Too short and physics is already "
        "good enough; too long and nothing predicts well.\n"
        f"Measured on held-out flies. Best trade-off at "
        f"{best['horizon_ms']:.0f} ms, where the model is "
        f"{best['improvement_%']:.0f}% more accurate than a straight-line guess.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    path = os.path.join(output_dir, "horizon_efficiency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")
    return merged, best


def main():
    os.makedirs(fc.OUTPUT_DIR, exist_ok=True)
    df = load_sweep()
    merged, best = plot(df, fc.OUTPUT_DIR)

    print(f"\n{'=' * 70}\nHOW FAR AHEAD IS IT WORTH PREDICTING?\n{'=' * 70}")
    table = merged[["horizon_ms", "median_error_mm_model",
                    "median_error_mm_base", "improvement_%"]].copy()
    table.columns = ["ahead (ms)", "LSTM (mm)", "physics (mm)", "better by (%)"]
    print(table.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    print(f"\nThe model helps most at {best['horizon_ms']:.0f} ms ahead, where it is\n"
          f"{best['improvement_%']:.0f}% more accurate than assuming the fly carries\n"
          f"straight on. Shorter than that, a straight line is already almost\n"
          f"right; longer than that, the fly has had time to do something the\n"
          f"model cannot anticipate either.")

    out = os.path.join(fc.OUTPUT_DIR, "horizon_efficiency.csv")
    table.to_csv(out, index=False)
    print(f"\nSaved '{out}'")


if __name__ == "__main__":
    main()
