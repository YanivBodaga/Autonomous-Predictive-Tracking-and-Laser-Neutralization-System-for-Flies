# -*- coding: utf-8 -*-
"""
Step 4 - the figures for the project book and the presentation.

Run this AFTER 03_continuous_model.py, which saves the trained weights and the
snapshots this script reads back.

What it produces
----------------
1. ONE FIGURE PER FLY, in per_fly_trajectories/, with four panels:

     (a) the fly's real path, and the path the MODEL predicts
     (b) the fly's real path, and the path SIMPLE PHYSICS predicts
     (c) the model's path drawn from 3-4 different training snapshots,
         so the improvement over training is visible on this one fly
     (d) how far each method is off, moment by moment, along that path

   Every panel is the same fly and the same stretch of time, so the four are
   directly comparable. The title says whether this fly was in the training
   set or the test set, because a good-looking prediction on a fly the model
   trained on proves nothing.

2. per_fly_summary.csv - one row per fly with its error numbers, so the
   hundreds of figures can be summarised in a table.

3. error_by_cluster.png - prediction error broken down by the movement
   clusters found in step 1. This is what connects the two halves of the
   project: it answers "which kinds of movement is the model good at".

Two ways of drawing a predicted path
------------------------------------
FREE ROLLOUT (panels a, b, c): the method is given the first few frames and
then runs on its own, feeding its own predictions back in. Nothing corrects
it, so small errors compound and the path drifts. This is the honest picture
of "predict the whole path from the start", and it is what makes the
comparison visible at a glance.

ONE STEP AHEAD (panel d): at every point on the REAL path, predict just the
next step from real history. Errors cannot accumulate. This is what the
reported accuracy numbers measure, and it is the fair way to compare methods.

Both are shown because they answer different questions, and quoting only the
flattering one would be misleading.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fly_common as fc

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required. Install with:  pip install torch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Longest stretch drawn per fly. A whole hour of tracking would be an unreadable
# scribble. Two seconds is the sweet spot: long enough that the methods visibly
# separate, short enough that a free-running prediction has not yet drifted so
# far that the panel becomes a picture of two unrelated squiggles.
ROLLOUT_FRAMES = fc.env_int("FLY_ROLLOUT_FRAMES", 120)   # 120 frames = 2 s
MAX_EPOCH_PANELS = 4          # how many snapshots to overlay in panel (c)
FIGURE_DPI = fc.env_int("FLY_FIGURE_DPI", 110)
MAX_FLIES = fc.env_int("FLY_MAX_FIGURES", None)          # None = every fly

# error_by_cluster.png needs a soft-DTW assignment per moment, which is the
# expensive part, so it runs on a subsample.
CLUSTER_SAMPLE = fc.env_int("FLY_CLUSTER_SAMPLE", 1500)

NROWS_PER_FILE = fc.env_int("FLY_NROWS", None)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# position of speed(mm/s) inside the per-frame feature vector built by
# bout_arrays(); kept as a name so the ordering is stated once
FEATURE_INDEX_SPEED = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() \
    else os.getcwd()


# ---------------------------------------------------------------------------
# Loading what step 3 saved
# ---------------------------------------------------------------------------
def load_model_module():
    """Import 03_continuous_model.py, whose name cannot be typed as an import.

    Reusing its NextStepLSTM class guarantees this script rebuilds exactly the
    architecture that was trained, rather than a copy that could drift out of
    sync with it.
    """
    path = os.path.join(SCRIPT_DIR, "03_continuous_model.py")
    if not os.path.exists(path):
        sys.exit(f"Cannot find '{path}' - it defines the model architecture.")
    spec = importlib.util.spec_from_file_location("continuous_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bundle():
    path = os.path.join(fc.OUTPUT_DIR, "next_step_lstm.pt")
    if not os.path.exists(path):
        sys.exit(f"'{path}' not found. Run 03_continuous_model.py first.")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if "checkpoints" not in bundle:
        sys.exit(f"'{path}' has no saved snapshots. It was written by an older\n"
                 "version of 03_continuous_model.py - re-run that script.")
    return bundle


def build_predictor(model_module, bundle, state_dict):
    """A plain function: (n, window, n_features) float array -> (n, 2) mm.

    Normalisation is folded in, so callers work in raw feature units and never
    have to remember to standardise.
    """
    mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
    std = np.asarray(bundle["feature_std"], dtype=np.float64)

    model = model_module.NextStepLSTM(
        len(bundle["feature_names"]),
        hidden=bundle.get("hidden_size", 64),
        layers=bundle.get("num_layers", 2),
        dropout=bundle.get("dropout", 0.1),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    @torch.no_grad()
    def predict(X):
        Xn = torch.from_numpy((np.asarray(X, np.float64) - mean) / std).float()
        return model(Xn.to(DEVICE)).cpu().numpy()

    return predict


# ---------------------------------------------------------------------------
# Turning one bout into the arrays the panels need
# ---------------------------------------------------------------------------
def bout_arrays(bout, window):
    """Positions, heading, and per-frame features for one movement bout.

    The features are exactly the ones the model was trained on: per-frame
    displacement in the fly's own frame, speed, turn rate, heading change.
    """
    x = bout["position_x(mm)"].to_numpy(np.float64)
    y = bout["position_y(mm)"].to_numpy(np.float64)
    heading = bout["heading_unwrapped(rad)"].to_numpy(np.float64)
    speed = bout["speed(mm/s)"].to_numpy(np.float64)
    ang_vel = bout["angular_velocity(rad/s)"].to_numpy(np.float64)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    forward = dx * cos_h - dy * sin_h
    lateral = dx * sin_h + dy * cos_h
    d_heading = np.diff(heading, prepend=heading[0])

    feats = np.column_stack([forward, lateral, speed, ang_vel, d_heading])
    return x, y, heading, feats


def free_rollout(predict, feats_seed, start_xy, start_heading, n_steps, horizon):
    """Let the model drive: feed its own predictions back in, step after step.

    The model outputs a displacement over `horizon` frames in the fly's current
    body frame. To continue from there the simulated fly needs a new heading and
    a new set of per-frame features, so:

      * the heading turns to face the direction the model just moved in
      * the predicted displacement is spread evenly over the `horizon` frames

    Both are approximations - the model predicts where the fly goes, not how it
    holds its body - but they are the minimum needed to keep stepping, and any
    error they introduce shows up honestly as drift in the drawn path.

    Returns (n_steps + 1, 2) world positions, beginning at start_xy.
    """
    window = feats_seed.shape[0]
    feats = list(feats_seed)
    pos = np.array(start_xy, dtype=np.float64)
    heading = float(start_heading)
    path = [pos.copy()]

    for _ in range(n_steps):
        X = np.asarray(feats[-window:], dtype=np.float64)[None, :, :]
        df, dl = predict(X)[0].astype(np.float64)

        c, s = np.cos(heading), np.sin(heading)
        pos = pos + np.array([df * c - dl * s, df * s + dl * c])
        path.append(pos.copy())

        step_len = float(np.hypot(df, dl))
        # A near-zero step carries no direction information, so leave the
        # heading alone rather than reading noise out of atan2.
        d_theta = float(np.arctan2(dl, df)) if step_len > 1e-9 else 0.0
        heading += d_theta

        per_frame = np.array([df / horizon, dl / horizon,
                              step_len / (horizon * fc.DT),
                              d_theta / (horizon * fc.DT),
                              d_theta / horizon])
        feats.extend([per_frame] * horizon)

    return np.asarray(path)


def physics_rollout(x, y, last_idx, n_steps, horizon):
    """Constant velocity: keep the last observed velocity forever.

    A straight line in arena coordinates. This is the same assumption the
    reported baseline makes, extended over many steps instead of one, and it is
    exactly what you would predict with no model at all.
    """
    vx = x[last_idx] - x[last_idx - 1]
    vy = y[last_idx] - y[last_idx - 1]
    steps = np.arange(n_steps + 1)[:, None]
    start = np.array([x[last_idx], y[last_idx]])
    return start + steps * np.array([vx, vy]) * horizon


def one_step_errors(predict, x, y, heading, feats, window, horizon):
    """Error at every point along the REAL path, for both methods.

    Each prediction starts from real history, so nothing accumulates. These are
    the numbers behind the accuracy the project reports.
    """
    n = len(x)
    starts = np.arange(0, n - window - horizon)
    if len(starts) == 0:
        return None

    X = np.stack([feats[s:s + window] for s in starts])
    last = starts + window - 1
    target = last + horizon

    c, s = np.cos(-heading[last]), np.sin(-heading[last])
    gdx, gdy = x[target] - x[last], y[target] - y[last]
    Y = np.column_stack([gdx * c - gdy * s, gdx * s + gdy * c])

    B = np.column_stack([feats[last, 0] * horizon, feats[last, 1] * horizon])
    P = predict(X)

    return {
        "time_s": last,
        "model": np.linalg.norm(P - Y, axis=1),
        "physics": np.linalg.norm(B - Y, axis=1),
    }


# ---------------------------------------------------------------------------
# The per-fly figure
# ---------------------------------------------------------------------------
def draw_fly_figure(fly_id, split_label, bout, predictors, epoch_list,
                    window, horizon, out_dir):
    """One image per fly: model path, physics path, epochs, and the errors."""
    x, y, heading, feats = bout_arrays(bout, window)
    if len(x) < window + horizon + 10:
        return None

    # Draw the most active stretch of this fly's longest bout, not simply its
    # first frames. A bout begins the moment the fly crosses the speed
    # threshold, so its opening is the slowest part of it - the one stretch
    # where every method looks equally fine and nothing is learned from the
    # picture. This choice affects only WHICH two seconds are drawn; every
    # number the project reports is computed over all moments, not this one.
    offset = 0
    if len(x) > ROLLOUT_FRAMES:
        speed = feats[:, FEATURE_INDEX_SPEED]
        rolling = np.convolve(speed, np.ones(ROLLOUT_FRAMES) / ROLLOUT_FRAMES,
                              mode="valid")
        offset = int(np.argmax(rolling))

    n = min(len(x) - offset, ROLLOUT_FRAMES)
    if n < window + horizon + 10:
        return None

    sl = slice(offset, offset + n)
    x, y, heading, feats = x[sl], y[sl], heading[sl], feats[sl]
    start_time_s = float(bout["time(s)"].to_numpy()[offset])

    last_idx = window - 1
    n_steps = (n - window) // horizon
    if n_steps < 3:
        return None

    seed = feats[:window]
    start_xy = (x[last_idx], y[last_idx])

    # The point every free-running path is compared against: where the fly
    # really was after exactly the same number of frames. Using one index for
    # both the marker and the measurement keeps the picture and the number
    # describing the same thing.
    end_idx = min(last_idx + n_steps * horizon, n - 1)
    true_end = np.array([x[end_idx], y[end_idx]])

    final_epoch = epoch_list[-1]
    model_path = free_rollout(predictors[final_epoch], seed, start_xy,
                              heading[last_idx], n_steps, horizon)
    phys_path = physics_rollout(x, y, last_idx, n_steps, horizon)
    true_path = np.column_stack([x, y])

    errs = one_step_errors(predictors[final_epoch], x, y, heading, feats,
                           window, horizon)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ax_model, ax_phys = axes[0]
    ax_epoch, ax_err = axes[1]

    def draw_truth(ax):
        ax.plot(true_path[:end_idx + 1, 0], true_path[:end_idx + 1, 1],
                color="0.25", linewidth=2.4,
                label="real path (ground truth)", zorder=3)
        ax.scatter(*start_xy, color="black", s=70, marker="o", zorder=6,
                   label="start")
        ax.scatter(true_end[0], true_end[1], color="0.25", s=110,
                   marker="*", zorder=6, label="where it really ended up")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)

    # --- (a) the model's own path -----------------------------------------
    draw_truth(ax_model)
    ax_model.plot(model_path[:, 0], model_path[:, 1], color="crimson",
                  linewidth=2.2, linestyle="-", label="LSTM prediction", zorder=4)
    ax_model.scatter(model_path[-1, 0], model_path[-1, 1], color="crimson",
                     s=90, marker="X", zorder=6, label="LSTM end")
    drift_model = float(np.linalg.norm(model_path[-1] - true_end))
    ax_model.set_title(f"(a) What the MODEL predicts\n"
                       f"ends {drift_model:.2f} mm from the real end point",
                       fontsize=11, color="crimson")
    ax_model.legend(fontsize=8, loc="best")

    # --- (b) the physics path ---------------------------------------------
    draw_truth(ax_phys)
    ax_phys.plot(phys_path[:, 0], phys_path[:, 1], color="steelblue",
                 linewidth=2.2, label="constant velocity", zorder=4)
    ax_phys.scatter(phys_path[-1, 0], phys_path[-1, 1], color="steelblue",
                    s=90, marker="P", zorder=6, label="physics end")
    drift_phys = float(np.linalg.norm(phys_path[-1] - true_end))
    ax_phys.set_title(f"(b) What SIMPLE PHYSICS predicts\n"
                      f"ends {drift_phys:.2f} mm from the real end point",
                      fontsize=11, color="steelblue")
    ax_phys.legend(fontsize=8, loc="best")

    # --- (c) the same fly, at several points during training ---------------
    draw_truth(ax_epoch)
    shades = plt.cm.autumn(np.linspace(0.75, 0.0, len(epoch_list)))
    for colour, epoch in zip(shades, epoch_list):
        path = free_rollout(predictors[epoch], seed, start_xy,
                            heading[last_idx], n_steps, horizon)
        drift = float(np.linalg.norm(path[-1] - true_end))
        label = ("epoch 0 (untrained)" if epoch == 0 else f"epoch {epoch}")
        ax_epoch.plot(path[:, 0], path[:, 1], color=colour, linewidth=1.9,
                      alpha=0.95, label=f"{label} - off by {drift:.2f} mm",
                      zorder=4)
    ax_epoch.set_title("(c) The same prediction, at different stages of training\n"
                       "later epochs should hug the real path more closely",
                       fontsize=11)
    ax_epoch.legend(fontsize=7.5, loc="best")

    # --- (d) the honest, non-accumulating comparison -----------------------
    if errs is not None:
        t = errs["time_s"] * fc.DT + start_time_s
        ax_err.plot(t, errs["physics"], color="steelblue", linewidth=1.3,
                    alpha=0.85, label="constant velocity")
        ax_err.plot(t, errs["model"], color="crimson", linewidth=1.3,
                    alpha=0.85, label="LSTM")
        ax_err.fill_between(t, errs["model"], errs["physics"],
                            where=errs["model"] < errs["physics"],
                            color="mediumseagreen", alpha=0.3,
                            label="model is closer")
        win = 100 * float((errs["model"] < errs["physics"]).mean())
        ax_err.set_title(
            f"(d) Error at each moment, predicting one step from real history\n"
            f"model closer at {win:.0f}% of moments  |  "
            f"median {np.median(errs['model']):.3f} mm vs "
            f"{np.median(errs['physics']):.3f} mm",
            fontsize=11)
        ax_err.set_xlabel("time in the recording (s)")
        ax_err.set_ylabel(f"error predicting {horizon * fc.DT * 1000:.0f} ms ahead (mm)")
        ax_err.grid(alpha=0.3)
        ax_err.legend(fontsize=8)
    else:
        ax_err.set_visible(False)

    seen = ("TEST fly - the model never saw this one"
            if split_label == "test"
            else f"{split_label.upper()} fly - the model trained on this one, "
                 f"so judge it on test flies")
    colour = "darkgreen" if split_label == "test" else "darkorange"
    fig.suptitle(
        f"Fly {fly_id}   |   {seen}\n"
        f"Its most active {n_steps * horizon * fc.DT:.1f} s, starting at "
        f"t = {start_time_s:.0f} s. Panels (a)-(c) let each method run free for "
        f"all {n_steps * horizon} frames with no correction;\n"
        f"panel (d) instead predicts only "
        f"{horizon * fc.DT * 1000:.0f} ms ahead from real history each time, "
        f"which is what the reported accuracy measures.",
        fontsize=11.5, color=colour)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, f"fly_{fly_id:04d}_{split_label}.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    row = {"fly": fly_id, "split": split_label,
           "start_time_s": start_time_s,
           "frames_drawn": int(n_steps * horizon),
           "rollout_end_error_model_mm": drift_model,
           "rollout_end_error_physics_mm": drift_phys}
    if errs is not None:
        row.update({
            "onestep_median_model_mm": float(np.median(errs["model"])),
            "onestep_median_physics_mm": float(np.median(errs["physics"])),
            "onestep_win_rate": float((errs["model"] < errs["physics"]).mean()),
        })
    return row


def longest_bout(fly_df):
    bouts = fc.extract_movement_bouts(fly_df)
    return max(bouts, key=len) if bouts else None


def per_fly_figures(model_module, bundle):
    window = int(bundle["window"])
    horizon = int(bundle["horizon"])
    checkpoints = bundle["checkpoints"]

    # 3-4 snapshots spanning the whole of training, always including the
    # untrained model and the final one, so the comparison has both ends.
    available = sorted(checkpoints)
    if len(available) <= MAX_EPOCH_PANELS:
        epoch_list = available
    else:
        picks = np.linspace(0, len(available) - 1, MAX_EPOCH_PANELS)
        epoch_list = sorted({available[int(round(p))] for p in picks})

    print(f"Rebuilding the model at epochs {epoch_list}...")
    predictors = {e: build_predictor(model_module, bundle, checkpoints[e])
                  for e in epoch_list}

    split = bundle["fly_split"]
    label_of = {}
    for name, ids in split.items():
        for f in ids:
            label_of[int(f)] = name

    out_dir = os.path.join(fc.OUTPUT_DIR, "per_fly_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nDrawing one figure per fly into '{out_dir}'")
    print(f"(each figure: model path, physics path, {len(epoch_list)} training "
          f"snapshots, and the error over time)\n")

    rows, drawn, skipped = [], 0, 0
    for fly_id, _condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        if MAX_FLIES is not None and drawn >= MAX_FLIES:
            break
        bout = longest_bout(fly_df)
        if bout is None:
            skipped += 1
            continue
        row = draw_fly_figure(fly_id, label_of.get(fly_id, "unused"), bout,
                              predictors, epoch_list, window, horizon, out_dir)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        drawn += 1
        if drawn % 25 == 0:
            print(f"  {drawn} figures written...", flush=True)

    if not rows:
        sys.exit("No fly produced a long enough movement stretch to draw.")

    summary = pd.DataFrame(rows)
    path = os.path.join(fc.OUTPUT_DIR, "per_fly_summary.csv")
    summary.to_csv(path, index=False)

    print(f"\n{drawn} figures written, {skipped} flies skipped "
          f"(no movement stretch long enough to draw).")
    print(f"Saved '{path}'")

    test_rows = summary[summary["split"] == "test"]
    if len(test_rows) and "onestep_win_rate" in test_rows:
        print(f"\nAcross the {len(test_rows)} TEST flies (the ones that count):")
        print(f"  model beats physics on "
              f"{100 * (test_rows['onestep_median_model_mm'] < test_rows['onestep_median_physics_mm']).mean():.0f}% "
              f"of flies")
        print(f"  median per-fly error: "
              f"{test_rows['onestep_median_model_mm'].median():.4f} mm (model) vs "
              f"{test_rows['onestep_median_physics_mm'].median():.4f} mm (physics)")

    return summary


# ---------------------------------------------------------------------------
# Error broken down by movement cluster
# ---------------------------------------------------------------------------
def error_by_cluster(model_module, bundle):
    """Which kinds of movement is the model actually good at?

    Step 1 grouped short stretches of movement into clusters. This takes a
    sample of test moments, works out which cluster each one belongs to, and
    reports the error separately for each. That is what ties the clustering
    half of the project to the prediction half: not just "the model is X% more
    accurate", but "the model's advantage comes from the turning clusters".
    """
    centroid_path = os.path.join(fc.OUTPUT_DIR, "cluster_centroids.npy")
    if not os.path.exists(centroid_path):
        print(f"\nSkipping error_by_cluster.png: '{centroid_path}' not found.\n"
              f"Re-run 01_cluster_ksweep.py to write it, then run this script "
              f"again.")
        return None

    try:
        from tslearn.metrics import cdist_soft_dtw_normalized
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        from tslearn.utils import to_time_series_dataset
    except ImportError:
        print("\nSkipping error_by_cluster.png: tslearn is not installed.")
        return None

    centroids = np.load(centroid_path)
    meta_path = os.path.join(fc.OUTPUT_DIR, "cluster_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    seg_len = int(meta.get("segment_length", centroids.shape[1]))

    window = int(bundle["window"])
    horizon = int(bundle["horizon"])
    predict = build_predictor(model_module, bundle, bundle["state_dict"])

    test_flies = set(int(f) for f in bundle["fly_split"]["test"])
    rng = np.random.default_rng(0)

    print(f"\nAssigning test moments to the {len(centroids)} movement clusters "
          f"from step 1...")

    segments, err_model, err_phys = [], [], []
    for fly_id, _condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        if fly_id not in test_flies:
            continue
        for bout in fc.extract_movement_bouts(fly_df):
            x, y, heading, feats = bout_arrays(bout, window)
            n = len(x)
            # a moment needs `seg_len` frames of context behind it for the
            # cluster assignment, and `horizon` frames ahead for the target
            lo, hi = max(seg_len, window), n - horizon - 1
            if hi <= lo:
                continue
            take = min(6, hi - lo)
            for i in rng.choice(np.arange(lo, hi), size=take, replace=False):
                seg = bout.iloc[i - seg_len:i]
                aligned = fc.align_window(seg)
                segments.append(fc.features_matrix(aligned))

                X = feats[i - window:i][None, :, :]
                c, s = np.cos(-heading[i - 1]), np.sin(-heading[i - 1])
                gdx = x[i - 1 + horizon] - x[i - 1]
                gdy = y[i - 1 + horizon] - y[i - 1]
                true = np.array([gdx * c - gdy * s, gdx * s + gdy * c])
                base = np.array([feats[i - 1, 0] * horizon,
                                 feats[i - 1, 1] * horizon])
                err_model.append(float(np.linalg.norm(predict(X)[0] - true)))
                err_phys.append(float(np.linalg.norm(base - true)))

            if len(segments) >= CLUSTER_SAMPLE:
                break
        if len(segments) >= CLUSTER_SAMPLE:
            break

    if len(segments) < 50:
        print("Skipping error_by_cluster.png: too few usable test moments.")
        return None

    scaled = TimeSeriesScalerMeanVariance().fit_transform(
        to_time_series_dataset(segments))
    labels = cdist_soft_dtw_normalized(scaled, centroids).argmin(axis=1)

    df = pd.DataFrame({"cluster": labels,
                       "model_mm": err_model,
                       "physics_mm": err_phys})
    stats = (df.groupby("cluster")
               .agg(n=("model_mm", "size"),
                    model_mm=("model_mm", "median"),
                    physics_mm=("physics_mm", "median"))
               .reset_index())
    stats["improvement_%"] = 100 * (stats["physics_mm"] - stats["model_mm"]) / stats["physics_mm"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    pos = np.arange(len(stats))
    ax.bar(pos - 0.2, stats["physics_mm"], width=0.4, color="steelblue",
           label="constant velocity")
    ax.bar(pos + 0.2, stats["model_mm"], width=0.4, color="crimson",
           label="LSTM")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"cluster {int(c)}\n(n={int(n)})"
                        for c, n in zip(stats["cluster"], stats["n"])],
                       fontsize=8)
    ax.set_ylabel("median error (mm)")
    ax.set_title("Error per movement cluster")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[1]
    colours = ["mediumseagreen" if v > 0 else "indianred"
               for v in stats["improvement_%"]]
    ax.bar(pos, stats["improvement_%"], color=colours)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"cluster {int(c)}" for c in stats["cluster"]], fontsize=9)
    ax.set_ylabel("how much better the model is (%)")
    ax.set_title("Where the learned model earns its place")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Prediction accuracy by movement type, on test flies only. "
                 "Cluster numbers match clusters_plot.png from step 1.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    path = os.path.join(fc.OUTPUT_DIR, "error_by_cluster.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    stats_path = os.path.join(fc.OUTPUT_DIR, "error_by_cluster.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Saved '{path}'")
    print(f"Saved '{stats_path}'")
    print("\n" + stats.to_string(index=False,
                                 float_format=lambda v: f"{v:8.4f}"))
    return stats


def main():
    os.makedirs(fc.OUTPUT_DIR, exist_ok=True)
    model_module = load_model_module()
    bundle = load_bundle()

    print(f"{'=' * 78}\nREPORT FIGURES\n{'=' * 78}")
    print(f"Model predicts {int(bundle['horizon']) * fc.DT * 1000:.0f} ms ahead "
          f"from {int(bundle['window'])} frames of history.")
    print(f"Snapshots available: {sorted(bundle['checkpoints'])}")

    per_fly_figures(model_module, bundle)

    # The per-fly figures are the deliverable and are already on disk by now.
    # The cluster breakdown is a bonus that depends on step 1 having been run
    # with the same settings, so a failure here must not throw away the work
    # above - it is reported and the script still exits cleanly.
    try:
        error_by_cluster(model_module, bundle)
    except Exception as exc:
        print(f"\nCould not build error_by_cluster.png: {exc}")
        print("Everything else above was written successfully. This step needs\n"
              "01_cluster_ksweep.py to have run into the same output folder.")

    print(f"\nDone. Everything is in '{fc.OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
