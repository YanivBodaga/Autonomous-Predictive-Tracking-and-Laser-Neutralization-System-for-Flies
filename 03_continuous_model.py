# -*- coding: utf-8 -*-
"""
Step 3 - predict where the fly will be one frame (~17 ms) from now.

This is the model that directly answers the project's goal. Unlike the
TrajLearn route (which predicts which grid cell the fly moves to next), this
one predicts a continuous position in millimetres.

What it predicts
----------------
Given a window of the last WINDOW frames, predict the fly's DISPLACEMENT over
the next frame, expressed in the fly's own reference frame at the end of the
window (forward / sideways rather than arena x / y). Predicting a displacement
rather than an absolute position, in a body-centred frame, means the model
learns "how does this fly turn and accelerate" instead of memorising where in
the arena things tend to happen.

What it is compared against
---------------------------
CONSTANT VELOCITY: assume the fly keeps its current velocity for one more
frame. This is the honest baseline - it is what you would do with no model at
all, and at 17 ms it is already a strong predictor. A learned model is only
worth its complexity if it beats this, so both are evaluated on identical test
windows and reported side by side.

Also reported is STOPPED (assume the fly does not move), which shows how much
of the apparent accuracy is just the fly being slow.

Evaluation is on movement bouts only, split by fly so that no fly contributes
to both training and test.

Outputs (in the output folder):
    prediction_results.csv      per-model error metrics on the test set
    training_history.csv        per-epoch validation numbers
    epoch_checkpoints.csv       test-set score of each saved snapshot
    learning_curve.png          per-epoch error vs. the fixed baselines
    epoch_progression.png       test-set accuracy at epoch 0, 5, 10, ...
    error_by_condition.png      error against speed and turn rate
    prediction_error_plot.png   error distributions
    prediction_examples.png     individual moments, true vs. predicted
    next_step_lstm.pt           weights, snapshots, and everything needed
                                to rebuild the model at any saved epoch
    fly_split.json              which flies went to train / val / test
"""

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
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is required. Install with:  pip install torch")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW = fc.env_int("FLY_WINDOW", 10)     # frames of history fed to the model (~0.17 s)
HORIZON = fc.env_int("FLY_HORIZON", 1)    # frames ahead to predict (1 = ~17 ms)
VALIDATION_FRACTION = 0.1 # fraction of FLIES held out
TEST_FRACTION = 0.2
RANDOM_STATE = 42

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 512
MAX_EPOCHS = fc.env_int("FLY_MAX_EPOCHS", 30)
LEARNING_RATE = 1e-3
PATIENCE = fc.env_int("FLY_PATIENCE", 5)

# Snapshot the model every N epochs (plus epoch 0, before any training). These
# snapshots are what makes "the model improves as it trains" a measurement
# rather than a claim: every snapshot is scored on the same held-out test set,
# and 04_report_figures.py redraws the same fly's trajectory from each one.
CHECKPOINT_EVERY = fc.env_int("FLY_CHECKPOINT_EVERY", 5)

MAX_WINDOWS_PER_FLY = fc.env_int("FLY_MAX_WINDOWS", 4000)  # cap so one recording can't dominate
NROWS_PER_FILE = fc.env_int("FLY_NROWS", None)             # e.g. 200000 for a quick test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model inputs, per frame in the window. All are body-centred or scalar, so
# nothing leaks the fly's absolute position or compass orientation.
FEATURE_NAMES = [
    "forward_step(mm)",     # displacement along the fly's heading
    "lateral_step(mm)",     # displacement perpendicular to it
    "speed(mm/s)",
    "angular_velocity(rad/s)",
    "delta_heading(rad)",   # heading change since the previous frame
]


# ---------------------------------------------------------------------------
# Building windows
# ---------------------------------------------------------------------------
def bout_to_windows(bout, window=WINDOW, horizon=HORIZON):
    """One movement bout -> (X, Y, baseline, history, times) arrays.

    Each row is ONE MOMENT: a window of history ending at some frame, and the
    displacement over the next `horizon` frames from that frame. A single fly
    contributes many such moments.

    X        (n, window, n_features)  model input
    Y        (n, 2)                   true next-frame displacement, body frame
    baseline (n, 2)                   constant-velocity prediction, body frame
    history  (n, window, 2)           the window's own path, same body frame
    times    (n,)                     recording time of each prediction moment

    `history` and `times` are not used for training. They exist so results can
    be drawn and labelled: plotting where the fly came from, and saying which
    fly and which second it was, is what makes the error figures concrete.
    """
    x = bout["position_x(mm)"].to_numpy(np.float64)
    y = bout["position_y(mm)"].to_numpy(np.float64)
    heading = bout["heading_unwrapped(rad)"].to_numpy(np.float64)
    speed = bout["speed(mm/s)"].to_numpy(np.float64)
    ang_vel = bout["angular_velocity(rad/s)"].to_numpy(np.float64)

    n = len(x)
    if n < window + horizon + 1:
        return None

    # frame-to-frame displacement, rotated into each frame's own heading
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    forward = dx * cos_h - dy * sin_h
    lateral = dx * sin_h + dy * cos_h
    d_heading = np.diff(heading, prepend=heading[0])

    features = np.column_stack([forward, lateral, speed, ang_vel, d_heading])

    starts = np.arange(0, n - window - horizon)
    if len(starts) == 0:
        return None

    X = np.stack([features[s:s + window] for s in starts])

    # target: displacement from the window's last frame to `horizon` frames on,
    # expressed in the heading frame of that last frame
    last = starts + window - 1
    target = last + horizon
    gdx = x[target] - x[last]
    gdy = y[target] - y[last]
    c, s = np.cos(-heading[last]), np.sin(-heading[last])
    Y = np.column_stack([gdx * c - gdy * s, gdx * s + gdy * c])

    # constant-velocity baseline, in the same frame: keep the last observed
    # per-frame step for `horizon` more frames
    bx, by = forward[last] * horizon, lateral[last] * horizon
    baseline = np.column_stack([bx, by])

    # the window's own path, in the same frame as Y: the last frame sits at
    # (0, 0) and the fly faces +x there, so every window is directly comparable
    idx = starts[:, None] + np.arange(window)[None, :]
    hx = x[idx] - x[last][:, None]
    hy = y[idx] - y[last][:, None]
    history = np.stack([hx * c[:, None] - hy * s[:, None],
                        hx * s[:, None] + hy * c[:, None]], axis=-1)

    times = bout["time(s)"].to_numpy()[last]

    return (X.astype(np.float32), Y.astype(np.float32),
            baseline.astype(np.float32), history.astype(np.float32),
            times.astype(np.float32))


def collect_windows(rng):
    print("Loading raw data and building prediction windows...")
    per_fly = {}

    for fly_id, _condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        parts = []
        for bout in fc.extract_movement_bouts(fly_df):
            out = bout_to_windows(bout)
            if out is not None:
                parts.append(out)
        if not parts:
            continue

        X = np.concatenate([p[0] for p in parts])
        Y = np.concatenate([p[1] for p in parts])
        B = np.concatenate([p[2] for p in parts])
        H = np.concatenate([p[3] for p in parts])
        T = np.concatenate([p[4] for p in parts])

        if len(X) > MAX_WINDOWS_PER_FLY:
            keep = rng.choice(len(X), MAX_WINDOWS_PER_FLY, replace=False)
            X, Y, B, H, T = X[keep], Y[keep], B[keep], H[keep], T[keep]

        # who and when, so each plotted example can be identified
        M = np.column_stack([np.full(len(X), fly_id, np.float32), T])
        per_fly[fly_id] = (X, Y, B, H, M)

    if not per_fly:
        sys.exit("No usable windows - try lowering MIN_SPEED_MM_S in fly_common.py.")

    total = sum(len(v[0]) for v in per_fly.values())
    print(f"\n{total:,} windows from {len(per_fly)} flies "
          f"(window {WINDOW} frames, horizon {HORIZON} frame = {HORIZON * fc.DT * 1000:.0f} ms).")
    return per_fly


def split_by_fly(per_fly, rng):
    fly_ids = np.array(sorted(per_fly))
    rng.shuffle(fly_ids)
    n = len(fly_ids)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    n_val = max(1, int(round(n * VALIDATION_FRACTION)))
    if n_test + n_val >= n:
        sys.exit(f"Too few flies ({n}) to split.")

    groups = {
        "train": fly_ids[: n - n_val - n_test],
        "val": fly_ids[n - n_val - n_test: n - n_test],
        "test": fly_ids[n - n_test:],
    }

    out = {}
    for name, ids in groups.items():
        X = np.concatenate([per_fly[f][0] for f in ids])
        Y = np.concatenate([per_fly[f][1] for f in ids])
        B = np.concatenate([per_fly[f][2] for f in ids])
        H = np.concatenate([per_fly[f][3] for f in ids])
        M = np.concatenate([per_fly[f][4] for f in ids])
        out[name] = (X, Y, B, H, M)
        print(f"  {name:5s}: {len(ids):3d} flies, {len(X):,} windows")

    # Which fly went where is needed again later: 04_report_figures.py draws a
    # trajectory for every fly and has to label each one train or test, since a
    # good-looking prediction on a training fly proves nothing.
    assignment = {name: [int(f) for f in ids] for name, ids in groups.items()}
    return out, assignment


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class NextStepLSTM(nn.Module):
    """LSTM over the history window, predicting a 2-D displacement."""

    def __init__(self, n_features, hidden=HIDDEN_SIZE, layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


@torch.no_grad()
def predict_in_batches(model, X, batch_size=4096):
    """Run inference in chunks.

    With all 11 recordings this can be well over a million windows; pushing
    them through the LSTM in one call would exhaust memory long before the
    maths became a problem, so evaluation is chunked just like training.
    """
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size].to(DEVICE)
        outputs.append(model(chunk).cpu())
    return torch.cat(outputs) if outputs else torch.empty(0, 2)


def train_model(splits):
    X_tr, Y_tr, *_ = splits["train"]
    X_va, Y_va, B_va, *_ = splits["val"]

    # The baselines don't train, so they are flat references the learning curve
    # can be read against: the epoch where the model's curve drops below the
    # constant-velocity line is the epoch it starts being worth using at all.
    baseline_val_mm = float(np.median(euclidean_error(B_va, Y_va)))
    stopped_val_mm = float(np.median(euclidean_error(np.zeros_like(Y_va), Y_va)))

    # standardise inputs using training statistics only
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(0) + 1e-8

    def prep(X, Y):
        return (torch.from_numpy((X - mean) / std).float(),
                torch.from_numpy(Y).float())

    Xtr, Ytr = prep(X_tr, Y_tr)
    Xva, Yva = prep(X_va, Y_va)

    model = NextStepLSTM(X_tr.shape[-1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Targets are per-frame displacements of ~0.1 mm. SmoothL1's default
    # beta=1.0 would put every sample deep in the quadratic region, making the
    # loss effectively plain MSE and highly sensitive to the rare large jumps
    # that tracking glitches produce. beta=0.1 puts the transition near the
    # typical step size, so ordinary steps get a smooth quadratic loss while
    # outliers are down-weighted linearly.
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr),
        batch_size=BATCH_SIZE, shuffle=True)

    # The training log is the main thing anyone watching this will read, so it
    # explains itself rather than assuming the reader knows the jargon.
    print(f"""
{'=' * 78}
TRAINING  -  learning to predict where the fly is {HORIZON * fc.DT * 1000:.0f} ms from now
{'=' * 78}
Running on {DEVICE}. Learning from {len(Xtr):,} moments, and checking progress
against {len(Xva):,} moments from different flies it never trains on.

WHAT WE ARE COMPETING AGAINST (measured on the held-out flies):
  Simple physics  - assume the fly keeps its current speed and direction.
                    Typically misses by {baseline_val_mm:.4f} mm.
  Assume standing still - typically misses by {stopped_val_mm:.4f} mm.

  The model is only useful if it misses by LESS than {baseline_val_mm:.4f} mm,
  because physics is free and needs no training at all.

WHAT EACH COLUMN BELOW MEANS:
  epoch          one full pass over all the training data
  fit            internal training score. Lower is better. It has no units and
                 no physical meaning - only its downward trend matters.
  check          the same score, but on flies the model never trains on.
                 This is what decides when training stops.
  typical miss   how far off the model is on a typical prediction, in
                 millimetres, on those held-out flies.  <-- THE NUMBER THAT MATTERS
  beat physics   out of all held-out moments, the share where the model landed
                 closer to the truth than simple physics did.
  verdict        whether 'typical miss' is better than the {baseline_val_mm:.4f} mm physics mark.

{'-' * 78}
{'epoch':>5} {'fit':>9} {'check':>9} {'typical miss':>14} {'beat physics':>14}   verdict
{'-' * 78}""")
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Epoch 0 is the untrained model - random weights, no learning at all. It is
    # the honest starting point every later snapshot is measured against, so the
    # improvement over training is visible rather than assumed.
    checkpoints = {0: snapshot()}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * len(xb)
        train_loss = total / len(Xtr)

        val_pred = predict_in_batches(model, Xva)
        val_loss = loss_fn(val_pred, Yva).item()
        val_mm = torch.linalg.norm(val_pred - Yva, dim=1).median().item()

        # Win rate on the validation set: the fraction of individual moments
        # where the model is closer than constant velocity. Median error and
        # win rate can disagree, so both are tracked.
        val_win = float((euclidean_error(val_pred.numpy(), Y_va)
                         < euclidean_error(B_va, Y_va)).mean())
        verdict = ("BETTER than physics" if val_mm < baseline_val_mm
                   else "still worse than physics")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_median_mm": val_mm,
                        "val_win_rate": val_win,
                        "baseline_median_mm": baseline_val_mm,
                        "stopped_median_mm": stopped_val_mm})

        print(f"{epoch:>5} {train_loss:>9.5f} {val_loss:>9.5f} "
              f"{val_mm:>11.4f} mm {100 * val_win:>13.1f}%   {verdict}")

        if epoch % CHECKPOINT_EVERY == 0:
            checkpoints[epoch] = snapshot()

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"{'-' * 78}\n"
                      f"Stopped early: no improvement for {PATIENCE} epochs in a row.\n"
                      f"This is normal and expected - it means more training would\n"
                      f"only make the model memorise the training flies.")
                break
    else:
        print(f"{'-' * 78}\n"
              f"Reached the {MAX_EPOCHS}-epoch limit without stalling. If the numbers\n"
              f"were still improving, raise MAX_EPOCHS and run again.")

    # Always snapshot the last epoch actually trained, so the progression never
    # stops short of where training really ended.
    last_epoch = history[-1]["epoch"]
    if last_epoch not in checkpoints:
        checkpoints[last_epoch] = snapshot()

    if best_state is not None:
        model.load_state_dict(best_state)

    best = pd.DataFrame(history).loc[pd.DataFrame(history)["val_loss"].idxmin()]
    print(f"\nKeeping the model from epoch {int(best['epoch'])}, the best one seen\n"
          f"(typical miss {best['val_median_mm']:.4f} mm vs physics "
          f"{baseline_val_mm:.4f} mm).")
    print(f"Saved {len(checkpoints)} snapshots along the way "
          f"(epochs {sorted(checkpoints)}), so the improvement over training\n"
          f"can be measured and drawn, not just described.")

    return model, mean, std, pd.DataFrame(history), checkpoints


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def euclidean_error(pred, true):
    return np.linalg.norm(pred - true, axis=1)


def evaluate(model, mean, std, splits):
    X_te, Y_te, B_te, *_ = splits["test"]

    Xn = torch.from_numpy((X_te - mean) / std).float()
    pred = predict_in_batches(model, Xn).numpy()

    errors = {
        "LSTM (learned)": euclidean_error(pred, Y_te),
        "Constant velocity (baseline)": euclidean_error(B_te, Y_te),
        "Stopped (no movement)": euclidean_error(np.zeros_like(Y_te), Y_te),
    }

    rows = []
    for name, err in errors.items():
        rows.append({
            "model": name,
            "median_error_mm": np.median(err),
            "mean_error_mm": err.mean(),
            "p90_error_mm": np.percentile(err, 90),
            "rmse_mm": np.sqrt((err ** 2).mean()),
        })
    results = pd.DataFrame(rows)

    baseline_median = results.loc[
        results["model"] == "Constant velocity (baseline)", "median_error_mm"].iloc[0]
    results["vs_baseline_%"] = (
        100 * (baseline_median - results["median_error_mm"]) / baseline_median)

    print(f"""
{'=' * 78}
FINAL RESULTS  -  on {len(Y_te):,} moments from flies the model has never seen
{'=' * 78}
Each row is one way of guessing where the fly will be in
{HORIZON * fc.DT * 1000:.0f} ms. Lower numbers are better; all distances are millimetres.

  median_error_mm  the typical miss (half the guesses are better than this)
  mean_error_mm    the average miss, which a few bad guesses can inflate
  p90_error_mm     the miss on the worst 10% of moments
  rmse_mm          an average that punishes large misses extra hard
  vs_baseline_%    how much better (+) or worse (-) than simple physics
{'-' * 78}""")
    print(results.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # Median error compares the two methods on average; this says how often the
    # learned model actually wins. A model can have a better median while still
    # losing most individual cases, so both are worth reporting.
    win_rate = 100 * (errors["LSTM (learned)"]
                      < errors["Constant velocity (baseline)"]).mean()
    print(f"\nHow often, not just how much: the model landed closer to the truth\n"
          f"than simple physics on {win_rate:.1f}% of the {len(Y_te):,} moments.\n"
          f"(Above 50% means it wins more often than it loses. This can disagree\n"
          f" with the median above - one says how often, the other how much.)")

    lstm_median = results.loc[
        results["model"] == "LSTM (learned)", "median_error_mm"].iloc[0]
    improvement = 100 * (baseline_median - lstm_median) / baseline_median
    horizon_ms = HORIZON * fc.DT * 1000
    print(f"\n{'-' * 78}\nWHAT THIS MEANS\n{'-' * 78}")
    if improvement > 5:
        print(f"The model is a clear improvement: {improvement:.1f}% more accurate than\n"
              f"simple physics on a typical prediction ({lstm_median:.4f} mm vs\n"
              f"{baseline_median:.4f} mm). Learning the fly's behaviour paid off.")
    elif improvement > 0:
        print(f"The model is only {improvement:.1f}% better than simple physics\n"
              f"({lstm_median:.4f} mm vs {baseline_median:.4f} mm) - a small gain.\n\n"
              f"This is expected and not a failure. In {horizon_ms:.0f} ms a fly barely has\n"
              f"time to change what it was already doing, so 'it keeps going the same\n"
              f"way' is a very strong guess. A learned model earns its keep when you\n"
              f"predict further ahead.\n\n"
              f"To show this, raise FLY_HORIZON (6 = 0.1 s, 30 = 0.5 s) and compare.")
    else:
        print(f"Simple physics wins here: {lstm_median:.4f} mm for the model versus\n"
              f"{baseline_median:.4f} mm for physics.\n\n"
              f"This is a real, reportable finding, not a broken model. Over {horizon_ms:.0f} ms a\n"
              f"fly cannot deviate much from the direction it was already travelling,\n"
              f"so assuming it carries straight on is close to unbeatable.\n\n"
              f"The interesting question is where that stops being true. Raise\n"
              f"FLY_HORIZON (6 = 0.1 s, 30 = 0.5 s) and find the point where the\n"
              f"model overtakes physics - that crossover is the real result.")

    return results, errors, pred


def plot_learning_curve(history, output_dir):
    """Model error per epoch against the two fixed baselines.

    This is the head-to-head view over training: the baselines are horizontal
    because they never learn, so the epoch where the model's curve crosses the
    constant-velocity line is the epoch it becomes worth using. If the curve
    never crosses, that is the honest answer - physics wins at this horizon.

    The right panel tracks the win rate, which answers a different question:
    not "is the model better on average" but "how often is it better". A model
    can improve its median while still losing most individual moments.
    """
    baseline = history["baseline_median_mm"].iloc[0]
    stopped = history["stopped_median_mm"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(history["epoch"], history["val_median_mm"], marker="o",
            color="crimson", label="LSTM (validation)")
    ax.axhline(baseline, color="steelblue", linestyle="--",
               label=f"constant velocity ({baseline:.4f} mm)")
    ax.axhline(stopped, color="0.5", linestyle=":",
               label=f"stopped ({stopped:.4f} mm)")

    crossed = history.loc[history["val_median_mm"] < baseline, "epoch"]
    if len(crossed):
        first = int(crossed.iloc[0])
        ax.axvline(first, color="darkgreen", alpha=0.4)
        ax.annotate(f"overtakes physics\nat epoch {first}",
                    xy=(first, baseline), xytext=(8, 18),
                    textcoords="offset points", fontsize=9, color="darkgreen")
    else:
        ax.set_title("Median error per epoch - never overtakes the baseline",
                     color="firebrick")
    if len(crossed):
        ax.set_title("Median error per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("median error (mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(history["epoch"], 100 * history["val_win_rate"], marker="o",
            color="darkgreen")
    ax.axhline(50, color="0.5", linestyle="--",
               label="50% = as good as a coin flip vs. physics")
    ax.set_xlabel("epoch")
    ax.set_ylabel("% of moments where LSTM is closer")
    ax.set_ylim(0, 100)
    ax.set_title("Win rate against constant velocity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "learning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def score_checkpoints(model, mean, std, splits, checkpoints):
    """Score every saved snapshot on the SAME held-out test set.

    The learning curve is measured on validation data, because that is what
    training is allowed to look at. This function does something different and
    stronger for the report: it takes the untrained model and each later
    snapshot, and scores them all on the test set - flies that played no part
    in training or in any stopping decision.

    The result is a straight answer to "does the model get better as it
    trains", with a number per snapshot rather than an impression.
    """
    X_te, Y_te, B_te, *_ = splits["test"]
    Xn = torch.from_numpy((X_te - mean) / std).float()

    baseline_err = euclidean_error(B_te, Y_te)
    stopped_err = euclidean_error(np.zeros_like(Y_te), Y_te)

    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n{'-' * 78}\nScoring each snapshot on the test set "
          f"({len(Y_te):,} unseen moments)\n{'-' * 78}")
    print(f"{'epoch':>6} {'typical miss':>14} {'beats physics on':>18}   {'vs physics':>11}")

    rows = []
    for epoch in sorted(checkpoints):
        model.load_state_dict(checkpoints[epoch])
        pred = predict_in_batches(model, Xn).numpy()
        err = euclidean_error(pred, Y_te)
        median = float(np.median(err))
        base_median = float(np.median(baseline_err))
        rows.append({
            "epoch": epoch,
            "median_error_mm": median,
            "mean_error_mm": float(err.mean()),
            "p90_error_mm": float(np.percentile(err, 90)),
            "rmse_mm": float(np.sqrt((err ** 2).mean())),
            "win_rate": float((err < baseline_err).mean()),
            "baseline_median_mm": base_median,
            "stopped_median_mm": float(np.median(stopped_err)),
            "vs_baseline_%": 100 * (base_median - median) / base_median,
        })
        r = rows[-1]
        print(f"{epoch:>6} {median:>11.4f} mm {100 * r['win_rate']:>17.1f}%   "
              f"{r['vs_baseline_%']:>+10.1f}%")

    model.load_state_dict(saved)   # restore the best model for everything else
    return pd.DataFrame(rows)


def plot_epoch_progression(ckpt_df, output_dir):
    """Test-set accuracy at each saved snapshot: epoch 0, 5, 10, ...

    This is the figure for the "does training actually help" slide. Everything
    is measured on the same unseen flies, so the only thing changing between
    points is how long the model has trained.
    """
    baseline = ckpt_df["baseline_median_mm"].iloc[0]
    stopped = ckpt_df["stopped_median_mm"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.plot(ckpt_df["epoch"], ckpt_df["median_error_mm"], marker="o",
            color="crimson", linewidth=2, markersize=7, label="LSTM (test set)")
    ax.axhline(baseline, color="steelblue", linestyle="--", linewidth=2,
               label=f"physics baseline ({baseline:.4f} mm)")
    ax.axhline(stopped, color="0.55", linestyle=":",
               label=f"assume stopped ({stopped:.4f} mm)")

    for _, r in ckpt_df.iterrows():
        ax.annotate(f"{r['median_error_mm']:.3f}",
                    xy=(r["epoch"], r["median_error_mm"]),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color="crimson")

    start = ckpt_df["median_error_mm"].iloc[0]
    end = ckpt_df["median_error_mm"].iloc[-1]
    ax.set_title(f"Accuracy improves with training\n"
                 f"untrained {start:.3f} mm  ->  trained {end:.3f} mm "
                 f"({100 * (start - end) / start:.0f}% better)", fontsize=11)
    ax.set_xlabel("epoch (0 = untrained model)")
    ax.set_ylabel("median error on the test set (mm)")
    ax.set_xticks(ckpt_df["epoch"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ckpt_df["epoch"], 100 * ckpt_df["win_rate"], marker="o",
            color="darkgreen", linewidth=2, markersize=7)
    ax.axhline(50, color="0.55", linestyle="--",
               label="50% = no better than physics")
    ax.set_xlabel("epoch (0 = untrained model)")
    ax.set_ylabel("% of test moments where the LSTM is closer")
    ax.set_ylim(0, 100)
    ax.set_xticks(ckpt_df["epoch"])
    ax.set_title("How often the model beats physics, per snapshot")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "epoch_progression.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def plot_error_by_speed(splits, pred, output_dir, n_bins=8):
    """Error against how fast, and how sharply, the fly was moving.

    This is the figure that explains WHEN the learned model is worth having.
    Constant velocity is an excellent guess for a fly walking in a straight
    line and a poor one for a fly turning hard, so breaking the error down by
    speed and by turn rate shows exactly where the model earns its place -
    and, just as usefully, where it does not.
    """
    X_te, Y_te, B_te, *_ = splits["test"]

    err_model = euclidean_error(pred, Y_te)
    err_base = euclidean_error(B_te, Y_te)

    # conditions at the moment of prediction = the last frame of the window
    speed = X_te[:, -1, FEATURE_NAMES.index("speed(mm/s)")]
    turn = np.abs(X_te[:, -1, FEATURE_NAMES.index("angular_velocity(rad/s)")])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for ax, values, label in ((axes[0], speed, "speed at the moment (mm/s)"),
                              (axes[1], turn, "|turn rate| at the moment (rad/s)")):
        # quantile bins, so every bin holds a comparable number of moments.
        # Bins with too few moments are dropped, because a median over a
        # handful of points is noise dressed up as a data point.
        min_per_bin = max(5, min(20, len(values) // (4 * n_bins)))
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        centres, m_med, b_med, counts = [], [], [], []
        if len(edges) >= 3:
            idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                sel = idx == b
                if sel.sum() < min_per_bin:
                    continue
                # the bin's median, not its midpoint: the top quantile bin is
                # very wide, and its midpoint would sit far to the right of
                # where its moments actually are, stretching the axis
                centres.append(float(np.median(values[sel])))
                m_med.append(np.median(err_model[sel]))
                b_med.append(np.median(err_base[sel]))
                counts.append(int(sel.sum()))

        # Say so plainly rather than leaving an empty pair of axes that looks
        # like a broken figure.
        if len(centres) < 2:
            ax.text(0.5, 0.5,
                    f"Not enough test moments to break down by\n{label}.\n\n"
                    f"{len(values):,} moments available; this plot needs at "
                    f"least\n{2 * min_per_bin:,} spread over two or more bins.",
                    ha="center", va="center", fontsize=10, color="0.35",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(label)
            continue

        ax.plot(centres, b_med, marker="s", color="steelblue", linewidth=2,
                label="physics baseline")
        ax.plot(centres, m_med, marker="o", color="crimson", linewidth=2,
                label="LSTM")
        ax.fill_between(centres, m_med, b_med,
                        where=np.array(m_med) < np.array(b_med),
                        color="mediumseagreen", alpha=0.25,
                        label="model is better here")
        ax.set_xlabel(label)
        ax.set_ylabel("median error (mm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # Descriptive titles, not conclusions. The expectation is that the gap
    # widens with speed and with turn rate, but the figure has to be readable
    # as evidence either way - a title asserting a result the plot does not
    # show would be worse than useless in a report.
    axes[0].set_title("Error vs. how fast the fly was going")
    axes[1].set_title("Error vs. how sharply it was turning")
    fig.suptitle(
        "Prediction error by movement condition, on the test set. Wherever the "
        "red line sits below the blue one, the model is\nthe better choice; "
        "where they meet, a straight-line guess is already good enough and the "
        "model adds nothing.",
        fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    path = os.path.join(output_dir, "error_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def plot_predictions(splits, pred, output_dir, n_examples=12):
    """Draw individual predictions: where the fly came from, and where each
    method says it goes next.

    Produces ONE figure with `n_examples` panels, no matter how large the test
    set is - the panels are a representative sample spanning easiest to
    hardest, not one image per prediction. Raise `n_examples` for more panels.

    Everything is in the window's own frame: the fly sits at (0, 0) at the
    moment of prediction and is facing +x, so all panels are comparable. The
    grey line is the observed past, and the three markers are the true next
    position and the two predictions of it.

    This is the figure that makes the error table concrete - a median error of
    a few hundredths of a millimetre is hard to picture until you see how far
    the markers actually sit from each other.
    """
    _X, Y_te, B_te, H_te, M_te = splits["test"]

    # Show a spread of difficulty rather than a random sample, which at this
    # horizon would be dominated by near-straight, essentially trivial cases.
    err = euclidean_error(pred, Y_te)
    err_baseline = euclidean_error(B_te, Y_te)
    order = np.argsort(err)
    picks = np.unique(np.linspace(0, len(order) - 1, n_examples).astype(int))
    picks = order[picks]

    wins = err < err_baseline
    win_rate = 100 * wins.mean()

    cols = 4
    rows_grid = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_grid, cols,
                             figsize=(cols * 3.4, rows_grid * 3.2),
                             squeeze=False)
    axes = axes.flatten()

    def draw_markers(ax, i, with_labels):
        ax.scatter(0, 0, color="black", s=28, zorder=6,
                   label="now" if with_labels else None)
        ax.scatter(*Y_te[i], color="seagreen", s=70, marker="*", zorder=7,
                   label="true next" if with_labels else None)
        ax.scatter(*pred[i], color="crimson", s=38, marker="X", zorder=7,
                   label="LSTM" if with_labels else None)
        ax.scatter(*B_te[i], color="steelblue", s=38, marker="P", zorder=7,
                   label="const. velocity" if with_labels else None)

    for ax_i, (ax, i) in enumerate(zip(axes, picks)):
        hist = H_te[i]
        ax.plot(hist[:, 0], hist[:, 1], color="0.6", linewidth=1.2,
                marker="o", markersize=2, label="past" if ax_i == 0 else None)
        draw_markers(ax, i, with_labels=(ax_i == 0))

        # State the comparison outright. The LSTM's own error means little on
        # its own; what matters is whether it beat the free baseline on this
        # example, so the title carries both numbers and the verdict.
        won = err[i] < err_baseline[i]
        fly_id, t_s = int(M_te[i, 0]), float(M_te[i, 1])
        ax.set_title(
            f"fly {fly_id}, t = {t_s:.1f}s\n"
            f"LSTM {err[i]:.3f}  vs  const.vel {err_baseline[i]:.3f} mm\n"
            f"{'LSTM closer' if won else 'baseline closer'}",
            fontsize=8, color="darkgreen" if won else "firebrick")
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

        # The history spans a couple of millimetres while the methods differ by
        # hundredths of one, so at this scale all three markers land on top of
        # each other. The inset zooms into just the prediction neighbourhood,
        # which is where the actual comparison happens.
        points = np.vstack([[0.0, 0.0], Y_te[i], pred[i], B_te[i]])
        centre = points.mean(axis=0)
        span = max(np.abs(points - centre).max() * 1.6, 1e-3)

        inset = ax.inset_axes([0.60, 0.60, 0.38, 0.38])
        inset.plot(hist[:, 0], hist[:, 1], color="0.6", linewidth=1.0)
        draw_markers(inset, i, with_labels=False)
        inset.set_xlim(centre[0] - span, centre[0] + span)
        inset.set_ylim(centre[1] - span, centre[1] + span)
        inset.set_aspect("equal")
        inset.set_xticks([]); inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor("0.4")
        inset.patch.set_alpha(0.95)

    for j in range(len(picks), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    n_flies_shown = len(np.unique(M_te[picks, 0]))
    fig.suptitle(
        f"Each panel is ONE MOMENT from the test set: a fly, at one instant, "
        f"predicted {HORIZON * fc.DT * 1000:.0f} ms ahead.\n"
        f"{len(picks)} moments shown, drawn from {n_flies_shown} different "
        f"flies, ordered easiest to hardest. Across the whole test set the "
        f"LSTM is closer than the baseline on {win_rate:.0f}% of moments.",
        fontsize=10.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    path = os.path.join(output_dir, "prediction_examples.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def plot_errors(errors, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names = list(errors)
    clipped = [np.clip(errors[k], 0, np.percentile(errors[k], 99)) for k in names]
    axes[0].hist(clipped, bins=50, label=names, density=True)
    axes[0].set_xlabel("prediction error (mm)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Error distribution (99th percentile clipped)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # set tick labels separately: boxplot's `labels` kwarg was renamed to
    # `tick_labels` in matplotlib 3.9, so passing either one breaks on some
    # versions.
    axes[1].boxplot([errors[k] for k in names], showfliers=False)
    axes[1].set_xticks(range(1, len(names) + 1))
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("prediction error (mm)")
    axes[1].set_title("Error comparison")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "prediction_error_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved '{path}'")


def main():
    os.makedirs(fc.OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    per_fly = collect_windows(rng)
    print("\nSplitting by fly (no fly appears in more than one partition):")
    splits, assignment = split_by_fly(per_fly, rng)

    model, mean, std, history, checkpoints = train_model(splits)
    results, errors, pred = evaluate(model, mean, std, splits)

    def save_csv(df, name):
        path = os.path.join(fc.OUTPUT_DIR, name)
        df.to_csv(path, index=False)
        print(f"Saved '{path}'")

    save_csv(results, "prediction_results.csv")
    save_csv(history, "training_history.csv")

    # Snapshots scored on the test set - the "training makes it better" evidence
    ckpt_df = score_checkpoints(model, mean, std, splits, checkpoints)
    save_csv(ckpt_df, "epoch_checkpoints.csv")

    plot_learning_curve(history, fc.OUTPUT_DIR)
    plot_epoch_progression(ckpt_df, fc.OUTPUT_DIR)
    plot_errors(errors, fc.OUTPUT_DIR)
    plot_error_by_speed(splits, pred, fc.OUTPUT_DIR)
    plot_predictions(splits, pred, fc.OUTPUT_DIR)

    # One bundle with everything 04_report_figures.py needs to rebuild the model
    # at any saved epoch: the weights, the normalisation, and the window/horizon
    # the windows were built with.
    bundle_path = os.path.join(fc.OUTPUT_DIR, "next_step_lstm.pt")
    torch.save({"state_dict": model.state_dict(),
                "checkpoints": checkpoints,
                "feature_mean": mean, "feature_std": std,
                "window": WINDOW, "horizon": HORIZON,
                "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "feature_names": FEATURE_NAMES,
                "fly_split": assignment,
                "max_windows_per_fly": MAX_WINDOWS_PER_FLY,
                "nrows_per_file": NROWS_PER_FILE},
               bundle_path)
    print(f"Saved '{bundle_path}'")

    split_path = os.path.join(fc.OUTPUT_DIR, "fly_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"horizon_frames": HORIZON,
                   "horizon_ms": HORIZON * fc.DT * 1000,
                   "window_frames": WINDOW,
                   "checkpoint_epochs": sorted(checkpoints),
                   **assignment}, f, indent=2)
    print(f"Saved '{split_path}'")

    print(f"\nNext: run 04_report_figures.py to draw a trajectory figure for "
          f"every fly,\nand the breakdown by movement cluster.")


if __name__ == "__main__":
    main()
