# -*- coding: utf-8 -*-
"""
Shared utilities for the fly-trajectory project.

Handles the Kim & Dickinson (Current Biology 2017) raw CSV format and turns it
into the two representations the rest of the pipeline needs:

  * MOVEMENT SEGMENTS  - short, individually aligned windows used for clustering
                         (see 01_cluster_ksweep.py)
  * MOVEMENT BOUTS     - contiguous stretches where the fly is actually moving,
                         used both for TrajLearn tokenisation
                         (02_prepare_trajlearn.py) and for the continuous
                         next-frame predictor (03_continuous_model.py)

Why bouts matter: in this dataset the flies are stationary most of the time.
Measured on the raw data, the median frame-to-frame displacement is ~0.02 mm
and ~68% of frames move less than 0.1 mm. Training or evaluating a "where will
it be next" model on all frames mostly measures "it is still sitting still",
so every downstream script works on movement bouts only.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration shared by all scripts
# ---------------------------------------------------------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is undefined when pasted cell-by-cell into Colab/Jupyter
    SCRIPT_DIR = os.getcwd()

# Both locations can be overridden with environment variables, which is how
# the Colab notebook points the scripts at a Google Drive folder without
# editing any code. Locally, the defaults just use this script's own folder.
RAW_DATA_DIR = os.environ.get("FLY_DATA_DIR", SCRIPT_DIR)   # searched recursively for *.csv
OUTPUT_DIR = os.environ.get("FLY_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "outputs"))

FRAME_RATE_HZ = 60.0
DT = 1.0 / FRAME_RATE_HZ

# --- movement-bout detection ---
MIN_SPEED_MM_S = 2.0    # a fly is "moving" above this speed
MIN_BOUT_FRAMES = 60    # ignore bouts shorter than ~1 second
SMOOTH_SPEED_FRAMES = 5 # rolling window used to de-noise the speed signal

def env_int(name, default):
    """Read an integer setting from the environment.

    Lets the Colab notebook tune the scripts without editing them. The value
    "none" maps to None, which the scripts read as "no limit".
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() == "none":
        return None
    return int(raw)


def env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


RELEVANT_COLUMNS = [
    "fly",
    "time(s)",
    "position_x(mm)",
    "position_y(mm)",
    "velocity_x(mm/s)",
    "velocity_y(mm/s)",
    "heading(radians)",
    "velocity_heading(radians/s)",
]

MOTION_FEATURE_COLUMNS = [
    "heading_sin",
    "heading_cos",
    "speed(mm/s)",
    "angular_velocity(rad/s)",
]


# ---------------------------------------------------------------------------
# Loading raw files
# ---------------------------------------------------------------------------
def find_raw_csv_files(raw_data_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR):
    """Every *.csv under raw_data_dir, excluding anything we ourselves wrote."""
    file_paths = sorted(glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True))
    output_dir_abs = os.path.abspath(output_dir)
    file_paths = [f for f in file_paths if not os.path.abspath(f).startswith(output_dir_abs)]
    if not file_paths:
        sys.exit(
            f"No CSV files found under '{raw_data_dir}'.\n"
            "Place the raw tracking CSVs anywhere under this folder and re-run."
        )
    return file_paths


def find_fly_data_header_row(file_path):
    """0-indexed line number of the 'fly,time(s),...' header, or None.

    The Kim & Dickinson files start with a metadata block and an 'Arena data'
    table; the real per-frame table comes later. Plain CSVs with the header on
    line 0 work too, since we just search for the header line.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("fly,time(s)"):
                return i
    return None


def read_fly_data(file_path, nrows=None):
    """Read one raw file's 'Fly data' table, or None if it has none."""
    header_row = find_fly_data_header_row(file_path)
    if header_row is None:
        print(f"  '{os.path.basename(file_path)}': no 'fly,time(s),...' table, skipping.")
        return None
    try:
        df = pd.read_csv(
            file_path,
            skiprows=header_row,
            usecols=RELEVANT_COLUMNS,
            dtype=np.float32,
            skipinitialspace=True,
            nrows=nrows,
        )
    except Exception as e:
        print(f"  '{os.path.basename(file_path)}': could not read ({e}), skipping.")
        return None
    df.dropna(subset=RELEVANT_COLUMNS, inplace=True)
    return df


def iter_flies(raw_data_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR, nrows_per_file=None):
    """Yield (global_fly_id, condition_name, fly_dataframe) across all raw files.

    Fly IDs are made globally unique and continuous, so flies coming from
    different recordings never collide. `condition_name` is the source file's
    stem (e.g. 'dataset_03_largeArena_dark'), kept so results can be broken
    down by experimental condition later if wanted.
    """
    file_paths = find_raw_csv_files(raw_data_dir, output_dir)
    next_global_fly_id = 1

    for file_path in file_paths:
        df = read_fly_data(file_path, nrows=nrows_per_file)
        if df is None or df.empty:
            continue

        condition = os.path.splitext(os.path.basename(file_path))[0]
        offset = next_global_fly_id - df["fly"].min()
        n_flies = df["fly"].nunique()

        for local_fly_id, fly_df in df.groupby("fly"):
            yield int(local_fly_id + offset), condition, fly_df

        next_global_fly_id = int(df["fly"].max() + offset) + 1
        print(f"  Loaded '{condition}': {len(df):,} rows, {n_flies} flies.")
        del df


# ---------------------------------------------------------------------------
# Derived motion features
# ---------------------------------------------------------------------------
def add_motion_features(fly_df):
    """Sort by time and attach speed / unwrapped heading / angular velocity.

    Heading is unwrapped so it is continuous (no +-pi jumps), which matters
    both for computing relative headings and for rotating coordinates.
    """
    g = fly_df.sort_values("time(s)").reset_index(drop=True)
    g["speed(mm/s)"] = np.hypot(g["velocity_x(mm/s)"], g["velocity_y(mm/s)"])
    g["heading_unwrapped(rad)"] = np.unwrap(g["heading(radians)"].to_numpy())
    g["angular_velocity(rad/s)"] = g["velocity_heading(radians/s)"]
    return g


def align_window(window):
    """Bring one window to the canonical initial state and featurise it.

    Position is translated so the window starts at (0, 0), then rotated by
    -heading(0) so the fly starts facing angle 0. This is the "multiply by the
    heading angle" step: it uses the measured heading at the window's first
    frame rather than an angle estimated from two noisy position samples.

    Returns a dict with rotation-invariant motion features (used for
    clustering / prediction) plus the aligned positions (used for plotting).
    """
    heading_unwrapped = window["heading_unwrapped(rad)"].to_numpy()
    heading0 = heading_unwrapped[0]
    heading_rel = heading_unwrapped - heading0

    x0 = window["position_x(mm)"].to_numpy()[0]
    y0 = window["position_y(mm)"].to_numpy()[0]
    x_c = window["position_x(mm)"].to_numpy() - x0
    y_c = window["position_y(mm)"].to_numpy() - y0
    cos0, sin0 = np.cos(-heading0), np.sin(-heading0)

    return {
        "heading_sin": np.sin(heading_rel),
        "heading_cos": np.cos(heading_rel),
        "speed(mm/s)": window["speed(mm/s)"].to_numpy(),
        "angular_velocity(rad/s)": window["angular_velocity(rad/s)"].to_numpy(),
        "x_aligned": x_c * cos0 - y_c * sin0,
        "y_aligned": x_c * sin0 + y_c * cos0,
        "t_in_window(s)": window["time(s)"].to_numpy() - window["time(s)"].to_numpy()[0],
        "start_time(s)": float(window["time(s)"].to_numpy()[0]),
    }


def features_matrix(aligned):
    """Stack an aligned window's motion features into a (T, 4) array."""
    return np.column_stack([aligned[c] for c in MOTION_FEATURE_COLUMNS])


# ---------------------------------------------------------------------------
# Segments (for clustering) and bouts (for prediction)
# ---------------------------------------------------------------------------
def extract_segments(fly_df, segment_length, segments_per_fly):
    """Cut one fly's recording into evenly-spaced aligned windows.

    A fly is tracked for up to ~100,000 frames and obviously does many
    different things over that hour, so we sample several short windows spread
    across the whole recording rather than treating the fly as one unit.
    """
    g = add_motion_features(fly_df)
    n = len(g)
    if n < segment_length:
        return []

    max_start = n - segment_length
    n_possible = max_start // segment_length + 1
    if segments_per_fly is None or n_possible <= segments_per_fly:
        starts = range(0, max_start + 1, segment_length)
    else:
        starts = sorted(set(np.linspace(0, max_start, segments_per_fly, dtype=int).tolist()))

    out = []
    for start in starts:
        aligned = align_window(g.iloc[start:start + segment_length])
        aligned["features"] = features_matrix(aligned)
        out.append(aligned)
    return out


def extract_movement_bouts(fly_df,
                           min_speed=MIN_SPEED_MM_S,
                           min_frames=MIN_BOUT_FRAMES,
                           smooth_frames=SMOOTH_SPEED_FRAMES):
    """Contiguous stretches where the fly is genuinely moving.

    Returns a list of DataFrames (already sorted and feature-augmented). The
    speed signal is smoothed first so brief dips below the threshold don't
    chop one real bout into many fragments.
    """
    g = add_motion_features(fly_df)
    if len(g) < min_frames:
        return []

    smoothed = (
        g["speed(mm/s)"]
        .rolling(smooth_frames, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    moving = smoothed >= min_speed
    if not moving.any():
        return []

    # find contiguous True runs
    padded = np.concatenate([[False], moving, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    return [g.iloc[s:e].reset_index(drop=True)
            for s, e in zip(starts, ends) if (e - s) >= min_frames]
