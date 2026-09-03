# -*- coding: utf-8 -*-
"""
Step 1 - find the best number of movement clusters, then cluster.

Sweeps k = K_MIN .. K_MAX. For each k it runs soft-DTW time-series K-Means on
the movement segments and scores the result with the silhouette coefficient,
then picks the k with the highest score and saves that clustering.

Reading the silhouette score
----------------------------
Silhouette compares, for each segment, how close it is to its own cluster
versus the nearest other cluster. It ranges from -1 to 1:
    ~1   segment sits firmly inside its own cluster
    ~0   segment sits on the border between two clusters
    <0   segment is probably in the wrong cluster

Be aware that values close to 1 are unrealistic for behavioural data like
this. Fly movement is a continuum (turns of every sharpness, speeds of every
magnitude) rather than a set of cleanly separated categories, so scores in the
0.2-0.4 range already indicate useful structure. A low-but-positive peak is a
finding worth reporting, not a failure - it says the clusters are convenient
labels carved out of a continuum.

Outputs (in outputs/):
    ksweep_silhouette.csv    score for every k tried
    ksweep_silhouette.png    score vs k, with the chosen k marked
    fly_segments_clustered.csv   per-frame rows with the final cluster labels
    clusters_plot.png        aligned trajectories per cluster (visual check)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import fly_common as fc

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_soft_dtw_normalized
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.utils import to_time_series_dataset
    from sklearn.metrics import silhouette_samples
except ImportError:
    sys.exit("Missing packages. Run:  pip install tslearn scikit-learn")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEGMENT_LENGTH = 40      # frames per segment (~0.67 s at 60 Hz)
SEGMENTS_PER_FLY = 15    # evenly spaced across each fly's whole recording
K_MIN = fc.env_int("FLY_K_MIN", 2)
K_MAX = fc.env_int("FLY_K_MAX", 15)
RANDOM_STATE = 42
MAX_ITER = 15

# Silhouette needs a full pairwise soft-DTW distance matrix (O(n^2)), and the
# sweep repeats the whole clustering for every k, so both are capped.
# Lower SWEEP_MAX_SEGMENTS is the main lever if the sweep is taking too long.
SILHOUETTE_MAX_SAMPLES = fc.env_int("FLY_SILHOUETTE_MAX", 1200)
SWEEP_MAX_SEGMENTS = fc.env_int("FLY_SWEEP_MAX", 1500)

# Set to a number to read only the first N rows of each raw file (quick test).
NROWS_PER_FILE = fc.env_int("FLY_NROWS", None)


def build_segments():
    print("Loading raw data and extracting movement segments...")
    segments, fly_ids, conditions = [], [], []

    for fly_id, condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        for seg in fc.extract_segments(fly_df, SEGMENT_LENGTH, SEGMENTS_PER_FLY):
            segments.append(seg)
            fly_ids.append(fly_id)
            conditions.append(condition)

    if not segments:
        sys.exit("No segments extracted - check SEGMENT_LENGTH and the input data.")

    print(f"\n{len(segments):,} segments from {len(set(fly_ids))} flies.")
    return segments, np.array(fly_ids), np.array(conditions)


def scaled_dataset(segments):
    """Stack segments into a tslearn dataset, z-scoring each feature.

    The four features live on very different scales (sin/cos in [-1,1], speed
    in mm/s, angular velocity in rad/s). Without standardising, whichever has
    the largest raw numbers would dominate the soft-DTW distance.
    """
    dataset = to_time_series_dataset([s["features"] for s in segments])
    return TimeSeriesScalerMeanVariance().fit_transform(dataset)


def silhouette_for(scaled, labels, rng, max_samples=SILHOUETTE_MAX_SAMPLES):
    """Overall + per-cluster silhouette, on a stratified subsample if large."""
    n = len(labels)
    if n > max_samples:
        idx = []
        for c in np.unique(labels):
            c_idx = np.where(labels == c)[0]
            take = max(1, round(len(c_idx) / n * max_samples))
            idx.extend(rng.choice(c_idx, size=min(take, len(c_idx)), replace=False))
        idx = np.array(sorted(idx))
    else:
        idx = np.arange(n)

    if len(np.unique(labels[idx])) < 2:
        return np.nan, {}

    distances = cdist_soft_dtw_normalized(scaled[idx])
    scores = silhouette_samples(distances, labels[idx], metric="precomputed")
    per_cluster = (
        pd.DataFrame({"cluster": labels[idx], "s": scores})
        .groupby("cluster")["s"].mean().to_dict()
    )
    return float(scores.mean()), per_cluster


def sweep_k(scaled, rng, output_dir):
    """Cluster at every k in the range and score it.

    The sweep is the longest part of the whole project - each k is a full
    clustering run, so the total can be hours. Two things make an
    interruption harmless:

      * results are written to disk after EVERY k, not at the end
      * on startup any k already present in that file is skipped

    So if Colab disconnects, the machine sleeps, or you stop the cell, you
    just re-run the script and it picks up from where it stopped. To force a
    clean re-run, delete outputs/ksweep_silhouette.csv first.
    """
    n = len(scaled)
    if n > SWEEP_MAX_SEGMENTS:
        sweep_idx = rng.choice(n, SWEEP_MAX_SEGMENTS, replace=False)
        sweep_data = scaled[np.sort(sweep_idx)]
        print(f"\nSweeping on a {SWEEP_MAX_SEGMENTS:,}-segment subsample "
              f"(of {n:,}) to keep runtime reasonable.")
    else:
        sweep_data = scaled

    path = os.path.join(output_dir, "ksweep_silhouette.csv")
    rows, done = [], set()
    if os.path.exists(path):
        try:
            previous = pd.read_csv(path)
            rows = previous.to_dict("records")
            done = set(previous["k"].astype(int))
            if done:
                print(f"\nResuming: k = {sorted(done)} already computed "
                      f"(from a previous run). Delete\n'{path}' to start over.")
        except Exception as e:
            print(f"\nCould not read previous results ({e}); starting fresh.")

    print(f"\nSweeping k = {K_MIN}..{K_MAX} (soft-DTW K-Means)")
    print(f"{'k':>4} {'silhouette':>12} {'time':>8}")
    print("-" * 26)

    for k in range(K_MIN, K_MAX + 1):
        if k in done:
            existing = next(r for r in rows if int(r["k"]) == k)
            print(f"{k:>4} {existing['silhouette']:>12.4f} {'skipped':>8}")
            continue

        t0 = time.time()
        model = TimeSeriesKMeans(n_clusters=k, metric="softdtw",
                                 max_iter=MAX_ITER, random_state=RANDOM_STATE,
                                 n_jobs=-1)
        labels = model.fit_predict(sweep_data)
        # A per-k generator, so each k's silhouette subsample is identical
        # whether it was computed in a fresh run or after resuming.
        score, _ = silhouette_for(sweep_data, labels,
                                  np.random.default_rng(RANDOM_STATE + k))
        elapsed = time.time() - t0
        rows.append({"k": k, "silhouette": score, "seconds": elapsed})
        print(f"{k:>4} {score:>12.4f} {elapsed:>7.0f}s", flush=True)

        # save immediately so this k is never recomputed
        pd.DataFrame(rows).sort_values("k").to_csv(path, index=False)

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def plot_sweep(sweep_df, best_k, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["k"], sweep_df["silhouette"], marker="o")
    ax.axvline(best_k, color="crimson", linestyle="--",
               label=f"best k = {best_k}")
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("mean silhouette (soft-DTW)")
    ax.set_title("Cluster separation vs. number of clusters")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "ksweep_silhouette.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def save_results(segments, fly_ids, conditions, labels, output_dir):
    rows = []
    for seg_id, (seg, fly, cond, cluster) in enumerate(
            zip(segments, fly_ids, conditions, labels)):
        for k in range(len(seg["t_in_window(s)"])):
            rows.append({
                "fly": fly,
                "condition": cond,
                "segment_id": seg_id,
                "segment_start_time(s)": seg["start_time(s)"],
                "t_in_segment(s)": seg["t_in_window(s)"][k],
                "x_aligned": seg["x_aligned"][k],
                "y_aligned": seg["y_aligned"][k],
                "heading_sin": seg["heading_sin"][k],
                "heading_cos": seg["heading_cos"][k],
                "speed(mm/s)": seg["speed(mm/s)"][k],
                "angular_velocity(rad/s)": seg["angular_velocity(rad/s)"][k],
                "cluster": int(cluster),
            })
    result = pd.DataFrame(rows)
    path = os.path.join(output_dir, "fly_segments_clustered.csv")
    result.to_csv(path, index=False)
    print(f"Saved '{path}'")
    return result


def plot_clusters(result, output_dir, max_per_cluster=60):
    clusters = sorted(result["cluster"].unique())
    n = len(clusters)
    rows_grid = min(4, n)
    cols_grid = (n + rows_grid - 1) // rows_grid
    fig, axes = plt.subplots(rows_grid, cols_grid,
                             figsize=(cols_grid * 4.5, rows_grid * 4),
                             squeeze=False)
    axes = axes.flatten()
    rng = np.random.default_rng(0)

    for i, cluster_id in enumerate(clusters):
        ax = axes[i]
        data = result[result["cluster"] == cluster_id]
        seg_ids = data["segment_id"].unique()
        total = len(seg_ids)
        if total > max_per_cluster:
            seg_ids = rng.choice(seg_ids, max_per_cluster, replace=False)
        shown = data[data["segment_id"].isin(seg_ids)]

        sns.lineplot(data=shown, x="x_aligned", y="y_aligned",
                     units="segment_id", estimator=None, color="steelblue",
                     linewidth=0.6, alpha=0.5, ax=ax, legend=False)
        ax.scatter(0, 0, color="black", s=20, zorder=5)
        ax.set_title(f"Cluster {cluster_id} ({total} segments)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    path = os.path.join(output_dir, "clusters_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def main():
    os.makedirs(fc.OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    segments, fly_ids, conditions = build_segments()
    scaled = scaled_dataset(segments)

    sweep_df = sweep_k(scaled, rng, fc.OUTPUT_DIR)   # saves after every k

    best_row = sweep_df.loc[sweep_df["silhouette"].idxmax()]
    best_k = int(best_row["k"])
    best_score = float(best_row["silhouette"])

    print(f"\nBest k = {best_k}  (silhouette {best_score:.4f})")
    if best_score < 0.25:
        print("Note: this is a low score. Fly movement is a continuum rather than\n"
              "      a set of cleanly separated behaviours, so the clusters are\n"
              "      useful labels but not sharply distinct groups. Worth stating\n"
              "      explicitly rather than presenting the clusters as clean-cut.")

    plot_sweep(sweep_df, best_k, fc.OUTPUT_DIR)

    print(f"\nClustering all {len(scaled):,} segments with k = {best_k}...")
    model = TimeSeriesKMeans(n_clusters=best_k, metric="softdtw",
                             max_iter=MAX_ITER, random_state=RANDOM_STATE,
                             n_jobs=-1)
    labels = model.fit_predict(scaled)

    overall, per_cluster = silhouette_for(scaled, labels, rng)
    print(f"\nFinal overall silhouette: {overall:.4f}")
    print("\nPer-cluster silhouette (higher = cleaner cluster):")
    for cluster_id in sorted(per_cluster):
        count = int((labels == cluster_id).sum())
        print(f"  Cluster {cluster_id}: {per_cluster[cluster_id]:>7.4f}  ({count} segments)")

    result = save_results(segments, fly_ids, conditions, labels, fc.OUTPUT_DIR)
    plot_clusters(result, fc.OUTPUT_DIR)

    # Save the cluster centres themselves, not just the labels. With these,
    # 04_report_figures.py can take any moment from the prediction model's test
    # set and say which movement cluster it belongs to - which is what lets the
    # two halves of the project be compared instead of just sitting side by side.
    centroid_path = os.path.join(fc.OUTPUT_DIR, "cluster_centroids.npy")
    np.save(centroid_path, model.cluster_centers_)
    print(f"Saved '{centroid_path}'")

    meta_path = os.path.join(fc.OUTPUT_DIR, "cluster_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"best_k": best_k,
                   "best_silhouette": best_score,
                   "final_silhouette": None if np.isnan(overall) else float(overall),
                   "segment_length": SEGMENT_LENGTH,
                   "segments_per_fly": SEGMENTS_PER_FLY,
                   "n_segments": int(len(scaled)),
                   "n_flies": int(len(set(fly_ids.tolist()))),
                   "feature_columns": fc.MOTION_FEATURE_COLUMNS,
                   "per_cluster_silhouette": {str(c): float(v)
                                              for c, v in per_cluster.items()},
                   "cluster_sizes": {str(c): int((labels == c).sum())
                                     for c in sorted(set(labels.tolist()))}},
                  f, indent=2)
    print(f"Saved '{meta_path}'")

    print("\nDone. The cluster labels in fly_segments_clustered.csv can now be\n"
          "used as an extra input feature for the prediction models.")


if __name__ == "__main__":
    main()
