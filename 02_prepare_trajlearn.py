# -*- coding: utf-8 -*-
"""
Step 2 - turn fly trajectories into TrajLearn input.

TrajLearn is a transformer over sequences of discrete spatial cells. In the
original paper those cells are H3 hexagons covering a city, and each token is
one hex the vehicle passed through. Two things have to be adapted for flies:

1. SCALE. H3's finest resolutions are still metres across, while our arena is
   65-170 mm wide. This script replaces H3 with a hexagonal grid defined
   directly in millimetres, using axial (q, r) coordinates. Everything
   downstream (vocabulary, neighbour lists, axial-coordinate embeddings) is
   produced in exactly the format TrajLearn's own preprocess.py emits, so the
   unmodified TrajLearn training code can consume it.

2. SAMPLING RATE. At 60 Hz the fly barely moves between frames: the median
   frame-to-frame step is ~0.02 mm and ~68% of frames move under 0.1 mm. With
   any sane cell size the fly stays in the same cell for most frames, so a
   token-level "next cell" model would learn to predict "same cell again" and
   score >95% while being useless. Two filters fix this, and together they
   restore the assumption TrajLearn was built on - that consecutive tokens are
   genuinely different places:

     * keep only MOVEMENT BOUTS, where the fly is actually travelling
     * collapse runs of the same cell, so each token is a NEW cell

   The resulting model answers "which cell does the fly move to next", i.e.
   the direction it commits to - not where it is 17 ms from now. That precise
   short-horizon question is handled by 03_continuous_model.py instead.

TRAIN/VAL/TEST SPLIT. TrajLearn's dataloader splits data.txt purely by line
order (train = first lines, val = middle, test = final lines). To split by fly
rather than by line - so no fly appears in both training and test - this script
writes the lines grouped by fly, train flies first, and reports the exact
ratios to put in the config. The generated configs.yaml already has them.

Outputs (in outputs/trajlearn/<DATASET_NAME>/):
    data.txt, vocab.txt, mapping.json, neighbors.json, embeddings.npy
    split_info.json     which flies went to train/val/test, and the ratios
  plus outputs/trajlearn/configs.yaml
"""

import json
import os
import sys

import numpy as np

import fly_common as fc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "flies"

# Hexagon circumradius in mm. Smaller = finer detail but a bigger vocabulary
# and more quantisation noise; larger = coarser but easier to learn.
# 1.5 mm is roughly a fly body length and gives a few thousand cells.
HEX_SIZE_MM = fc.env_float("FLY_HEX_SIZE_MM", 1.5)

MIN_TOKENS_PER_SEQUENCE = 8   # drop bouts that yield very short cell sequences
EMBEDDING_DIM = 128           # axial-coordinate embeddings, as TrajLearn does

VALIDATION_FRACTION = 0.1     # fraction of FLIES (not lines) held out
TEST_FRACTION = 0.2
RANDOM_STATE = 42

NROWS_PER_FILE = fc.env_int("FLY_NROWS", None)   # e.g. 200000 for a quick test


# ---------------------------------------------------------------------------
# Hexagonal grid in millimetres (replaces H3)
# ---------------------------------------------------------------------------
def xy_to_axial(x, y, size=HEX_SIZE_MM):
    """Pointy-top hex grid: (x, y) in mm -> integer axial coordinates (q, r).

    Uses the standard cube-rounding method so each point maps to the hexagon
    whose centre it is genuinely closest to.
    """
    q_f = (np.sqrt(3) / 3.0 * x - 1.0 / 3.0 * y) / size
    r_f = (2.0 / 3.0 * y) / size

    # Convert to cube coordinates and round, then repair the single coordinate
    # with the largest rounding error so the constraint x + y + z == 0 still
    # holds. Repairing the y coordinate leaves (q, r) untouched, so only the
    # x and z branches change the result.
    x_c, z_c = q_f, r_f
    y_c = -x_c - z_c
    rx, ry, rz = np.round(x_c), np.round(y_c), np.round(z_c)
    dx, dy, dz = np.abs(rx - x_c), np.abs(ry - y_c), np.abs(rz - z_c)

    fix_x = (dx > dy) & (dx > dz)
    fix_y = (~fix_x) & (dy > dz)
    fix_z = (~fix_x) & (~fix_y)
    rx = np.where(fix_x, -ry - rz, rx)          # fix_x and fix_z are mutually
    rz = np.where(fix_z, -rx - ry, rz)          # exclusive, so this is safe

    return rx.astype(np.int64), rz.astype(np.int64)


def axial_neighbours(q, r):
    """The six hexes adjacent to (q, r)."""
    return [(q + 1, r), (q + 1, r - 1), (q, r - 1),
            (q - 1, r), (q - 1, r + 1), (q, r + 1)]


def cell_name(q, r):
    return f"{q}_{r}"


# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------
def bout_to_tokens(bout):
    """One movement bout -> list of hex cell names, consecutive repeats removed."""
    q, r = xy_to_axial(bout["position_x(mm)"].to_numpy(),
                       bout["position_y(mm)"].to_numpy())

    # keep only frames where the cell actually changed
    changed = np.ones(len(q), dtype=bool)
    changed[1:] = (q[1:] != q[:-1]) | (r[1:] != r[:-1])
    q, r = q[changed], r[changed]

    return [cell_name(a, b) for a, b in zip(q, r)]


def collect_sequences():
    """Walk every fly, returning {fly_id: [token sequences]} plus stats."""
    print("Loading raw data and building movement bouts...")
    per_fly = {}
    n_bouts = n_frames_kept = n_frames_total = 0

    for fly_id, _condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        n_frames_total += len(fly_df)
        sequences = []
        for bout in fc.extract_movement_bouts(fly_df):
            n_bouts += 1
            n_frames_kept += len(bout)
            tokens = bout_to_tokens(bout)
            if len(tokens) >= MIN_TOKENS_PER_SEQUENCE:
                sequences.append(tokens)
        if sequences:
            per_fly[fly_id] = sequences

    if not per_fly:
        sys.exit(
            "No usable sequences. Try lowering MIN_SPEED_MM_S / MIN_BOUT_FRAMES\n"
            "in fly_common.py, or reducing HEX_SIZE_MM / MIN_TOKENS_PER_SEQUENCE."
        )

    total_seqs = sum(len(v) for v in per_fly.values())
    total_tokens = sum(len(s) for v in per_fly.values() for s in v)
    print(f"\n{n_bouts:,} movement bouts found "
          f"({n_frames_kept:,} of {n_frames_total:,} frames = "
          f"{100 * n_frames_kept / max(n_frames_total, 1):.1f}% - the rest is the fly sitting still).")
    print(f"{total_seqs:,} token sequences from {len(per_fly)} flies, "
          f"{total_tokens:,} tokens total "
          f"(mean length {total_tokens / total_seqs:.1f}).")
    return per_fly


# ---------------------------------------------------------------------------
# Split by fly, then write in TrajLearn's format
# ---------------------------------------------------------------------------
def split_flies(per_fly, rng):
    fly_ids = np.array(sorted(per_fly))
    rng.shuffle(fly_ids)

    n = len(fly_ids)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    n_val = max(1, int(round(n * VALIDATION_FRACTION)))
    if n_test + n_val >= n:
        sys.exit(f"Too few flies ({n}) to split - lower TEST/VALIDATION_FRACTION.")

    train = fly_ids[: n - n_val - n_test]
    val = fly_ids[n - n_val - n_test: n - n_test]
    test = fly_ids[n - n_test:]
    # plain ints so the split can be written to JSON
    return ([int(v) for v in train], [int(v) for v in val], [int(v) for v in test])


def generate_axial_embeddings(vocab, embedding_dim, rng):
    """Axial-coordinate embeddings, mirroring TrajLearn's preprocess.py.

    Cells that are near each other on the grid get similar initial embeddings,
    which gives the transformer a useful spatial prior at initialisation.
    Row 0 is the EOT (end-of-trajectory) token.
    """
    axial = np.array([[int(v) for v in name.split("_")] for name in vocab], dtype=float)
    axial -= axial[0]

    projection = rng.standard_normal((2, embedding_dim))
    projected = axial @ projection

    # rank-map onto a normal distribution, as the reference implementation does
    normal_samples = rng.normal(0, 0.02, projected.size)
    flat = projected.flatten()
    order = np.argsort(flat)
    mapped = np.empty_like(flat)
    mapped[order] = np.sort(normal_samples)

    eot = rng.normal(0, 0.02, (1, embedding_dim))
    return np.concatenate([eot, mapped.reshape(projected.shape)], axis=0)


def write_trajlearn_dataset(per_fly, train, val, test, out_dir, rng):
    os.makedirs(out_dir, exist_ok=True)

    # Line order defines the split, because TrajLearn slices data.txt by index.
    ordered_lines, ordered_flies = [], []
    for group in (train, val, test):
        for fly_id in group:
            for seq in per_fly[fly_id]:
                ordered_lines.append(seq)
                ordered_flies.append(fly_id)

    n_lines = len(ordered_lines)
    n_train = sum(len(per_fly[f]) for f in train)
    n_val = sum(len(per_fly[f]) for f in val)
    n_test = sum(len(per_fly[f]) for f in test)

    vocab = sorted({t for seq in ordered_lines for t in seq},
                   key=lambda s: tuple(int(v) for v in s.split("_")))

    embeddings = generate_axial_embeddings(vocab, EMBEDDING_DIM, rng)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    vocab_with_eot = ["EOT"] + vocab
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_with_eot) + "\n")

    mapping = {name: i for i, name in enumerate(vocab_with_eot)}
    with open(os.path.join(out_dir, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    # neighbour lists power TrajLearn's spatial-continuity constraint
    in_vocab = set(vocab)
    neighbours = {}
    for name in vocab:
        q, r = (int(v) for v in name.split("_"))
        neighbours[mapping[name]] = [
            mapping[cell_name(nq, nr)]
            for nq, nr in axial_neighbours(q, r)
            if cell_name(nq, nr) in in_vocab
        ]
    with open(os.path.join(out_dir, "neighbors.json"), "w", encoding="utf-8") as f:
        json.dump(neighbours, f)

    eot = mapping["EOT"]
    with open(os.path.join(out_dir, "data.txt"), "w", encoding="utf-8") as f:
        for seq in ordered_lines:
            f.write(" ".join(str(mapping[t]) for t in seq) + f" {eot}\n")

    # Ratios that reproduce the by-fly split under TrajLearn's index slicing.
    # TrajLearn slices with int(n * ratio), which truncates: the exact ratio
    # n_test/n can evaluate to 98.999... and silently lose a line, shifting the
    # boundary so a fly leaks across partitions. Aiming at the midpoint of the
    # valid interval makes the truncation land on exactly n_test.
    test_ratio = (n_test + 0.5) / n_lines
    val_ratio = n_val / n_lines

    split_info = {
        "hex_size_mm": HEX_SIZE_MM,
        "vocabulary_size_including_EOT": len(vocab_with_eot),
        "total_sequences": n_lines,
        "train": {"n_flies": len(train), "n_sequences": n_train, "fly_ids": train},
        "validation": {"n_flies": len(val), "n_sequences": n_val, "fly_ids": val},
        "test": {"n_flies": len(test), "n_sequences": n_test, "fly_ids": test},
        "validation_ratio_for_config": val_ratio,
        "test_ratio_for_config": test_ratio,
        "note": ("Lines in data.txt are ordered train, then validation, then test, "
                 "grouped by fly. TrajLearn slices data.txt by line index, so using "
                 "the ratios above reproduces a split where no fly appears in more "
                 "than one partition."),
    }
    with open(os.path.join(out_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nVocabulary: {len(vocab_with_eot):,} cells (including EOT), "
          f"hex size {HEX_SIZE_MM} mm")
    print(f"Sequences:  {n_train:,} train ({len(train)} flies) | "
          f"{n_val:,} val ({len(val)} flies) | {n_test:,} test ({len(test)} flies)")
    print(f"Config ratios: validation_ratio={val_ratio:.6f}  test_ratio={test_ratio:.6f}")

    verify_split(n_lines, val_ratio, test_ratio, n_train, n_val, n_test)
    return val_ratio, test_ratio, len(vocab_with_eot)


def verify_split(n_lines, val_ratio, test_ratio, n_train, n_val, n_test):
    """Replicate TrajLearn's own slicing and confirm it reproduces our split.

    TrajectoryBatchDataset slices data.txt with int(n * ratio), which
    truncates. If that lands even one line off, a fly ends up in two
    partitions and the evaluation is quietly invalid - so this is checked
    rather than assumed.
    """
    n_vt = int(n_lines * (val_ratio + test_ratio))
    n_t = int(n_lines * test_ratio)
    idx = list(range(n_lines))
    got = (len(idx[:-n_vt]), len(idx[-n_vt:-n_t]), len(idx[-n_t:]))
    want = (n_train, n_val, n_test)

    if got == want:
        print(f"Split verified: TrajLearn's slicing reproduces "
              f"{got[0]}/{got[1]}/{got[2]} train/val/test lines, by fly.")
    else:
        sys.exit(
            f"\nSplit verification FAILED.\n"
            f"  TrajLearn would slice: train={got[0]} val={got[1]} test={got[2]}\n"
            f"  We intended:           train={want[0]} val={want[1]} test={want[2]}\n"
            f"Do not train on this data - flies would leak between partitions."
        )


def write_config(out_root, val_ratio, test_ratio):
    config = f"""# Auto-generated by 02_prepare_trajlearn.py
# Run training with:  python3 main.py configs.yaml
# Run testing with:   python3 main.py configs.yaml --test
{DATASET_NAME}:
  data_dir: ./data
  dataset: {DATASET_NAME}
  model_checkpoint_directory: ./checkpoints

  # These ratios reproduce the by-fly split - do not change them by hand,
  # they must match the line ordering written into data.txt.
  validation_ratio: {val_ratio:.6f}
  test_ratio: {test_ratio:.6f}
  delimiter: " "

  min_input_length: 4
  max_input_length: 12
  test_input_length: 8
  test_prediction_length: 1

  batch_size: 128
  device: cuda          # change to cpu if no GPU is available
  max_epochs: 50
  block_size: 24
  learning_rate: 0.0006
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  decay_lr: true
  warmup_iters: 200
  lr_decay_iters: 5000
  min_lr: 0.00006
  seed: {RANDOM_STATE}

  n_layer: 4
  n_head: 4
  n_embd: {EMBEDDING_DIM}
  bias: false
  dropout: 0.1
  custom_initialization: true
  train_from_checkpoint_if_exist: false
  patience: 5

  continuity: true
  beam_width: 3
  store_predictions: true
"""
    path = os.path.join(out_root, "configs.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(config)
    print(f"Saved '{path}'")


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    out_root = os.path.join(fc.OUTPUT_DIR, "trajlearn")
    out_dir = os.path.join(out_root, DATASET_NAME)
    os.makedirs(out_root, exist_ok=True)

    per_fly = collect_sequences()
    train, val, test = split_flies(per_fly, rng)
    val_ratio, test_ratio, vocab_size = write_trajlearn_dataset(
        per_fly, train, val, test, out_dir, rng)
    write_config(out_root, val_ratio, test_ratio)

    print(f"""
Next steps
----------
1. git clone https://github.com/amir-ni/Trajectory-prediction
2. Copy '{out_dir}' into the cloned repo as  ./data/{DATASET_NAME}
3. Copy '{os.path.join(out_root, 'configs.yaml')}' into the repo root
4. python3 main.py configs.yaml            # train
   python3 main.py configs.yaml --test     # evaluate

The vocabulary is {vocab_size:,} cells. If training struggles, raise
HEX_SIZE_MM (fewer, larger cells) and re-run this script.""")


if __name__ == "__main__":
    main()
