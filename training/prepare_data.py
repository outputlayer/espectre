#!/usr/bin/env python3
"""Prepare WiFi CSI data — v5: multi-session training for drift invariance.

Uses recordings from TWO sessions at different times.
Session 1: train_{empty_v4, lying_v6, sitting_v5, walking_v5}
Session 2: train_{empty_v5, lying_v7, sitting_v6, walking_v6}

Multi-session data teaches the CNN to extract drift-invariant features.
"""
import json
import numpy as np
import sys
from pathlib import Path

NUM_NODES = 3
NUM_SUBCARRIERS = 56
NUM_FEATURES = NUM_NODES * NUM_SUBCARRIERS
SUBSAMPLE_RATE = 5
WINDOW_SIZE = 40
WINDOW_STRIDE = 10
VAL_RATIO = 0.3

CLASSES = {"empty": 0, "lying": 1, "walking": 2, "sitting": 3}
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "prepared"


def parse_recording(filepath):
    node_latest = {i: np.zeros(NUM_SUBCARRIERS) for i in range(1, NUM_NODES + 1)}
    frames = []
    with open(filepath) as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            for node in data.get("nodes", []):
                nid = node.get("node_id")
                if nid and 1 <= nid <= NUM_NODES:
                    amp = node.get("amplitude", [])
                    arr = np.zeros(NUM_SUBCARRIERS)
                    arr[:min(NUM_SUBCARRIERS, len(amp))] = amp[:NUM_SUBCARRIERS]
                    node_latest[nid] = arr
            frame = np.stack([node_latest[i] for i in range(1, NUM_NODES + 1)])
            frames.append(frame)
    arr = np.array(frames, dtype=np.float32)
    print(f"  Parsed {filepath.name}: {arr.shape[0]} frames ({arr.shape[0]/109:.0f}s)")
    return arr


def l2_normalize(data):
    n = data.shape[0]
    flat = data.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    flat = flat / norms
    return flat.reshape(n, NUM_NODES, NUM_SUBCARRIERS)


def create_windows(data, window_size, stride):
    n = data.shape[0]
    flat = data.reshape(n, -1)
    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(flat[start:start + window_size])
    if windows:
        return np.array(windows, dtype=np.float32)
    return np.zeros((0, window_size, flat.shape[1]), dtype=np.float32)


def augment_window(window, n_augments=3):
    augmented = []
    for _ in range(n_augments):
        w = window.copy()
        w += np.random.normal(0, 0.02, w.shape).astype(np.float32)
        drift = np.random.normal(0, 0.15, (1, w.shape[1])).astype(np.float32)
        w += drift
        scale = np.random.uniform(0.85, 1.15, (1, w.shape[1])).astype(np.float32)
        w *= scale
        augmented.append(w)
    return augmented


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    S1_DIR = DATA_DIR  # Session 1 files in data/recordings/ (symlinked or same dir)
    S2_DIR = DATA_DIR / "data_s2"

    # Multi-session recordings
    recordings = {
        "empty":   [S1_DIR / "train_empty_v4.jsonl",   S2_DIR / "train_empty_v5.jsonl"],
        "lying":   [S1_DIR / "train_lying_v6.jsonl",    S2_DIR / "train_lying_v7.jsonl"],
        "walking": [S1_DIR / "train_walking_v5.jsonl",  S2_DIR / "train_walking_v6.jsonl"],
        "sitting": [S1_DIR / "train_sitting_v5.jsonl",  S2_DIR / "train_sitting_v6.jsonl"],
    }

    for name, paths in recordings.items():
        for path in paths:
            if not path.exists():
                print(f"ERROR: {path} not found"); sys.exit(1)

    # 1. Parse all sessions
    print("=== Parsing multi-session recordings ===")
    session_data = {}  # {class: [(session_idx, data), ...]}
    for name, paths in recordings.items():
        session_data[name] = []
        for i, p in enumerate(paths):
            raw = parse_recording(p)
            session_data[name].append((i, raw))

    # 2. L2 normalize each session independently
    print("\n=== L2 normalization (per-session) ===")
    for name in session_data:
        for j, (sid, data) in enumerate(session_data[name]):
            session_data[name][j] = (sid, l2_normalize(data))

    # 3. Compute baseline from EACH session's empty data, then subtract per-session
    print("\n=== Per-session baseline subtraction ===")
    session_baselines = {}
    for sid, data in session_data["empty"]:
        bl = data.mean(axis=0)
        session_baselines[sid] = bl
        print(f"  Session {sid+1} baseline: mean={bl.mean():.6f}, std={bl.std():.6f}")

    # Save combined baseline (average of all sessions) for inference
    combined_baseline = np.mean([session_baselines[s] for s in session_baselines], axis=0)
    np.save(OUTPUT_DIR / "baseline.npy", combined_baseline)
    print(f"  Combined baseline: mean={combined_baseline.mean():.6f}")

    # Subtract each session's own baseline
    for name in session_data:
        for j, (sid, data) in enumerate(session_data[name]):
            bl = session_baselines[sid]
            session_data[name][j] = (sid, data - bl[np.newaxis, :, :])

    # 4. Concatenate all sessions per class
    print(f"\n=== Merging sessions ===")
    raw_data = {}
    for name in session_data:
        parts = [data for _, data in session_data[name]]
        raw_data[name] = np.concatenate(parts, axis=0)
        sizes = [data.shape[0] for _, data in session_data[name]]
        print(f"  {name}: {' + '.join(str(s) for s in sizes)} = {raw_data[name].shape[0]} frames")

    # 5. Subsample
    print(f"\n=== Subsample (1/{SUBSAMPLE_RATE}) ===")
    for name in raw_data:
        orig = raw_data[name].shape[0]
        raw_data[name] = raw_data[name][::SUBSAMPLE_RATE]
        print(f"  {name}: {orig} -> {raw_data[name].shape[0]} frames")

    # 6. Temporal split + windows + augmentation
    print(f"\n=== Temporal split (70/30) ===")
    train_X, train_y = [], []
    val_X, val_y = [], []

    for name, label in CLASSES.items():
        data = raw_data[name]
        n = data.shape[0]
        split = int(n * (1 - VAL_RATIO))

        tw = create_windows(data[:split], WINDOW_SIZE, WINDOW_STRIDE)
        vw = create_windows(data[split:], WINDOW_SIZE, WINDOW_STRIDE)

        aug_windows = []
        for i in range(tw.shape[0]):
            aug_windows.extend(augment_window(tw[i], n_augments=3))
        if aug_windows:
            tw = np.concatenate([tw, np.array(aug_windows, dtype=np.float32)], axis=0)

        print(f"  {name}: train={tw.shape[0]} (incl. 3x aug), val={vw.shape[0]}")
        train_X.append(tw)
        train_y.append(np.full(tw.shape[0], label, dtype=np.int64))
        val_X.append(vw)
        val_y.append(np.full(vw.shape[0], label, dtype=np.int64))

    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_val = np.concatenate(val_X, axis=0)
    y_val = np.concatenate(val_y, axis=0)

    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"\n  Total: train={X_train.shape[0]}, val={X_val.shape[0]}")
    print(f"  Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # 7. Global normalization
    print("\n=== Global normalization ===")
    flat = X_train.reshape(-1, X_train.shape[-1])
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    np.save(OUTPUT_DIR / "feat_mean.npy", feat_mean)
    np.save(OUTPUT_DIR / "feat_std.npy", feat_std)

    # 8. Save
    print("\n=== Saving ===")
    for arr_name, arr in [("X_train", X_train), ("y_train", y_train),
                           ("X_val", X_val), ("y_val", y_val)]:
        np.save(OUTPUT_DIR / f"{arr_name}.npy", arr)

    print(f"  X_train: {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")
    print(f"  X_val:   {X_val.shape} ({X_val.nbytes/1e6:.1f} MB)")

    print(f"\n=== Pipeline: MULTI-SESSION ===")
    print(f"  L2 norm → per-session baseline → global norm → drift augmentation")
    print(f"  Sessions: 2 (different times of day)")
    print(f"  Goal: drift-invariant features")


if __name__ == "__main__":
    main()
