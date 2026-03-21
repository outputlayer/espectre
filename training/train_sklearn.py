#!/usr/bin/env python3
"""Train classifier using scikit-learn on WiFi CSI data."""
import json, os, sys, math, random
from collections import Counter
import numpy as np

RECORDINGS_DIR = "data/recordings"
CLASSES = ["absent", "present_still", "present_moving", "active"]

def classify_name(fname):
    f = fname.lower()
    if "empty" in f or "absent" in f: return 0
    if "still" in f or "sitting" in f: return 1
    if "walking" in f or "moving" in f: return 2
    if "active" in f or "exercise" in f: return 3
    return None

def extract_features(frames):
    samples = []
    prev_amps = None
    prev_prev_amps = None
    motion_ema = 0.0
    turb_ema = 0.0
    amp_ema = np.zeros(56)
    WINDOW = 20
    motion_window = []
    turb_window = []

    for frame in frames:
        feat = frame.get("features", {})
        nodes = frame.get("nodes", [])
        vitals = frame.get("vital_signs", {})

        amps = []
        for node in nodes:
            amps = node.get("amplitude", [])
            break
        if not amps: continue

        a = np.zeros(56)
        a[:min(56, len(amps))] = amps[:56]

        amp_mean = a.mean()
        amp_std = a.std()
        amp_range = a.max() - a.min()
        low, mid, high = a[:18], a[18:37], a[37:]
        low_m, mid_m, high_m = low.mean(), mid.mean(), high.mean()
        low_s, mid_s, high_s = low.std(), mid.std(), high.std()

        sc_idx = [12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]
        selected = a[sc_idx]
        turbulence = selected.std()

        if prev_amps is not None:
            diff_sq = np.mean((a - prev_amps)**2)
            abs_diff = np.mean(np.abs(a - prev_amps))
        else:
            diff_sq = abs_diff = 0

        if prev_prev_amps is not None and prev_amps is not None:
            d1 = a - prev_amps
            d2 = prev_amps - prev_prev_amps
            accel = np.mean((d1 - d2)**2)
        else:
            accel = 0

        prev_prev_amps = prev_amps
        prev_amps = a.copy()

        motion_ema = motion_ema * 0.85 + diff_sq * 0.15
        turb_ema = turb_ema * 0.9 + turbulence * 0.1
        amp_ema = amp_ema * 0.95 + a * 0.05
        ema_dev = np.mean((a - amp_ema)**2)

        motion_window.append(diff_sq)
        turb_window.append(turbulence)
        if len(motion_window) > WINDOW: motion_window.pop(0)
        if len(turb_window) > WINDOW: turb_window.pop(0)

        if len(motion_window) >= 5:
            mw = np.array(motion_window)
            tw = np.array(turb_window)
            motion_mean_w, motion_std_w, motion_max_w = mw.mean(), mw.std(), mw.max()
            turb_mean_w, turb_std_w = tw.mean(), tw.std()
        else:
            motion_mean_w = motion_std_w = motion_max_w = 0
            turb_mean_w = turb_std_w = 0

        variance = feat.get("variance", 0)
        mbp = feat.get("motion_band_power", 0)
        bbp = feat.get("breathing_band_power", 0)
        sp = feat.get("spectral_power", 0)
        df = feat.get("dominant_freq_hz", 0)
        cp = feat.get("change_points", 0)
        rssi = feat.get("mean_rssi", 0)

        hr_conf = vitals.get("heartbeat_confidence", 0)
        br_conf = vitals.get("breathing_confidence", 0)
        sig_q = vitals.get("signal_quality", 0)

        fv = [
            amp_mean, amp_std, amp_range, low_m, mid_m, high_m,
            low_s, mid_s, high_s,
            turbulence, turb_ema,
            diff_sq, abs_diff, motion_ema, accel,
            ema_dev,
            motion_mean_w, motion_std_w, motion_max_w,
            turb_mean_w, turb_std_w,
            variance, mbp, bbp, sp, df, cp, rssi,
            hr_conf, br_conf, sig_q,
        ]
        samples.append(fv)
    return samples

def load_data():
    all_X, all_y = [], []
    for fname in sorted(os.listdir(RECORDINGS_DIR)):
        if not fname.startswith("train_") or not fname.endswith(".jsonl"): continue
        cls = classify_name(fname)
        if cls is None: continue
        path = os.path.join(RECORDINGS_DIR, fname)
        frames = []
        with open(path) as f:
            for line in f:
                try: frames.append(json.loads(line))
                except: continue
        features = extract_features(frames)
        all_X.extend(features)
        all_y.extend([cls] * len(features))
        print(f"  {fname}: {len(features)} -> {CLASSES[cls]}")
        sys.stdout.flush()
    return np.array(all_X), np.array(all_y)

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    print("Loading recordings...")
    sys.stdout.flush()
    X, y = load_data()
    print(f"\nTotal: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {dict(Counter(y))}")
    sys.stdout.flush()

    # Balance classes
    min_class = min(Counter(y).values())
    target = min(min_class, 5000)
    balanced_idx = []
    for c in range(4):
        cls_idx = np.where(y == c)[0]
        np.random.seed(42)
        chosen = np.random.choice(cls_idx, target, replace=False)
        balanced_idx.extend(chosen)
    np.random.shuffle(balanced_idx)
    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx]
    print(f"Balanced: {len(X_bal)} ({target} per class)")
    sys.stdout.flush()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    sys.stdout.flush()

    # === 1. MLP ===
    print("\n=== MLP Classifier ===")
    sys.stdout.flush()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        verbose=True,
        learning_rate_init=0.001,
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_val)
    mlp_acc = accuracy_score(y_val, y_pred)
    print(f"\nMLP Val accuracy: {mlp_acc*100:.1f}%")
    print(classification_report(y_val, y_pred, target_names=CLASSES))
    sys.stdout.flush()

    # === 2. Random Forest ===
    print("\n=== Random Forest ===")
    sys.stdout.flush()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, y_pred_rf)
    print(f"RF Val accuracy: {rf_acc*100:.1f}%")
    print(classification_report(y_val, y_pred_rf, target_names=CLASSES))
    sys.stdout.flush()

    # Feature importance
    print("\nTop 10 features (RF):")
    feat_names = [
        'amp_mean','amp_std','amp_range','low_m','mid_m','high_m',
        'low_s','mid_s','high_s','turbulence','turb_ema',
        'diff_sq','abs_diff','motion_ema','accel','ema_dev',
        'motion_mean_w','motion_std_w','motion_max_w',
        'turb_mean_w','turb_std_w',
        'variance','mbp','bbp','sp','df','cp','rssi',
        'hr_conf','br_conf','sig_q',
    ]
    imp = rf.feature_importances_
    top = np.argsort(imp)[::-1][:10]
    for i in top:
        print(f"  {feat_names[i]:20s}: {imp[i]:.4f}")
    sys.stdout.flush()

    # === 3. Gradient Boosting ===
    print("\n=== Gradient Boosting ===")
    sys.stdout.flush()
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42, learning_rate=0.1)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_val)
    gb_acc = accuracy_score(y_val, y_pred_gb)
    print(f"GB Val accuracy: {gb_acc*100:.1f}%")
    print(classification_report(y_val, y_pred_gb, target_names=CLASSES))
    sys.stdout.flush()

    # Save best model info
    best_name = "MLP" if mlp_acc >= rf_acc and mlp_acc >= gb_acc else ("RF" if rf_acc >= gb_acc else "GB")
    best_acc = max(mlp_acc, rf_acc, gb_acc)
    print(f"\n=== Best: {best_name} ({best_acc*100:.1f}%) ===")

    # Export MLP weights for Rust
    if mlp_acc >= rf_acc * 0.95:  # prefer MLP if close (deployable in Rust)
        print("\nExporting MLP weights for Rust integration...")
        model_export = {
            "type": "mlp",
            "architecture": [X.shape[1]] + list(mlp.hidden_layer_sizes) + [4],
            "classes": CLASSES,
            "accuracy": float(mlp_acc),
            "n_samples": int(len(X)),
            "normalization": {
                "means": scaler.mean_.tolist(),
                "stds": scaler.scale_.tolist(),
            },
            "layers": [],
        }
        for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            model_export["layers"].append({
                "weights": W.tolist(),
                "biases": b.tolist(),
            })
        with open("data/mlp_model.json", 'w') as f:
            json.dump(model_export, f)
        print(f"Saved MLP model to /data/mlp_model.json")

    print(f"\nSummary:")
    print(f"  MLP:  {mlp_acc*100:.1f}%")
    print(f"  RF:   {rf_acc*100:.1f}%")
    print(f"  GB:   {gb_acc*100:.1f}%")
    sys.stdout.flush()
