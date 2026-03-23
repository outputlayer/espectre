# ESPectre Sense

**WiFi CSI room presence detection & activity classification with ESP32-S3 and deep learning.**

Uses WiFi Channel State Information (CSI) from 3 ESP32-S3 nodes to classify room activity in real-time. No cameras, no wearables — just WiFi signals.

## How It Works

```
ESP32-S3 Nodes (×3)          Rust Server (Docker)              Browser
┌──────────────┐        ┌───────────────────────────┐    ┌──────────────┐
│ WiFi CSI data │──UDP──→│  CSINetLight CNN (pure    │    │  Patient     │
│ 56 subcarrier │  5005  │  Rust, no ML frameworks)  │─WS→│  Monitoring  │
│ amplitudes    │        │  4-class classification   │4001│  Dashboard   │
│ ~21 fps/node  │        │  empty|lying|sitting|walk │    │              │
└──────────────┘        └───────────────────────────┘    └──────────────┘
```

### CSINetLight CNN

| | |
|-------|---------|
| **Input** | 3 nodes × 56 subcarriers = 168 features × 40 frames (~1.8s window) |
| **Preprocessing** | L2 normalization → baseline subtraction → global normalization |
| **Architecture** | Conv1d(168→128,k=7) → BN → ReLU → Pool → Conv1d(128→256,k=5) → BN → ReLU → Pool → Conv1d(256→128,k=3) → BN → ReLU → GlobalAvgPool → Dense(128→64→4) |
| **Classes** | `empty`, `lying`, `sitting`, `walking` |
| **Inference** | Pure Rust, zero ML dependencies, ~1ms per frame |
| **Post-processing** | Temporal voting (5-frame window), adaptive baseline on empty detection |

### Signal Processing Pipeline

1. **L2 Normalization** — Per-packet division by L2 norm removes ESP32 AGC (Automatic Gain Control) artifacts that cause amplitude scale shifts
2. **Baseline Subtraction** — Subtract mean of empty-room CSI to isolate human-caused signal changes
3. **Global Normalization** — Zero-mean, unit-variance normalization using training-set statistics
4. **CNN Inference** — CSINetLight processes 40-frame sliding windows
5. **Softmax + Temporal Voting** — Average probabilities over last 5 predictions to smooth class bouncing
6. **Adaptive Baseline** — When model detects empty room with >85% confidence, slowly updates baseline (rate=0.005)

## WiFi CSI Drift — The Hard Problem

> **TL;DR:** The model works well within a single session but accuracy degrades across sessions due to environmental CSI drift. This is a fundamental limitation of WiFi sensing, not a bug.

### What is CSI drift?

WiFi Channel State Information is not stable over time. Even with fixed sensors in a fixed room, the CSI baseline drifts due to:

- **Oscillator temperature drift** — ESP32 crystal frequency shifts with temperature
- **AGC (Automatic Gain Control) changes** — the radio adjusts gain unpredictably
- **Humidity and weather** — affects signal propagation
- **Neighbor WiFi interference** — nearby APs on overlapping channels
- **Building micro-vibrations** — subtle structural movement

Research confirms this is a fundamental issue: [CSI-Bench (2024)](https://arxiv.org/abs/2410.22652) documented a **41 F1-point drop** when testing WiFi sensing models across different environments/sessions vs. within the same session.

### What this means in practice

- **Same session:** Model achieves good accuracy (85-95% F1) distinguishing empty/lying/sitting/walking
- **Next day:** Accuracy can drop to near-random because the baseline has shifted. The model may predict a single class for everything.
- **After recalibration:** Accuracy recovers immediately within the new session

### Our mitigations

| Strategy | Status | Effect |
|----------|--------|--------|
| **L2 normalization** | Implemented | Removes AGC artifacts. Reduced drift metric from 1.28 to 0.001 |
| **Adaptive baseline** | Implemented | Slowly updates baseline when room is empty. Has chicken-and-egg problem: if model never predicts empty, baseline never updates |
| **Manual recalibration** | Implemented | User triggers baseline reset via API/UI. Captures 500 frames (~25s) of empty room. Breaks the chicken-and-egg cycle |
| **Multi-session training** | In progress | Record same activities at different times so CNN learns drift-invariant features. Primary research-backed solution |
| **Adversarial training** | Planned | Force feature extractor to be environment-invariant. Literature suggests +15-25 F1 points |

### Recalibration

When the model is stuck (predicting the same class regardless of activity), use the Recalibrate button in the dashboard or call the API:

```bash
# Start recalibration (room must be EMPTY)
curl -X POST http://SERVER:3030/api/v1/recalibrate

# Check progress
curl http://SERVER:3030/api/v1/recalibrate/status
```

Leave the room empty for ~25 seconds while recalibration captures new baseline frames.

### Multi-session training (recommended)

For best accuracy, record training data from **multiple sessions** at different times of day:

```
Session 1 (morning):  empty_v4, lying_v6, sitting_v5, walking_v5
Session 2 (evening):  empty_v5, lying_v7, sitting_v6, walking_v6
Session 3 (next day): empty_v6, lying_v8, sitting_v7, walking_v7
```

More sessions = more drift-invariant model. 3+ sessions recommended.

## Quick Start

### 1. Flash ESP32-S3 Nodes

```bash
cd firmware
idf.py set-target esp32s3 && idf.py build && idf.py flash
python provision.py --port /dev/ttyUSB0 --ssid "WiFi" --password "Pass" --target-ip SERVER_IP
```

### 2. Deploy

```bash
docker build -t espectre-sense -f docker/Dockerfile .
docker run -d --name espectre-sense \
  -p 3030:4000 -p 3031:4001 -p 5005:5005/udp \
  -e CSI_SOURCE=esp32 \
  -v espectre-data:/app/data \
  espectre-sense
```

Dashboard: `http://SERVER:3030/ui/heatmap.html`

### 3. First-time setup

1. Open the dashboard
2. Ensure the room is **empty**
3. Click **Recalibrate** and wait ~25 seconds
4. The model should now correctly detect empty room

### 4. Train Your Own Model

Record data via `http://SERVER:3030/ui/train.html` (3-5 min per class), then:

```bash
cd training && pip install -r requirements.txt
python prepare_data.py   # JSONL → L2 norm → baseline subtract → sliding windows
python train_dl.py       # Train CNN, export weights to server/models/
```

Rebuild Docker to deploy updated model.

## Project Structure

```
espectre-sense/
├── firmware/                 # ESP32-S3 CSI firmware (ESP-IDF)
├── server/
│   ├── src/main.rs           # UDP receiver, WS, REST API
│   ├── src/dl_classifier.rs  # CSINetLight inference (pure Rust)
│   └── models/               # Weights + normalization params
├── training/
│   ├── prepare_data.py       # Data preprocessing pipeline
│   ├── csi_model.py          # CSINet & CSINetLight architectures
│   └── train_dl.py           # Training with temporal split
├── ui/
│   ├── heatmap.html          # Patient monitoring dashboard
│   ├── sleep.html            # Sleep tracking
│   └── train.html            # Data recording
└── docker/Dockerfile
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sensing/latest` | GET | Latest classification + CSI data |
| `/api/v1/recalibrate` | POST | Start baseline recalibration (room must be empty) |
| `/api/v1/recalibrate/status` | GET | Recalibration progress |
| `/api/v1/recording/start` | POST | Start labeled CSI recording |
| `/api/v1/recording/stop` | POST | Stop recording |
| `/api/v1/recording/list` | GET | List all recordings |
| `/ws/sensing` | WS | Real-time data stream |

## Hardware

3× ESP32-S3 boards in triangle layout + WiFi AP + Linux server (Docker, ~128MB RAM).

## Known Limitations

- **CSI drift** means the model needs recalibration after environment changes (temperature shifts, hours passing). See [drift section](#wifi-csi-drift--the-hard-problem) above.
- **Single room** — model is trained per-room. Moving sensors to a new room requires retraining.
- **Sitting vs walking confusion** — the most common misclassification. Multi-session training data helps.
- **Single person** — designed for monitoring one person. Multiple people produce overlapping CSI signatures.

## References

- [CSI-Bench: A Large-Scale WiFi Sensing Benchmark (2024)](https://arxiv.org/abs/2410.22652) — Documents cross-environment accuracy drops
- [WiFi CSI-based HAR survey (2023)](https://arxiv.org/abs/2310.07628) — Comprehensive review of WiFi sensing approaches
- [EfficientFi (2022)](https://arxiv.org/abs/2204.01548) — Adversarial domain adaptation for WiFi sensing

## License

MIT
