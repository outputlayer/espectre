# ESPectre Sense

**WiFi CSI room presence detection & activity classification with ESP32-S3 and deep learning.**

Uses WiFi Channel State Information (CSI) from 3 ESP32-S3 nodes to classify room activity in real-time. No cameras, no wearables — just WiFi signals.

## How It Works

```
ESP32-S3 Nodes (×3)          Rust Server (Docker)              Browser
┌──────────────┐        ┌───────────────────────────┐    ┌──────────────┐
│ WiFi CSI data │──UDP──→│  CSINetLight CNN (pure    │    │   Heatmap    │
│ 56 subcarrier │  5005  │  Rust, no ML frameworks)  │─WS→│   Timeline   │
│ amplitudes    │        │  4-class classification   │4001│   Training   │
│ ~21 fps/node  │        │  empty|lying|sitting|walk │    │              │
└──────────────┘        └───────────────────────────┘    └──────────────┘
```

### CSINetLight CNN

| | |
|-------|---------|
| **Input** | 3 nodes × 56 subcarriers = 168 features × 40 frames (~1.8s window) |
| **Preprocessing** | Subsample 1/5 → baseline subtraction → normalization |
| **Architecture** | Conv1d(168→128,k=7) → BN → ReLU → Pool → Conv1d(128→256,k=5) → BN → ReLU → Pool → Conv1d(256→128,k=3) → BN → ReLU → GlobalAvgPool → Dense(128→64→4) |
| **Classes** | `empty`, `lying`, `sitting`, `walking` |
| **Inference** | Pure Rust, zero ML dependencies, ~1ms per frame |
| **Training** | Temporal split (70/30), augmentation (noise + scaling), class balancing |

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

### 3. Train Your Own Model

Record data via `http://SERVER:3030/ui/train.html` (3-5 min per class), then:

```bash
cd training && pip install -r requirements.txt
python prepare_data.py   # JSONL → sliding windows
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
│   ├── heatmap.html          # Real-time dashboard
│   ├── sleep.html            # Sleep tracking
│   └── train.html            # Data recording
└── docker/Dockerfile
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/sensing/latest` | Latest classification + CSI data |
| `GET /api/v1/vital-signs` | Heart rate & breathing estimates |
| `POST /api/v1/recording/start` | Start labeled CSI recording |
| `POST /api/v1/recording/stop` | Stop recording |
| `WS /ws/sensing` | Real-time data stream |

## Hardware

3× ESP32-S3 boards in triangle layout + WiFi AP + Linux server (Docker, ~128MB RAM).

## License

MIT
