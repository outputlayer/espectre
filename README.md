# ESPectre

**WiFi CSI room presence detection, movement tracking, vital signs & sleep quality analysis using ESP32-S3 sensors.**

ESPectre turns standard WiFi signals into a room-sensing system — detecting people, classifying activity, estimating heart/breathing rate, and analyzing sleep quality. No cameras, no wearables.

## Quick Start (4 Steps)

### Step 1: Flash ESP32-S3 Firmware

You need 3x ESP32-S3 boards arranged in a triangle in the room.

```bash
# Install ESP-IDF v5.2
# https://docs.espressif.com/projects/esp-idf/en/v5.2/esp32s3/get-started/

cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash

# Write WiFi credentials to each board (stored in NVS, never in code)
python provision.py --port /dev/ttyUSB0 --ssid "YourWiFi" --password "YourPass" --target-ip YOUR_SERVER_IP
```

Repeat for all 3 boards. Each board streams CSI data via UDP to port 5005 on your server.

### Step 2: Deploy Server

```bash
git clone https://github.com/outputlayer/espectre.git
cd espectre

docker build -t espectre -f docker/Dockerfile .
docker run -d \
  --name espectre \
  -p 3030:4000 \
  -p 3031:4001 \
  -p 5005:5005/udp \
  -v espectre-data:/app/data \
  -e CSI_SOURCE=esp32 \
  espectre
```

Open dashboards:
- **Heatmap:** `http://YOUR_SERVER:3030/ui/heatmap.html`
- **Sleep Monitor:** `http://YOUR_SERVER:3030/ui/sleep.html`

At this point you have real-time presence detection using the built-in ESPectre algorithm (no ML training needed).

### Step 3: Collect Training Data

Open the training UI: `http://YOUR_SERVER:3030/ui/train.html`

Record labeled CSI data by performing activities while the system captures sensor readings:

| Button | What to do | Duration |
|--------|-----------|----------|
| **Empty Room** | Leave room, close door | 3-5 min |
| **Sitting Still** | Sit at desk normally | 3-5 min |
| **Walking** | Walk around the room | 2-3 min |
| **Active** | Exercise, wave arms | 2-3 min |
| **Lying Down** | Lie in bed, stay still | 3-5 min |
| **Empty + Door Open** | Leave room, door open | 3-5 min |

Repeat at different times of day and with different conditions (lights on/off, door open/closed) for best results.

### Step 4: Train & Deploy Model

```bash
cd training
pip install -r requirements.txt
python train_sklearn.py
```

This trains 3 models (MLP, Random Forest, Gradient Boosting) and exports the best MLP weights for the Rust server. Copy the updated `trained_mlp.rs` to the server and rebuild Docker.

## How It Works

```
ESP32-S3 Nodes (x3)          Rust Server (Docker)          Browser
┌──────────────┐        ┌─────────────────────┐     ┌──────────────┐
│ WiFi CSI data │──UDP──→│ ESPectre Algorithm  │     │   Heatmap    │
│ 56 subcarrier │  5005  │ ├─ Turbulence       │──WS→│   Sleep      │
│ amplitudes    │        │ ├─ Motion energy    │ 4001│   Training   │
│               │        │ ├─ Vital signs (HR) │     │              │
│               │        │ └─ MLP Classifier   │     │              │
└──────────────┘        └─────────────────────┘     └──────────────┘
```

The ESPectre algorithm processes CSI amplitude data per node:
1. **Turbulence** — frame-to-frame amplitude changes (body disrupts WiFi multipath)
2. **Peak detection** — 90th percentile turbulence (robust to WiFi spikes)
3. **Motion energy** — current motion vs calibrated baseline
4. **MLP Classifier** — trained neural network for 4-class classification

Combined score: `turb×0.35 + peak×0.25 + motion×0.20 + variability×0.20` → 0-10 scale

## Project Structure

```
espectre/
├── firmware/               # ESP32-S3 firmware (ESP-IDF v5.2, C)
│   ├── main/main.c         # CSI collection + UDP streaming
│   ├── main/csi_collector.c # WiFi CSI frame capture
│   ├── main/stream_sender.c # UDP frame transmission
│   ├── main/nvs_config.c   # WiFi credentials from NVS
│   ├── main/ota_update.c   # Over-the-air firmware updates
│   └── provision.py        # Flash WiFi credentials via NVS
├── server/src/             # Rust sensing server (Axum + Tokio)
│   ├── main.rs             # UDP receiver, WS, REST API, classification
│   ├── espectre.rs         # ESPectre motion detection algorithm
│   ├── vital_signs.rs      # Heart rate & breathing rate from CSI
│   ├── trained_mlp.rs      # Trained MLP classifier weights
│   └── adaptive_classifier.rs # Runtime model adaptation
├── training/               # ML training pipeline (Python)
│   ├── train_sklearn.py    # Train MLP/RF/GB on recorded CSI data
│   └── requirements.txt    # Python dependencies
├── ui/                     # Web dashboards (vanilla HTML/JS/Canvas)
│   ├── heatmap.html        # Real-time spatial heatmap + spectrograms
│   ├── sleep.html          # Sleep quality dashboard with chart zoom
│   └── train.html          # Training data collection UI
└── docker/Dockerfile       # Multi-stage build (Rust → Debian slim)
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health status |
| `/api/v1/sensing/latest` | GET | Latest sensing data |
| `/api/v1/vital-signs` | GET | Heart rate & breathing rate |
| `/api/v1/sleep/history?hours=N` | GET | Sleep log for last N hours |
| `/api/v1/recording/start` | POST | Start recording CSI data |
| `/api/v1/recording/stop` | POST | Stop recording |
| `/api/v1/recording/list` | GET | List all recordings |
| `/ws/sensing` | WS | Real-time WebSocket stream |

## Sleep Quality Analysis

Data logged every 10s to `/app/data/sleep_log.jsonl`. The dashboard calculates:

- **Duration** — first-to-last still period
- **Efficiency** — % time actually asleep
- **Restlessness** — movement events per hour
- **Avg HR/BR** — weighted by signal confidence
- **Quality Score** — 0-100% composite

Charts support drag-to-zoom, scroll zoom, and double-click to reset.

## Hardware

- **3x ESP32-S3** development boards (any with WiFi CSI support)
- **WiFi access point** in the monitored room
- **Server** — any Linux machine or VPS (Docker, ~128MB RAM)

```
        [Node 3]
       /        \
      /   Room    \
     /              \
[Node 2] ────── [Node 1]
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `auto` | `esp32`, `simulate`, `auto` |
| `--http-port` | `4000` | HTTP API port |
| `--ws-port` | `4001` | WebSocket port |
| `--tick-ms` | `100` | Processing interval (ms) |
| `--bind-addr` | `127.0.0.1` | Bind address (`0.0.0.0` for Docker) |
| `--ui-path` | `ui` | Static UI files path |

## Training Results

| Model | Accuracy | Deployable in Rust |
|-------|----------|-------------------|
| MLP (current) | 83.0% | Yes (weight arrays) |
| Random Forest | 82.1% | No |
| Gradient Boosting | 87.3% | No (used as teacher for distillation) |

More training data improves accuracy. Target: 5+ recording sessions across different conditions.

## License

MIT
