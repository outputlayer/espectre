# ESPectre

**WiFi CSI room presence detection, movement tracking, vital signs monitoring & sleep quality analysis using ESP32-S3 sensors.**

ESPectre uses Channel State Information (CSI) from standard WiFi signals to detect human presence, track movement patterns, estimate heart rate and breathing rate, and analyze sleep quality — all without cameras or wearables.

## Features

- **Presence Detection** — detect whether a room is occupied using WiFi signal perturbations
- **Movement Tracking** — classify activity levels (absent → still → moving → active) in real-time
- **Vital Signs** — estimate heart rate (HR) and breathing rate (BR) from micro-movements in WiFi signals
- **Sleep Monitoring** — record overnight data with automatic sleep quality scoring
- **Real-time Heatmap** — IDW-interpolated spatial visualization with Everforest Dark theme
- **Multi-node Triangulation** — 3x ESP32-S3 nodes in triangle formation for spatial coverage
- **Persistent Calibration** — baseline calibration saved to disk, no recalibration on restart
- **REST API + WebSocket** — full HTTP API and real-time WebSocket streaming

## How It Works

```
ESP32-S3 Nodes (x3)          Server (Docker)              Browser
┌──────────────┐        ┌─────────────────────┐     ┌──────────────┐
│ WiFi CSI data │──UDP──→│ ESPectre Algorithm  │     │   Heatmap    │
│ 56 subcarrier │  5005  │ ├─ Turbulence score │──WS─→│   Sleep      │
│ amplitudes    │        │ ├─ Motion energy    │ 4001│   Dashboard  │
│ per frame     │        │ ├─ Vital signs (HR) │     │              │
│               │        │ └─ Classification   │     │              │
└──────────────┘        └─────────────────────┘     └──────────────┘
```

The ESPectre algorithm processes raw CSI amplitude data from each node:

1. **Turbulence** — mean frame-to-frame amplitude difference (body movement disrupts WiFi multipath)
2. **Peak detection** — 90th percentile turbulence (robust to single-frame WiFi spikes)
3. **Motion energy** — ratio of current motion to calibrated baseline with dead zones
4. **Variability** — standard deviation of rolling turbulence window

Combined score: `turb×0.35 + peak×0.25 + motion×0.20 + variability×0.20` → 0-10 scale

## Architecture

```
espectre/
├── server/src/           # Rust sensing server (Axum + Tokio)
│   ├── main.rs           # UDP receiver, WebSocket, REST API, classification
│   ├── espectre.rs       # ESPectre motion detection algorithm
│   ├── vital_signs.rs    # Heart rate & breathing rate detection from CSI
│   ├── trained_mlp.rs    # Trained MLP classifier (4-class, 81% accuracy)
│   └── ...
├── firmware/             # ESP32-S3 firmware (ESP-IDF v5.2, C)
│   ├── main/main.c       # CSI collection + UDP streaming
│   ├── main/csi_collector.c/h
│   ├── main/stream_sender.c/h
│   └── provision.py      # Flash WiFi credentials via NVS
├── ui/
│   ├── heatmap.html      # Real-time spatial heatmap + spectrograms
│   └── sleep.html        # Sleep quality dashboard with historical graphs
└── docker/Dockerfile     # Multi-stage build (Rust builder + Debian slim)
```

## Quick Start

### 1. Flash ESP32-S3 Firmware

```bash
# Set up ESP-IDF v5.2 environment
. ~/esp/esp-idf/export.sh

cd firmware
idf.py build
idf.py flash

# Provision WiFi credentials (stored in NVS, not in code)
python provision.py --port /dev/ttyUSB0 --ssid "YourWiFi" --password "YourPass" --target-ip YOUR_SERVER_IP
```

### 2. Run Server with Docker

```bash
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

### 3. Open Dashboard

- **Heatmap:** `http://YOUR_SERVER:3030/ui/heatmap.html`
- **Sleep Monitor:** `http://YOUR_SERVER:3030/ui/sleep.html`

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Server health status |
| `GET /api/v1/sensing/latest` | Latest sensing data (JSON) |
| `GET /api/v1/vital-signs` | Heart rate & breathing rate |
| `GET /api/v1/sleep/history?hours=N` | Sleep log data for last N hours |
| `WS /ws/sensing` | Real-time WebSocket stream |

## Hardware Requirements

- **3x ESP32-S3** development boards (any ESP32-S3 with WiFi CSI support)
- **WiFi access point** in the monitored room
- **Server** — any Linux machine or VPS (Docker, ~128MB RAM)

### Sensor Placement

Place 3 ESP32-S3 nodes in a triangle formation covering the room:

```
        [Node 3]
       /        \
      /   Room    \
     /              \
[Node 2] ────── [Node 1]
```

The triangulation enables spatial coverage. Each node independently measures CSI perturbations; the server fuses scores from all nodes.

## Sleep Quality Analysis

ESPectre records vital signs and motion data every 10 seconds to `/app/data/sleep_log.jsonl`. The sleep dashboard calculates:

- **Sleep Duration** — time from first still period to last
- **Sleep Efficiency** — percentage of time actually asleep (still/absent)
- **Restlessness Index** — movement events per hour
- **Average HR/BR** — weighted by signal confidence
- **Overall Quality Score** — 0-100% composite metric

> WiFi CSI vital signs work best during sleep — the person is stationary for extended periods, giving high signal integration time.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `CSI_SOURCE` | `auto` | Data source: `esp32`, `simulate`, `auto` |
| `RUST_LOG` | `info` | Log level: `debug`, `info`, `warn`, `error` |

| CLI Flag | Default | Description |
|---|---|---|
| `--http-port` | `4000` | HTTP API port |
| `--ws-port` | `4001` | WebSocket port |
| `--udp-port` | `5005` | ESP32 UDP receive port |
| `--tick-ms` | `100` | Processing interval (ms) |
| `--bind-addr` | `127.0.0.1` | Bind address (`0.0.0.0` for Docker) |
| `--ui-path` | `ui` | Path to static UI files |

## Data Format

### ESP32 CSI Frame (UDP, binary)

```
Magic:          0xC5110001 (4 bytes, little-endian)
Node ID:        u8
N Antennas:     u8
N Subcarriers:  u8 (typically 56)
Frequency MHz:  u16
Sequence:       u32
RSSI:           i8
Noise Floor:    i8
Payload:        [I, Q] pairs × N_antennas × N_subcarriers
```

### Sleep Log Entry (JSONL)

```json
{"ts":"2025-03-21T01:32:17Z","hr":62.5,"hr_conf":0.55,"br":14.2,"br_conf":0.48,"motion":0.12,"class":"present_still"}
```

## Training Data

The sleep log data (`sleep_log.jsonl`) can be used to train ML models for:
- Sleep phase classification (awake / light / deep / REM)
- Improved presence detection with personalized baselines
- Activity recognition from temporal motion patterns

## Tech Stack

- **Firmware:** C (ESP-IDF v5.2) on ESP32-S3
- **Server:** Rust (Axum + Tokio async runtime)
- **UI:** Vanilla HTML/CSS/JS with Canvas rendering
- **Deploy:** Docker multi-stage build (~35MB image)
- **Theme:** Everforest Dark color palette

## License

MIT
