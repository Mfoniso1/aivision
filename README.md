# 🍓 Raspberry Pi Gaze Intent-Eye Client (`ai_vision_model`)

A lightweight, standalone, high-performance edge client designed to capture local camera frames, stream them to a self-hosted AI model server, and drive pan-tilt servos (e.g., MG995) in real-time using a local PID control loop based on human "intent to interact."

This repository has been fully optimized for edge deployment (such as on the Raspberry Pi) with **zero local machine learning package overhead** (no local MediaPipe, no local PyTorch, no heavy dependencies).

---

## 🔬 Core System Architecture

```
 +-----------------------------------------------------------------+
 |                    EXTERNAL SELF-HOSTED MODEL                   |
 |  (Any server hosting your vision/intent model, e.g. FastAPI,    |
 |   Ollama, custom CNN, or VLM endpoint)                          |
 |  - Accepts JPEG images (via HTTP POST or WebSocket)             |
 |  - Returns JSON intent/coordinate telemetry                    |
 +-----------------------------------------------------------------+
                                 ^
                                 | Configurable API Calls
                                 | (HTTP POST or WebSocket)
                                 v
 +-----------------------------------------------------------------+
 |                 EDGE / TARGET DEVICE (Raspberry Pi)             |
 |  - Runs `intent_detector.py` (Lightweight Python Client)        |
 |  - Reads camera frame -> Compresses to JPEG                     |
 |  - Sends frame to Self-Hosted Model API                         |
 |  - Receives target JSON -> Filters coordinates via EMA          |
 |  - Feeds Dwell-Time State Machine -> Updates PID Loop           |
 |  - Drives physical servos via pigpio/gpiozero (or Simulation)   |
 +-----------------------------------------------------------------+
```

### 🧠 Research-Backed Intent Logic
1. **The Engagement Cone (Gaze Deflection):** Uses the Euclidean norm of head yaw and pitch angular deflection relative to the camera axis. Gaze deviation $\le 22.0^\circ$ registers "potential intent to interact."
2. **Dwell-Time State Machine:** Transitions between `IDLE` $\rightarrow$ `SEARCHING` $\rightarrow$ `POTENTIAL_INTENT` $\rightarrow$ `ENGAGED`. Requires the user to look at the device for at least $0.6$ seconds before triggering locked tracking, preventing accidental glances from swinging the camera.
3. **EMA Coordinate Smoothing:** Applies a real-time Exponential Moving Average filter on target coordinates to prevent high-frequency servo jitter and grinding.

---

## 📦 Project Structure

```
.
├── intent_detector.py      # Core edge client (Camera capture, API stream, PID, Servos)
├── config.json             # Central configuration (pins, PID gains, API url, intent cone)
├── requirements.txt        # Minimal Python dependencies
└── README.md               # Deployment manual
```

---

## 🛠️ Step-by-Step Raspberry Pi Setup

### 1. Install System Requirements & Enable PWM Daemon
Log into your Raspberry Pi terminal and run:

```bash
# 1. Update packages and install the pigpio system daemon (crucial for jitter-free PWM)
sudo apt-get update
sudo apt-get install pigpio python3-pigpio -y

# 2. Enable and start the background hardware PWM service
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### 2. Install Python Dependencies
Inside your project directory, run:
```bash
pip install -r requirements.txt
```

### 3. Connect the Servos (MG995)
Attach your pan-tilt servo motor control lines directly to your Raspberry Pi GPIO header pins:
* **Pan Servo (Horizontal):** GPIO 18 (Pin 12)
* **Tilt Servo (Vertical):** GPIO 23 (Pin 16)
* **Power:** Ensure your MG995 servos are powered by an external 5V/6V supply (do not power them directly from the Pi's 5V pin, as current draw can cause a brownout/reset). Connect the Pi's GND pin to the external power supply GND.

---

## ⚙️ Configuration (`config.json`)

Configure your server endpoint, camera dimensions, servo pins, and PID constants directly inside `config.json`:

```json
{
  "api": {
    "mode": "HTTP",
    "url": "http://<YOUR_MODEL_SERVER_IP>:8000/predict",
    "timeout_seconds": 2.0
  },
  "camera": {
    "index": 0,
    "width": 640,
    "height": 480,
    "jpeg_quality": 80
  },
  "servos": {
    "pan_pin": 18,
    "tilt_pin": 23,
    "pid": {
      "kp_pan": 24.0,
      "ki_pan": 0.05,
      "kd_pan": 0.4,
      "kp_tilt": 24.0,
      "ki_tilt": 0.05,
      "kd_tilt": 0.4
    }
  },
  "intent": {
    "engagement_cone_deg": 22.0,
    "dwell_time_seconds": 0.6,
    "ema_beta": 0.65
  }
}
```

---

## 🚀 Execution

To start the standalone client loop:

```bash
python intent_detector.py
```

### 🔄 Auto-Actuation Modes
* **Pi Hardware Mode:** If `pigpio` or `gpiozero` is detected, it immediately drives your physical servos using hardware PWM.
* **Console Simulation Mode:** If run on a non-Pi device (such as your local Windows/Mac PC for testing), the script gracefully falls back to logging the calculated virtual servo angles, allowing you to debug the camera feed, API responses, and Dwell State Machine without crashing.
