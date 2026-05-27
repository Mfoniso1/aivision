# Intent-Eye AI // Kinetic Pan-Tilt Servo Tracker

An autonomous computer vision system designed for **Raspberry Pi** and standard **USB/Pi Cameras**. It utilizes a 3D Head Pose gaze estimation model to calculate human "intent to interact" and controls a dual-axis pan-tilt mechanism using high-torque **MG995 metal gear servos** driven by a closed-loop PID controller.

A stunning **Cyberpunk Space-Tech Telemetry Web Dashboard** allows you to view live optical overlays, monitor real-time angles, plot historical trajectories, tune PID coefficients on the fly, and take manual joystick override.

---

## 1. Hardware Connection Architecture

> [!WARNING]
> **DO NOT POWER THE MG995 SERVOS DIRECTLY FROM THE RASPBERRY PI'S 5V PINS!**
> Standard SG90 servos draw up to 600mA, but heavy-duty MG995 metal gear servos pull up to **1.2A to 1.5A under load or stall**. Powering them directly from the Pi will instantly trigger a voltage brownout and crash the Pi's CPU, and can permanently damage the Pi's GPIO rail.

### Recommended Wiring Blueprint

To power and control the servos safely, you need an **external 5V (or 6V) DC power supply rated for at least 2.5A to 3.0A**.

```
                           +----------------------+
                           |   External 5V / 6V   |
                           |   Power Supply       |
                           |   (2.5A - 3.0A+)     |
                           +--+---------------+---+
                              |               |
                              | 5V (Red)      | GND (Black)
                              v               v
                        +-----+---------------+-----+
                        |                           |
                        |   Common Junction Block   |
                        |                           |
                        +--+--------+--------+------+
                           |        |        |
         +-----------------+        |        +---------------------+
         | 5V Power                 | 5V Power                     | GND
         v                          v                              v
   +-----+-----+              +-----+-----+                  +-----+-----+
   |           |              |           |                  |           |
   |   MG995   |              |   MG995   |                  | Raspberry |
   |   PAN     |              |   TILT    |                  | Pi 4 / 5  |
   |   SERVO   |              |   SERVO   |                  | GPIO GND  |
   |  (Pin 18) |              |  (Pin 23) |                  |  (Pin 6)  |
   +-----+-----+              +-----+-----+                  +-----+-----+
         |                          |                              |
         | Signal                   | Signal                       |
         | (Yellow/Orange)          | (Yellow/Orange)              |
         v                          v                              |
   +-----+--------------------------+------------------------------+-----+
   | GPIO 18 (PWM)               GPIO 23                         GND     |
   |                                                                     |
   |                    RASPBERRY PI GPIO HEADER RAILS                   |
   +---------------------------------------------------------------------+
```

### Connector Details (Standard MG995 Pinout)
*   **Brown Wire:** Ground (GND) -> Connect to External GND **AND** Raspberry Pi GPIO Ground (e.g., Pin 6, 9, 14, 20, 25, 30, 34, 39).
*   **Red Wire:** Power (VCC, 4.8V - 7.2V) -> Connect to External Power supply (+) terminal.
*   **Yellow/Orange Wire:** PWM Signal -> 
    *   **Pan Servo**: Connect to Raspberry Pi **GPIO 18** (Pin 12 on header).
    *   **Tilt Servo**: Connect to Raspberry Pi **GPIO 23** (Pin 16 on header).

---

## 2. Software Installation & Setup

### A. Clone & Establish Virtual Environment
First, ensure python is installed on your machine. Open a terminal/powershell inside the `ai_vision_model` directory:

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows (Powershell)
.\venv\Scripts\Activate.ps1

# Activate on Linux / Raspberry Pi
source venv/bin/activate
```

### B. Install Python Dependencies
Install the required core computer vision, server, and utility libraries:

```bash
pip install -r requirements.txt
```

### C. (Raspberry Pi Only) Setup Jitter-Free Hardware PWM
To prevent the MG995 from twitching or jittering (which occurs with standard software PWM libraries due to CPU thread scheduling delays), we utilize the hardware-accurate **pigpio** daemon:

```bash
# Install the pigpio package and system daemon
sudo apt-get update
sudo apt-get install pigpio python3-pigpio -y

# Start and enable the pigpio background service
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

*Note: If `pigpio` is not running or not installed, the software will automatically fall back to `gpiozero` software PWM, and if not running on Linux, it will enter simulation mode.*

---

## 3. Execution & Deployment

### A. Testing on Windows (High-Fidelity Simulation Mode)
You can run and verify the entire AI pipeline, head-pose angles, intent detection, and PID tracking loops on your laptop using a standard webcam before mounting the hardware:

```powershell
python backend/app.py
```

1. The terminal will indicate that the hardware driver is in `SIMULATION` mode.
2. Open your web browser and navigate to **`http://localhost:8000`**.
3. You will see a beautiful dark-mode space dashboard.
4. Align your head with the camera:
   - When looking away, the status will show `SEARCHING` and the virtual servos will sweep horizontally.
   - When looking directly at the camera, the status will switch to `ENGAGED` (Intent Detected). The targeting crosshair will lock onto your nose and spin.
   - Look around or move: the **Virtual Gauges** and **Chrono Telemetry Waveform** will show the active PID loop smoothly adjusting the virtual pan and tilt servos to center your face!
5. Toggle to **Manual Override** to test joystick control and sliders, or tune the PID constants.

### B. Running on Raspberry Pi (Real-Time Servo Control Mode)
Once your servos and camera are wired according to the blueprint, run:

```bash
python backend/app.py
```

1. The console will report `[SERVO] Initialized MG995 hardware on GPIO 18/23 using pigpio.`
2. Access the dashboard from any device on your local network by navigating to: `http://<YOUR_PI_IP_ADDRESS>:8000` (e.g. `http://192.168.1.45:8000`).
3. Sit back and watch the physical pan/tilt camera autonomously rotate to find, engage, lock onto, and track you based on your intent!

---

## 4. Tuning the PID Loop

The MG995 is an extremely powerful, high-torque servo. Out-of-the-box servos can overshoot or oscillate wildly if the feedback loop gains are too aggressive.
Our dashboard allows you to tune these variables in real-time under **[03] COMMAND & TUNING CENTER**:

*   **$K_p$ (Proportional Gain):** Determines how fast the servo responds to target error. If the face is far from the center, a higher $K_p$ commands a larger speed step.
    *   *Symptom of too high:* Fast shaking, violent oscillation around the target center.
    *   *Symptom of too low:* Extremely slow, sluggish tracking.
*   **$K_i$ (Integral Gain):** Corrects steady-state tracking offsets. Fills in the missing torque if the camera gets stuck slightly off-center.
    *   *Symptom of too high:* Constant drifting back and forth.
    *   *Symptom of too low:* Camera centers near you but stops short of absolute precision.
*   **$K_d$ (Derivative Gain):** Provides dampening against sudden accelerations, smoothing the approach as the face returns to the center.
    *   *Symptom of too high:* Grinding sounds or stiff, jerky movements.
    *   *Symptom of too low:* Overshooting the center, swinging past your nose, then having to correct back.

### Real-Time Commit
Modify the values on the panel and click **COMMIT PID COEFFS** to inject the new parameters into the active control loop instantly without restarting the server!
