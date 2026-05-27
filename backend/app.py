import sys
import os
import subprocess

# ----------------- Self-Bootstrap Virtual Environment -----------------
# Check if running inside a virtual environment. If not, automatically re-execute
# this script within the local virtual environment context (venv).
if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The local venv folder is in the parent directory of backend/
    venv_python = os.path.abspath(os.path.join(current_dir, "..", "venv", "bin", "python"))
    if os.name == 'nt':
        venv_python = os.path.abspath(os.path.join(current_dir, "..", "venv", "Scripts", "python.exe"))
        
    if os.path.exists(venv_python) and sys.executable != venv_python:
        print(f"[BOOTSTRAP] System Python detected. Re-executing inside venv: {venv_python}")
        result = subprocess.run([venv_python] + sys.argv)
        sys.exit(result.returncode)
# ----------------------------------------------------------------------

import cv2
import asyncio
import json
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Import our custom components
from vision import IntentVisionEngine
from servo_controller import MG995ServoDriver, PIDController

app = FastAPI(title="Intent-Eye AI System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Variables
latest_frame = None
latest_data = {}
frame_lock = threading.Lock()
is_running = True

# Control states
control_mode = "AUTO"  # AUTO or MANUAL
search_sweep_enabled = True
sweep_direction = 1  # 1 = Right, -1 = Left
sweep_angle = 90.0
sweep_speed = 0.8    # Degrees per frame

# PID Parameters for MG995 Tracking
# MG995 is high-torque, so we use gentle proportional gains and derivative dampening to prevent overshoot
kp_pan, ki_pan, kd_pan = 24.0, 0.05, 0.4
kp_tilt, ki_tilt, kd_tilt = 24.0, 0.05, 0.4

# Initialize Controller & Drivers
servos = MG995ServoDriver(pan_pin=18, tilt_pin=23)
pan_pid = PIDController(kp_pan, ki_pan, kd_pan, output_limits=(-12.0, 12.0))
tilt_pid = PIDController(kp_tilt, ki_tilt, kd_tilt, output_limits=(-12.0, 12.0))

vision_engine = IntentVisionEngine(intent_threshold_deg=22.0, filter_beta=0.65)

# Connected clients
active_websockets = []

def get_system_metrics():
    """
    Retrieves system metrics like CPU load and temperature.
    Includes fallbacks if running on non-Linux/non-Pi platforms.
    """
    metrics = {
        "cpu_usage": 10.0,
        "temperature": 45.0,
        "platform": "Simulated"
    }
    
    # Try importing psutil for CPU usage
    try:
        import psutil
        metrics["cpu_usage"] = round(psutil.cpu_percent(), 1)
        metrics["platform"] = os.name
    except ImportError:
        # Mock CPU fluctuation slightly
        metrics["cpu_usage"] = round(10.0 + 5.0 * np.sin(time.time() / 10.0), 1)
        
    # Try reading Raspberry Pi core temperature
    if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_raw = int(f.read().strip())
                metrics["temperature"] = round(temp_raw / 1000.0, 1)
                metrics["platform"] = "Raspberry Pi"
        except Exception:
            pass
            
    return metrics

def camera_and_control_thread():
    """
    Dedicated background thread to capture video, run AI vision,
    calculate PID, and command the servos at 30 FPS.
    """
    global latest_frame, latest_data, is_running, sweep_angle, sweep_direction
    
    print("[BACKEND] Starting camera capture and vision tracking thread...")
    
    # Open camera interface (0 is standard webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Could not open primary camera. Retrying index 1...")
        cap = cv2.VideoCapture(1)
        
    # Set lower resolution to optimize FPS on Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Time vars for FPS computation
    fps_start_time = time.time()
    fps_counter = 0
    calculated_fps = 30.0
    
    # Wait briefly for camera startup
    time.sleep(1.0)
    
    while is_running:
        start_loop = time.time()
        
        ret, frame = cap.read()
        if not ret:
            # Create a mock black frame with rotating circular target if no webcam is available
            # This is awesome for testing in headless or Docker/VM environments!
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a simulated bouncing face circle if no webcam detected
            cx = int(320 + 150 * np.cos(time.time() * 0.8))
            cy = int(240 + 100 * np.sin(time.time() * 1.2))
            cv2.circle(frame, (cx, cy), 50, (0, 255, 0), 2)
            cv2.circle(frame, (cx - 15, cy - 10), 5, (0, 255, 0), -1) # eyes
            cv2.circle(frame, (cx + 15, cy - 10), 5, (0, 255, 0), -1)
            cv2.ellipse(frame, (cx, cy + 15), (20, 10), 0, 0, 180, (0, 255, 0), 2) # mouth
            
        # Flip frame horizontally to act like a mirror for natural visual tracking
        frame = cv2.flip(frame, 1)
        
        # Run intent AI tracking on frame
        tracking_data, annotated_frame = vision_engine.process_frame(frame)
        
        # FPS Calculation
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            calculated_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
        tracking_data["fps"] = round(calculated_fps, 1)
        
        # ----------------- Servo Control Loop -----------------
        state = tracking_data["state"]
        p_angle, t_angle = servos.pan_angle, servos.tilt_angle
        
        if control_mode == "AUTO":
            if state == "ENGAGED":
                # Active locked PID tracking
                # Error is the offset from center coordinates (0.5)
                # Pan: if target is right of center (x > 0.5), we need to pan right (decrease pan angle)
                # Note: target coordinates are from camera mirror POV
                error_x = 0.5 - tracking_data["target_x"]
                error_y = 0.5 - tracking_data["target_y"]
                
                # Compute PID adjustments
                # Scale error to degree adjustments
                pan_delta = pan_pid.compute(error_x)
                tilt_delta = tilt_pid.compute(error_y)
                
                # Command servos
                p_angle, t_angle = servos.update_position(pan_delta, tilt_delta)
                
            elif state == "SEARCHING" and search_sweep_enabled:
                # Slowly sweep pan servo horizontally to search for human presence
                pan_pid.reset()
                tilt_pid.reset()
                
                # Gently return tilt to center (90)
                tilt_error = 90.0 - servos.tilt_angle
                tilt_delta = 0.05 * tilt_error  # P-only recovery
                
                sweep_angle += sweep_direction * sweep_speed
                if sweep_angle >= servos.pan_limits[1]:
                    sweep_direction = -1
                elif sweep_angle <= servos.pan_limits[0]:
                    sweep_direction = 1
                    
                p_angle, t_angle = servos.set_absolute_position(sweep_angle, servos.tilt_angle + tilt_delta)
                
            else:
                # IDLE or LOST: hold positions, reset PID accumulation
                pan_pid.reset()
                tilt_pid.reset()
        else:
            # MANUAL MODE: PID is reset, servos hold manual values
            pan_pid.reset()
            tilt_pid.reset()
            
        # Append current physical angles to global telemetry
        tracking_data["pan_angle"] = p_angle
        tracking_data["tilt_angle"] = t_angle
        tracking_data["servo_mode"] = servos.mode
        tracking_data["control_mode"] = control_mode
        tracking_data["search_sweep"] = search_sweep_enabled
        
        # Add system metrics
        tracking_data.update(get_system_metrics())
        
        # Update thread-safe globals
        with frame_lock:
            latest_frame = annotated_frame.copy()
            latest_data = tracking_data.copy()
            
        # Maintain exactly 30 FPS to avoid overloading Raspberry Pi CPU
        elapsed = time.time() - start_loop
        sleep_time = max(0.001, (1.0 / 30.0) - elapsed)
        time.sleep(sleep_time)
        
    cap.release()
    servos.cleanup()
    print("[BACKEND] Camera thread terminated and hardware cleaned up.")

# Start backend thread on startup
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=camera_and_control_thread, daemon=True).start()

@app.on_event("shutdown")
def shutdown_event():
    global is_running
    is_running = False

# ----------------- HTTP Routes -----------------
# Serve files from the frontend workspace directory
# Ensure frontend files exist
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

@app.get("/")
def get_dashboard():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend dashboard not built yet. Create index.html."}

@app.get("/style.css")
def get_css():
    return FileResponse(os.path.join(frontend_dir, "style.css"))

@app.get("/main.js")
def get_js():
    return FileResponse(os.path.join(frontend_dir, "main.js"))

# Live annotated video stream generator
def generate_mjpeg_stream():
    global latest_frame
    while is_running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            # Encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if not ret:
                time.sleep(0.05)
                continue
            frame_bytes = jpeg.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Rate-limit the stream to avoid network congestion
        time.sleep(0.04)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ----------------- WebSocket Communication -----------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global control_mode, search_sweep_enabled, kp_pan, ki_pan, kd_pan, kp_tilt, ki_tilt, kd_tilt
    
    await websocket.accept()
    active_websockets.append(websocket)
    print(f"[WS] Client connected. Active clients: {len(active_websockets)}")
    
    try:
        # Asynchronous task to push high-speed telemetry to the UI
        async def telemetry_pusher():
            while websocket in active_websockets:
                with frame_lock:
                    telemetry = latest_data.copy() if latest_data else {}
                    
                if telemetry:
                    # Inject current PID values for feedback in settings
                    telemetry["pid"] = {
                        "kp_p": pan_pid.kp, "ki_p": pan_pid.ki, "kd_p": pan_pid.kd,
                        "kp_t": tilt_pid.kp, "ki_t": tilt_pid.ki, "kd_t": tilt_pid.kd
                    }
                    try:
                        await websocket.send_text(json.dumps(telemetry))
                    except Exception:
                        break
                await asyncio.sleep(0.05) # 20 Hz push is incredibly fluid and lightweight
                
        # Start pusher task in background
        pusher_task = asyncio.create_task(telemetry_pusher())
        
        while True:
            # Wait for client control commands
            data = await websocket.receive_text()
            cmd = json.loads(data)
            
            action = cmd.get("action")
            if action == "set_mode":
                control_mode = cmd.get("mode", "AUTO")
                print(f"[WS] Control Mode updated to: {control_mode}")
                
            elif action == "set_profile":
                profile = cmd.get("profile", "TRACKING")
                vision_engine.profile = profile
                print(f"[WS] Active Vision Profile updated to: {profile}")
                
            elif action == "set_sweep":
                search_sweep_enabled = bool(cmd.get("enabled", True))
                print(f"[WS] Search Sweep toggled: {search_sweep_enabled}")
                
            elif action == "manual_set":
                if control_mode == "MANUAL":
                    pan = float(cmd.get("pan", servos.pan_angle))
                    tilt = float(cmd.get("tilt", servos.tilt_angle))
                    servos.set_absolute_position(pan, tilt)
                    
            elif action == "manual_step":
                if control_mode == "MANUAL":
                    p_delta = float(cmd.get("pan_delta", 0.0))
                    t_delta = float(cmd.get("tilt_delta", 0.0))
                    servos.update_position(p_delta, t_delta)
                    
            elif action == "update_pid":
                # Dynamically update PID constants on the fly
                kp_p = float(cmd.get("kp_p", pan_pid.kp))
                ki_p = float(cmd.get("ki_p", pan_pid.ki))
                kd_p = float(cmd.get("kd_p", pan_pid.kd))
                
                kp_t = float(cmd.get("kp_t", tilt_pid.kp))
                ki_t = float(cmd.get("ki_t", tilt_pid.ki))
                kd_t = float(cmd.get("kd_t", tilt_pid.kd))
                
                pan_pid.kp, pan_pid.ki, pan_pid.kd = kp_p, ki_p, kd_p
                tilt_pid.kp, tilt_pid.ki, tilt_pid.kd = kp_t, ki_t, kd_t
                
                print(f"[WS] PID params updated: PAN({kp_p},{ki_p},{kd_p}) TILT({kp_t},{ki_t},{kd_t})")
                
    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        try:
            pusher_task.cancel()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    print("[BACKEND] Launching Intent-Eye AI Web Server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
