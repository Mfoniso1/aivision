import time
import os
import sys
import json
import threading
import cv2
import numpy as np
import requests

# Try importing WebSocket client for streaming mode
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

# Try importing Adafruit PCA9685 ServoKit
try:
    from adafruit_servokit import ServoKit
    HAS_SERVOKIT = True
except ImportError:
    HAS_SERVOKIT = False

# =====================================================================
# 1. CORE CONTROL & ACTUATION MODULES
# =====================================================================

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-10.0, 10.0)):
        """
        Proportional-Integral-Derivative feedback controller.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0:
            dt = 0.01  # Prevent divide-by-zero
            
        p_term = self.kp * error
        
        # Integral with anti-windup clamping
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        
        min_lim, max_lim = self.output_limits
        if output < min_lim:
            output = min_lim
            self.integral -= error * dt # anti-windup clamp
        elif output > max_lim:
            output = max_lim
            self.integral -= error * dt # anti-windup clamp
            
        self.last_error = error
        self.last_time = now
        
        return output

class MG995ServoDriver:
    def __init__(self, pan_channel=0, tilt_channel=2):
        """
        Dual-axis servo controller designed for physical MG995 servos.
        Uses Adafruit PCA9685 16-Channel I2C driver (ServoKit) on Pi.
        Falls back to high-fidelity console SIMULATION on non-Pi platforms.
        """
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        
        # Matches your exact hardware limits (0 to 180 deg)
        self.pan_limits = (0.0, 180.0)
        self.tilt_limits = (0.0, 180.0)
        
        # Matches your exact startup positions
        self.pan_angle = 180.0
        self.tilt_angle = 90.0
        
        self.mode = "SIMULATION"
        self.kit = None
        
        self.init_hardware()
        
    def init_hardware(self):
        if HAS_SERVOKIT:
            try:
                self.kit = ServoKit(channels=16)
                self.mode = "SERVOKIT"
                # Center servos slowly/gently on startup
                self.write_angle(self.pan_channel, self.pan_angle)
                self.write_angle(self.tilt_channel, self.tilt_angle)
                print(f"[SERVO] Adafruit PCA9685 I2C Driver active. Channels: Pan={self.pan_channel}, Tilt={self.tilt_channel}")
                return
            except Exception as e:
                print(f"[SERVO] Adafruit PCA9685 initialization failed: {e}")
                
        self.mode = "SIMULATION"
        print("[SERVO] Running in Simulation Mode. Virtual angles will be logged.")

    def write_angle(self, channel, angle):
        if channel == self.pan_channel:
            self.pan_angle = max(self.pan_limits[0], min(self.pan_limits[1], angle))
            target_angle = self.pan_angle
        else:
            self.tilt_angle = max(self.tilt_limits[0], min(self.tilt_limits[1], angle))
            target_angle = self.tilt_angle
            
        if self.mode == "SERVOKIT" and self.kit:
            try:
                self.kit.servo[channel].angle = target_angle
            except Exception as e:
                print(f"[SERVO ERROR] Failed to write angle {target_angle} to channel {channel}: {e}")

    def update_position(self, pan_delta, tilt_delta):
        # Move horizontal
        self.pan_angle += pan_delta
        self.write_angle(self.pan_channel, self.pan_angle)
        
        # Move vertical
        self.tilt_angle += tilt_delta
        self.write_angle(self.tilt_channel, self.tilt_angle)
        
        return round(self.pan_angle, 1), round(self.tilt_angle, 1)

    def set_absolute_position(self, pan_angle, tilt_angle):
        self.write_angle(self.pan_channel, pan_angle)
        self.write_angle(self.tilt_channel, tilt_angle)
        return round(self.pan_angle, 1), round(self.tilt_angle, 1)

    def cleanup(self):
        print("[SERVO] Cleaning up. PCA9685 connection closed.")

# =====================================================================
# 2. MAIN EDGE INTENT TRACKER SYSTEM
# =====================================================================

class StandaloneIntentTracker:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.load_config()
        
        # Initialize Actuation
        servo_cfg = self.config.get("servos", {})
        pid_cfg = servo_cfg.get("pid", {})
        self.driver = MG995ServoDriver(
            pan_channel=servo_cfg.get("pan_channel", 0),
            tilt_channel=servo_cfg.get("tilt_channel", 2)
        )
        
        self.pan_pid = PIDController(
            pid_cfg.get("kp_pan", 24.0),
            pid_cfg.get("ki_pan", 0.05),
            pid_cfg.get("kd_pan", 0.4),
            output_limits=(-12.0, 12.0)
        )
        self.tilt_pid = PIDController(
            pid_cfg.get("kp_tilt", 24.0),
            pid_cfg.get("ki_tilt", 0.05),
            pid_cfg.get("kd_tilt", 0.4),
            output_limits=(-12.0, 12.0)
        )
        
        # Intent Dwell State Machine Variables
        intent_cfg = self.config.get("intent", {})
        self.cone_threshold = intent_cfg.get("engagement_cone_deg", 22.0)
        self.dwell_threshold = intent_cfg.get("dwell_time_seconds", 0.6)
        self.ema_beta = intent_cfg.get("ema_beta", 0.65)
        
        self.state = "IDLE"  # IDLE, SEARCHING, POTENTIAL_INTENT, ENGAGED
        self.potential_start_time = None
        self.last_success_time = time.time()
        
        # EMA smoothed coordinates
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        
        # Camera Sweep Parameters
        self.sweep_angle = 90.0
        self.sweep_direction = 1
        self.sweep_speed = 0.8
        
        # Video Capture Setup
        cam_cfg = self.config.get("camera", {})
        self.cam_index = cam_cfg.get("index", 0)
        self.cam_w = cam_cfg.get("width", 640)
        self.cam_h = cam_cfg.get("height", 480)
        self.jpeg_quality = cam_cfg.get("jpeg_quality", 80)
        
        # Threading and Loop parameters
        self.running = False
        self.ws_conn = None
        
    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            print(f"[CONFIG] Successfully loaded: {self.config_path}")
        except Exception as e:
            print(f"[CONFIG] Error loading {self.config_path}, using defaults. Error: {e}")
            self.config = {}

    def get_api_details(self):
        api_cfg = self.config.get("api", {})
        mode = api_cfg.get("mode", "HTTP").upper()
        url = api_cfg.get("url", "http://localhost:8000/predict")
        timeout = api_cfg.get("timeout_seconds", 2.0)
        return mode, url, timeout

    def send_frame_http(self, url, jpeg_bytes, timeout):
        """
        Sends frame using HTTP multipart POST.
        """
        try:
            files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
            response = requests.post(url, files=files, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[API] HTTP Server Error {response.status_code}")
                return None
        except Exception as e:
            print(f"[API] HTTP Connection Error: {e}")
            return None

    def send_frame_ws(self, url, jpeg_bytes):
        """
        Sends frame using high-speed WebSockets.
        """
        if not HAS_WEBSOCKET:
            print("[API] websocket-client library missing. Install via pip install websocket-client.")
            return None
            
        try:
            if not self.ws_conn:
                print(f"[API] Connecting WebSocket to {url} ...")
                self.ws_conn = websocket.create_connection(url, timeout=2.0)
                print("[API] WebSocket Connected.")
                
            # Send binary image
            self.ws_conn.send_binary(jpeg_bytes)
            # Receive response JSON string
            resp_str = self.ws_conn.recv()
            return json.loads(resp_str)
        except Exception as e:
            print(f"[API] WebSocket Error: {e}")
            # Reset connection to trigger retry next iteration
            if self.ws_conn:
                try:
                    self.ws_conn.close()
                except:
                    pass
                self.ws_conn = None
            return None

    def process_telemetry(self, telemetry):
        """
        Processes AI coordinate and gaze telemetry through the Dwell-Time State Machine.
        """
        if not telemetry:
            self.driver_sweep_or_idle()
            return
            
        face_detected = telemetry.get("face_detected", False)
        
        if face_detected:
            self.last_success_time = time.time()
            
            # Fetch Gaze Angular Deflection
            yaw = telemetry.get("yaw", 0.0)
            pitch = telemetry.get("pitch", 0.0)
            gaze_deflection = np.sqrt(yaw**2 + pitch**2)
            
            # Fetch tracking coordinate centers
            target_x = telemetry.get("target_x", 0.5)
            target_y = telemetry.get("target_y", 0.5)
            
            # 1. Apply EMA smoothing on coordinates to limit servo jitter
            self.smooth_x = self.ema_beta * self.smooth_x + (1 - self.ema_beta) * target_x
            self.smooth_y = self.ema_beta * self.smooth_y + (1 - self.ema_beta) * target_y
            
            # 2. Check if gaze is inside our intent engagement cone
            in_cone = gaze_deflection <= self.cone_threshold
            
            # 3. State Machine Logic
            if in_cone:
                if self.state in ["IDLE", "SEARCHING"]:
                    self.state = "POTENTIAL_INTENT"
                    self.potential_start_time = time.time()
                    print("[INTENT] Potential interaction intent detected. Initiating dwell confirmation...")
                    
                elif self.state == "POTENTIAL_INTENT":
                    dwell_duration = time.time() - self.potential_start_time
                    if dwell_duration >= self.dwell_threshold:
                        self.state = "ENGAGED"
                        print(f"[INTENT] LOCK ENGAGED! Gaze stayed in cone for {round(dwell_duration, 2)}s.")
                        
            else:
                # Gaze is looking away, reset intent state
                if self.state in ["POTENTIAL_INTENT", "ENGAGED"]:
                    print(f"[INTENT] Lost intent (deflection {round(gaze_deflection, 1)} deg). Transitioning to search...")
                self.state = "SEARCHING"
                self.potential_start_time = None
                
            # 4. Action Loop based on State
            if self.state == "ENGAGED":
                # Active Locked Tracking via PID Loop
                error_x = 0.5 - self.smooth_x
                error_y = 0.5 - self.smooth_y
                
                pan_delta = self.pan_pid.compute(error_x)
                tilt_delta = self.tilt_pid.compute(error_y)
                
                p_angle, t_angle = self.driver.update_position(pan_delta, tilt_delta)
                print(f"[TRACKING] Active Servo Adjust -> Pan: {p_angle} Tilt: {t_angle} | Intent Score: {telemetry.get('intent_score', 100)}%")
                
            else:
                # Potential intent or search mode: perform standard searching sweep
                self.driver_sweep_or_idle()
                
        else:
            # No face detected
            self.driver_sweep_or_idle()

    def driver_sweep_or_idle(self):
        """
        Handles searching behavior or returning to center when no face or intent is confirmed.
        """
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.potential_start_time = None
        
        # Transition to SEARCHING or IDLE if offline for long
        if time.time() - self.last_success_time > 3.0:
            if self.state != "IDLE":
                print("[INTENT] Target lost. Servos holding in search IDLE.")
            self.state = "IDLE"
            # Return tilt gently to center
            tilt_error = 90.0 - self.driver.tilt_angle
            tilt_delta = 0.05 * tilt_error
            self.driver.set_absolute_position(self.driver.pan_angle, self.driver.tilt_angle + tilt_delta)
        else:
            self.state = "SEARCHING"
            # Slowly sweep pan horizontally
            self.sweep_angle += self.sweep_direction * self.sweep_speed
            if self.sweep_angle >= self.driver.pan_limits[1]:
                self.sweep_direction = -1
            elif self.sweep_angle <= self.driver.pan_limits[0]:
                self.sweep_direction = 1
                
            # Keep tilt centered
            tilt_error = 90.0 - self.driver.tilt_angle
            tilt_delta = 0.05 * tilt_error
            self.driver.set_absolute_position(self.sweep_angle, self.driver.tilt_angle + tilt_delta)

    def run(self):
        self.running = True
        print("[SYSTEM] Booting standalone intent detector client loop...")
        
        # Open Camera
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print(f"[FATAL] Cannot open primary camera index {self.cam_index}. Exiting.")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        
        # Short startup delay
        time.sleep(1.0)
        
        api_mode, api_url, api_timeout = self.get_api_details()
        print(f"[SYSTEM] Camera capture active ({self.cam_w}x{self.cam_h}). Target: {api_url} ({api_mode})")
        
        fps_start = time.time()
        fps_counter = 0
        
        try:
            while self.running:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to capture image frame from camera.")
                    time.sleep(0.1)
                    continue
                    
                # Mirror frame for natural visual coordination
                frame = cv2.flip(frame, 1)
                
                # Compress to JPEG
                ret, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ret:
                    print("[WARNING] JPEG compression failed.")
                    continue
                    
                jpeg_bytes = jpeg_buf.tobytes()
                
                # 2. Send image to self-hosted model
                telemetry = None
                if api_mode == "WEBSOCKET":
                    telemetry = self.send_frame_ws(api_url, jpeg_bytes)
                else:
                    telemetry = self.send_frame_http(api_url, jpeg_bytes, api_timeout)
                    
                # 3. Feed the telemetry to our local actuation/intent engine
                self.process_telemetry(telemetry)
                
                # FPS Logging
                fps_counter += 1
                if time.time() - fps_start >= 5.0:
                    avg_fps = fps_counter / (time.time() - fps_start)
                    print(f"[SYSTEM LOG] Running at {round(avg_fps, 1)} Hz | Active State: {self.state} | Servo Mode: {self.driver.mode}")
                    fps_counter = 0
                    fps_start = time.time()
                    
                # Force precise 30 FPS timing limit
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, (1.0 / 30.0) - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[SYSTEM] Termination signal received.")
        finally:
            self.running = False
            cap.release()
            self.driver.cleanup()
            if self.ws_conn:
                try:
                    self.ws_conn.close()
                except:
                    pass
            print("[SYSTEM] Camera, sockets, and hardware drivers terminated cleanly. Safe shutdown completed.")

if __name__ == "__main__":
    # Check if a different config file was supplied via CLI
    config_file = "config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        
    tracker = StandaloneIntentTracker(config_path=config_file)
    tracker.run()
