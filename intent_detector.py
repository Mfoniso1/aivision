import time
import os
import sys
import json
import threading
import cv2
import numpy as np
import requests
import config

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

# Try importing Google MediaPipe for local mode
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

class LocalGazeEstimator:
    def __init__(self):
        if not HAS_MEDIAPIPE:
            raise ImportError(
                "[ERROR] mediapipe is not installed. To run in 'LOCAL' mode, "
                "please run 'pip install mediapipe' on your system."
            )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Critical: enables iris center tracking (points 468 and 473)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Processes BGR frame, calculates facial landmarks, and computes gaze yaw/pitch.
        """
        # Convert BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Coordinates for gaze tracking (Left Eye outer 33, inner 133, pupil 468)
        l_outer = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        l_inner = np.array([landmarks[133].x, landmarks[133].y, landmarks[133].z])
        l_eye_center = (l_outer + l_inner) / 2.0
        l_iris = np.array([landmarks[468].x, landmarks[468].y, landmarks[468].z])
        
        # Coordinates for Right Eye (outer 263, inner 362, pupil 473)
        r_outer = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
        r_inner = np.array([landmarks[362].x, landmarks[362].y, landmarks[362].z])
        r_eye_center = (r_outer + r_inner) / 2.0
        r_iris = np.array([landmarks[473].x, landmarks[473].y, landmarks[473].z])
        
        # Calculate eye width dimensions as normalizers
        l_width = np.linalg.norm(l_outer - l_inner)
        r_width = np.linalg.norm(r_outer - r_inner)
        
        # Compute horizontal and vertical pupil displacements
        # Normalize by eye width to stay invariant to distance from camera
        l_gaze_x = (l_iris[0] - l_eye_center[0]) / (l_width + 1e-6)
        l_gaze_y = (l_iris[1] - l_eye_center[1]) / (l_width * 0.55 + 1e-6)
        
        r_gaze_x = (r_iris[0] - r_eye_center[0]) / (r_width + 1e-6)
        r_gaze_y = (r_iris[1] - r_eye_center[1]) / (r_width * 0.55 + 1e-6)
        
        # Average left & right gaze vectors
        gaze_x = (l_gaze_x + r_gaze_x) / 2.0
        gaze_y = (l_gaze_y + r_gaze_y) / 2.0
        
        # Map pupil displacement coordinates to approximate deflection degrees
        yaw_deg = -gaze_x * 45.0
        pitch_deg = -gaze_y * 45.0
        
        # Compute face center (normalized coords, 0.0 to 1.0)
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        target_x = np.mean(x_coords)
        target_y = np.mean(y_coords)
        
        # Compute intent score (100% when looking straight, drops as gaze deflects)
        gaze_deflection = np.sqrt(yaw_deg**2 + pitch_deg**2)
        intent_score = max(0, min(100, int(100 * (1.0 - (gaze_deflection / 45.0)))))
        
        return {
            "face_detected": True,
            "target_x": target_x,
            "target_y": target_y,
            "yaw": float(yaw_deg),
            "pitch": float(pitch_deg),
            "intent_score": intent_score
        }

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
    def __init__(self, pan_channel=None, tilt_channel=None):
        """
        Dual-axis servo controller designed for physical MG995 servos.
        Uses Adafruit PCA9685 16-Channel I2C driver (ServoKit) on Pi.
        Falls back to high-fidelity console SIMULATION on non-Pi platforms.
        """
        self.pan_channel = pan_channel if pan_channel is not None else config.PAN_CHANNEL
        self.tilt_channel = tilt_channel if tilt_channel is not None else config.TILT_CHANNEL
        
        # Read ranges and startup angles directly from config.py
        self.pan_limits = (config.PAN_MIN, config.PAN_MAX)
        self.tilt_limits = (config.TILT_MIN, config.TILT_MAX)
        
        self.pan_angle = config.START_PAN
        self.tilt_angle = config.START_TILT
        
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
    def __init__(self):
        # Initialize Actuation using direct python config values
        self.driver = MG995ServoDriver()
        
        # Output step limits constrained by config.SERVO_STEP_LIMIT
        self.pan_pid = PIDController(
            config.KP_PAN,
            config.KI_PAN,
            config.KD_PAN,
            output_limits=(-config.SERVO_STEP_LIMIT, config.SERVO_STEP_LIMIT)
        )
        self.tilt_pid = PIDController(
            config.KP_TILT,
            config.KI_TILT,
            config.KD_TILT,
            output_limits=(-config.SERVO_STEP_LIMIT, config.SERVO_STEP_LIMIT)
        )
        
        # Intent Dwell State Machine Variables from config
        self.cone_threshold = config.ENGAGEMENT_CONE_DEG
        self.dwell_threshold = config.DWELL_TIME_SECONDS
        self.ema_beta = config.SERVO_SMOOTHING
        
        self.state = "IDLE"  # IDLE, SEARCHING, POTENTIAL_INTENT, ENGAGED
        self.potential_start_time = None
        self.last_success_time = time.time()
        
        # EMA smoothed coordinates
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        
        # Camera Sweep Parameters
        self.sweep_angle = config.START_PAN
        self.sweep_direction = 1
        self.sweep_speed = 0.8
        
        # Video Capture Setup
        self.video_source = config.VIDEO_SOURCE
        self.cam_w = config.FRAME_WIDTH
        self.cam_h = config.FRAME_HEIGHT
        self.jpeg_quality = config.JPEG_QUALITY
        
        # Local offline gaze mesh engine
        self.local_gaze = None
        if config.API_MODE == "LOCAL":
            print("[SYSTEM] Initializing Google MediaPipe Face Mesh locally...")
            self.local_gaze = LocalGazeEstimator()
            print("[SYSTEM] Local MediaPipe face mesh loaded successfully.")
        
        # Threading and Loop parameters
        self.running = False
        self.ws_conn = None

    def get_api_details(self):
        return config.API_MODE, config.API_URL, config.API_TIMEOUT

    def send_frame_http(self, url, jpeg_bytes, timeout):
        """
        Sends frame using HTTP multipart POST. Supports secure API keys.
        """
        try:
            files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
            headers = {}
            api_key = getattr(config, "API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                headers["X-API-Key"] = api_key
                
            response = requests.post(url, files=files, headers=headers, timeout=timeout)
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
                headers = []
                api_key = getattr(config, "API_KEY", "")
                if api_key:
                    headers.append(f"Authorization: Bearer {api_key}")
                    headers.append(f"X-API-Key: {api_key}")
                self.ws_conn = websocket.create_connection(url, timeout=2.0, header=headers)
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
                    print("[INTENT] Potential interaction intent detected. Confirming dwell time...")
                    
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
                
                # Apply deadzone (convert normalized errors to pixel dimensions)
                pixel_error_x = error_x * config.FRAME_WIDTH
                pixel_error_y = error_y * config.FRAME_HEIGHT
                
                if abs(pixel_error_x) < config.DEADZONE_X:
                    error_x = 0.0
                if abs(pixel_error_y) < config.DEADZONE_Y:
                    error_y = 0.0
                
                pan_delta = self.pan_pid.compute(error_x)
                tilt_delta = self.tilt_pid.compute(error_y)
                
                p_angle, t_angle = self.driver.update_position(pan_delta, tilt_delta)
                print(f"[TRACKING] Servo Adjust -> Pan: {p_angle} Tilt: {t_angle} | Intent Score: {telemetry.get('intent_score', 100)}%")
                
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
            # Return tilt gently to startup center
            tilt_error = config.START_TILT - self.driver.tilt_angle
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
            tilt_error = config.START_TILT - self.driver.tilt_angle
            tilt_delta = 0.05 * tilt_error
            self.driver.set_absolute_position(self.sweep_angle, self.driver.tilt_angle + tilt_delta)

    def run(self):
        self.running = True
        print("[SYSTEM] Booting standalone intent detector client loop...")
        
        # Parse video source (support remote stream URLs or integer camera indices)
        source = self.video_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        print(f"[SYSTEM] Attempting to open video source: {source}...")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"[FATAL] Cannot open video source: {source}. Exiting.")
            return
            
        print(f"[SYSTEM] Successfully opened video source: {source}")
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
                
                # 2. Process image (natively in-memory or over network)
                telemetry = None
                if api_mode == "LOCAL":
                    if self.local_gaze:
                        telemetry = self.local_gaze.process_frame(frame)
                elif api_mode == "WEBSOCKET":
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
    tracker = StandaloneIntentTracker()
    tracker.run()
