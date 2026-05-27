import cv2
import numpy as np
import mediapipe as mp
import time
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class IntentVisionEngine:
    def __init__(self, intent_threshold_deg=20.0, filter_beta=0.7):
        """
        Initializes the Intent AI Vision Engine using MediaPipe Tasks API.
        Supports dynamic profiles: TRACKING, SECURITY, HEALTH.
        """
        self.intent_threshold_deg = intent_threshold_deg
        self.filter_beta = filter_beta
        self.profile = "TRACKING"  # TRACKING, SECURITY, HEALTH
        
        # 1. Download face landmarker model if not present
        model_filename = "face_landmarker.task"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(self.model_path):
            print(f"[VISION] Modern model '{model_filename}' not found. Downloading from Google repository...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("[VISION] Model downloaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to download model: {e}")
                raise e
        
        # 2. Initialize MediaPipe Face Landmarker with Blendshapes enabled
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,  # Enable blendshape extraction
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        print("[VISION] MediaPipe Tasks FaceLandmarker with Blendshapes initialized.")
        
        # Generic 3D model points of a human head (Anthropometric 3D model)
        self.model_3d_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, -330.0, -65.0),        # Chin (index 152)
            (-225.0, 170.0, -135.0),     # Right eye outer corner (index 33)
            (225.0, 170.0, -135.0),      # Left eye outer corner (index 263)
            (-150.0, -150.0, -125.0),    # Right mouth corner (index 61)
            (150.0, -150.0, -125.0)      # Left mouth corner (index 291)
        ], dtype=np.float32)
        
        self.face_landmark_indices = [1, 152, 33, 263, 61, 291]
        
        # Smoothed state variables (Exponential Moving Average)
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.smooth_roll = 0.0
        self.smooth_target_x = 0.5
        self.smooth_target_y = 0.5
        
        # General state variables
        self.state = "IDLE"  # Toggles dynamically based on active profile
        self.last_seen_time = 0.0
        self.lost_hold_duration = 1.5
        self.intent_score = 0.0
        
        # --- Time-based / Blendshape Tracking Variables ---
        self.blink_start_time = None
        self.blink_durations = []
        self.first_detected_time = None
        
        # Mode-specific telemetry caches
        self.evasion_score = 0.0
        self.fatigue_index = 0.0
        self.distress_score = 0.0
        self.smile_score = 0.0
        self.speaking_score = 0.0
        self.wink_detected = "NONE"
        self.blink_duration_current = 0.0
        
    def estimate_head_pose(self, landmarks, width, height):
        """
        Calculates the 3D head pose (yaw, pitch, roll) from 2D landmarks.
        """
        image_points = []
        for idx in self.face_landmark_indices:
            pt = landmarks[idx]
            image_points.append((pt.x * width, pt.y * height))
        image_points = np.array(image_points, dtype=np.float32)
        
        focal_length = width
        center_x, center_y = width / 2.0, height / 2.0
        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            self.model_3d_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None, None, None
            
        rmat, _ = cv2.Rodrigues(rvec)
        
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            pitch_rad = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw_rad = np.arctan2(-rmat[2, 0], sy)
            roll_rad = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch_rad = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw_rad = np.arctan2(-rmat[2, 0], sy)
            roll_rad = 0.0
            
        pitch = np.degrees(pitch_rad)
        yaw = np.degrees(yaw_rad)
        roll = np.degrees(roll_rad)
        
        yaw = -yaw
        
        self.smooth_yaw = self.filter_beta * self.smooth_yaw + (1 - self.filter_beta) * yaw
        self.smooth_pitch = self.filter_beta * self.smooth_pitch + (1 - self.filter_beta) * pitch
        self.smooth_roll = self.filter_beta * self.smooth_roll + (1 - self.filter_beta) * roll
        
        nose_tip_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        gaze_vector_3d = np.array([[0.0, 0.0, 600.0]], dtype=np.float32)
        
        nose_2d, _ = cv2.projectPoints(nose_tip_3d, rvec, tvec, camera_matrix, dist_coeffs)
        gaze_2d, _ = cv2.projectPoints(gaze_vector_3d, rvec, tvec, camera_matrix, dist_coeffs)
        
        p1 = (int(nose_2d[0][0][0]), int(nose_2d[0][0][1]))
        p2 = (int(gaze_2d[0][0][0]), int(gaze_2d[0][0][1]))
        
        return self.smooth_yaw, self.smooth_pitch, self.smooth_roll, p1, p2

    def process_frame(self, frame):
        """
        Processes a single BGR video frame to track landmarks, calculate pose, and detect intent.
        :param frame: Standard OpenCV BGR frame.
        :return: Dict containing tracking data, and the annotated BGR frame.
        """
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process detection
        detection_result = self.detector.detect(mp_image)
        
        tracking_data = {
            "face_detected": False,
            "target_x": 0.5,
            "target_y": 0.5,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "intent_score": 0.0,
            "state": self.state,
            "fps": 0.0,
            # Multi-profile parameters
            "profile": self.profile,
            "evasion_score": round(self.evasion_score, 1),
            "fatigue_index": round(self.fatigue_index, 1),
            "distress_score": round(self.distress_score, 1),
            "smile_score": round(self.smile_score, 1),
            "speaking_score": round(self.speaking_score, 1),
            "wink_detected": self.wink_detected,
            "blink_duration": round(self.blink_duration_current, 2)
        }
        
        annotated_frame = frame.copy()
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            tracking_data["face_detected"] = True
            
            # 1. Calculate head pose
            yaw, pitch, roll, p1, p2 = self.estimate_head_pose(landmarks, width, height)
            
            if yaw is not None:
                tracking_data["yaw"] = round(yaw, 2)
                tracking_data["pitch"] = round(pitch, 2)
                tracking_data["roll"] = round(roll, 2)
                
                # Center of the face (nose tip) in normalized [0.0 - 1.0] coordinates
                raw_target_x = landmarks[1].x
                raw_target_y = landmarks[1].y
                
                # Smooth the pixel tracking coordinate
                self.smooth_target_x = self.filter_beta * self.smooth_target_x + (1 - self.filter_beta) * raw_target_x
                self.smooth_target_y = self.filter_beta * self.smooth_target_y + (1 - self.filter_beta) * raw_target_y
                
                tracking_data["target_x"] = round(self.smooth_target_x, 4)
                tracking_data["target_y"] = round(self.smooth_target_y, 4)
                
                # 2. Compute Intent Score based on head gaze deflection from camera axis
                gaze_deflection = np.sqrt(yaw**2 + pitch**2)
                if gaze_deflection <= self.intent_threshold_deg:
                    self.intent_score = max(0.0, 100.0 - (gaze_deflection / self.intent_threshold_deg) * 100.0)
                else:
                    self.intent_score = max(0.0, self.intent_score - 10.0)
                tracking_data["intent_score"] = round(self.intent_score, 1)
                
                self.last_seen_time = time.time()
                
                # 3. Extract Blendshapes for Multi-Profile Analysis
                if detection_result.face_blendshapes:
                    blendshapes = detection_result.face_blendshapes[0]
                    
                    # Fetch essential blendshape scalars
                    blink_l = next((b.score for b in blendshapes if b.category_name == "eyeBlinkLeft"), 0.0)
                    blink_r = next((b.score for b in blendshapes if b.category_name == "eyeBlinkRight"), 0.0)
                    jaw_open = next((b.score for b in blendshapes if b.category_name == "jawOpen"), 0.0)
                    brow_down_l = next((b.score for b in blendshapes if b.category_name == "browDownLeft"), 0.0)
                    brow_down_r = next((b.score for b in blendshapes if b.category_name == "browDownRight"), 0.0)
                    squint_l = next((b.score for b in blendshapes if b.category_name == "eyeSquintLeft"), 0.0)
                    squint_r = next((b.score for b in blendshapes if b.category_name == "eyeSquintRight"), 0.0)
                    smile_l = next((b.score for b in blendshapes if b.category_name == "mouthSmileLeft"), 0.0)
                    smile_r = next((b.score for b in blendshapes if b.category_name == "mouthSmileRight"), 0.0)
                    
                    # Compute blink durations
                    is_blinking = (blink_l > 0.65) and (blink_r > 0.65)
                    if is_blinking:
                        if self.blink_start_time is None:
                            self.blink_start_time = time.time()
                        self.blink_duration_current = time.time() - self.blink_start_time
                    else:
                        if self.blink_start_time is not None:
                            self.blink_durations.append(time.time() - self.blink_start_time)
                            if len(self.blink_durations) > 10:
                                self.blink_durations.pop(0)
                            self.blink_start_time = None
                        self.blink_duration_current = 0.0
                        
                    # Wink detection
                    if abs(blink_l - blink_r) > 0.55:
                        self.wink_detected = "LEFT" if blink_l > blink_r else "RIGHT"
                    else:
                        self.wink_detected = "NONE"
                        
                    # Smooth scores
                    self.smile_score = 0.8 * self.smile_score + 0.2 * ((smile_l + smile_r) / 2.0 * 100.0)
                    self.speaking_score = 0.8 * self.speaking_score + 0.2 * (jaw_open * 100.0)
                    self.distress_score = 0.8 * self.distress_score + 0.2 * ((brow_down_l + brow_down_r + squint_l + squint_r) / 4.0 * 100.0)
                
                # --- Multi-Profile State Decision Matrices ---
                if self.profile == "TRACKING":
                    # Classic Tracking States
                    if self.intent_score > 40.0:
                        self.state = "ENGAGED"
                    else:
                        self.state = "SEARCHING"
                        
                elif self.profile == "SECURITY":
                    # Security Evasion and Dwell States
                    if self.first_detected_time is None:
                        self.first_detected_time = time.time()
                        
                    dwell_time = time.time() - self.first_detected_time
                    
                    # Face is near but user is intentionally looking away from camera axis
                    x_coordinates = [pt.x * width for pt in landmarks]
                    y_coordinates = [pt.y * height for pt in landmarks]
                    x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
                    y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))
                    face_area = (x_max - x_min) * (y_max - y_min) / (width * height)
                    
                    # If looking away and face is close -> Increment Evasion
                    if gaze_deflection > 32.0 and face_area > 0.06:
                        self.evasion_score = min(100.0, self.evasion_score + 6.0)
                    else:
                        self.evasion_score = max(0.0, self.evasion_score - 3.0)
                        
                    if self.evasion_score > 50.0:
                        self.state = "EVADING"
                    elif dwell_time > 8.0 and self.intent_score < 25.0:
                        self.state = "SUSPICIOUS_DWELL"
                    else:
                        self.state = "SEC_CLEAR"
                        
                elif self.profile == "HEALTH":
                    # Biometric Health / Drowsiness States
                    # Calculate Fatigue Index
                    if self.blink_duration_current > 0.4:
                        self.fatigue_index = min(100.0, (self.blink_duration_current / 2.0) * 100.0)
                    elif jaw_open > 0.55 and (blink_l > 0.5 or blink_r > 0.5):
                        # Yawning with sleepy eyes
                        self.fatigue_index = min(100.0, self.fatigue_index + 10.0)
                    else:
                        avg_blink = sum(self.blink_durations) / max(len(self.blink_durations), 1)
                        self.fatigue_index = max(0.0, self.fatigue_index - 2.0)
                        
                    # Unresponsive: eyes closed > 3.0s or head roll tilted past 35 degrees (possible collapse)
                    is_unresponsive = (self.blink_duration_current > 3.0) or (abs(roll) > 35.0)
                    
                    if is_unresponsive:
                        self.state = "UNRESPONSIVE"
                    elif self.fatigue_index > 65.0:
                        self.state = "FATIGUED"
                    elif self.distress_score > 55.0:
                        self.state = "DISTRESS"
                    else:
                        self.state = "HLTH_NORMAL"
                
                # --- HUD Annotation Rendering ---
                # Determine colors based on states
                if self.profile == "TRACKING":
                    hud_color = (0, 240, 255) if self.state == "ENGAGED" else (255, 183, 0)
                elif self.profile == "SECURITY":
                    hud_color = (0, 0, 255) if self.state in ["EVADING", "SUSPICIOUS_DWELL"] else (255, 183, 0)
                else:  # HEALTH
                    hud_color = (0, 0, 255) if self.state == "UNRESPONSIVE" else ((0, 165, 255) if self.state in ["FATIGUED", "DISTRESS"] else (0, 255, 120))
                
                # Draw gaze vector line
                cv2.line(annotated_frame, p1, p2, hud_color, 3, cv2.LINE_AA)
                cv2.circle(annotated_frame, p1, 5, (0, 0, 255), -1)
                
                # Face brackets
                x_coordinates = [pt.x * width for pt in landmarks]
                y_coordinates = [pt.y * height for pt in landmarks]
                x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
                y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))
                
                bracket_len = 20
                thickness = 2
                # TL
                cv2.line(annotated_frame, (x_min, y_min), (x_min + bracket_len, y_min), hud_color, thickness)
                cv2.line(annotated_frame, (x_min, y_min), (x_min, y_min + bracket_len), hud_color, thickness)
                # TR
                cv2.line(annotated_frame, (x_max, y_min), (x_max - bracket_len, y_min), hud_color, thickness)
                cv2.line(annotated_frame, (x_max, y_min), (x_max, y_min + bracket_len), hud_color, thickness)
                # BL
                cv2.line(annotated_frame, (x_min, y_max), (x_min + bracket_len, y_max), hud_color, thickness)
                cv2.line(annotated_frame, (x_min, y_max), (x_min, y_max - bracket_len), hud_color, thickness)
                # BR
                cv2.line(annotated_frame, (x_max, y_max), (x_max - bracket_len, y_max), hud_color, thickness)
                cv2.line(annotated_frame, (x_max, y_max), (x_max, y_max - bracket_len), hud_color, thickness)
                
                # Bounding lock crosshair
                center_px = (int(self.smooth_target_x * width), int(self.smooth_target_y * height))
                cv2.drawMarker(annotated_frame, center_px, hud_color, cv2.MARKER_CROSS, 20, 2)
                cv2.circle(annotated_frame, center_px, 6, hud_color, 1)
        else:
            # Face not detected
            self.first_detected_time = None
            self.blink_start_time = None
            self.blink_duration_current = 0.0
            
            # Decay score values slowly
            self.intent_score = max(0.0, self.intent_score - 15.0)
            self.evasion_score = max(0.0, self.evasion_score - 8.0)
            self.fatigue_index = max(0.0, self.fatigue_index - 4.0)
            self.distress_score = max(0.0, self.distress_score - 6.0)
            
            # Handle blackout/cover tampering detection in security mode
            if self.profile == "SECURITY":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                if avg_brightness < 6.0:  # extremely dark frame -> blocked sensor
                    self.state = "TAMPERED"
                else:
                    if self.state == "TAMPERED":
                        self.state = "IDLE"
                    elif self.state in ["EVADING", "SUSPICIOUS_DWELL", "SEC_CLEAR"]:
                        self.state = "LOST"
                        self.last_seen_time = time.time()
                    elif self.state == "LOST" and (time.time() - self.last_seen_time > self.lost_hold_duration):
                        self.state = "IDLE"
            else:
                if self.state in ["ENGAGED", "SEARCHING", "UNRESPONSIVE", "FATIGUED", "DISTRESS", "HLTH_NORMAL"]:
                    self.state = "LOST"
                    self.last_seen_time = time.time()
                elif self.state == "LOST" and (time.time() - self.last_seen_time > self.lost_hold_duration):
                    self.state = "IDLE"
                    
        # Synchronize tracking dictionary values
        tracking_data["state"] = self.state
        tracking_data["intent_score"] = round(self.intent_score, 1)
        tracking_data["evasion_score"] = round(self.evasion_score, 1)
        tracking_data["fatigue_index"] = round(self.fatigue_index, 1)
        tracking_data["distress_score"] = round(self.distress_score, 1)
        tracking_data["smile_score"] = round(self.smile_score, 1)
        tracking_data["speaking_score"] = round(self.speaking_score, 1)
        tracking_data["wink_detected"] = self.wink_detected
        tracking_data["blink_duration"] = round(self.blink_duration_current, 2)
        
        # General overlay metadata on top-left of video feed
        hud_col = (0, 240, 255) if self.state == "ENGAGED" else ((255, 183, 0) if self.state in ["SEARCHING", "LOST"] else (150, 150, 150))
        if self.profile == "SECURITY":
            hud_col = (0, 0, 255) if self.state in ["EVADING", "SUSPICIOUS_DWELL", "TAMPERED"] else (0, 255, 255)
        elif self.profile == "HEALTH":
            hud_col = (0, 0, 255) if self.state == "UNRESPONSIVE" else ((0, 165, 255) if self.state in ["FATIGUED", "DISTRESS"] else (0, 255, 120))
            
        cv2.putText(annotated_frame, f"PROFILE: {self.profile}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_col, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"STATUS: {self.state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_col, 2, cv2.LINE_AA)
        
        if tracking_data["face_detected"]:
            if self.profile == "TRACKING":
                cv2.putText(annotated_frame, f"Gaze Intent: {tracking_data['intent_score']}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1, cv2.LINE_AA)
            elif self.profile == "SECURITY":
                cv2.putText(annotated_frame, f"Evasion Index: {tracking_data['evasion_score']}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1, cv2.LINE_AA)
            elif self.profile == "HEALTH":
                cv2.putText(annotated_frame, f"Fatigue Index: {tracking_data['fatigue_index']}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Distress Index: {tracking_data['distress_score']}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1, cv2.LINE_AA)
                
        return tracking_data, annotated_frame
