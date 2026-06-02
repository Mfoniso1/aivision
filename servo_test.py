import sys
import os
import time
import json

# Try importing Adafruit PCA9685 ServoKit
try:
    from adafruit_servokit import ServoKit
    HAS_SERVOKIT = True
except ImportError:
    HAS_SERVOKIT = False

class MG995ServoDriver:
    def __init__(self, pan_channel=0, tilt_channel=2):
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        
        # Matches your exact hardware limits (0 to 180 deg)
        self.pan_limits = (0.0, 180.0)
        self.tilt_limits = (0.0, 180.0)
        
        # Matches your exact startup positions from sshkeyboard code
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
                # Move to startup positions
                self.write_angle(self.pan_channel, self.pan_angle)
                self.write_angle(self.tilt_channel, self.tilt_angle)
                print(f"[TESTER] Adafruit PCA9685 active. Connected to channels: Pan={self.pan_channel}, Tilt={self.tilt_channel}")
                return
            except Exception as e:
                print(f"[TESTER] Adafruit PCA9685 initialization failed: {e}")
                
        self.mode = "SIMULATION"
        print("[TESTER] Running in Simulation Mode. All angles will be virtually logged.")

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
                print(f"[PHYSICAL WRITER] Channel {channel} -> Angle: {target_angle} deg")
            except Exception as e:
                print(f"[SERVO ERROR] Failed to write angle {target_angle} to channel {channel}: {e}")
        else:
            print(f"[VIRTUAL WRITER] Channel {channel} -> Angle: {target_angle} deg (Simulation)")

    def cleanup(self):
        print("[TESTER] PCA9685 connection closed cleanly.")

def load_config():
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            return cfg.get("servos", {})
    except Exception:
        return {"pan_channel": 0, "tilt_channel": 2}

def run_calibration_sweep(driver):
    print("\n=========================================")
    print("  STEP 1: AUTOMATED CALIBRATION SWEEP")
    print("=========================================")
    
    # 1. Align to center
    print("[1/4] Centering Pan to 180 deg and Tilt to 90 deg...")
    driver.write_angle(driver.pan_channel, 180.0)
    driver.write_angle(driver.tilt_channel, 90.0)
    time.sleep(1.0)
    
    # 2. Sweep Pan horizontal axis
    print("[2/4] Sweeping Pan (Horizontal Axis)...")
    for angle in [135.0, 90.0, 45.0, 180.0]:
        print(f"  -> Commanding Pan: {angle} deg")
        driver.write_angle(driver.pan_channel, angle)
        time.sleep(0.8)
        
    # 3. Sweep Tilt vertical axis
    print("[3/4] Sweeping Tilt (Vertical Axis)...")
    for angle in [60.0, 90.0, 120.0, 90.0]:
        print(f"  -> Commanding Tilt: {angle} deg")
        driver.write_angle(driver.tilt_channel, angle)
        time.sleep(0.8)
        
    # 4. Final Centering
    print("[4/4] Calibration complete. Servos locked at startup positions.")
    driver.write_angle(driver.pan_channel, 180.0)
    driver.write_angle(driver.tilt_channel, 90.0)
    time.sleep(0.5)

def interactive_terminal(driver):
    print("\n=========================================")
    print("  STEP 2: INTERACTIVE MANUAL POSITIONING")
    print("=========================================")
    print(f"Active limits: Pan {driver.pan_limits} deg, Tilt {driver.tilt_limits} deg")
    print("Instructions:")
    print("  - Type 'pan <angle>' to command horizontal position (e.g. 'pan 120')")
    print("  - Type 'tilt <angle>' to command vertical position (e.g. 'tilt 60')")
    print("  - Type 'center' to return to Pan 180 deg, Tilt 90 deg")
    print("  - Type 'exit' or 'quit' to terminate")
    
    while True:
        try:
            cmd = input("\nServoCMD >> ").strip().lower()
            if not cmd:
                continue
            if cmd in ["exit", "quit", "q"]:
                break
            if cmd == "center":
                driver.write_angle(driver.pan_channel, 180.0)
                driver.write_angle(driver.tilt_channel, 90.0)
                continue
                
            parts = cmd.split()
            if len(parts) != 2:
                print("[ERROR] Invalid format. Use 'pan <0-180>' or 'tilt <0-180>'.")
                continue
                
            axis, val_str = parts[0], parts[1]
            try:
                val = float(val_str)
            except ValueError:
                print("[ERROR] Angle must be a valid number.")
                continue
                
            if axis == "pan":
                driver.write_angle(driver.pan_channel, val)
            elif axis == "tilt":
                driver.write_angle(driver.tilt_channel, val)
            else:
                print(f"[ERROR] Unknown axis '{axis}'. Use 'pan' or 'tilt'.")
        except KeyboardInterrupt:
            break
        except EOFError:
            print("[INFO] End of standard input. Exiting interactive console.")
            break

if __name__ == "__main__":
    print("[SYSTEM] Starting MG995 Standalone Calibration Tool...")
    servo_cfg = load_config()
    
    driver = MG995ServoDriver(
        pan_channel=servo_cfg.get("pan_channel", 0),
        tilt_channel=servo_cfg.get("tilt_channel", 2)
    )
    
    try:
        run_calibration_sweep(driver)
        interactive_terminal(driver)
    finally:
        driver.cleanup()
        print("[SYSTEM] Calibration Tool exited cleanly.")
