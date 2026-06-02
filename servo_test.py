import sys
import os
import time
import json

# Try importing Raspberry Pi hardware GPIO libraries
try:
    import pigpio
    HAS_PIGPIO = True
except ImportError:
    HAS_PIGPIO = False

try:
    from gpiozero import AngularServo
    HAS_GPIOZERO = True
except ImportError:
    HAS_GPIOZERO = False

class MG995ServoDriver:
    def __init__(self, pan_pin=18, tilt_pin=23):
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        
        self.pulse_min = 500.0
        self.pulse_max = 2500.0
        
        self.pan_limits = (10.0, 170.0)
        self.tilt_limits = (30.0, 150.0)
        
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        
        self.mode = "SIMULATION"
        self.pi = None
        self.servos = {}
        
        self.init_hardware()
        
    def init_hardware(self):
        # 1. Try PIGPIO (Hardware PWM)
        if HAS_PIGPIO:
            try:
                self.pi = pigpio.pi()
                if self.pi.connected:
                    self.mode = "PIGPIO"
                    self.pi.set_mode(self.pan_pin, pigpio.OUTPUT)
                    self.pi.set_mode(self.tilt_pin, pigpio.OUTPUT)
                    self.write_angle(self.pan_pin, self.pan_angle)
                    self.write_angle(self.tilt_pin, self.tilt_angle)
                    print(f"[TESTER] pigpio active. Connected to physical hardware servos on GPIO {self.pan_pin}/{self.tilt_pin}")
                    return
            except Exception as e:
                print(f"[TESTER] pigpio connection failed: {e}")
                
        # 2. Try GPIOZERO (Software PWM Fallback)
        if HAS_GPIOZERO:
            try:
                min_pw = self.pulse_min / 1000000.0
                max_pw = self.pulse_max / 1000000.0
                self.servos['pan'] = AngularServo(
                    self.pan_pin, min_angle=0, max_angle=180,
                    min_pulse_width=min_pw, max_pulse_width=max_pw
                )
                self.servos['tilt'] = AngularServo(
                    self.tilt_pin, min_angle=0, max_angle=180,
                    min_pulse_width=min_pw, max_pulse_width=max_pw
                )
                self.mode = "GPIOZERO"
                self.write_angle(self.pan_pin, self.pan_angle)
                self.write_angle(self.tilt_pin, self.tilt_angle)
                print(f"[TESTER] gpiozero active. Connected to physical hardware servos on GPIO {self.pan_pin}/{self.tilt_pin}")
                return
            except Exception as e:
                print(f"[TESTER] gpiozero initialization failed: {e}")
                
        # 3. Fallback to simulation
        self.mode = "SIMULATION"
        print("[TESTER] Running in Simulation Mode. All angles will be virtually logged.")

    def write_angle(self, pin, angle):
        # Cache current angles
        if pin == self.pan_pin:
            self.pan_angle = max(self.pan_limits[0], min(self.pan_limits[1], angle))
            angle = self.pan_angle
        else:
            self.tilt_angle = max(self.tilt_limits[0], min(self.tilt_limits[1], angle))
            angle = self.tilt_angle

        pulse_width = self.pulse_min + (angle / 180.0) * (self.pulse_max - self.pulse_min)
        
        if self.mode == "PIGPIO" and self.pi:
            self.pi.set_servo_pulsewidth(pin, int(pulse_width))
            print(f"[PHYSICAL WRITER] Pin {pin} -> Angle: {angle} deg | PulseWidth: {int(pulse_width)}us")
        elif self.mode == "GPIOZERO" and self.servos:
            key = 'pan' if pin == self.pan_pin else 'tilt'
            if key in self.servos:
                self.servos[key].angle = angle
                print(f"[PHYSICAL WRITER] Pin {pin} -> Angle: {angle} deg")
        else:
            print(f"[VIRTUAL WRITER] Pin {pin} -> Angle: {angle} deg (Simulation)")

    def cleanup(self):
        if self.mode == "PIGPIO" and self.pi:
            self.pi.set_servo_pulsewidth(self.pan_pin, 0)
            self.pi.set_servo_pulsewidth(self.tilt_pin, 0)
            self.pi.stop()
        elif self.mode == "GPIOZERO" and self.servos:
            for s in self.servos.values():
                s.close()
        print("[TESTER] Channels cleaned up safely.")

def load_config():
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            return cfg.get("servos", {})
    except Exception:
        return {"pan_pin": 18, "tilt_pin": 23}

def run_calibration_sweep(driver):
    print("\n=========================================")
    print("  STEP 1: AUTOMATED CALIBRATION SWEEP")
    print("=========================================")
    
    # 1. Align to center
    print("[1/4] Centering Pan and Tilt to 90 deg...")
    driver.write_angle(driver.pan_pin, 90.0)
    driver.write_angle(driver.tilt_pin, 90.0)
    time.sleep(1.0)
    
    # 2. Sweep Pan horizontal axis
    print("[2/4] Sweeping Pan (Horizontal Axis)...")
    for angle in [45.0, 90.0, 135.0, 90.0]:
        print(f"  -> Commanding Pan: {angle} deg")
        driver.write_angle(driver.pan_pin, angle)
        time.sleep(0.8)
        
    # 3. Sweep Tilt vertical axis
    print("[3/4] Sweeping Tilt (Vertical Axis)...")
    for angle in [60.0, 90.0, 120.0, 90.0]:
        print(f"  -> Commanding Tilt: {angle} deg")
        driver.write_angle(driver.tilt_pin, angle)
        time.sleep(0.8)
        
    # 4. Final Centering
    print("[4/4] Calibration complete. Servos locked at 90 deg center.")
    driver.write_angle(driver.pan_pin, 90.0)
    driver.write_angle(driver.tilt_pin, 90.0)
    time.sleep(0.5)

def interactive_terminal(driver):
    print("\n=========================================")
    print("  STEP 2: INTERACTIVE MANUAL POSITIONING")
    print("=========================================")
    print(f"Active limits: Pan {driver.pan_limits} deg, Tilt {driver.tilt_limits} deg")
    print("Instructions:")
    print("  - Type 'pan <angle>' to command horizontal position (e.g. 'pan 120')")
    print("  - Type 'tilt <angle>' to command vertical position (e.g. 'tilt 60')")
    print("  - Type 'center' to return both servos to 90 deg")
    print("  - Type 'exit' or 'quit' to terminate")
    
    while True:
        try:
            cmd = input("\nServoCMD >> ").strip().lower()
            if not cmd:
                continue
            if cmd in ["exit", "quit", "q"]:
                break
            if cmd == "center":
                driver.write_angle(driver.pan_pin, 90.0)
                driver.write_angle(driver.tilt_pin, 90.0)
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
                driver.write_angle(driver.pan_pin, val)
            elif axis == "tilt":
                driver.write_angle(driver.tilt_pin, val)
            else:
                print(f"[ERROR] Unknown axis '{axis}'. Use 'pan' or 'tilt'.")
        except KeyboardInterrupt:
            break
        except EOFError:
            # Handle non-interactive run gracefully
            print("[INFO] End of standard input. Exiting interactive console.")
            break

if __name__ == "__main__":
    print("[SYSTEM] Starting MG995 Standalone Calibration Tool...")
    servo_cfg = load_config()
    
    driver = MG995ServoDriver(
        pan_pin=servo_cfg.get("pan_pin", 18),
        tilt_pin=servo_cfg.get("tilt_pin", 23)
    )
    
    try:
        run_calibration_sweep(driver)
        interactive_terminal(driver)
    finally:
        driver.cleanup()
        print("[SYSTEM] Calibration Tool exited cleanly.")
