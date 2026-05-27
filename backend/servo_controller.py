import time
import os

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-10.0, 10.0)):
        """
        Standard Proportional-Integral-Derivative controller.
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
            
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (on error)
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        # Compute total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        min_lim, max_lim = self.output_limits
        if output < min_lim:
            output = min_lim
            # Revert integration step to prevent windup
            self.integral -= error * dt
        elif output > max_lim:
            output = max_lim
            # Revert integration step to prevent windup
            self.integral -= error * dt
            
        # Save state for next step
        self.last_error = error
        self.last_time = now
        
        return output

class MG995ServoDriver:
    def __init__(self, pan_pin=18, tilt_pin=23):
        """
        Dual-axis servo driver designed for standard MG995 servos (50Hz frequency).
        Supports pigpio (DMA hardware PWM), gpiozero (soft PWM fallback), or high-fidelity simulation.
        :param pan_pin: GPIO Pin for Pan Servo (default 18, supports hardware PWM)
        :param tilt_pin: GPIO Pin for Tilt Servo (default 23)
        """
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        
        # MG995 standard pulse widths (micro-seconds)
        # Standard: 500us = 0deg, 1500us = 90deg (center), 2500us = 180deg
        self.pulse_min = 500.0
        self.pulse_max = 2500.0
        self.pulse_center = 1500.0
        
        # Angles in degrees (0 to 180)
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        
        # Limits to prevent mechanical bind
        self.pan_limits = (10.0, 170.0)
        self.tilt_limits = (30.0, 150.0)
        
        self.mode = "SIMULATION"
        self.pi = None      # pigpio object
        self.servos = {}    # gpiozero objects
        
        self.init_hardware()
        
    def init_hardware(self):
        """
        Initializes the hardware drivers with nested fallbacks.
        """
        # 1. Try pigpio (Best: Hardware DMA-timed PWM eliminates MG995 jitter)
        try:
            import pigpio
            self.pi = pigpio.pi()
            if self.pi.connected:
                self.mode = "PIGPIO"
                # Initialize pins
                self.pi.set_mode(self.pan_pin, pigpio.OUTPUT)
                self.pi.set_mode(self.tilt_pin, pigpio.OUTPUT)
                # Send center pulse
                self.write_angle(self.pan_pin, self.pan_angle)
                self.write_angle(self.tilt_pin, self.tilt_angle)
                print(f"[SERVO] Initialized MG995 hardware on GPIO {self.pan_pin}/{self.tilt_pin} using pigpio.")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"[SERVO] pigpio daemon not running or connection failed: {e}")
            
        # 2. Try gpiozero (Fallback: Software PWM)
        try:
            from gpiozero import AngularServo
            # Convert pulse widths to fractional seconds required by gpiozero
            min_pw = self.pulse_min / 1000000.0
            max_pw = self.pulse_max / 1000000.0
            
            self.servos['pan'] = AngularServo(
                self.pan_pin, 
                min_angle=0, 
                max_angle=180, 
                min_pulse_width=min_pw, 
                max_pulse_width=max_pw
            )
            self.servos['tilt'] = AngularServo(
                self.tilt_pin, 
                min_angle=0, 
                max_angle=180, 
                min_pulse_width=min_pw, 
                max_pulse_width=max_pw
            )
            self.mode = "GPIOZERO"
            self.write_angle(self.pan_pin, self.pan_angle)
            self.write_angle(self.tilt_pin, self.tilt_angle)
            print(f"[SERVO] Initialized MG995 hardware on GPIO {self.pan_pin}/{self.tilt_pin} using gpiozero AngularServo.")
            return
        except (ImportError, Exception) as e:
            print(f"[SERVO] gpiozero/RPi.GPIO not available or failed: {e}")
            
        # 3. Fallback to simulation
        self.mode = "SIMULATION"
        print("[SERVO] Running in Simulation Mode. Virtual angles will be reported.")

    def write_angle(self, pin, angle):
        """
        Sends the pulse command corresponding to an angle (0 to 180 deg) to a given GPIO pin.
        """
        # Map 0-180 angle to pulse width (500us - 2500us)
        pulse_width = self.pulse_min + (angle / 180.0) * (self.pulse_max - self.pulse_min)
        
        if self.mode == "PIGPIO" and self.pi:
            self.pi.set_servo_pulsewidth(pin, int(pulse_width))
        elif self.mode == "GPIOZERO" and self.servos:
            key = 'pan' if pin == self.pan_pin else 'tilt'
            if key in self.servos:
                self.servos[key].angle = angle

    def update_position(self, pan_delta, tilt_delta):
        """
        Applies an incremental change to the current pan and tilt servo angles.
        :param pan_delta: Angle change for pan axis (degrees).
        :param tilt_delta: Angle change for tilt axis (degrees).
        """
        # Update Pan Axis
        self.pan_angle += pan_delta
        # Clamp to physical range
        self.pan_angle = max(self.pan_limits[0], min(self.pan_limits[1], self.pan_angle))
        self.write_angle(self.pan_pin, self.pan_angle)
        
        # Update Tilt Axis (invert depending on servo mounting direction, default is additive)
        self.tilt_angle += tilt_delta
        # Clamp to physical range
        self.tilt_angle = max(self.tilt_limits[0], min(self.tilt_limits[1], self.tilt_angle))
        self.write_angle(self.tilt_pin, self.tilt_angle)
        
        return round(self.pan_angle, 1), round(self.tilt_angle, 1)

    def set_absolute_position(self, pan_angle, tilt_angle):
        """
        Sets absolute angles manually (e.g. from sliders or joystick).
        """
        self.pan_angle = max(self.pan_limits[0], min(self.pan_limits[1], pan_angle))
        self.write_angle(self.pan_pin, self.pan_angle)
        
        self.tilt_angle = max(self.tilt_limits[0], min(self.tilt_limits[1], tilt_angle))
        self.write_angle(self.tilt_pin, self.tilt_angle)
        
        return round(self.pan_angle, 1), round(self.tilt_angle, 1)

    def cleanup(self):
        """
        Shuts down servo signals cleanly to prevent continuous load/heating when system is idle.
        """
        if self.mode == "PIGPIO" and self.pi:
            self.pi.set_servo_pulsewidth(self.pan_pin, 0) # 0 turns off PWM signals
            self.pi.set_servo_pulsewidth(self.tilt_pin, 0)
            self.pi.stop()
            print("[SERVO] pigpio PWM signals terminated cleanly.")
        elif self.mode == "GPIOZERO" and self.servos:
            for s in self.servos.values():
                s.close()
            print("[SERVO] gpiozero servos closed.")
