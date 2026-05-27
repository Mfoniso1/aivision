// WebSocket and Telemetry State variables
let socket = null;
let latencyStart = 0;
const historyLimit = 150;
const telemetryHistory = {
    pan: [],
    tilt: [],
    intent: []
};

// SVG Dial Helper: Maps 0-180 angle to stroke dashoffset (251.2 max circumference)
const DIAL_CIRCUMFERENCE = 251.2;
function updateDial(dialElement, valueElement, angle) {
    const ratio = Math.max(0, Math.min(180, angle)) / 180.0;
    const offset = DIAL_CIRCUMFERENCE * (1 - ratio);
    dialElement.style.strokeDashoffset = offset;
    valueElement.textContent = `${angle.toFixed(1)}°`;
}

// UI Elements caching
const statusGlow = document.getElementById("status-glow");
const statusLabel = document.getElementById("status-label");
const fpsVal = document.getElementById("fps-val");
const cpuVal = document.getElementById("cpu-val");
const tempVal = document.getElementById("temp-val");
const platformVal = document.getElementById("platform-val");
const latencyVal = document.getElementById("latency-val");

const yawVal = document.getElementById("yaw-val");
const pitchVal = document.getElementById("pitch-val");
const rollVal = document.getElementById("roll-val");

const intentVal = document.getElementById("intent-val");
const intentProgress = document.getElementById("intent-progress");

const panVal = document.getElementById("pan-val");
const tiltVal = document.getElementById("tilt-val");
const panDial = document.getElementById("pan-dial");
const tiltDial = document.getElementById("tilt-dial");

const reticle = document.getElementById("reticle");
const viewport = document.querySelector(".feed-viewport");

// Control Inputs
const autoModeToggle = document.getElementById("auto-mode-toggle");
const sweepToggle = document.getElementById("sweep-toggle");
const sweepRow = document.getElementById("sweep-control-row");
const manualPanel = document.getElementById("manual-panel");

const panSlider = document.getElementById("pan-slider");
const tiltSlider = document.getElementById("tilt-slider");
const panSliderVal = document.getElementById("pan-slider-val");
const tiltSliderVal = document.getElementById("tilt-slider-val");

const savePidBtn = document.getElementById("save-pid-btn");

// ----------------- Canvas Waveform Engine -----------------
const canvas = document.getElementById("telemetry-chart");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function drawWaveform() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const w = canvas.width;
    const h = canvas.height;
    
    // Draw Grid Lines (Futuristic horizontal subdivisions)
    ctx.strokeStyle = "rgba(0, 240, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 20; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
    
    // Scale mapping helpers
    // Pan and Tilt are 0 to 180 degrees. Intent is 0 to 100%.
    const getPosPanTilt = (val) => h - 15 - ((val / 180) * (h - 30));
    const getPosIntent = (val) => h - 15 - ((val / 100) * (h - 30));
    
    const count = telemetryHistory.pan.length;
    if (count < 2) return;
    
    const step = w / (historyLimit - 1);
    
    // Helper to draw a timeline trace
    function drawTrace(dataArray, color, mapFunc) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.shadowColor = color;
        ctx.shadowBlur = 4;
        
        ctx.moveTo(0, mapFunc(dataArray[0]));
        for (let i = 1; i < dataArray.length; i++) {
            const x = i * step;
            const y = mapFunc(dataArray[i]);
            ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0; // reset
    }
    
    // Draw historical trace lines
    drawTrace(telemetryHistory.pan, "#00f0ff", getPosPanTilt);   // Pan trace in Cyan
    drawTrace(telemetryHistory.tilt, "#ffb700", getPosPanTilt);  // Tilt trace in Amber
    drawTrace(telemetryHistory.intent, "#00ff66", getPosIntent); // Intent trace in Green
}

// ----------------- WebSocket Pipeline -----------------
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log("[WS] Tunnel connected.");
        statusGlow.className = "status-ring glow-idle";
        statusLabel.textContent = "SYS_ONLINE";
        latencyStart = Date.now();
        // Request latency probe
        socket.send(JSON.stringify({ action: "ping" }));
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Handle Latency calculation
        if (data.action === "pong") {
            const rtt = Date.now() - latencyStart;
            latencyVal.textContent = `${rtt}ms`;
            setTimeout(() => {
                latencyStart = Date.now();
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ action: "ping" }));
                }
            }, 2000);
            return;
        }
        
        // 1. Header indicators
        fpsVal.textContent = data.fps.toFixed(1);
        cpuVal.textContent = `${data.cpu_usage}%`;
        tempVal.textContent = `${data.temperature.toFixed(1)}°C`;
        platformVal.textContent = data.platform.toUpperCase();
        
        // 2. Head pose angles
        yawVal.textContent = `${data.yaw.toFixed(2)}°`;
        pitchVal.textContent = `${data.pitch.toFixed(2)}°`;
        rollVal.textContent = `${data.roll.toFixed(2)}°`;
        
        // 3. Servo indicators
        updateDial(panDial, panVal, data.pan_angle);
        updateDial(tiltDial, tiltVal, data.tilt_angle);
        
        // Sync sliders when manually viewing auto movement
        if (data.control_mode === "AUTO") {
            panSlider.value = data.pan_angle;
            tiltSlider.value = data.tilt_angle;
            panSliderVal.textContent = `${Math.round(data.pan_angle)}°`;
            tiltSliderVal.textContent = `${Math.round(data.tilt_angle)}°`;
        }
        
        // 4. Intent Confidence & Overlays
        intentVal.textContent = `${data.intent_score.toFixed(0)}%`;
        intentProgress.style.width = `${data.intent_score}%`;
        
        // Dynamic visibility of metric panels based on active profile
        const activeProfile = data.profile || "TRACKING";
        
        // Sync active button state
        document.querySelectorAll(".profile-btn").forEach(btn => btn.classList.remove("active"));
        if (activeProfile === "TRACKING") {
            document.getElementById("profile-tracking-btn").classList.add("active");
            document.getElementById("intent-gauge-wrapper").classList.remove("hide-metric");
            document.getElementById("evasion-gauge-wrapper").classList.add("hide-metric");
            document.getElementById("health-gauges-wrapper").classList.add("hide-metric");
        } else if (activeProfile === "SECURITY") {
            document.getElementById("profile-security-btn").classList.add("active");
            document.getElementById("intent-gauge-wrapper").classList.add("hide-metric");
            document.getElementById("evasion-gauge-wrapper").classList.remove("hide-metric");
            document.getElementById("health-gauges-wrapper").classList.add("hide-metric");
        } else if (activeProfile === "HEALTH") {
            document.getElementById("profile-health-btn").classList.add("active");
            document.getElementById("intent-gauge-wrapper").classList.add("hide-metric");
            document.getElementById("evasion-gauge-wrapper").classList.add("hide-metric");
            document.getElementById("health-gauges-wrapper").classList.remove("hide-metric");
        }

        // Update security and health UI metrics
        document.getElementById("evasion-val").textContent = `${Math.round(data.evasion_score)}%`;
        document.getElementById("evasion-progress").style.width = `${data.evasion_score}%`;
        
        document.getElementById("fatigue-val").textContent = `${Math.round(data.fatigue_index)}%`;
        document.getElementById("fatigue-progress").style.width = `${data.fatigue_index}%`;
        
        document.getElementById("distress-val").textContent = `${Math.round(data.distress_score)}%`;
        document.getElementById("distress-progress").style.width = `${data.distress_score}%`;

        // Transition targeting state class and LED colors based on active profile
        viewport.className = "card-body feed-viewport"; // Reset class
        
        if (activeProfile === "TRACKING") {
            if (data.state === "ENGAGED") {
                viewport.classList.add("engaged");
                statusGlow.className = "status-ring glow-engaged";
                statusLabel.textContent = "ENGAGED";
            } else if (data.state === "SEARCHING") {
                statusGlow.className = "status-ring glow-searching";
                statusLabel.textContent = "SEARCHING";
            } else if (data.state === "LOST") {
                statusGlow.className = "status-ring glow-lost";
                statusLabel.textContent = "TARGET_LOST";
            } else {
                statusGlow.className = "status-ring glow-idle";
                statusLabel.textContent = "STANDBY";
            }
        } else if (activeProfile === "SECURITY") {
            if (data.state === "EVADING") {
                viewport.classList.add("engaged");
                statusGlow.className = "status-ring glow-lost"; // Flashing red
                statusLabel.textContent = "SEC_EVADING";
            } else if (data.state === "SUSPICIOUS_DWELL") {
                statusGlow.className = "status-ring glow-lost"; // Flashing red
                statusLabel.textContent = "SEC_SUSPICIOUS";
            } else if (data.state === "TAMPERED") {
                statusGlow.className = "status-ring glow-lost"; // Flashing red
                statusLabel.textContent = "SENSOR_TAMPERED";
            } else if (data.state === "SEC_CLEAR") {
                statusGlow.className = "status-ring glow-engaged"; // Green
                statusLabel.textContent = "SEC_CLEAR";
            } else if (data.state === "LOST") {
                statusGlow.className = "status-ring glow-searching"; // Amber
                statusLabel.textContent = "TARGET_LOST";
            } else {
                statusGlow.className = "status-ring glow-idle";
                statusLabel.textContent = "STANDBY";
            }
        } else if (activeProfile === "HEALTH") {
            if (data.state === "UNRESPONSIVE") {
                viewport.classList.add("engaged");
                statusGlow.className = "status-ring glow-lost"; // Flashing red
                statusLabel.textContent = "UNRESPONSIVE";
            } else if (data.state === "FATIGUED") {
                statusGlow.className = "status-ring glow-searching"; // Amber
                statusLabel.textContent = "FATIGUED";
            } else if (data.state === "DISTRESS") {
                statusGlow.className = "status-ring glow-searching"; // Amber
                statusLabel.textContent = "DISTRESS";
            } else if (data.state === "HLTH_NORMAL") {
                statusGlow.className = "status-ring glow-engaged"; // Green
                statusLabel.textContent = "HLTH_NORMAL";
            } else if (data.state === "LOST") {
                statusGlow.className = "status-ring glow-searching"; // Amber
                statusLabel.textContent = "TARGET_LOST";
            } else {
                statusGlow.className = "status-ring glow-idle";
                statusLabel.textContent = "STANDBY";
            }
        }
        
        // Sync toggles and settings values on first load
        autoModeToggle.checked = (data.control_mode === "AUTO");
        sweepToggle.checked = data.search_sweep;
        
        if (data.control_mode === "AUTO") {
            manualPanel.classList.add("disabled");
            sweepRow.classList.remove("disabled");
        } else {
            manualPanel.classList.remove("disabled");
            sweepRow.classList.add("disabled");
        }
        
        // Sync PID inputs with active backend constants (if sent)
        if (data.pid) {
            document.getElementById("kp-p-input").placeholder = data.pid.kp_p;
            document.getElementById("ki-p-input").placeholder = data.pid.ki_p;
            document.getElementById("kd-p-input").placeholder = data.pid.kd_p;
            document.getElementById("kp-t-input").placeholder = data.pid.kp_t;
            document.getElementById("ki-t-input").placeholder = data.pid.ki_t;
            document.getElementById("kd-t-input").placeholder = data.pid.kd_t;
        }
        
        // 5. Append historical coordinates
        telemetryHistory.pan.push(data.pan_angle);
        telemetryHistory.tilt.push(data.tilt_angle);
        
        let primaryMetric = data.intent_score;
        if (activeProfile === "SECURITY") {
            primaryMetric = data.evasion_score;
        } else if (activeProfile === "HEALTH") {
            primaryMetric = data.fatigue_index;
        }
        telemetryHistory.intent.push(primaryMetric);
        
        if (telemetryHistory.pan.length > historyLimit) {
            telemetryHistory.pan.shift();
            telemetryHistory.tilt.shift();
            telemetryHistory.intent.shift();
        }
        
        // Redraw historical canvas plot
        drawWaveform();
    };
    
    socket.onclose = () => {
        console.log("[WS] Connection lost. Retrying in 3s...");
        statusGlow.className = "status-ring glow-lost";
        statusLabel.textContent = "DISCONNECTED";
        setTimeout(connectWebSocket, 3000);
    };
}

// ----------------- Controls Toggles & Tuning callbacks -----------------
autoModeToggle.addEventListener("change", (e) => {
    const isAuto = e.target.checked;
    const mode = isAuto ? "AUTO" : "MANUAL";
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            action: "set_mode",
            mode: mode
        }));
    }
    
    if (isAuto) {
        manualPanel.classList.add("disabled");
        sweepRow.classList.remove("disabled");
    } else {
        manualPanel.classList.remove("disabled");
        sweepRow.classList.add("disabled");
    }
});

sweepToggle.addEventListener("change", (e) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            action: "set_sweep",
            enabled: e.target.checked
        }));
    }
});

// Fine manual Slider controls
function sendManualSliderPosition() {
    if (socket && socket.readyState === WebSocket.OPEN && autoModeToggle.checked === false) {
        const panVal = parseFloat(panSlider.value);
        const tiltVal = parseFloat(tiltSlider.value);
        
        panSliderVal.textContent = `${Math.round(panVal)}°`;
        tiltSliderVal.textContent = `${Math.round(tiltVal)}°`;
        
        socket.send(JSON.stringify({
            action: "manual_set",
            pan: panVal,
            tilt: tiltVal
        }));
    }
}

panSlider.addEventListener("input", sendManualSliderPosition);
tiltSlider.addEventListener("input", sendManualSliderPosition);

// Save and Send PID Parameters
savePidBtn.addEventListener("click", () => {
    const kp_p = parseFloat(document.getElementById("kp-p-input").value);
    const ki_p = parseFloat(document.getElementById("ki-p-input").value);
    const kd_p = parseFloat(document.getElementById("kd-p-input").value);
    
    const kp_t = parseFloat(document.getElementById("kp-t-input").value);
    const ki_t = parseFloat(document.getElementById("ki-t-input").value);
    const kd_t = parseFloat(document.getElementById("kd-t-input").value);
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            action: "update_pid",
            kp_p: kp_p, ki_p: ki_p, kd_p: kd_p,
            kp_t: kp_t, ki_t: ki_t, kd_t: kd_t
        }));
        
        // Visual trigger click animation feedback
        savePidBtn.textContent = "COEFFS COMMITTED!";
        savePidBtn.style.borderColor = "var(--neon-green)";
        savePidBtn.style.color = "var(--neon-green)";
        
        setTimeout(() => {
            savePidBtn.textContent = "COMMIT PID COEFFS";
            savePidBtn.style.borderColor = "var(--neon-cyan)";
            savePidBtn.style.color = "var(--neon-cyan)";
        }, 1500);
    }
});

// ----------------- Virtual Precision Joystick logic -----------------
const joystick = document.getElementById("joystick");
const joystickBoundary = document.querySelector(".joystick-boundary");

let joystickActive = false;
let joystickStartX = 0;
let joystickStartY = 0;
let joystickTimer = null;

// Joystick speed factors (adjust scale for manual tracking velocity)
const speedFactor = 1.6;

function handleJoystickMove(clientX, clientY) {
    if (!joystickActive) return;
    
    const rect = joystickBoundary.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    let dx = clientX - centerX;
    let dy = clientY - centerY;
    
    // Clamp distance inside circular boundary radius (55px boundary minus knob radius)
    const maxRadius = 40;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > maxRadius) {
        dx = (dx / distance) * maxRadius;
        dy = (dy / distance) * maxRadius;
    }
    
    // Render knob position physically
    joystick.style.transform = `translate(${dx}px, ${dy}px)`;
    
    // Translate relative vectors to step inputs (-1.0 to 1.0)
    const normalizedDx = dx / maxRadius;
    const normalizedDy = dy / maxRadius;
    
    // Run high speed step updates if active
    if (!joystickTimer) {
        joystickTimer = setInterval(() => {
            if (socket && socket.readyState === WebSocket.OPEN && autoModeToggle.checked === false) {
                // Reverse coordinates to match panning/tilting directions
                const panStep = -normalizedDx * speedFactor;
                const tiltStep = -normalizedDy * speedFactor;
                
                socket.send(JSON.stringify({
                    action: "manual_step",
                    pan_delta: panStep,
                    tilt_delta: tiltStep
                }));
            }
        }, 50); // 20Hz step updates
    }
}

function stopJoystick() {
    if (!joystickActive) return;
    joystickActive = false;
    joystick.style.transform = "translate(0px, 0px)";
    joystick.style.transition = "transform 0.2s ease-out";
    
    if (joystickTimer) {
        clearInterval(joystickTimer);
        joystickTimer = null;
    }
}

// Mouse events
joystick.addEventListener("mousedown", (e) => {
    joystickActive = true;
    joystick.style.transition = "none";
    handleJoystickMove(e.clientX, e.clientY);
});

window.addEventListener("mousemove", (e) => {
    if (joystickActive) {
        handleJoystickMove(e.clientX, e.clientY);
    }
});

window.addEventListener("mouseup", stopJoystick);

// Touch events for mobile screens
joystick.addEventListener("touchstart", (e) => {
    joystickActive = true;
    joystick.style.transition = "none";
    const touch = e.touches[0];
    handleJoystickMove(touch.clientX, touch.clientY);
});

window.addEventListener("touchmove", (e) => {
    if (joystickActive) {
        const touch = e.touches[0];
        handleJoystickMove(touch.clientX, touch.clientY);
    }
});

window.addEventListener("touchend", stopJoystick);

// Profile selection handlers
document.getElementById("profile-tracking-btn").addEventListener("click", () => setVisionProfile("TRACKING"));
document.getElementById("profile-security-btn").addEventListener("click", () => setVisionProfile("SECURITY"));
document.getElementById("profile-health-btn").addEventListener("click", () => setVisionProfile("HEALTH"));

function setVisionProfile(profileName) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            action: "set_profile",
            profile: profileName
        }));
    }
}

// Initialize Socket Connection
connectWebSocket();
