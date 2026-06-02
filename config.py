# =========================
# CAMERA & API STREAM
# =========================

# API connection mode: "LOCAL" (offline MediaPipe), "HTTP" (remote backend), or "WEBSOCKET"
API_MODE = "LOCAL"

# Remote self-hosted AI model server endpoint (using custom port 5050 and isolated path)
API_URL = "http://127.0.0.1:5050/api/v1/intent"
API_TIMEOUT = 2.0

# Optional API authentication key for Cloud-hosted models (e.g. Hugging Face, Roboflow, AWS)
API_KEY = ""

# Video source: Can be a remote MJPEG stream URL or a physical camera index (0, 1, etc.)
VIDEO_SOURCE = "http://127.0.0.1:8000/stream.mjpg"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

FRAME_CENTER_X = FRAME_WIDTH // 2
FRAME_CENTER_Y = FRAME_HEIGHT // 2


# =========================
# SERVO SETTINGS
# =========================

PAN_CHANNEL = 0
TILT_CHANNEL = 2

PAN_MIN = 10
PAN_MAX = 170

TILT_MIN = 10
TILT_MAX = 170

START_PAN = 90
START_TILT = 90

# Exponential moving average filter factor (EMA beta)
SERVO_SMOOTHING = 0.05
SERVO_STEP_LIMIT = 3

# PID Controller tuning gains for active tracking alignment
KP_PAN = 24.0
KI_PAN = 0.05
KD_PAN = 0.4

KP_TILT = 24.0
KI_TILT = 0.05
KD_TILT = 0.4


# =========================
# AI & INTENT SETTINGS
# =========================

DETECTION_CONFIDENCE = 0.6
FRAME_SKIP = 2

# Deadzone bounds around target center (in pixels)
DEADZONE_X = 30
DEADZONE_Y = 30

# Dwell-time state machine intent confirmation parameters
ENGAGEMENT_CONE_DEG = 22.0
DWELL_TIME_SECONDS = 0.6
