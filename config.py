import os
import openwakeword
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NASA_API_KEY = os.environ.get("NASA_API_KEY", "YOUR_NASA_KEY_HERE")

# ============================================================
# AUDIO SETTINGS
# ============================================================

INPUT_DEVICE = "default"
MIC_SAMPLE_RATE = 16000
ASSISTANT_SAMPLE_RATE = 16000

VAD_FRAME_MS = 30
MIC_FRAME_SAMPLES = int(MIC_SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_FRAME_SAMPLES = int(ASSISTANT_SAMPLE_RATE * VAD_FRAME_MS / 1000)

SILENCE_DURATION = 0.8
MAX_RECORDING_SECONDS = 8

WAKE_THRESHOLD = 0.5

# ============================================================
# PATHS
# ============================================================

WAKE_WORD_MODEL = os.path.join(
    os.path.dirname(openwakeword.__file__),
    "resources",
    "models",
    "hey_jarvis_v0.1.onnx",
)

# NOTE: move this to your own machine's actual path, or better,
# make it relative to this project folder.
PIPER_VOICE_PATH = "/home/a_c/Downloads/piper_voices/en_GB-alan-medium.onnx"

# ============================================================
# WEBSOCKET
# ============================================================

WS_HOST = "localhost"
WS_PORT = 8765

# ============================================================
# VOICE INTENSITY (for UI glow)
# ============================================================

NOISE_FLOOR = 0.003
LOUD_RMS = 0.03
RESPONSE_POWER = 0.55

# ============================================================
# WEATHER DEFAULT LOCATION
# ============================================================

DEFAULT_LAT = 8.4822
DEFAULT_LON = 124.6472
DEFAULT_CITY_NAME = "Cagayan de Oro"

# ============================================================
# GEMINI MODEL FALLBACK CHAIN
# ============================================================

GEMINI_MODEL_FALLBACKS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]