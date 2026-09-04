from faster_whisper import WhisperModel
from piper import PiperVoice

import config

print("Loading Whisper...")

whisper_model = WhisperModel(
    "small.en",
    device="cpu",          # change to "cuda" if you have an NVIDIA GPU
    compute_type="int8",   # use "float16" if device="cuda"
)

print("Whisper loaded.")
print()

print("Loading voice...")

piper_voice = PiperVoice.load(config.PIPER_VOICE_PATH)

print("Voice loaded.")