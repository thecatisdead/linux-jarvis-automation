import numpy as np
import sounddevice as sd

import models
from audio_utils import calculate_intensity
from ws_server import broadcast_intensity

from piper import SynthesisConfig
import random





syn_config = SynthesisConfig(
    length_scale=0.95,
    noise_scale=0.8,
    noise_w_scale=0.9,   # was noise_w — wrong name
)
def speak(text):
    print(f"Jarvis: {text}")

    for audio_chunk in models.piper_voice.synthesize(text, syn_config=syn_config):

        audio_np = np.frombuffer(
            audio_chunk.audio_int16_bytes,
            dtype=np.int16,
        ).astype(np.float32) / 32767.0

        chunk_rms = float(np.sqrt(np.mean(np.square(audio_np))))
        broadcast_intensity(calculate_intensity(chunk_rms))

        sd.play(audio_np, samplerate=audio_chunk.sample_rate)
        sd.wait()

    broadcast_intensity(0.0)