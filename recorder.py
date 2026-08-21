import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad

import config
from audio_utils import calculate_intensity
from ws_server import broadcast_intensity

vad = webrtcvad.Vad(2)


def record_command(wait_timeout=None):
    print()
    print("=" * 60)
    print("GET READY...")
    print("=" * 60)

    # time.sleep(0.5)

    print()
    print("🎤 LISTENING...")
    print()

    silence_limit = int(config.SILENCE_DURATION * 1000 / config.VAD_FRAME_MS)

    max_frames = int(config.MAX_RECORDING_SECONDS * 1000 / config.VAD_FRAME_MS)

    if wait_timeout is not None:
        wait_frames = int(wait_timeout * 1000 / config.VAD_FRAME_MS)
    else:
        wait_frames = max_frames

    pre_speech_ms = 500
    pre_speech_frames = int(pre_speech_ms / config.VAD_FRAME_MS)

    pre_speech_buffer = deque(maxlen=pre_speech_frames)

    audio_chunks = []

    speech_started = False
    silence_frames = 0
    wait_frames_count = 0

    # Require several consecutive VAD-positive frames
    # before deciding that actual speech has started.
    speech_confirmation_frames = 0
    REQUIRED_SPEECH_FRAMES = 5

    with sd.InputStream(
        device=config.INPUT_DEVICE,
        samplerate=config.MIC_SAMPLE_RATE,
        blocksize=80 * config.MIC_SAMPLE_RATE // 1000,
        channels=1,
        dtype="float32",
        latency="high",
    ) as stream:

        for _ in range(max_frames):

            audio, overflowed = stream.read(config.MIC_FRAME_SAMPLES)

            if overflowed:
                print("Audio overflow")

            audio = np.squeeze(audio)

            if audio.ndim != 1:
                audio = audio.mean(axis=1)

            # ---------------------------------------------
            # MICROPHONE INTENSITY
            # ---------------------------------------------
            frame_rms = float(np.sqrt(np.mean(np.square(audio))))

            broadcast_intensity(calculate_intensity(frame_rms))

            # ---------------------------------------------
            # KEEP PRE-SPEECH AUDIO
            # ---------------------------------------------
            if not speech_started:
                pre_speech_buffer.append(audio.copy())

            # ---------------------------------------------
            # VAD
            # ---------------------------------------------
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            is_speech = vad.is_speech(
                audio_int16.tobytes(),
                config.ASSISTANT_SAMPLE_RATE,
            )

            # ---------------------------------------------
            # SPEECH DETECTED
            # ---------------------------------------------

            # if is_speech

            if is_speech and frame_rms >= 0.002:

                if not speech_started:

                    speech_confirmation_frames += 1

                    # Don't start recording from a single
                    # random VAD-positive frame.
                    if speech_confirmation_frames >= REQUIRED_SPEECH_FRAMES:

                        print("Speech detected.")

                        speech_started = True
                        silence_frames = 0

                        # Include the audio immediately before
                        # speech started.
                        audio_chunks.extend(list(pre_speech_buffer))

                        pre_speech_buffer.clear()

                        audio_chunks.append(audio.copy())

                else:

                    silence_frames = 0
                    audio_chunks.append(audio.copy())

            # ---------------------------------------------
            # NOT SPEECH
            # ---------------------------------------------
            else:

                if not speech_started:

                    # Speech was not continuous enough.
                    # Reset the confirmation counter.
                    speech_confirmation_frames = 0

                    wait_frames_count += 1

                    if wait_frames_count >= wait_frames:

                        print(f"No speech detected for " f"{wait_timeout} seconds.")

                        break

                else:

                    # Speech has already started, so this is
                    # silence inside/after the sentence.
                    audio_chunks.append(audio.copy())

                    silence_frames += 1

                    if silence_frames >= silence_limit:

                        print("Speech finished.")
                        break

    # ---------------------------------------------
    # NO SPEECH
    # ---------------------------------------------
    if not speech_started:

        broadcast_intensity(0.0)
        return None

    # ---------------------------------------------
    # RETURN AUDIO
    # ---------------------------------------------
    audio = np.concatenate(audio_chunks)

    broadcast_intensity(0.0)

    return audio
