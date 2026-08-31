import numpy as np
import sounddevice as sd
from openwakeword.model import Model

import config


def wait_for_wake_word():
    print()
    print("=" * 60)
    print("WAITING FOR WAKE WORD")
    print("=" * 60)
    print('Say: "Hey Jarvis"')
    print()

    model = Model(
        wakeword_model_paths=[config.WAKE_WORD_MODEL]
    )

    with sd.InputStream(
        device=config.INPUT_DEVICE,
        samplerate=config.MIC_SAMPLE_RATE,
        blocksize=80 * config.MIC_SAMPLE_RATE // 1000,
        channels=1,
        dtype="float32",
        latency="high",
    ) as stream:

        while True:

            audio, overflowed = stream.read(
                80 * config.MIC_SAMPLE_RATE // 1000
            )

            if overflowed:
                print("Audio overflow")

            audio = np.squeeze(audio)

            audio = (
                audio
                if audio.ndim == 1
                else audio.mean(axis=1)
            )

            audio_int16 = (
                np.clip(audio, -1.0, 1.0) * 32767
            ).astype(np.int16)

            prediction = model.predict(audio_int16)

            for name, score in prediction.items():

                if score >= config.WAKE_THRESHOLD:

                    print()
                    print("=" * 60)
                    print("WAKE WORD DETECTED!")
                    print("=" * 60)
                    print(f"{name}: {score:.3f}")

                    model.reset()
                    return

                    # -----------------------------------------
                    # WAKE WORD DETECTED
                    # Keep listening for the rest of the
                    # sentence.
                    #
                    # Example:
                    # "Hey Jarvis, open Brave"
                    # -----------------------------------------

                    # for _ in range(
                    #     int(3 * 1000 / config.VAD_FRAME_MS)
                    # ):

                    #     command_audio, overflowed = stream.read(
                    #         config.MIC_FRAME_SAMPLES
                    #     )

                    #     if overflowed:
                    #         print("Audio overflow")

                    #     command_audio = np.squeeze(
                    #         command_audio
                    #     )

                    #     if command_audio.ndim != 1:
                    #         command_audio = command_audio.mean(
                    #             axis=1
                    #         )

                    #     audio_chunks.append(
                    #         command_audio.copy()
                    #     )

                    # if audio_chunks:

                    #     return np.concatenate(
                    #         audio_chunks
                    #     )

                    # return None

                
def detect_jarvis(audio):
    model = Model(
        wakeword_model_paths=[config.WAKE_WORD_MODEL]
    )

    audio = np.squeeze(audio)

    if audio.ndim != 1:
        audio = audio.mean(axis=1)

    audio_int16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    prediction = model.predict(audio_int16)

    for name, score in prediction.items():

        if score >= config.WAKE_THRESHOLD:

            print()
            print(
                f"Jarvis detected: {score:.3f}"
            )

            model.reset()

            return True

    return False
               