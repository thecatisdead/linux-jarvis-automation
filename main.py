import sys


sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import sounddevice as sd

import random
import time
import config
import models  
from tts import speak
from wake_word import wait_for_wake_word, detect_jarvis
from recorder import record_command
from commands import execute_command
from ws_server import start_websocket_server


import memory


def print_startup_info():
    device_info = sd.query_devices(config.INPUT_DEVICE)

    print("=" * 60)
    print("VOICE ASSISTANT")
    print("=" * 60)
    print()
    print(f"Microphone: {device_info['name']}")
    print(f"Device:     {config.INPUT_DEVICE}")
    print(f"Native Hz:  {device_info['default_samplerate']}")
    print()
    print("Assistant is running.")
    print('Say "Hey Jarvis" to wake it.')
    print("Press Ctrl+C to stop.")
    print()


def main_loop():
    while True:
        # ------------------------------------------------
        # SLEEP
        # ------------------------------------------------
        wait_for_wake_word()

        # ------------------------------------------------
        # WAKE
        # ------------------------------------------------
        print()
        print("Assistant activated.")

        responses = [
            "Yes, sir?",
            "What is it, sir?",
            "What's your command, sir?",
            "How may I assist you, sir?",
            "I'm listening, sir.",
            "At your service, sir."
        ]

        speak(random.choice(responses))

        first_command = True
        no_jarvis_count = 0
        MAX_NO_JARVIS = 1

        # =================================================
        # CONTINUOUS CONVERSATION
        # =================================================
        while True:

            # Wait up to 10 seconds for speech
            audio = record_command(wait_timeout=5)

            # ------------------------------------------------
            # NO SPEECH → GO BACK TO SLEEP
            # ------------------------------------------------
            if audio is None:
                print()
                print("No speech for 10 seconds.")
                speak("Going back to sleep, sir.")
                break

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------
            audio = audio - np.mean(audio)
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))

            print()
            print(f"RMS:  {rms:.5f}")
            print(f"Peak: {peak:.5f}")

            if rms < 0.002:
                print("Microphone level too low.")
                continue

            if rms > 1e-6:
                audio *= (0.15 / rms)

            audio = np.clip(audio, -1.0, 1.0)

            

            # ------------------------------------------------
            # TRANSCRIBE
            # ------------------------------------------------
            print()
            print("Transcribing...")
            print()

            segments, info = models.whisper_model.transcribe(
                audio,
                language="en",
                temperature=0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,

            )

            text = " ".join(
                seg.text for seg in segments
            ).strip()

            # ------------------------------------------------
            # REQUIRE "JARVIS" DURING ACTIVE CONVERSATION
            # ------------------------------------------------
            
            if first_command:
                first_command = False

            else:
                if "jarvis" not in text.lower():

                    no_jarvis_count += 1

                    print()
                    print("No 'Jarvis' detected. Ignoring command.")
                    print(
                        f"No-Jarvis attempts: "
                        f"{no_jarvis_count}/{MAX_NO_JARVIS}"
                    )
                    print()

                    if no_jarvis_count >= MAX_NO_JARVIS:
                        print("Too many failed commands.")
                        print("Going back to sleep.")

                        speak("Going back to sleep, sir.")
                        break

                    continue

                # Valid Jarvis command → reset failure counter
                no_jarvis_count = 0

                text = text.lower().replace("jarvis", "", 1).strip()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------
            print("=" * 60)
            print("YOU SAID:")
            print(text if text else "[NOTHING DETECTED]")
            print("=" * 60)

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------
            if text:
                execute_command(text)

        
            # The inner loop listens for another command.

        print()
        print("Waiting for wake word...")

if __name__ == "__main__":
    print_startup_info()

    # Initialize Jarvis memory database
    memory.initialize_memory()

    start_websocket_server()

    try:
        main_loop()

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("VOICE ASSISTANT STOPPED")
        print("=" * 60)