import asyncio
import json
import time

import numpy as np
import sounddevice as sd
import websockets


# ============================================================
# CONFIG
# ============================================================

HOST = "localhost"
PORT = 8765

# YOUR WORKING MICROPHONE
INPUT_DEVICE = 2

# Device 2 supports 44100 natively.
# 48000 was also confirmed to work.
SAMPLE_RATE = 48000

BLOCK_SIZE = 1024
CHANNELS = 1


# ============================================================
# VOICE SENSITIVITY
# ============================================================

# Your observed microphone:
#
# Talking:
#     ~0.002 - 0.004
#
# Silence:
#     ~0.00055 - 0.00085
#

NOISE_FLOOR = 0.00075

# Anything above this becomes maximum intensity.
LOUD_RMS = 0.0045

# Response curve.
# Lower = more sensitive to quiet speech.
RESPONSE_POWER = 0.55


# ============================================================
# SMOOTHING
# ============================================================

# Attack = how quickly glow becomes bright.
ATTACK = 0.65

# Release = how quickly glow fades.
RELEASE = 0.18


# ============================================================
# DEBUG
# ============================================================

DEBUG = True


# ============================================================
# CLIENTS
# ============================================================

clients = set()


# ============================================================
# STATE
# ============================================================

current_intensity = 0.0


# ============================================================
# RMS -> INTENSITY
# ============================================================

def calculate_intensity(rms):

    # Remove microphone noise floor.
    signal = rms - NOISE_FLOOR

    if signal <= 0:
        target = 0.0

    else:

        # Normalize.
        target = signal / (
            LOUD_RMS - NOISE_FLOOR
        )

        target = np.clip(
            target,
            0.0,
            1.0
        )

        # Boost quiet speech.
        target = target ** RESPONSE_POWER

    return float(
        np.clip(
            target,
            0.0,
            1.0
        )
    )


# ============================================================
# BROADCAST
# ============================================================

async def broadcast(intensity):

    if not clients:
        return

    message = json.dumps({
        "intensity": round(
            float(intensity),
            3
        )
    })

    dead = []

    for client in list(clients):

        try:

            await client.send(message)

        except Exception:

            dead.append(client)

    for client in dead:

        clients.discard(client)


# ============================================================
# WEBSOCKET HANDLER
# ============================================================

async def websocket_handler(websocket):

    clients.add(websocket)

    print(
        f"\nClient connected "
        f"({len(clients)} total)"
    )

    try:

        await websocket.wait_closed()

    except Exception:
        pass

    finally:

        clients.discard(websocket)

        print(
            f"\nClient disconnected "
            f"({len(clients)} total)"
        )


# ============================================================
# MICROPHONE
# ============================================================

async def microphone_loop():

    global current_intensity

    loop = asyncio.get_running_loop()

    audio_queue = asyncio.Queue()


    # --------------------------------------------------------
    # AUDIO CALLBACK
    # --------------------------------------------------------

    def audio_callback(
        indata,
        frames,
        time_info,
        status
    ):

        if status and DEBUG:

            print(
                f"\nAudio status: {status}"
            )

        try:

            samples = indata[:, 0]

            # RMS
            rms = float(
                np.sqrt(
                    np.mean(
                        samples * samples
                    )
                )
            )

            loop.call_soon_threadsafe(
                audio_queue.put_nowait,
                rms
            )

        except Exception as e:

            print(
                f"\nAudio callback error: {e}"
            )


    # --------------------------------------------------------
    # SHOW DEVICE
    # --------------------------------------------------------

    device = sd.query_devices(
        INPUT_DEVICE,
        "input"
    )

    print()
    print("=" * 60)
    print("MICROPHONE")
    print("=" * 60)

    print(
        "Device:",
        device["name"]
    )

    print(
        "Device index:",
        INPUT_DEVICE
    )

    print(
        "Native sample rate:",
        device["default_samplerate"]
    )

    print(
        "Using sample rate:",
        SAMPLE_RATE
    )

    print(
        "Input channels:",
        device["max_input_channels"]
    )

    print("=" * 60)
    print()


    # --------------------------------------------------------
    # OPEN INPUT STREAM
    # --------------------------------------------------------

    with sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback
    ):

        print(
            "Microphone stream OPEN"
        )

        print(
            "Speak into your microphone..."
        )

        print()


        # ----------------------------------------------------
        # PROCESS AUDIO
        # ----------------------------------------------------

        while True:

            rms = await audio_queue.get()

            target = calculate_intensity(
                rms
            )


            # -----------------------------------------------
            # SMOOTHING
            # -----------------------------------------------

            if target > current_intensity:

                current_intensity += (
                    target -
                    current_intensity
                ) * ATTACK

            else:

                current_intensity += (
                    target -
                    current_intensity
                ) * RELEASE


            current_intensity = float(
                np.clip(
                    current_intensity,
                    0.0,
                    1.0
                )
            )


            # -----------------------------------------------
            # DEBUG
            # -----------------------------------------------

            if DEBUG:

                print(
                    f"\r"
                    f"RMS: {rms:.5f}   "
                    f"Target: {target:.3f}   "
                    f"Intensity: "
                    f"{current_intensity:.3f}   "
                    f"Clients: {len(clients)}",
                    end="",
                    flush=True
                )


            # -----------------------------------------------
            # SEND TO FLUTTER
            # -----------------------------------------------

            await broadcast(
                current_intensity
            )


# ============================================================
# SERVER
# ============================================================

async def main():

    print()
    print("=" * 60)
    print("VOICE GLOW SERVER")
    print("=" * 60)

    print(
        f"WebSocket: "
        f"ws://{HOST}:{PORT}"
    )

    print(
        f"Input device: "
        f"{INPUT_DEVICE}"
    )

    print(
        f"Sample rate: "
        f"{SAMPLE_RATE}"
    )

    print(
        f"Noise floor: "
        f"{NOISE_FLOOR}"
    )

    print(
        f"Loud RMS: "
        f"{LOUD_RMS}"
    )

    print("=" * 60)
    print()


    async with websockets.serve(
        websocket_handler,
        HOST,
        PORT
    ):

        print(
            "WebSocket server running."
        )

        print(
            "Waiting for Flutter..."
        )

        print()

        await microphone_loop()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print()
        print()
        print("Server stopped.")