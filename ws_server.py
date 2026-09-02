import asyncio
import json
import threading

import websockets

import config

websocket_clients = set()
ws_loop = None


async def websocket_handler(websocket):
    websocket_clients.add(websocket)
    print(f"[WS] Flutter connected ({len(websocket_clients)} client(s))")

    try:
        await websocket.wait_closed()
    except Exception as e:
        print(f"[WS] Client error: {e}")
    finally:
        websocket_clients.discard(websocket)
        print(f"[WS] Flutter disconnected ({len(websocket_clients)} client(s))")


async def websocket_server():
    print(f"[WS] Starting WebSocket server on ws://{config.WS_HOST}:{config.WS_PORT}")

    async with websockets.serve(
        websocket_handler,
        config.WS_HOST,
        config.WS_PORT,
        reuse_address=True,
    ):
        print(f"[WS] WebSocket server running on ws://{config.WS_HOST}:{config.WS_PORT}")
        await asyncio.Future()


def start_websocket_server():
    global ws_loop

    def runner():
        global ws_loop
        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)
        ws_loop.run_until_complete(websocket_server())

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    print("[WS] WebSocket thread started.")


def broadcast_intensity(intensity):
    if not websocket_clients or ws_loop is None:
        return

    message = json.dumps({"intensity": round(float(intensity), 3)})

    async def _send():
        dead = []
        for client in list(websocket_clients):
            try:
                await client.send(message)
            except Exception:
                dead.append(client)
        for client in dead:
            websocket_clients.discard(client)

    asyncio.run_coroutine_threadsafe(_send(), ws_loop)