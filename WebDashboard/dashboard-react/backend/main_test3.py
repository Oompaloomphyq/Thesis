#!/usr/bin/env python3
"""
main_test3.py — Manual Slot Simulator for Thesis Video Recording
=================================================================
No LiDAR. No model. Manually defined slot states sent over WebSocket.

Timeline:
  t=0s  → slots 1,4,7,8,9,10,13,14,15 occupied
  t+10s → slot 3 also becomes occupied

Usage:
  python main_test3.py

Open: http://localhost:3000/demo.html
"""

import asyncio
import json
import websockets
from datetime import datetime

WS_HOST = 'localhost'
WS_PORT = 5000

ALL_SLOTS = [f'slot_{i}' for i in range(1, 17)]

# ── Initial occupied slots ──────────────────────────────────
INITIAL_OCCUPIED = {
    'slot_1', 'slot_4', 'slot_7', 'slot_8', 'slot_9',
    'slot_10', 'slot_13', 'slot_14', 'slot_15', 'slot_11', 'slot_12',
}

# ── After 10 seconds, slot_3 joins ─────────────────────────
UPDATED_OCCUPIED = (INITIAL_OCCUPIED | {'slot_3'})


def build_message(occupied_set):
    occupied_count = len(occupied_set)
    return json.dumps({
        'type':           'slot_update',
        'timestamp':      datetime.now().isoformat(),
        'slots':          [{'id': sid, 'occupied': sid in occupied_set} for sid in ALL_SLOTS],
        'moving_cars':    [],
        'occupied_count': occupied_count,
        'free_count':     16 - occupied_count,
        'mode':           'live'
    })


def build_vacant_message():
    return build_message(set())


async def ws_handler(websocket):
    print("[WS] Dashboard connected")
    try:
        # Step 1: reset frontend state
        await websocket.send(build_vacant_message())
        print("  → Sent: all vacant")

        # Step 2: wait for car.glb to load
        await asyncio.sleep(4)

        # Step 3: send initial occupied state
        await websocket.send(build_message(INITIAL_OCCUPIED))
        print(f"  → Sent: {len(INITIAL_OCCUPIED)} slots occupied")

        # Step 4: wait 10 seconds, then slot_3 arrives (no reset needed — model already loaded)
        await asyncio.sleep(10)
        await websocket.send(build_message(UPDATED_OCCUPIED))
        print("  → Sent: slot_3 now also occupied")

        # Step 5: keep connection alive
        while True:
            await websocket.send(json.dumps({'type': 'ping'}))
            await asyncio.sleep(10)

    except Exception:
        pass
    finally:
        print("[WS] Dashboard disconnected")


async def main():
    print("\n[Simulator] No LiDAR. No model. Manual slot states only.")
    print(f"[Simulator] Initial occupied: {sorted(INITIAL_OCCUPIED)}")
    print(f"[Simulator] After 10s, slot_3 joins.\n")

    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, ping_interval=None)
    print(f"[WS] Server running → ws://{WS_HOST}:{WS_PORT}")
    print(f"[Simulator] Open http://localhost:3000")
    print(f"[Simulator] Press Ctrl+C to stop.\n")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\n[Simulator] Stopped.")
        server.close()


if __name__ == "__main__":
    asyncio.run(main())
