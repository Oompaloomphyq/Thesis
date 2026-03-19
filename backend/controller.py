#!/usr/bin/env python3
# current controller.py
"""
controller.py — Smart Parking System Controller
=================================================
Sits between main.py (detection) and the dashboard (Three.js).

What it does every frame:
  1. Receives slot_status + moving_cars from main.py
  2. Compares with previous frame → detects arrivals and departures
  3. Calls db_api.car_arrived(slot_id) for newly occupied slots
  4. Calls db_api.car_left(slot_id)    for newly vacated slots
  5. Logs raw LiDAR detection to lidar_detections table
  6. Broadcasts WebSocket message → dashboard updates in real time
     including moving car positions for smooth animation

WebSocket message format:
  {
    "type":           "slot_update",
    "timestamp":      "2026-03-08T10:00:00",
    "slots":          [{"id": "slot_1", "occupied": true}, ...],
    "moving_cars":    [{"id": "mv_0", "scene_x": 2.5, "scene_z": 0.5}, ...],
    "occupied_count": 5,
    "free_count":     11,
    "mode":           "live"
  }
"""

import asyncio
import json
import websockets
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import db_api

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
WS_HOST   = 'localhost'
WS_PORT   = 5000
ALL_SLOTS = [f'slot_{i}' for i in range(1, 17)]

# ═══════════════════════════════════════════════════════════
# COORDINATE MAPPING  (LiDAR KITTI → Tine's 3D scene)
# Derived from calibrated slot centers:
#   scene_x = 0.7019 * lidar_y + (-0.1109)
#   scene_z = 0.5 (fixed — driving lane between two rows)
# ═══════════════════════════════════════════════════════════
SCENE_X_SCALE  =  0.7019
SCENE_X_OFFSET = -0.1109
SCENE_Z_SCALE  = -0.547
SCENE_Z_OFFSET =  3.65

def lidar_to_scene(lidar_x, lidar_y):
    """
    Convert LiDAR KITTI coordinates to Tine's 3D scene coordinates.
    Derived from calibrated slot centers:
      scene_x = 0.7019 * lidar_y - 0.1109
      scene_z = -0.547 * lidar_x + 3.65
    """
    scene_x = SCENE_X_SCALE * lidar_y + SCENE_X_OFFSET
    scene_z = SCENE_Z_SCALE * lidar_x + SCENE_Z_OFFSET
    return round(scene_x, 3), round(scene_z, 3)


# ═══════════════════════════════════════════════════════════
# CONTROLLER CLASS
# ═══════════════════════════════════════════════════════════
class Controller:
    def __init__(self):
        self.prev_occupied     = set()
        self.connected_clients = set()
        self.ws_server         = None

        print("[Controller] Initialized")
        print(f"[Controller] WebSocket → ws://{WS_HOST}:{WS_PORT}")

    async def ws_handler(self, websocket):
        self.connected_clients.add(websocket)
        print(f"[WS] Dashboard connected ({len(self.connected_clients)} client(s))")
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.discard(websocket)
            print(f"[WS] Dashboard disconnected ({len(self.connected_clients)} client(s))")

    async def start_ws_server(self):
        self.ws_server = await websockets.serve(
            self.ws_handler, WS_HOST, WS_PORT, ping_interval=None
        )
        print(f"[WS] Server running → ws://{WS_HOST}:{WS_PORT}")

    async def broadcast(self, message: str):
        if not self.connected_clients:
            return
        await asyncio.gather(
            *[client.send(message) for client in self.connected_clients],
            return_exceptions=True
        )

    async def update(self, slot_status: dict, slot_detection_info: dict, moving_cars: list = None):
        """
        Parameters
        ----------
        slot_status         : {'slot_1': 'occupied', 'slot_2': 'vacant', ...}
        slot_detection_info : {'slot_1': {'x': 9.5, 'y': -11.2, 'score': 0.95}}
        moving_cars         : [{'x': 1.3, 'y': -4.9, 'score': 0.97}, ...]
        """
        if moving_cars is None:
            moving_cars = []

        curr_occupied = {
            sid for sid, status in slot_status.items()
            if status == 'occupied'
        }

        # ── Arrivals ──────────────────────────────────────────────────────────
        arrived = curr_occupied - self.prev_occupied
        for slot_id in sorted(arrived):
            try:
                db_api.car_arrived(slot_id)
            except Exception as e:
                print(f"[DB ERROR] car_arrived {slot_id}: {e}")
            if slot_id in slot_detection_info:
                info = slot_detection_info[slot_id]
                try:
                    db_api.log_lidar_detection(
                        x=info['x'], y=info['y'], z=0.0,
                        slot_id=slot_id,
                        confidence=info['score']
                    )
                except Exception as e:
                    print(f"[DB ERROR] log_lidar_detection {slot_id}: {e}")

        # ── Departures ────────────────────────────────────────────────────────
        left = self.prev_occupied - curr_occupied
        for slot_id in sorted(left):
            try:
                db_api.car_left(slot_id)
            except Exception as e:
                print(f"[DB ERROR] car_left {slot_id}: {e}")

        # ── Console summary ───────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%H:%M:%S")
        if arrived or left:
            print(f"\n[{timestamp}] Slot changes:")
            for sid in sorted(arrived):
                print(f"  ✅ ARRIVED  → {sid}")
            for sid in sorted(left):
                print(f"  🚗 LEFT     → {sid}")
        else:
            print(f"[{timestamp}] No changes — "
                  f"{len(curr_occupied)}/16 occupied | "
                  f"{len(moving_cars)} moving")

        # ── Convert moving cars to scene coordinates ──────────────────────────
        scene_moving_cars = []
        for i, car in enumerate(moving_cars):
            sx, sz = lidar_to_scene(car['x'], car['y'])
            scene_moving_cars.append({
                'id':      f'mv_{i}',
                'scene_x': sx,
                'scene_z': sz,
            })

        # ── Broadcast ─────────────────────────────────────────────────────────
        occupied_count = len(curr_occupied)
        msg = json.dumps({
            'type':           'slot_update',
            'timestamp':      datetime.now().isoformat(),
            'slots': [
                {'id': sid, 'occupied': sid in curr_occupied}
                for sid in ALL_SLOTS
            ],
            'moving_cars':    scene_moving_cars,
            'occupied_count': occupied_count,
            'free_count':     16 - occupied_count,
            'mode':           'live'
        })

        await self.broadcast(msg)
        self.prev_occupied = curr_occupied
