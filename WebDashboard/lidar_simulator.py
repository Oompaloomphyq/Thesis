"""
lidar_simulator.py — Smart Parking LiDAR Test Mode
====================================================
Simulates real LiDAR detections using 278 pre-recorded frames (001000-001277).

What it does:
  1. Reads each .npy frame one by one
  2. Maps LiDAR coordinates → parking slot IDs
  3. Compares with previous frame to detect arrivals/departures
  4. Inserts into MySQL (slots, parking_events, lidar_detections) — exactly like real LiDAR
  5. Sends WebSocket update to dashboard → slots turn RED/GREEN in real-time

Usage:
  python lidar_simulator.py                   # normal speed (1 frame per 2 sec)
  python lidar_simulator.py --speed 3         # 3x faster
  python lidar_simulator.py --frame 001050    # start from specific frame
  python lidar_simulator.py --confidence 0.5  # stricter detection threshold
  python lidar_simulator.py --no-loop         # play once, don't repeat

Requirements:
  pip install mysql-connector-python websockets numpy asyncio

Folder structure:
  lidar_simulator.py
  extracted/
    001000.npy
    001001.npy
    ...
    001277.npy
"""

import asyncio
import json
import numpy as np
import os
import argparse
import websockets
from datetime import datetime
import mysql.connector
from mysql.connector import Error

# ═══════════════════════════════════════════════════════════
# CONFIG — update these to match your setup
# ═══════════════════════════════════════════════════════════
NPY_FOLDER     = './extracted'
FRAME_INTERVAL = 2.0        # seconds between frames
CONF_THRESHOLD = 0.3        # min confidence to count a detection
WS_HOST        = 'localhost'
WS_PORT        = 5000
WS_PATH = '/ws/lidar'

DB_CONFIG = {
    'host':     'localhost',
    'port':     3306,
    'user':     'root',
    'password': '',
    'database': 'smart_parking'
}

# ═══════════════════════════════════════════════════════════
# SLOT DEFINITIONS — matches your ParkingScene.jsx exactly
# ═══════════════════════════════════════════════════════════
SLOT_POSITIONS = [
    {'id': 'slot_1',  'x': -8.0,  'z': -2.1, 'width': 1.6, 'depth': 4.2},
    {'id': 'slot_2',  'x': -5.9,  'z': -2.0, 'width': 1.4, 'depth': 4.1},
    {'id': 'slot_3',  'x': -3.9,  'z': -1.9, 'width': 1.4, 'depth': 4.0},
    {'id': 'slot_4',  'x': -1.8,  'z': -1.9, 'width': 1.4, 'depth': 3.9},
    {'id': 'slot_5',  'x':  0.3,  'z': -1.9, 'width': 1.4, 'depth': 3.8},
    {'id': 'slot_6',  'x':  2.5,  'z': -1.9, 'width': 1.4, 'depth': 3.7},
    {'id': 'slot_7',  'x':  4.6,  'z': -1.9, 'width': 1.4, 'depth': 3.7},
    {'id': 'slot_8',  'x':  6.7,  'z': -1.9, 'width': 1.4, 'depth': 3.7},
    {'id': 'slot_9',  'x':  8.8,  'z': -1.9, 'width': 1.4, 'depth': 3.7},
    {'id': 'slot_10', 'x': 11.4,  'z': -2.1, 'width': 1.6, 'depth': 4.2},
    {'id': 'slot_11', 'x':  5.9,  'z':  2.8, 'width': 2.6, 'depth': 1.2},
    {'id': 'slot_12', 'x':  3.1,  'z':  2.8, 'width': 2.7, 'depth': 1.2},
    {'id': 'slot_13', 'x':  0.2,  'z':  2.8, 'width': 2.7, 'depth': 1.2},
    {'id': 'slot_14', 'x': -2.7,  'z':  2.8, 'width': 2.7, 'depth': 1.2},
    {'id': 'slot_15', 'x': -5.6,  'z':  2.8, 'width': 2.7, 'depth': 1.2},
    {'id': 'slot_16', 'x': -8.4,  'z':  2.8, 'width': 2.6, 'depth': 1.2},
]

# ═══════════════════════════════════════════════════════════
# COORDINATE MAPPING: LiDAR → 3D Scene
# ═══════════════════════════════════════════════════════════
def lidar_to_scene(lx, ly):
    sx = (ly - (-17.0)) / (20.0 - (-17.0)) * (11.4 - (-8.0)) + (-8.0)
    sz = (lx -   1.5)  / (10.5  -   1.5)  * (2.8  - (-2.1)) + (-2.1)
    return round(sx, 3), round(sz, 3)

def point_in_slot(sx, sz, slot, margin=0.8):
    hw = (slot['width'] / 2) + margin
    hd = (slot['depth'] / 2) + margin
    return abs(sx - slot['x']) < hw and abs(sz - slot['z']) < hd

def get_occupied_slots(npy_path, conf_threshold):
    """Read one .npy frame → return {slot_id: detection_info}"""
    data   = np.load(npy_path, allow_pickle=True).item()
    boxes  = data['boxes']
    scores = data['scores']

    occupied = {}
    for i, box in enumerate(boxes):
        if scores[i] < conf_threshold:
            continue
        sx, sz = lidar_to_scene(box[0], box[1])
        for slot in SLOT_POSITIONS:
            if point_in_slot(sx, sz, slot) and slot['id'] not in occupied:
                occupied[slot['id']] = {
                    'lidar_x':    float(box[0]),
                    'lidar_y':    float(box[1]),
                    'lidar_z':    float(box[2]),
                    'scene_x':    sx,
                    'scene_z':    sz,
                    'length':     float(box[3]),
                    'width':      float(box[4]),
                    'height':     float(box[5]),
                    'confidence': float(scores[i]),
                }
    return occupied

# ═══════════════════════════════════════════════════════════
# DATABASE FUNCTIONS
# ═══════════════════════════════════════════════════════════
def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"[DB ERROR] {e}")
        return None

def db_car_arrived(slot_id, detection):
    """Car detected in slot → insert vehicle + parking_event + lidar_detection."""
    conn = get_db()
    if not conn: return
    try:
        cur = conn.cursor()

        # 1. Create vehicle
        cur.execute("INSERT INTO vehicles (first_seen) VALUES (NOW())")
        vehicle_id = cur.lastrowid

        # 2. Log parking event
        cur.execute("""
            INSERT INTO parking_events (vehicle_id, slot_id, status, time_in)
            VALUES (%s, %s, 'occupied', NOW())
        """, (vehicle_id, slot_id))

        # 3. Update slot status
        cur.execute(
            "UPDATE slots SET status = 'occupied' WHERE slot_id = %s",
            (slot_id,)
        )

        # 4. Log raw LiDAR detection
        cur.execute("""
            INSERT INTO lidar_detections
            (slot_id, x, y, z, length, width, height, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            slot_id,
            detection['lidar_x'], detection['lidar_y'], detection['lidar_z'],
            detection['length'],  detection['width'],   detection['height'],
            detection['confidence']
        ))

        conn.commit()
        print(f"  [DB] ✅ Car ARRIVED  → {slot_id} (vehicle #{vehicle_id}, conf: {detection['confidence']:.2f})")
    except Error as e:
        print(f"  [DB ERROR] car_arrived {slot_id}: {e}")
    finally:
        conn.close()

def db_car_left(slot_id):
    """Car left slot → close parking_event + update slot to vacant."""
    conn = get_db()
    if not conn: return
    try:
        cur = conn.cursor(dictionary=True)

        # Find open event
        cur.execute("""
            SELECT event_id, time_in FROM parking_events
            WHERE slot_id = %s AND status = 'occupied' AND time_out IS NULL
            ORDER BY time_in DESC LIMIT 1
        """, (slot_id,))
        event = cur.fetchone()

        if not event:
            return

        # Calculate duration
        time_out = datetime.now()
        duration = max(1, int((time_out - event['time_in']).total_seconds() / 60))

        cur2 = conn.cursor()

        # Close event
        cur2.execute("""
            UPDATE parking_events
            SET status = 'vacant', time_out = %s, duration_mins = %s
            WHERE event_id = %s
        """, (time_out, duration, event['event_id']))

        # Update slot
        cur2.execute(
            "UPDATE slots SET status = 'vacant' WHERE slot_id = %s",
            (slot_id,)
        )

        conn.commit()
        print(f"  [DB] 🚗 Car LEFT     → {slot_id} (parked {duration} min)")
    except Error as e:
        print(f"  [DB ERROR] car_left {slot_id}: {e}")
    finally:
        conn.close()

def db_get_all_slots():
    """Get current slot statuses from DB for WebSocket broadcast."""
    conn = get_db()
    if not conn: return []
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT slot_id, status FROM slots ORDER BY slot_id")
        return cur.fetchall()
    except Error as e:
        print(f"  [DB ERROR] get_all_slots: {e}")
        return []
    finally:
        conn.close()

def db_reset_all_slots():
    """Reset all slots to vacant at startup (clean state)."""
    conn = get_db()
    if not conn: return
    try:
        cur = conn.cursor()
        cur.execute("UPDATE slots SET status = 'vacant'")
        cur.execute("""
            UPDATE parking_events SET status = 'vacant',
            time_out = NOW(), duration_mins = 0
            WHERE status = 'occupied' AND time_out IS NULL
        """)
        conn.commit()
        print("[DB] 🔄 All slots reset to vacant")
    except Error as e:
        print(f"[DB ERROR] reset: {e}")
    finally:
        conn.close()

# ═══════════════════════════════════════════════════════════
# WEBSOCKET
# ═══════════════════════════════════════════════════════════
connected_clients = set()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Client connected ({len(connected_clients)} total)")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected ({len(connected_clients)} total)")

async def ws_broadcast(message):
    if connected_clients:
        await asyncio.gather(
            *[c.send(message) for c in connected_clients],
            return_exceptions=True
        )

# ═══════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════
async def simulate(args):
    files = sorted([f for f in os.listdir(NPY_FOLDER) if f.endswith('.npy')])
    if not files:
        print(f"❌ No .npy files in {NPY_FOLDER}")
        return

    # Start frame index
    start_idx = 0
    if args.frame:
        target = f"{args.frame}.npy"
        if target in files:
            start_idx = files.index(target)

    interval = FRAME_INTERVAL / args.speed

    print(f"\n{'='*55}")
    print(f"  🚗 Smart Parking LiDAR Simulator")
    print(f"  📁 Folder     : {NPY_FOLDER}")
    print(f"  🎞  Frames     : {len(files)} ({files[0]} → {files[-1]})")
    print(f"  ⚡ Speed      : {args.speed}x  ({interval:.1f}s/frame)")
    print(f"  🎯 Confidence : > {args.confidence}")
    print(f"  🔁 Loop       : {'Yes' if not args.no_loop else 'No'}")
    print(f"  🌐 WebSocket  : ws://{WS_HOST}:{WS_PORT}/ws/lidar")
    print(f"{'='*55}\n")

    # Reset DB to clean state
    db_reset_all_slots()

    prev_occupied = set()

    while True:
        for i in range(start_idx, len(files)):
            fname  = files[i]
            fpath  = os.path.join(NPY_FOLDER, fname)
            frame_num = fname.replace('.npy', '')

            # Detect occupied slots from this frame
            occupied_map = get_occupied_slots(fpath, args.confidence)
            curr_occupied = set(occupied_map.keys())

            # ── Cars that just ARRIVED (in curr but not prev) ──
            arrived = curr_occupied - prev_occupied
            for slot_id in arrived:
                db_car_arrived(slot_id, occupied_map[slot_id])

            # ── Cars that just LEFT (in prev but not curr) ──
            left = prev_occupied - curr_occupied
            for slot_id in left:
                db_car_left(slot_id)

            # ── Broadcast to dashboard via WebSocket ──
            db_slots = db_get_all_slots()
            occupied_count = sum(1 for s in db_slots if s['status'] == 'occupied')

            msg = json.dumps({
                'type':           'slot_update',
                'frame':          frame_num,
                'timestamp':      datetime.now().isoformat(),
                'slots': [
                    {'id': s['slot_id'], 'occupied': s['status'] == 'occupied'}
                    for s in db_slots
                ],
                'occupied_count': occupied_count,
                'free_count':     16 - occupied_count,
                'mode':           'test_simulation',
            })
            await ws_broadcast(msg)

            # ── Console log ──
            print(f"[{frame_num}] Occupied: {len(curr_occupied):2d}/16 "
                  f"| +{len(arrived)} arrived, -{len(left)} left "
                  f"| Clients: {len(connected_clients)}")

            prev_occupied = curr_occupied
            await asyncio.sleep(interval)

        if args.no_loop:
            print("\n✅ Simulation complete (all frames played)")
            break

        # Loop back
        start_idx = 0
        print(f"\n🔁 Looping back to {files[0]}...\n")
        db_reset_all_slots()
        prev_occupied = set()

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
async def main(args):
    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, ping_interval=None)
    print(f"✅ WebSocket server running → ws://{WS_HOST}:{WS_PORT}/ws/lidar")
    print(f"✅ WebSocket server running → ws://{WS_HOST}:{WS_PORT}/ws/lidar")
    await asyncio.gather(
        server.wait_closed(),
        simulate(args),
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart Parking LiDAR Simulator')
    parser.add_argument('--speed',      type=float, default=1.0,
                        help='Playback speed (default: 1.0)')
    parser.add_argument('--frame',      type=str,   default=None,
                        help='Start from frame e.g. --frame 001050')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Min confidence threshold (default: 0.3)')
    parser.add_argument('--no-loop',    action='store_true',
                        help='Play frames once only, do not loop')
    args = parser.parse_args()
    asyncio.run(main(args))
