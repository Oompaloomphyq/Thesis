#!/usr/bin/env python3
"""
simulate.py — Smart Parking Simulation using pre-recorded .npy frames
======================================================================
Replays 001000.npy → 001277.npy using your calibrated slot_match logic,
writes results to MySQL, and broadcasts to the dashboard via WebSocket.

Usage:
  python simulate.py                        # normal (10 FPS)
  python simulate.py --fps 5               # slower
  python simulate.py --frame 001050        # start from specific frame
  python simulate.py --confidence 0.5      # stricter detection threshold

Requirements:
  pip install websockets mysql-connector-python numpy
"""

import asyncio
import argparse
import json
import math
import os
import sys
import numpy as np
from datetime import datetime

# controller.py and db_api.py must be in the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from controller import Controller

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
NPY_FOLDER      = os.path.expanduser('~/Thesis/npy')
DEFAULT_FPS     = 10
CONF_THRESHOLD  = 0.3

# ═══════════════════════════════════════════════════════════
# SLOT CENTERS — calibrated from your real LiDAR data
# ═══════════════════════════════════════════════════════════
SLOT_CENTERS = [
    {"slot_id": "slot_1",  "x": 9.590, "y": -11.240, "type": "angled"},
    {"slot_id": "slot_2",  "x": 9.555, "y":  -8.240, "type": "angled"},
    {"slot_id": "slot_3",  "x": 9.531, "y":  -5.240, "type": "angled"},
    {"slot_id": "slot_4",  "x": 9.331, "y":  -2.240, "type": "angled"},
    {"slot_id": "slot_5",  "x": 9.273, "y":   0.760, "type": "angled"},
    {"slot_id": "slot_6",  "x": 8.480, "y":   3.760, "type": "angled"},
    {"slot_id": "slot_7",  "x": 8.461, "y":   7.400, "type": "angled"},
    {"slot_id": "slot_8",  "x": 9.130, "y":  10.400, "type": "angled"},
    {"slot_id": "slot_9",  "x": 9.072, "y":  13.400, "type": "angled"},
    {"slot_id": "slot_10", "x": 8.526, "y":  16.400, "type": "angled"},
    {"slot_id": "slot_11", "x": 1.549, "y":   7.043, "type": "parallel"},
    {"slot_id": "slot_12", "x": 1.500, "y":   2.556, "type": "parallel"},
    {"slot_id": "slot_13", "x": 1.992, "y":  -1.931, "type": "parallel"},
    {"slot_id": "slot_14", "x": 2.330, "y":  -6.418, "type": "parallel"},
    {"slot_id": "slot_15", "x": 2.741, "y": -10.905, "type": "parallel"},
    {"slot_id": "slot_16", "x": 2.588, "y": -15.392, "type": "parallel"},
]

ANGLED_THRESHOLD = 1.5
PARALLEL_HALF_X  = 1.0
PARALLEL_HALF_Y  = 2.5

# ═══════════════════════════════════════════════════════════
# SLOT MATCHING — same logic as slot_match.py
# ═══════════════════════════════════════════════════════════
def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_match(veh_x, veh_y, slot):
    if slot["type"] == "angled":
        return distance(veh_x, veh_y, slot["x"], slot["y"]) <= ANGLED_THRESHOLD
    else:
        return (
            slot["x"] - PARALLEL_HALF_X <= veh_x <= slot["x"] + PARALLEL_HALF_X and
            slot["y"] - PARALLEL_HALF_Y <= veh_y <= slot["y"] + PARALLEL_HALF_Y
        )


def match_slots(npy_path, conf_threshold):
    """
    Read one .npy frame → run slot matching → return slot_status + slot_detection_info.
    """
    data   = np.load(npy_path, allow_pickle=True).item()
    boxes  = data['boxes']
    scores = data['scores']

    slot_status         = {s["slot_id"]: "vacant" for s in SLOT_CENTERS}
    slot_detection_info = {}
    matched_ids         = set()
    moving_cars         = []

    for det_idx, (box, score) in enumerate(zip(boxes, scores)):
        if score < conf_threshold:
            continue
        veh_x, veh_y = float(box[0]), float(box[1])

        for slot in SLOT_CENTERS:
            if is_match(veh_x, veh_y, slot):
                sid = slot["slot_id"]
                slot_status[sid] = "occupied"
                matched_ids.add(det_idx)
                slot_detection_info[sid] = {
                    "x":      round(veh_x, 3),
                    "y":      round(veh_y, 3),
                    "score":  round(float(score), 3),
                    "length": round(float(box[3]), 3),
                    "width":  round(float(box[4]), 3),
                    "height": round(float(box[5]), 3),
                }
                break

    for det_idx, (box, score) in enumerate(zip(boxes, scores)):
        if score < conf_threshold:
            continue
        if det_idx not in matched_ids:
            moving_cars.append({
                "x":     round(float(box[0]), 3),
                "y":     round(float(box[1]), 3),
                "score": round(float(score),  3),
            })

    return slot_status, slot_detection_info, moving_cars


# ═══════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════
async def simulate(args):
    # Collect and sort .npy files
    all_files = sorted([
        f for f in os.listdir(NPY_FOLDER)
        if f.endswith('.npy')
    ])

    if not all_files:
        print(f"❌ No .npy files found in {NPY_FOLDER}")
        return

    # Start from specific frame if requested
    start_idx = 0
    if args.frame:
        target = f"{args.frame}.npy"
        if target in all_files:
            start_idx = all_files.index(target)
            print(f"  Starting from frame: {target}")
        else:
            print(f"  ⚠ Frame {target} not found — starting from beginning")

    files     = all_files[start_idx:]
    interval  = 1.0 / args.fps

    print(f"\n{'='*55}")
    print(f"  🚗 Smart Parking Simulate")
    print(f"  📁 Folder     : {NPY_FOLDER}")
    print(f"  🎞  Frames     : {len(files)} ({files[0]} → {files[-1]})")
    print(f"  ⚡ FPS        : {args.fps}  ({interval:.2f}s/frame)")
    print(f"  🎯 Confidence : >= {args.confidence}")
    print(f"  🌐 WebSocket  : ws://localhost:5000")
    print(f"{'='*55}\n")

    # Start controller (WebSocket server + DB)
    controller = Controller()
    await controller.start_ws_server()

    print("Waiting for dashboard to connect...")
    print("Press Ctrl+C to stop.\n")

    # Give dashboard time to connect before starting
    await asyncio.sleep(2.0)

    frame_count    = 0
    total_frames   = len(files)

    for fname in files:
        fpath     = os.path.join(NPY_FOLDER, fname)
        frame_num = fname.replace('.npy', '')
        frame_count += 1

        # Run slot matching on this frame
        slot_status, slot_detection_info, moving_cars = match_slots(
            fpath, args.confidence
        )

        occupied_count = sum(1 for s in slot_status.values() if s == 'occupied')

        # Send to MySQL + broadcast to dashboard
        await controller.update(slot_status, slot_detection_info)

        # Console log
        print(f"  [{frame_count:4d}/{total_frames}] Frame {frame_num} "
              f"| Occupied: {occupied_count:2d}/16 "
              f"| Moving: {len(moving_cars)} "
              f"| Clients: {len(controller.connected_clients)}")

        await asyncio.sleep(interval)

    print(f"\n{'='*55}")
    print(f"  ✅ Simulation complete — {frame_count} frames played")
    print(f"{'='*55}\n")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
async def main(args):
    await simulate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart Parking Simulator')
    parser.add_argument('--fps',        type=float, default=DEFAULT_FPS,
                        help=f'Playback speed in FPS (default: {DEFAULT_FPS})')
    parser.add_argument('--frame',      type=str,   default=None,
                        help='Start from specific frame e.g. --frame 001050')
    parser.add_argument('--confidence', type=float, default=CONF_THRESHOLD,
                        help=f'Min confidence threshold (default: {CONF_THRESHOLD})')
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n⛔ Simulation stopped by user.")
