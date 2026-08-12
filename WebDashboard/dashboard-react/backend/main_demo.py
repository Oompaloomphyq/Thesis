#!/usr/bin/env python3
"""
main_demo.py — Single-Frame Demo for Thesis Video Recording
============================================================
Loads ONE raw .bin file, runs the full pipeline ONCE, then broadcasts
the result over WebSocket and stays idle forever (no looping, no DB writes).

Usage:
  python main_demo.py

Open the demo page at: http://localhost:3000/demo.html
"""

import asyncio
import numpy as np
import torch
import os
import sys
import math
import warnings
import json
import websockets
from pathlib import Path
from datetime import datetime
from collections import deque

warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

sys.path.append('/home/phyq/Thesis/OpenPCDet')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# =============================================================================
# CONFIGURATION — only thing you need to change
# =============================================================================
FRAME_FILE = '/home/phyq/Thesis/Dataset_Test/march16/live_raw_frames7/000022.bin'

CFG_FILE   = '/home/phyq/Thesis/OpenPCDet/tools/cfgs/kitti_models/pointpillar_custom2.yaml'
CKPT_FILE  = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom2/default/ckpt/checkpoint_epoch_150.pth'
SCORE_THRESHOLD = 0.3

WS_HOST = 'localhost'
WS_PORT = 5000

PARKING_CONFIG = {
    'x_min': -20, 'x_max': 15,
    'y_min': 6.2, 'y_max': 10,
    'z_min': 0,   'z_max': 12,
    'ground_y_livox': 8.3,
    'tilt_factor': 0.04,
    'ransac_distance_threshold': 0.15,
    'ransac_iterations': 100,
    'ransac_z_max': 0.5,
    'ransac_min_inliers': 100,
}

RANSAC_THRESHOLD   = 0.15
RANSAC_ITERATIONS  = 100
RANSAC_MIN_INLIERS = 100
MIN_POINTS_FOR_INFERENCE = 500

SLOT_CENTERS = [
    {"slot_id": "slot_1",  "x": 8.590, "y": -12.240, "type": "angled"},
    {"slot_id": "slot_2",  "x": 8.555, "y":  -9.240, "type": "angled"},
    {"slot_id": "slot_3",  "x": 8.531, "y":  -6.240, "type": "angled"},
    {"slot_id": "slot_4",  "x": 8.331, "y":  -3.240, "type": "angled"},
    {"slot_id": "slot_5",  "x": 8.273, "y":  -0.600, "type": "angled"},
    {"slot_id": "slot_6",  "x": 8.280, "y":   2.060, "type": "angled"},
    {"slot_id": "slot_7",  "x": 8.461, "y":   5.200, "type": "angled"},
    {"slot_id": "slot_8",  "x": 8.130, "y":   8.000, "type": "angled"},
    {"slot_id": "slot_9",  "x": 8.072, "y":  11.000, "type": "angled"},
    {"slot_id": "slot_10", "x": 8.000, "y":  14.000, "type": "angled"},
    {"slot_id": "slot_11", "x": 1.000, "y":   7.043, "type": "parallel"},
    {"slot_id": "slot_12", "x": 1.000, "y":   2.556, "type": "parallel"},
    {"slot_id": "slot_13", "x": 1.092, "y":  -1.931, "type": "parallel"},
    {"slot_id": "slot_14", "x": 1.030, "y":  -6.418, "type": "parallel"},
    {"slot_id": "slot_15", "x": 1.041, "y": -10.905, "type": "parallel"},
    {"slot_id": "slot_16", "x": 1.088, "y": -15.392, "type": "parallel"},
]

ANGLED_THRESHOLD = 1.5
PARALLEL_HALF_X  = 1.0
PARALLEL_HALF_Y  = 2.5
ALL_SLOTS        = [f'slot_{i}' for i in range(1, 17)]

SCENE_X_SCALE  =  0.7019
SCENE_X_OFFSET = -0.1109
SCENE_Z_SCALE  = -0.547
SCENE_Z_OFFSET =  3.65

# =============================================================================
# PIPELINE STEPS
# =============================================================================
def load_raw(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def preprocess(points_raw, config):
    valid = np.all(np.isfinite(points_raw), axis=1)
    pts   = points_raw[valid]
    mask  = (
        (pts[:, 0] >= config['x_min']) & (pts[:, 0] <= config['x_max']) &
        (pts[:, 1] >= config['y_min']) & (pts[:, 1] <= config['y_max']) &
        (pts[:, 2] >= config['z_min']) & (pts[:, 2] <= config['z_max'])
    )
    pts = pts[mask]
    if len(pts) == 0:
        return pts
    x_livox   = pts[:, 0]
    y_livox   = pts[:, 1]
    z_livox   = pts[:, 2]
    intensity = pts[:, 3] / 255.0
    y_corrected = y_livox - (z_livox * config['tilt_factor'])
    x_kitti   = z_livox
    y_kitti   = x_livox
    z_kitti   = config['ground_y_livox'] - y_corrected
    return np.column_stack([x_kitti, y_kitti, z_kitti, intensity]).astype(np.float32)

def ransac_ground_removal(points):
    if len(points) < RANSAC_MIN_INLIERS:
        return points
    xyz        = points[:, :3]
    candidates = xyz[xyz[:, 2] < 0.5]
    if len(candidates) < 3:
        return points
    best_inliers = None
    best_count   = 0
    rng = np.random.default_rng(42)
    for _ in range(RANSAC_ITERATIONS):
        idx        = rng.choice(len(candidates), 3, replace=False)
        p1, p2, p3 = candidates[idx[0]], candidates[idx[1]], candidates[idx[2]]
        normal     = np.cross(p2 - p1, p3 - p1)
        nlen       = np.linalg.norm(normal)
        if nlen < 1e-6:
            continue
        normal  = normal / nlen
        d       = -np.dot(normal, p1)
        dists   = np.abs(xyz @ normal + d)
        inliers = dists < RANSAC_THRESHOLD
        count   = np.sum(inliers)
        if count > best_count:
            best_count   = count
            best_inliers = inliers
    if best_inliers is None or best_count < RANSAC_MIN_INLIERS:
        return points
    non_ground = points[~best_inliers]
    print(f"  [RANSAC] Removed {best_count} ground pts → {len(non_ground)} remaining")
    return non_ground

class DummyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
    def __len__(self): return 0
    def __getitem__(self, index): return None

def load_model():
    print("=" * 60)
    print("  Loading PointPillars model...")
    print("=" * 60)
    os.chdir('/home/phyq/Thesis/OpenPCDet/tools')
    logger  = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    dataset = DummyDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, Path('.'), logger)
    model   = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    print("  Model loaded")
    print("=" * 60 + "\n")
    return model, dataset

def run_inference(model, dataset, points):
    if len(points) < MIN_POINTS_FOR_INFERENCE:
        print(f"  [Infer] Too few points ({len(points)}) — skipping")
        return np.zeros((0, 8), dtype=np.float32)
    input_dict = {'points': points.copy(), 'frame_id': 0}
    data_dict  = dataset.prepare_data(data_dict=input_dict)
    data_dict  = dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    boxes  = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    mask   = scores >= SCORE_THRESHOLD
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    detections = np.hstack([boxes, scores.reshape(-1, 1)]).astype(np.float32)
    print(f"  [Infer] {len(detections)} detections")
    return detections

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def is_match(veh_x, veh_y, slot):
    if slot["type"] == "angled":
        return distance(veh_x, veh_y, slot["x"], slot["y"]) <= ANGLED_THRESHOLD
    else:
        return (slot["x"] - PARALLEL_HALF_X <= veh_x <= slot["x"] + PARALLEL_HALF_X and
                slot["y"] - PARALLEL_HALF_Y <= veh_y <= slot["y"] + PARALLEL_HALF_Y)

def match_slots(detections):
    slot_status         = {s["slot_id"]: "vacant" for s in SLOT_CENTERS}
    slot_detection_info = {}
    matched_ids         = set()
    moving_cars         = []
    for det_idx, det in enumerate(detections):
        veh_x, veh_y, score = det[0], det[1], det[7]
        for slot in SLOT_CENTERS:
            if is_match(veh_x, veh_y, slot):
                sid = slot["slot_id"]
                slot_status[sid] = "occupied"
                matched_ids.add(det_idx)
                slot_detection_info[sid] = {
                    "x": round(float(veh_x), 3),
                    "y": round(float(veh_y), 3),
                    "score": round(float(score), 3),
                }
                break
    for det_idx, det in enumerate(detections):
        if det_idx not in matched_ids:
            moving_cars.append({
                "x": round(float(det[0]), 3),
                "y": round(float(det[1]), 3),
                "score": round(float(det[7]), 3),
            })
    return slot_status, moving_cars, slot_detection_info

def lidar_to_scene(lidar_x, lidar_y):
    scene_x = SCENE_X_SCALE * lidar_y + SCENE_X_OFFSET
    scene_z = SCENE_Z_SCALE * lidar_x + SCENE_Z_OFFSET
    return round(scene_x, 3), round(scene_z, 3)

def build_vacant_message():
    return json.dumps({
        'type':           'slot_update',
        'timestamp':      datetime.now().isoformat(),
        'slots':          [{'id': sid, 'occupied': False} for sid in ALL_SLOTS],
        'moving_cars':    [],
        'occupied_count': 0,
        'free_count':     16,
        'mode':           'live'
    })

def build_ws_message(slot_status, moving_cars):
    curr_occupied = {sid for sid, s in slot_status.items() if s == 'occupied'}
    scene_moving_cars = []
    for i, car in enumerate(moving_cars):
        sx, sz = lidar_to_scene(car['x'], car['y'])
        scene_moving_cars.append({'id': f'mv_{i}', 'scene_x': sx, 'scene_z': sz})
    return json.dumps({
        'type':           'slot_update',
        'timestamp':      datetime.now().isoformat(),
        'slots':          [{'id': sid, 'occupied': sid in curr_occupied} for sid in ALL_SLOTS],
        'moving_cars':    scene_moving_cars,
        'occupied_count': len(curr_occupied),
        'free_count':     16 - len(curr_occupied),
        'mode':           'live'
    })

# =============================================================================
# MAIN
# =============================================================================
async def main():
    # --- Run pipeline once ---
    print(f"\nLoading frame: {FRAME_FILE}")
    raw_points   = load_raw(FRAME_FILE)
    print(f"  [Load]       {len(raw_points):,} raw points")

    kitti_points = preprocess(raw_points, PARKING_CONFIG)
    print(f"  [Preprocess] {len(raw_points):,} → {len(kitti_points):,} pts")

    clean_points = ransac_ground_removal(kitti_points)

    model, dataset = load_model()
    detections     = run_inference(model, dataset, clean_points)

    slot_status, moving_cars, slot_detection_info = match_slots(detections)

    occupied = sum(1 for v in slot_status.values() if v == 'occupied')
    print(f"\n  Result: {occupied}/16 slots occupied")
    for sid, status in slot_status.items():
        if status == 'occupied':
            print(f"    {sid}: OCCUPIED")

    static_message  = build_ws_message(slot_status, moving_cars)
    vacant_message  = build_vacant_message()

    # --- WebSocket server: send static result to every connecting client ---
    connected_clients = set()

    async def ws_handler(websocket):
        connected_clients.add(websocket)
        print(f"[WS] Dashboard connected — sending static frame")
        try:
            # Reset prevOccupied on frontend, wait for car model to load, then send once
            await websocket.send(vacant_message)
            await asyncio.sleep(4)
            await websocket.send(static_message)
            # Keep connection alive with a neutral ping — frontend ignores unknown types
            # so no re-renders happen, no reconnect cycle, no blinking
            while True:
                await websocket.send(json.dumps({'type': 'ping'}))
                await asyncio.sleep(10)
        except Exception:
            pass
        finally:
            connected_clients.discard(websocket)
            print("[WS] Dashboard disconnected")

    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, ping_interval=None)
    print(f"\n[WS] Server running → ws://{WS_HOST}:{WS_PORT}")
    print(f"[Demo] Open http://localhost:3000/demo.html in your browser")
    print(f"[Demo] Static frame is live. Press Ctrl+C to stop.\n")

    try:
        await asyncio.Future()   # run forever
    except KeyboardInterrupt:
        print("\n[Demo] Stopped.")
        server.close()

if __name__ == "__main__":
    asyncio.run(main())
