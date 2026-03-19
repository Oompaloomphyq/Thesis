#!/usr/bin/env python3
#current main.py
"""
main.py — Real-Time Parking Detection Pipeline
Raspberry Pi (raw Livox points) → Preprocess → RANSAC → PointPillars → Slot Match → Controller
"""

import asyncio
import socket
import re
import csv
import numpy as np
import torch
import time
import os
import sys
import re
import math
import warnings
from pathlib import Path
from collections import deque
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

sys.path.append('/home/phyq/Thesis/OpenPCDet')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from controller import Controller

# =============================================================================
# CONFIGURATION
# =============================================================================
RECEIVE_PORT  = 9999
BUFFER_SIZE   = 65536
MAX_POINTS    = 120000

CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/tools/cfgs/kitti_models/pointpillar_custom2.yaml'
CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom2/default/ckpt/checkpoint_epoch_150.pth'

# CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom/old_150epochs/pointpillar_custom.yaml'
# CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom/old_150epochs/ckpt/checkpoint_epoch_150.pth'
SCORE_THRESHOLD = 0.3

PARKING_CONFIG = {
    'x_min': -20, 'x_max': 15,
    'y_min': 6.2,  'y_max': 10,
    'z_min': 0,    'z_max': 12,
    'ground_y_livox': 8.3,
    'tilt_factor': 0.04,
}

RANSAC_THRESHOLD   = 0.15
RANSAC_ITERATIONS  = 100
RANSAC_MIN_INLIERS = 100

MIN_POINTS_FOR_INFERENCE = 1000
INFERENCE_INTERVAL       = .5

# =============================================================================
# SLOT CENTERS (updated calibrated values)
# =============================================================================
SLOT_CENTERS = [
    # --- Angled slots (top row) ---
    {"slot_id": "slot_1",  "x": 8.590, "y": -12.240, "type": "angled"},
    {"slot_id": "slot_2",  "x": 8.555, "y":  -9.240, "type": "angled"},
    {"slot_id": "slot_3",  "x": 8.531, "y":  -6.240, "type": "angled"},
    {"slot_id": "slot_4",  "x": 8.331, "y":  -3.240, "type": "angled"},
    {"slot_id": "slot_5",  "x": 8.273, "y":  -0.600, "type": "angled"},
    {"slot_id": "slot_6",  "x": 7.280, "y":   2.060, "type": "angled"},
    {"slot_id": "slot_7",  "x": 8.461, "y":   5.200, "type": "angled"},
    {"slot_id": "slot_8",  "x": 8.130, "y":   8.000, "type": "angled"},
    {"slot_id": "slot_9",  "x": 8.072, "y":  11.000, "type": "angled"},
    {"slot_id": "slot_10", "x": 7.000, "y":  13.000, "type": "angled"},

    # --- Parallel slots (bottom row) ---
    {"slot_id": "slot_11", "x": 1.000, "y":   7.043, "type": "parallel"},
    {"slot_id": "slot_12", "x": 1.000, "y":   2.556, "type": "parallel"},
    {"slot_id": "slot_13", "x": 1.092, "y":  -1.931, "type": "parallel"},
    {"slot_id": "slot_14", "x": 1.030, "y":  -6.418, "type": "parallel"},
    {"slot_id": "slot_15", "x": 1.041, "y": -10.905, "type": "parallel"},
    {"slot_id": "slot_16", "x": 1.088, "y": -15.392, "type": "parallel"},
]

ANGLED_THRESHOLD = 1.3
PARALLEL_HALF_X  = 1.0
PARALLEL_HALF_Y  = 2.5

# =============================================================================
# TEMPORAL SMOOTHING
# =============================================================================
HISTORY_SIZE        = 4
OCCUPIED_THRESHOLD = 3

slot_history = {
    f"slot_{i}": deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    for i in range(1, 17)
}

def smooth_slot_status(raw_status: dict) -> dict:
    smoothed = {}
    for slot_id, history in slot_history.items():
        detected = 1 if raw_status.get(slot_id) == "occupied" else 0
        history.append(detected)
        smoothed[slot_id] = "occupied" if sum(history) >= OCCUPIED_THRESHOLD else "vacant"
    return smoothed

# =============================================================================
# LATENCY LOGGING
# =============================================================================
LATENCY_CSV = '/home/phyq/Thesis/WebDashboard/dashboard-react/backend/latency_log.csv'

def init_csv():
    if not Path(LATENCY_CSV).exists():
        with open(LATENCY_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'timestamp',
                'dur_preprocess', 'dur_ransac', 'dur_inference',
                'dur_slotmatch', 'dur_broadcast', 'dur_total'
            ])

def log_latency(frame_id, t_start, t_preprocess, t_ransac, t_inference, t_slotmatch, t_broadcast):
    dur_preprocess = round(t_preprocess - t_start,      4)
    dur_ransac     = round(t_ransac     - t_preprocess, 4)
    dur_inference  = round(t_inference  - t_ransac,     4)
    dur_slotmatch  = round(t_slotmatch  - t_inference,  4)
    dur_broadcast  = round(t_broadcast  - t_slotmatch,  4)
    dur_total      = round(t_broadcast  - t_start,      4)

    # Print to terminal
    print(f"  [LATENCY] preprocess={dur_preprocess:.3f}s | ransac={dur_ransac:.3f}s | "
          f"inference={dur_inference:.3f}s | slotmatch={dur_slotmatch:.3f}s | "
          f"broadcast={dur_broadcast:.3f}s | TOTAL={dur_total:.3f}s")

    # Append to CSV
    with open(LATENCY_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            dur_preprocess, dur_ransac, dur_inference,
            dur_slotmatch, dur_broadcast, dur_total
        ])

# =============================================================================
# STEP 1 — PREPROCESS
# =============================================================================
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

# =============================================================================
# STEP 2 — RANSAC
# =============================================================================
def ransac_ground_removal(points):
    if len(points) < RANSAC_MIN_INLIERS:
        return points
    xyz = points[:, :3]

    # Only use near-ground points as candidates (matches preprocess_ransac.py)
    candidates = xyz[xyz[:, 2] < 0.5]
    if len(candidates) < 3:
        return points

    best_inliers = None
    best_count   = 0
    rng = np.random.default_rng(42)
    for _ in range(RANSAC_ITERATIONS):
        idx         = rng.choice(len(candidates), 3, replace=False)
        p1, p2, p3  = candidates[idx[0]], candidates[idx[1]], candidates[idx[2]]
        normal      = np.cross(p2 - p1, p3 - p1)
        nlen        = np.linalg.norm(normal)
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

# =============================================================================
# STEP 3 — POINTPILLARS
# =============================================================================
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
    print(f"  ✓ Model loaded")
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

# =============================================================================
# STEP 4 — SLOT MATCHING
# =============================================================================
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
                "x":     round(float(det[0]), 3),
                "y":     round(float(det[1]), 3),
                "score": round(float(det[7]), 3),
            })

    return slot_status, moving_cars, slot_detection_info

# =============================================================================
# MAIN LOOP
# =============================================================================
async def main():
    model, dataset = load_model()

    controller = Controller()
    await controller.start_ws_server()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RECEIVE_PORT))
    sock.settimeout(0.01)

    # no pattern needed anymore — binary unpacking
    point_buffer    = deque(maxlen=MAX_POINTS)
    data_received   = False
    last_infer_time = time.time()
    frame_id        = 0
    init_csv()

    print(f"Listening on UDP port {RECEIVE_PORT} (binary mode)...")
    print("Waiting for data from Raspberry Pi...\n")

    try:
        while True:
            # ── RECEIVE (binary) ──────────────────────────────────────────────
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
                if not data_received:
                    print(f"✓ Connected to Raspberry Pi at {addr[0]} (binary stream)\n")
                    data_received = True

                # Each point = 4 float32 = 16 bytes
                num_floats = len(data) // 4
                if num_floats < 4 or (num_floats % 4) != 0:
                    continue  # incomplete or malformed packet

                pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)

                for pt in pts:
                    x, y, z, r = pt[0], pt[1], pt[2], pt[3]
                    if not (x == 0.0 and y == 0.0 and z == 0.0):
                        point_buffer.append([x, y, z, r])

            except socket.timeout:
                pass

            # ── PIPELINE ──────────────────────────────────────────────────────
            now = time.time()
            if (now - last_infer_time) >= INFERENCE_INTERVAL:
                last_infer_time = now

                if len(point_buffer) < MIN_POINTS_FOR_INFERENCE:
                    print(f"  Waiting... ({len(point_buffer)}/{MIN_POINTS_FOR_INFERENCE})")
                    await asyncio.sleep(0)
                    continue

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Pipeline ({len(point_buffer)} pts)...")

                frame_id    += 1
                t_start      = time.time()

                raw_points   = np.array(point_buffer, dtype=np.float32)
                kitti_points = preprocess(raw_points, PARKING_CONFIG)
                t_preprocess = time.time()
                print(f"  [Preprocess] {len(raw_points)} → {len(kitti_points)} pts")

                if len(kitti_points) == 0:
                    await asyncio.sleep(0)
                    continue

                clean_points = ransac_ground_removal(kitti_points)
                t_ransac     = time.time()

                detections   = run_inference(model, dataset, clean_points)
                t_inference  = time.time()

                raw_status, moving_cars, slot_detection_info = match_slots(detections)
                slot_status = smooth_slot_status(raw_status)
                t_slotmatch  = time.time()

                await controller.update(slot_status, slot_detection_info, moving_cars)
                t_broadcast  = time.time()

                log_latency(frame_id, t_start, t_preprocess, t_ransac,
                            t_inference, t_slotmatch, t_broadcast)

            await asyncio.sleep(0)

    except KeyboardInterrupt:
        print("\nPipeline stopped.")
    finally:
        sock.close()
        print("Socket closed.")

if __name__ == "__main__":
    asyncio.run(main())