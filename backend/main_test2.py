#!/usr/bin/env python3
"""
main_test.py — Offline Pipeline Test using Prerecorded Raw .bin Files
======================================================================
Reads raw .bin files from TEST_DIR, runs the full pipeline:
  preprocess → RANSAC → PointPillars → slot match → dashboard WebSocket

Logs each detection to MySQL lidar_detections table WITH the source filename
so frame-level ground truth comparison is possible.

Usage:
  python main_test.py

Drop raw .bin files into TEST_DIR below.

NOTE: MySQL table must have a frame_file column. Run this once if not yet added:
  ALTER TABLE lidar_detections ADD COLUMN frame_file VARCHAR(255) DEFAULT NULL;
"""

import asyncio
import numpy as np
import torch
import time
import os
import sys
import math
import warnings
import glob
import csv
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
from controller import Controller

# =============================================================================
# CONFIGURATION — synced from main.py
# =============================================================================
TEST_DIR       = "/home/phyq/Thesis/Dataset_Test/march16/live_raw_frames8"
LOOP_FRAMES    = False       # False = stop after last frame (better for evaluation)
FRAME_INTERVAL = 0.5         # seconds between frames

# CSV output — one row per frame showing system's slot decisions
RESULTS_CSV = '/home/phyq/Thesis/Dataset_Test/system_predictions.csv'

CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/tools/cfgs/kitti_models/pointpillar_custom2.yaml'
CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom2/default/ckpt/checkpoint_epoch_150.pth'
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

MIN_POINTS_FOR_INFERENCE = 1000   # synced from main.py (was 500 in old main_test)

# =============================================================================
# TEMPORAL SMOOTHING — synced from main.py
# =============================================================================
HISTORY_SIZE       = 4
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
# SLOT CENTERS — synced from main.py
# =============================================================================
SLOT_CENTERS = [
    # --- Angled slots ---
    {"slot_id": "slot_1",  "x": 8.590, "y": -12.240, "type": "angled"},
    {"slot_id": "slot_2",  "x": 8.555, "y":  -9.240, "type": "angled"},
    {"slot_id": "slot_3",  "x": 8.531, "y":  -6.240, "type": "angled"},
    {"slot_id": "slot_4",  "x": 8.331, "y":  -3.240, "type": "angled"},
    {"slot_id": "slot_5",  "x": 8.273, "y":  -0.600, "type": "angled"},
    {"slot_id": "slot_6",  "x": 7.280, "y":   2.060, "type": "angled"},   # x=7.280 from main.py
    {"slot_id": "slot_7",  "x": 8.461, "y":   5.200, "type": "angled"},
    {"slot_id": "slot_8",  "x": 8.130, "y":   8.000, "type": "angled"},
    {"slot_id": "slot_9",  "x": 8.072, "y":  11.000, "type": "angled"},
    {"slot_id": "slot_10", "x": 7.000, "y":  13.000, "type": "angled"},   # x=7.000, y=13.000 from main.py

    # --- Parallel slots ---
    {"slot_id": "slot_11", "x": 1.000, "y":   7.043, "type": "parallel"},
    {"slot_id": "slot_12", "x": 1.000, "y":   2.556, "type": "parallel"},
    {"slot_id": "slot_13", "x": 1.092, "y":  -1.931, "type": "parallel"},
    {"slot_id": "slot_14", "x": 1.030, "y":  -6.418, "type": "parallel"},
    {"slot_id": "slot_15", "x": 1.041, "y": -10.905, "type": "parallel"},
    {"slot_id": "slot_16", "x": 1.088, "y": -15.392, "type": "parallel"},
]

ANGLED_THRESHOLD = 1.3    # synced from main.py (was 1.5 in old main_test)
PARALLEL_HALF_X  = 1.0
PARALLEL_HALF_Y  = 2.5

# =============================================================================
# CSV LOGGING
# =============================================================================
SLOT_IDS = [f"slot_{i}" for i in range(1, 17)]

def init_results_csv():
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_file'] + SLOT_IDS)
    print(f"  [CSV] Results will be saved to: {RESULTS_CSV}")

def log_frame_result(filename, smoothed_status):
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [filename] + [smoothed_status.get(sid, 'vacant') for sid in SLOT_IDS]
        writer.writerow(row)

# =============================================================================
# STEP 1 — LOAD RAW FILE
# =============================================================================
def load_raw(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

# =============================================================================
# STEP 2 — PREPROCESS
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
# STEP 3 — RANSAC GROUND REMOVAL
# =============================================================================
def ransac_ground_removal(points):
    if len(points) < RANSAC_MIN_INLIERS:
        return points
    xyz = points[:, :3]
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

# =============================================================================
# STEP 4 — POINTPILLARS INFERENCE
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
    print("  ✓ Model loaded")
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
    for i, det in enumerate(detections):
        print(f"    [{i}] x={det[0]:.2f} y={det[1]:.2f} score={det[7]:.3f}")
    return detections

# =============================================================================
# STEP 5 — SLOT MATCHING
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
                    "x":     round(float(veh_x), 3),
                    "y":     round(float(veh_y), 3),
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

    occupied = sum(1 for v in slot_status.values() if v == "occupied")
    print(f"  [Slots] {occupied}/16 occupied (raw, before smoothing)")
    return slot_status, moving_cars, slot_detection_info

# =============================================================================
# MAIN LOOP
# =============================================================================
async def main():
    frame_files = sorted([Path(f) for f in glob.glob(os.path.join(TEST_DIR, '*.bin'))])
    if len(frame_files) == 0:
        print(f"[ERROR] No .bin files found in {TEST_DIR}")
        return

    print(f"Found {len(frame_files)} raw frames in {TEST_DIR}\n")

    model, dataset = load_model()

    controller = Controller()
    await controller.start_ws_server()

    init_results_csv()

    print("=" * 60)
    print("  OFFLINE PIPELINE TEST")
    print(f"  Frames     : {len(frame_files)}")
    print(f"  Loop       : {LOOP_FRAMES}")
    print(f"  Interval   : {FRAME_INTERVAL}s per frame")
    print(f"  ground_y   : {PARKING_CONFIG['ground_y_livox']}")
    print(f"  Threshold  : ANGLED={ANGLED_THRESHOLD}  PARALLEL±X={PARALLEL_HALF_X} ±Y={PARALLEL_HALF_Y}")
    print(f"  Smoothing  : {OCCUPIED_THRESHOLD}/{HISTORY_SIZE} frames to confirm")
    print("=" * 60 + "\n")

    frame_idx = 0

    try:
        while True:
            file_path = frame_files[frame_idx % len(frame_files)]
            filename  = file_path.name   # e.g. "frame_001.bin"

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Frame {frame_idx + 1}/{len(frame_files)} — {filename}")

            # Step 1 — Load
            raw_points = load_raw(str(file_path))
            print(f"  [Load]       {len(raw_points):,} raw points")

            # Step 2 — Preprocess
            kitti_points = preprocess(raw_points, PARKING_CONFIG)
            print(f"  [Preprocess] {len(raw_points):,} → {len(kitti_points):,} pts")

            if len(kitti_points) == 0:
                print("  [Skip] No points after crop")
                frame_idx += 1
                await asyncio.sleep(FRAME_INTERVAL)
                continue

            # Step 3 — RANSAC
            clean_points = ransac_ground_removal(kitti_points)

            # Step 4 — Inference
            detections = run_inference(model, dataset, clean_points)

            # Step 5 — Slot matching (raw)
            raw_status, moving_cars, slot_detection_info = match_slots(detections)

            # Step 5b — Temporal smoothing
            smoothed_status = smooth_slot_status(raw_status)
            occupied_after  = sum(1 for v in smoothed_status.values() if v == "occupied")
            print(f"  [Smooth] {occupied_after}/16 occupied (after smoothing)")

            # Step 5c — Save frame result to CSV
            log_frame_result(filename, smoothed_status)
            print(f"  [CSV]    Logged → {filename}")

            # Step 6 — Send to dashboard
            await controller.update(smoothed_status, slot_detection_info, moving_cars)

            # Advance frame
            frame_idx += 1
            if frame_idx >= len(frame_files):
                if LOOP_FRAMES:
                    print("\n  [Loop] Restarting from frame 1...\n")
                    frame_idx = 0
                    # Reset smoothing history on loop so old frames don't bleed into next pass
                    for h in slot_history.values():
                        h.clear()
                        h.extend([0] * HISTORY_SIZE)
                else:
                    print("\n✓ All frames processed.")
                    break

            await asyncio.sleep(FRAME_INTERVAL)

    except KeyboardInterrupt:
        print("\nTest stopped.")

if __name__ == "__main__":
    asyncio.run(main())
