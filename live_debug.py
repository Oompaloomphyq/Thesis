#!/usr/bin/env python3
"""
live_debug.py — Live LiDAR Visualizer with Detection Boxes
===========================================================
Receives raw points from Raspberry Pi, preprocesses, runs PointPillars,
and shows everything in a live Open3D window so you can visually verify
if detections are accurate.

Controls:
  Q / ESC  → quit
  R        → reset camera view

Usage:
  python live_debug.py
"""

import socket
import sys
import os
import time
import threading
import numpy as np
import open3d as o3d
import torch
import warnings
from pathlib import Path
from collections import deque

warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

sys.path.append('/home/phyq/Thesis/OpenPCDet')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# ═══════════════════════════════════════════════════════════
# CONFIG — matches main.py exactly
# ═══════════════════════════════════════════════════════════
RECEIVE_PORT    = 9999
BUFFER_SIZE     = 65536
MAX_POINTS      = 120000
SCORE_THRESHOLD = 0.3
INFERENCE_EVERY = 3.0

CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/tools/cfgs/kitti_models/pointpillar_custom2.yaml'
CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom2/default/ckpt/checkpoint_epoch_150.pth'

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

# ═══════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════
COLOR_GROUND   = [0.5, 0.5, 0.5]
COLOR_POINTS   = [0.2, 0.8, 0.2]
COLOR_BOX_HIGH = [1.0, 0.0, 0.0]
COLOR_BOX_LOW  = [1.0, 0.6, 0.0]

# ═══════════════════════════════════════════════════════════
# PREPROCESSING — matches main.py exactly
# ═══════════════════════════════════════════════════════════
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
    x_livox     = pts[:, 0]
    y_livox     = pts[:, 1]
    z_livox     = pts[:, 2]
    intensity   = pts[:, 3] / 255.0
    y_corrected = y_livox - (z_livox * config['tilt_factor'])
    x_kitti     = z_livox
    y_kitti     = x_livox
    z_kitti     = config['ground_y_livox'] - y_corrected
    return np.column_stack([x_kitti, y_kitti, z_kitti, intensity]).astype(np.float32)

# ═══════════════════════════════════════════════════════════
# RANSAC — matches main.py exactly
# ═══════════════════════════════════════════════════════════
def ransac_ground_removal(points):
    if len(points) < RANSAC_MIN_INLIERS:
        return points, points[:0]
    xyz = points[:, :3]

    candidates = xyz[xyz[:, 2] < 0.5]
    if len(candidates) < 3:
        return points, points[:0]

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
        return points, points[:0]
    return points[~best_inliers], points[best_inliers]

# ═══════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════
class DummyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
    def __len__(self): return 0
    def __getitem__(self, index): return None

def load_model():
    print("Loading PointPillars model...")
    os.chdir('/home/phyq/Thesis/OpenPCDet/tools')
    logger  = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    dataset = DummyDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, Path('.'), logger)
    model   = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    print("✓ Model loaded!\n")
    return model, dataset

def run_inference(model, dataset, points):
    if len(points) < 500:
        return []
    input_dict = {'points': points.copy(), 'frame_id': 0}
    data_dict  = dataset.prepare_data(data_dict=input_dict)
    data_dict  = dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    boxes  = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    mask   = scores >= SCORE_THRESHOLD
    results = []
    for box, score in zip(boxes[mask], scores[mask]):
        results.append({'box': box, 'score': float(score)})
    return results

# ═══════════════════════════════════════════════════════════
# OPEN3D BOX DRAWING
# ═══════════════════════════════════════════════════════════
def make_box_lines(box, score):
    cx, cy, cz, dx, dy, dz, yaw = box
    color = COLOR_BOX_HIGH if score >= 0.6 else COLOR_BOX_LOW

    hw, hl, hh = dx/2, dy/2, dz/2
    corners = np.array([
        [ hw,  hl, -hh], [ hw, -hl, -hh], [-hw, -hl, -hh], [-hw,  hl, -hh],
        [ hw,  hl,  hh], [ hw, -hl,  hh], [-hw, -hl,  hh], [-hw,  hl,  hh],
    ])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0],
                  [sin_y,  cos_y, 0],
                  [0,      0,     1]])
    corners = (R @ corners.T).T + np.array([cx, cy, cz])
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls

# ═══════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════
point_buffer = deque(maxlen=MAX_POINTS)
buffer_lock  = threading.Lock()
running      = True

# ═══════════════════════════════════════════════════════════
# RECEIVER THREAD — binary, matches main.py
# ═══════════════════════════════════════════════════════════
def receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RECEIVE_PORT))
    sock.settimeout(0.1)
    print(f"Listening for binary points on UDP port {RECEIVE_PORT}...")
    connected = False
    while running:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            if not connected:
                print(f"✓ Receiving binary stream from {addr[0]}\n")
                connected = True

            num_floats = len(data) // 4
            if num_floats < 4 or (num_floats % 4) != 0:
                continue

            pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
            with buffer_lock:
                for pt in pts:
                    x, y, z, r = pt[0], pt[1], pt[2], pt[3]
                    if not (x == 0.0 and y == 0.0 and z == 0.0):
                        point_buffer.append([x, y, z, r])

        except socket.timeout:
            continue
    sock.close()

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    global running

    model, dataset = load_model()

    t = threading.Thread(target=receiver_thread, daemon=True)
    t.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Live LiDAR Debug', width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.point_size       = 2.0

    pcd_ground    = o3d.geometry.PointCloud()
    pcd_nonground = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_ground)
    vis.add_geometry(pcd_nonground)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(frame)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    grid_lines = []
    grid_pts   = []
    for i in range(-60, 61, 5):
        grid_pts.extend([[i, -60, 0], [i, 60, 0]])
        grid_lines.append([len(grid_pts)-2, len(grid_pts)-1])
        grid_pts.extend([[-60, i, 0], [60, i, 0]])
        grid_lines.append([len(grid_pts)-2, len(grid_pts)-1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(grid_pts)
    grid.lines  = o3d.utility.Vector2iVector(grid_lines)
    grid.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2]] * len(grid_lines))
    vis.add_geometry(grid)

    active_boxes = []
    last_infer   = time.time()
    frame_count  = 0

    print("=" * 60)
    print("  LIVE DEBUG VISUALIZER")
    print("  Green points = non-ground (input to PointPillars)")
    print("  Gray points  = ground (removed by RANSAC)")
    print("  Red boxes    = detections (score >= 0.6)")
    print("  Orange boxes = detections (score 0.3–0.6)")
    print("  Close window to quit")
    print("=" * 60 + "\n")

    try:
        while True:
            now = time.time()

            if (now - last_infer) >= INFERENCE_EVERY:
                last_infer = now

                with buffer_lock:
                    if len(point_buffer) < 500:
                        vis.poll_events()
                        vis.update_renderer()
                        continue
                    raw = np.array(point_buffer, dtype=np.float32)

                kitti_pts = preprocess(raw, PARKING_CONFIG)
                if len(kitti_pts) == 0:
                    vis.poll_events()
                    vis.update_renderer()
                    continue

                non_ground, ground = ransac_ground_removal(kitti_pts)

                frame_count += 1
                print(f"[Frame {frame_count}] "
                      f"Raw={len(raw):,} → KITTI={len(kitti_pts):,} → "
                      f"NonGround={len(non_ground):,} | Ground={len(ground):,}")

                detections = run_inference(model, dataset, non_ground)
                print(f"  Detections: {len(detections)}")
                for i, d in enumerate(detections):
                    b = d['box']
                    print(f"    [{i}] x={b[0]:.2f} y={b[1]:.2f} z={b[2]:.2f} "
                          f"dx={b[3]:.2f} dy={b[4]:.2f} dz={b[5]:.2f} "
                          f"score={d['score']:.3f}")

                if len(non_ground) > 0:
                    pcd_nonground.points = o3d.utility.Vector3dVector(non_ground[:, :3])
                    pcd_nonground.colors = o3d.utility.Vector3dVector(
                        np.tile(COLOR_POINTS, (len(non_ground), 1))
                    )
                else:
                    pcd_nonground.points = o3d.utility.Vector3dVector([])
                    pcd_nonground.colors = o3d.utility.Vector3dVector([])

                if len(ground) > 0:
                    pcd_ground.points = o3d.utility.Vector3dVector(ground[:, :3])
                    pcd_ground.colors = o3d.utility.Vector3dVector(
                        np.tile(COLOR_GROUND, (len(ground), 1))
                    )
                else:
                    pcd_ground.points = o3d.utility.Vector3dVector([])
                    pcd_ground.colors = o3d.utility.Vector3dVector([])

                vis.update_geometry(pcd_nonground)
                vis.update_geometry(pcd_ground)

                for box_ls in active_boxes:
                    vis.remove_geometry(box_ls, reset_bounding_box=False)
                active_boxes.clear()

                for d in detections:
                    ls = make_box_lines(d['box'], d['score'])
                    vis.add_geometry(ls, reset_bounding_box=False)
                    active_boxes.append(ls)

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        running = False
        vis.destroy_window()

if __name__ == '__main__':
    main()