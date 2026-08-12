#!/usr/bin/env python3
"""
replay_debug.py — Offline LiDAR Visualizer using Pre-recorded Raw .bin Files
=============================================================================
Replays raw 4-channel (x, y, z, r) .bin frames through the full pipeline:
  preprocess → RANSAC ground removal → PointPillars inference → Open3D display

The .bin files are raw Livox output (same format as stream_raw.py UDP stream),
so the full preprocessing pipeline is applied to each frame exactly as
live_debug.py does with live UDP data — no Pi or sensor required.

Controls:
  SPACE        → pause / resume auto-playback
  N            → next frame (works when paused OR playing)
  R            → reset camera view
  Q / ESC      → quit

Usage:
  # Use default BIN_DIR defined below
  python replay_debug.py

  # Override folder and delay from command line
  python replay_debug.py --bin_dir /path/to/raw/bins --delay 1.5
"""

import sys
import os
import time
import argparse
import numpy as np
import open3d as o3d
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

sys.path.append('/home/phyq/Thesis/OpenPCDet')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# ═══════════════════════════════════════════════════════════
# CONFIG — edit these to match your setup
# ═══════════════════════════════════════════════════════════
BIN_DIR         = '/home/phyq/Thesis/Dataset_Test/march16/live_raw_frames4'   # folder containing raw .bin files
DELAY           = 1.0                               # seconds between frames (auto-play)
SCORE_THRESHOLD = 0.3

CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/tools/cfgs/kitti_models/pointpillar_custom2.yaml'
CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom2/default/ckpt/checkpoint_epoch_150.pth'

# Preprocessing config — matches live_debug.py exactly
PARKING_CONFIG = {
    'x_min': -20, 'x_max': 15,
    'y_min': 6.5,  'y_max': 10,
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
# PREPROCESSING — matches live_debug.py exactly
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
# RANSAC — matches live_debug.py exactly
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
    if len(points) < 50:
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
    return [{'box': box, 'score': float(score)}
            for box, score in zip(boxes[mask], scores[mask])]

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
# LOAD .BIN FILES
# ═══════════════════════════════════════════════════════════
def load_bin_files(bin_dir):
    files = sorted(Path(bin_dir).glob('*.bin'))
    if not files:
        print(f"ERROR: No .bin files found in {bin_dir}")
        sys.exit(1)
    print(f"Found {len(files)} .bin files in {bin_dir}\n")
    return files

def read_bin(filepath):
    """Read raw 4-channel (x, y, z, r) .bin file."""
    data = np.fromfile(filepath, dtype=np.float32)
    if len(data) % 4 != 0:
        print(f"  WARNING: {filepath.name} has {len(data)} floats — not divisible by 4, skipping")
        return None
    return data.reshape(-1, 4)

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_dir', type=str, default=BIN_DIR,
                        help='Folder containing raw 4-channel .bin files')
    parser.add_argument('--delay',   type=float, default=DELAY,
                        help='Seconds between frames in auto-play mode (default: 1.0)')
    args = parser.parse_args()

    bin_files = load_bin_files(args.bin_dir)
    model, dataset = load_model()

    # ── Open3D setup ─────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Replay Debug', width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.point_size       = 2.0

    pcd_ground    = o3d.geometry.PointCloud()
    pcd_nonground = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_ground)
    vis.add_geometry(pcd_nonground)

    # frame_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    # vis.add_geometry(frame_axis)

    # Grid
    grid_pts, grid_lines = [], []
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

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    # ── Shared state ─────────────────────────────────────────
    state = {
        'frame_idx':    0,
        'paused':       False,
        'next_frame':   False,
        'active_boxes': [],
        'last_time':    time.time(),
    }

    # ── Key callbacks ─────────────────────────────────────────
    def on_space(vis):
        state['paused'] = not state['paused']
        print(f"  [{'PAUSED' if state['paused'] else 'PLAYING'}]")
        return False

    def on_n(vis):
        state['next_frame'] = True
        return False

    def on_reset(vis):
        c = vis.get_view_control()
        c.set_zoom(0.1)
        c.set_front([0, 0, -1])
        c.set_lookat([0, 0, 0])
        c.set_up([0, -1, 0])
        return False

    vis.register_key_callback(ord(' '), on_space)
    vis.register_key_callback(ord('N'), on_n)
    vis.register_key_callback(ord('R'), on_reset)

    # ── Process one frame ─────────────────────────────────────
    def process_frame(idx):
        filepath = bin_files[idx]
        print(f"\n[Frame {idx + 1}/{len(bin_files)}] {filepath.name}")

        raw = read_bin(filepath)
        if raw is None:
            return

        kitti_pts = preprocess(raw, PARKING_CONFIG)
        if len(kitti_pts) == 0:
            print("  WARNING: No points after preprocessing — skipping")
            return

        non_ground, ground = ransac_ground_removal(kitti_pts)

        print(f"  Raw={len(raw):,} → KITTI={len(kitti_pts):,} → "
              f"NonGround={len(non_ground):,} | Ground={len(ground):,}")

        detections = run_inference(model, dataset, non_ground)
        print(f"  Detections: {len(detections)}")
        for i, d in enumerate(detections):
            b = d['box']
            print(f"    [{i}] x={b[0]:.2f} y={b[1]:.2f} z={b[2]:.2f} "
                  f"dx={b[3]:.2f} dy={b[4]:.2f} dz={b[5]:.2f} "
                  f"score={d['score']:.3f}")

        # Update point clouds
        if len(non_ground) > 0:
            pcd_nonground.points = o3d.utility.Vector3dVector(non_ground[:, :3])
            pcd_nonground.colors = o3d.utility.Vector3dVector(
                np.tile(COLOR_POINTS, (len(non_ground), 1)))
        else:
            pcd_nonground.points = o3d.utility.Vector3dVector([])
            pcd_nonground.colors = o3d.utility.Vector3dVector([])

        if len(ground) > 0:
            pcd_ground.points = o3d.utility.Vector3dVector(ground[:, :3])
            pcd_ground.colors = o3d.utility.Vector3dVector(
                np.tile(COLOR_GROUND, (len(ground), 1)))
        else:
            pcd_ground.points = o3d.utility.Vector3dVector([])
            pcd_ground.colors = o3d.utility.Vector3dVector([])

        vis.update_geometry(pcd_nonground)
        vis.update_geometry(pcd_ground)

        # Swap bounding boxes
        for box_ls in state['active_boxes']:
            vis.remove_geometry(box_ls, reset_bounding_box=False)
        state['active_boxes'].clear()

        for d in detections:
            ls = make_box_lines(d['box'], d['score'])
            vis.add_geometry(ls, reset_bounding_box=False)
            state['active_boxes'].append(ls)

    # ── Print controls ────────────────────────────────────────
    print("=" * 60)
    print("  REPLAY DEBUG VISUALIZER")
    print(f"  {len(bin_files)} frames  |  delay={args.delay}s  |  {args.bin_dir}")
    print()
    print("  Green points = non-ground (fed into PointPillars)")
    print("  Gray  points = ground (removed by RANSAC)")
    print("  Red   boxes  = detections (score >= 0.6)")
    print("  Orange boxes = detections (score 0.3–0.6)")
    print()
    print("  SPACE → pause / resume")
    print("  N     → next frame")
    print("  R     → reset camera")
    print("  Close window to quit")
    print("=" * 60 + "\n")

    # Process first frame immediately on launch
    process_frame(0)

    # ── Main loop ─────────────────────────────────────────────
    try:
        while True:
            now = time.time()

            should_advance = (
                (not state['paused'] and (now - state['last_time']) >= args.delay)
                or state['next_frame']
            )

            if should_advance:
                state['next_frame'] = False
                state['last_time']  = now
                state['frame_idx']  = (state['frame_idx'] + 1) % len(bin_files)
                process_frame(state['frame_idx'])

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        vis.destroy_window()

if __name__ == '__main__':
    main()