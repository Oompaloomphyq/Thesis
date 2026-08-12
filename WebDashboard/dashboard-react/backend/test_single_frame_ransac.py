#!/usr/bin/env python3
"""
test_single_frame.py
Runs the full pipeline on a single raw .npy frame:
  raw .npy (Livox) → preprocess (pre_process1.py logic, no RANSAC) → PointPillars → BEV

Usage:
    cd /home/phyq/Thesis/OpenPCDet
    conda activate cpark
    python test_single_frame.py
"""

import os
import sys
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

sys.path.append('/home/phyq/Thesis/OpenPCDet')
os.chdir('/home/phyq/Thesis/OpenPCDet/tools')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# =============================================================================
# SETTINGS — change these if needed
# =============================================================================
INPUT_NPY = '/home/phyq/Thesis/SUSTechPOINTS/data/final_dataset_raw/lidar/001175.bin'
CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/cfgs/kitti_models/pointpillar_custom.yaml'
CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom/ransaced_retrain/ckpt/checkpoint_epoch_150.pth'
SCORE_THRESHOLD = 0.3

# =============================================================================
# PARKING CONFIG — exact copy from pre_process1.py
# =============================================================================
PARKING_CONFIG = {
    'x_min': -20,
    'x_max': 30,
    'y_min': 6,
    'y_max': 10,
    'z_min': 0,
    'z_max': 15,
    'ground_y_livox': 8.2,
}

# =============================================================================
# STEP 1 — PREPROCESS (identical to pre_process1.py, NO RANSAC)
# =============================================================================
def preprocess(points_raw, config):
    # Remove invalid points
    valid_mask = np.all(np.isfinite(points_raw), axis=1)
    points = points_raw[valid_mask]

    # Crop in Livox space
    mask = (
        (points[:, 0] >= config['x_min']) & (points[:, 0] <= config['x_max']) &
        (points[:, 1] >= config['y_min']) & (points[:, 1] <= config['y_max']) &
        (points[:, 2] >= config['z_min']) & (points[:, 2] <= config['z_max'])
    )
    points = points[mask]

    if len(points) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x_livox   = points[:, 0]
    y_livox   = points[:, 1]
    z_livox   = points[:, 2]
    intensity = points[:, 3]

    # Coordinate transformation (Livox → KITTI)
    x_kitti = z_livox
    y_kitti = x_livox
    z_kitti = config['ground_y_livox'] - y_livox   # ground at Z=0

    return np.column_stack([x_kitti, y_kitti, z_kitti, intensity]).astype(np.float32)

# =============================================================================
# STEP 2 — LOAD MODEL
# =============================================================================
class DummyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
    def __len__(self): return 0
    def __getitem__(self, index): return None

def load_model():
    print("Loading PointPillars...")
    logger  = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    dataset = DummyDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, Path('.'), logger)
    model   = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    print("  ✓ Model ready\n")
    return model, dataset

# =============================================================================
# STEP 3 — INFERENCE
# =============================================================================
def run_inference(model, dataset, points):
    input_dict = {'points': points.copy(), 'frame_id': 0}
    data_dict  = dataset.prepare_data(data_dict=input_dict)
    data_dict  = dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    boxes  = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    mask   = scores >= SCORE_THRESHOLD
    return boxes[mask], scores[mask]

# =============================================================================
# STEP 4 — BEV VISUALIZATION
# Portrait orientation matching training BEV (Image 1):
#   - Parking lot runs vertically (top to bottom)
#   - KITTI X (= Livox Z, depth) → vertical axis
#   - KITTI Y (= Livox X, lateral) → horizontal axis
# =============================================================================
def visualize_bev(kitti_points, boxes, scores):
    fig, ax = plt.subplots(figsize=(5, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # --- Point cloud ---
    if len(kitti_points) > 0:
        z      = kitti_points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors = plt.cm.jet(z_norm)
        # horizontal = KITTI Y (lateral), vertical = KITTI X (forward/depth)
        ax.scatter(kitti_points[:, 1], kitti_points[:, 0],
                   c=colors, s=0.8, alpha=0.7, linewidths=0)

    # --- Bounding boxes ---
    for box, score in zip(boxes, scores):
        cx, cy = box[0], box[1]   # KITTI X, KITTI Y
        l,  w  = box[3], box[4]
        yaw    = box[6]

        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners_local = np.array([
            [ l/2,  w/2],
            [ l/2, -w/2],
            [-l/2, -w/2],
            [-l/2,  w/2],
        ])
        corners = (rot @ corners_local.T).T + np.array([cx, cy])

        # plot: x-axis = KITTI Y, y-axis = KITTI X  (same as scatter above)
        poly = plt.Polygon(
            list(zip(corners[:, 1], corners[:, 0])),
            fill=False, edgecolor='orange', linewidth=1.5
        )
        ax.add_patch(poly)
        ax.text(cy, cx, f'{score:.2f}',
                color='yellow', fontsize=6, ha='center', va='center')

    # Axis limits matching KITTI space after pre_process1.py crop:
    #   KITTI X = Livox Z: z_min=0 to z_max=15  →  0 to 15
    #   KITTI Y = Livox X: x_min=-20 to x_max=30 → -20 to 30
    ax.set_xlim(30, -20)    # KITTI Y, flipped so left side of lot is on left
    ax.set_ylim(-1, 16)     # KITTI X, full crop range

    ax.set_xlabel('KITTI Y  (Livox X, lateral) [m]', color='white', fontsize=8)
    ax.set_ylabel('KITTI X  (Livox Z, depth) [m]',   color='white', fontsize=8)
    ax.set_title(f'BEV — {len(boxes)} detection(s) | score ≥ {SCORE_THRESHOLD}',
                 color='white', fontsize=10)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.grid(True, color='#2a2a2a', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('bev_test_output.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    print("\nBEV saved to: bev_test_output.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Single Frame Test Pipeline")
    print("="*60)
    print(f"  Input : {INPUT_NPY}")
    print(f"  Config: {CFG_FILE}")
    print(f"  Ckpt  : {CKPT_FILE}")
    print("="*60 + "\n")

    # Load raw frame
    raw = np.load(INPUT_NPY)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 4)

    print(f"Raw Livox points : {len(raw):,}")
    print(f"  Livox X : {raw[:,0].min():.2f}  to  {raw[:,0].max():.2f}")
    print(f"  Livox Y : {raw[:,1].min():.2f}  to  {raw[:,1].max():.2f}")
    print(f"  Livox Z : {raw[:,2].min():.2f}  to  {raw[:,2].max():.2f}")

    # Preprocess
    kitti_pts = preprocess(raw, PARKING_CONFIG)
    print(f"\nAfter preprocess : {len(kitti_pts):,} points")

    if len(kitti_pts) > 0:
        print(f"  KITTI X : {kitti_pts[:,0].min():.2f}  to  {kitti_pts[:,0].max():.2f}")
        print(f"  KITTI Y : {kitti_pts[:,1].min():.2f}  to  {kitti_pts[:,1].max():.2f}")
        print(f"  KITTI Z : {kitti_pts[:,2].min():.2f}  to  {kitti_pts[:,2].max():.2f}")

    if len(kitti_pts) < 100:
        print("\n[ERROR] Too few points after preprocessing — check input file.")
        sys.exit(1)

    # Inference
    model, dataset = load_model()
    boxes, scores  = run_inference(model, dataset, kitti_pts)

    print(f"Detections : {len(boxes)}")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  [{i+1:2d}] x={b[0]:6.2f}  y={b[1]:7.2f}  z={b[2]:5.2f} "
              f" l={b[3]:.2f}  w={b[4]:.2f}  h={b[5]:.2f}  score={s:.3f}")

    # Visualize
    visualize_bev(kitti_pts, boxes, scores)