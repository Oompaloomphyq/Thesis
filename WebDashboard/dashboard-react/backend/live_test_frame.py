#!/usr/bin/env python3
# live_test_frame.py
import sys
import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
# CONFIG — must match training exactly
# =============================================================================
PARKING_CONFIG = {
    'x_min': -20, 'x_max': 30,
    'y_min': 6,   'y_max': 10,
    'z_min': 0,   'z_max': 15,
    'ground_y_livox': 8.2,
}

CFG_FILE  = '/home/phyq/Thesis/OpenPCDet/cfgs/kitti_models/pointpillar_custom2.yaml'
CKPT_FILE = '/home/phyq/Thesis/OpenPCDet/output/kitti_models/pointpillar_custom2/default/ckpt/checkpoint_epoch_150.pth'
SCORE_THRESHOLD = 0.3

# =============================================================================
# PREPROCESS — identical to pre_process1.py, NO RANSAC
# =============================================================================
def preprocess(points_raw):
    cfg = PARKING_CONFIG
    valid = np.all(np.isfinite(points_raw), axis=1)
    pts   = points_raw[valid]

    mask = (
        (pts[:, 0] >= cfg['x_min']) & (pts[:, 0] <= cfg['x_max']) &
        (pts[:, 1] >= cfg['y_min']) & (pts[:, 1] <= cfg['y_max']) &
        (pts[:, 2] >= cfg['z_min']) & (pts[:, 2] <= cfg['z_max'])
    )
    pts = pts[mask]
    if len(pts) == 0:
        return pts

    x_kitti = pts[:, 2]                             # Livox Z → KITTI X
    y_kitti = pts[:, 0]                             # Livox X → KITTI Y
    z_kitti = cfg['ground_y_livox'] - pts[:, 1]    # ground_y - Livox Y → KITTI Z
    intensity = pts[:, 3]

    return np.column_stack([x_kitti, y_kitti, z_kitti, intensity]).astype(np.float32)

# =============================================================================
# MODEL
# =============================================================================
class DummyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
    def __len__(self): return 0
    def __getitem__(self, index): return None

def load_model():
    print("Loading PointPillars model...")
    logger  = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    dataset = DummyDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, Path('.'), logger)
    model   = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    print("  ✓ Model ready\n")
    return model, dataset

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
# BEV VISUALIZATION
# — KITTI space: X = forward (long axis), Y = lateral (short axis)
# — BEV top-down: plot X on vertical axis, Y on horizontal axis
#   so the parking lot appears in the same portrait orientation as training
# =============================================================================
def visualize_bev(kitti_points, boxes, scores, title="BEV — No RANSAC (matches training)"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Plot point cloud: X vertical (forward), Y horizontal (lateral)
    if len(kitti_points) > 0:
        z_vals  = kitti_points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_norm  = (z_vals - z_min) / (z_max - z_min + 1e-6)
        colors  = plt.cm.jet(z_norm)
        # Scatter: x-axis = KITTI Y (lateral), y-axis = KITTI X (forward)
        ax.scatter(kitti_points[:, 1], kitti_points[:, 0],
                   c=colors, s=0.5, alpha=0.6, linewidths=0)

    # Draw bounding boxes
    for i, (box, score) in enumerate(zip(boxes, scores)):
        cx, cy = box[0], box[1]   # KITTI X (forward), Y (lateral)
        l,  w  = box[3], box[4]   # length, width
        yaw    = box[6]

        corners_local = np.array([
            [ l/2,  w/2],
            [ l/2, -w/2],
            [-l/2, -w/2],
            [-l/2,  w/2],
        ])
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (rot @ corners_local.T).T + np.array([cx, cy])

        # corners[:,0]=KITTI X (plot as Y axis), corners[:,1]=KITTI Y (plot as X axis)
        poly = plt.Polygon(
            list(zip(corners[:, 1], corners[:, 0])),
            fill=False, edgecolor='orange', linewidth=1.2
        )
        ax.add_patch(poly)
        ax.text(cy, cx, f'{score:.2f}', color='yellow',
                fontsize=5, ha='center', va='center')

    ax.set_xlabel('KITTI Y — lateral (m)', color='white', fontsize=9)
    ax.set_ylabel('KITTI X — forward (m)', color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=10, pad=8)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    ax.set_xlim(-25, 25)   # KITTI Y range
    ax.set_ylim(-5, 30)    # KITTI X range

    ax.invert_xaxis()      # match training BEV orientation

    ax.grid(True, color='#333', linewidth=0.4)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_live_frame.py <path_to_frame.npy>")
        print("Example: python test_live_frame.py /home/phyq/Thesis/new_data_processed/demoed/000000.npy")
        sys.exit(1)

    npy_path = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"  Input: {npy_path}")
    print(f"{'='*60}\n")

    # Load raw .npy frame (shape: [N, 4] — x,y,z,r in Livox space)
    raw = np.load(npy_path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 4)
    print(f"Raw points: {len(raw)}")
    print(f"  Livox X: {raw[:,0].min():.2f} to {raw[:,0].max():.2f}")
    print(f"  Livox Y: {raw[:,1].min():.2f} to {raw[:,1].max():.2f}")
    print(f"  Livox Z: {raw[:,2].min():.2f} to {raw[:,2].max():.2f}")

    # Preprocess (NO RANSAC — matches training)
    kitti_pts = preprocess(raw)
    print(f"\nAfter preprocess: {len(kitti_pts)} points")
    if len(kitti_pts) > 0:
        print(f"  KITTI X: {kitti_pts[:,0].min():.2f} to {kitti_pts[:,0].max():.2f}")
        print(f"  KITTI Y: {kitti_pts[:,1].min():.2f} to {kitti_pts[:,1].max():.2f}")
        print(f"  KITTI Z: {kitti_pts[:,2].min():.2f} to {kitti_pts[:,2].max():.2f}")

    if len(kitti_pts) < 100:
        print("\n[ERROR] Too few points after preprocessing!")
        print("  Check that your .npy file contains raw Livox data (not already preprocessed)")
        sys.exit(1)

    # Inference
    model, dataset = load_model()
    boxes, scores  = run_inference(model, dataset, kitti_pts)
    print(f"\nDetections: {len(boxes)}")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  [{i+1}] x={b[0]:.2f} y={b[1]:.2f} z={b[2]:.2f} "
              f"l={b[3]:.2f} w={b[4]:.2f} h={b[5]:.2f} score={s:.3f}")

    # Visualize
    visualize_bev(kitti_pts, boxes, scores)