#!/usr/bin/env python3
"""
demo_stream.py — Live Point Cloud Visualizer (No Inference)
============================================================
Receives raw points from Raspberry Pi via stream_raw.py and
displays them in a live Open3D window. No PointPillars inference.
Use this to verify the stream is working and to inspect the raw
point cloud before running detection.

Usage:
  python demo_stream.py
"""

import socket
import re
import sys
import os
import time
import threading
import numpy as np
import open3d as o3d
import warnings
from collections import deque

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
RECEIVE_PORT  = 9999
BUFFER_SIZE   = 65536
MAX_POINTS    = 120000
UPDATE_EVERY  = 0.5    # seconds between display updates

PARKING_CONFIG = {
    'x_min': -20, 'x_max': 30,
    'y_min': 6,   'y_max': 10,
    'z_min': 0,   'z_max': 15,
    'ground_y_livox': 8.2,
}

# ═══════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════
point_buffer = deque(maxlen=MAX_POINTS)
buffer_lock  = threading.Lock()
running      = True

# ═══════════════════════════════════════════════════════════
# RECEIVER THREAD
# ═══════════════════════════════════════════════════════════
def receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RECEIVE_PORT))
    sock.settimeout(0.1)
    pattern = re.compile(
        r'x=([-\d.]+)\s+y=([-\d.]+)\s+z=([-\d.]+)\s+r=([-\d]+)',
        re.IGNORECASE
    )
    print(f"Listening for raw points on UDP port {RECEIVE_PORT}...")
    connected = False
    while running:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            if not connected:
                print(f"✓ Receiving from {addr[0]}\n")
                connected = True
            text    = data.decode('utf-8', errors='ignore')
            matches = pattern.findall(text)
            with buffer_lock:
                for m in matches:
                    try:
                        x, y, z, r = float(m[0]), float(m[1]), float(m[2]), float(m[3])
                        if not (x == 0 and y == 0 and z == 0):
                            point_buffer.append([x, y, z, r])
                    except ValueError:
                        continue
        except socket.timeout:
            continue
    sock.close()

# ═══════════════════════════════════════════════════════════
# PREPROCESS (Livox → KITTI, no RANSAC)
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
    x_kitti   = pts[:, 2]
    y_kitti   = pts[:, 0]
    z_kitti   = config['ground_y_livox'] - pts[:, 1]
    intensity = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))
    return np.column_stack([x_kitti, y_kitti, z_kitti, intensity]).astype(np.float32)

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    global running

    # Start UDP receiver
    t = threading.Thread(target=receiver_thread, daemon=True)
    t.start()

    # Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Live Stream Viewer', width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.point_size       = 2.0

    # Point clouds: raw (blue) and cropped/transformed (green)
    pcd_raw     = o3d.geometry.PointCloud()
    pcd_cropped = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_raw)
    vis.add_geometry(pcd_cropped)

    # Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(frame)

    # Camera view (bird's eye)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.05)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    # Grid
    grid_lines = []
    grid_pts   = []
    for i in range(-30, 31, 2):
        grid_pts.extend([[i, -30, 0], [i, 30, 0]])
        grid_lines.append([len(grid_pts)-2, len(grid_pts)-1])
        grid_pts.extend([[-30, i, 0], [30, i, 0]])
        grid_lines.append([len(grid_pts)-2, len(grid_pts)-1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(grid_pts)
    grid.lines  = o3d.utility.Vector2iVector(grid_lines)
    grid.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2]] * len(grid_lines))
    vis.add_geometry(grid)

    last_update = time.time()
    frame_count = 0

    print("=" * 60)
    print("  LIVE STREAM VIEWER")
    print("  Blue  points = raw Livox (all points)")
    print("  Green points = after crop + KITTI transform")
    print("  Close window to quit")
    print("=" * 60 + "\n")

    try:
        while True:
            now = time.time()

            if (now - last_update) >= UPDATE_EVERY:
                last_update = now

                with buffer_lock:
                    if len(point_buffer) == 0:
                        vis.poll_events()
                        vis.update_renderer()
                        continue
                    raw = np.array(point_buffer, dtype=np.float32)

                # Show raw points in blue (Livox space XYZ)
                pcd_raw.points = o3d.utility.Vector3dVector(raw[:, :3])
                pcd_raw.colors = o3d.utility.Vector3dVector(
                    np.tile([0.2, 0.4, 1.0], (len(raw), 1))
                )

                # Show cropped + transformed points in green (KITTI space)
                kitti_pts = preprocess(raw, PARKING_CONFIG)
                if len(kitti_pts) > 0:
                    # Color by Z height
                    z_vals = kitti_pts[:, 2]
                    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
                    import matplotlib.pyplot as plt
                    colors = plt.cm.jet(z_norm)[:, :3]
                    pcd_cropped.points = o3d.utility.Vector3dVector(kitti_pts[:, :3])
                    pcd_cropped.colors = o3d.utility.Vector3dVector(colors)
                else:
                    pcd_cropped.points = o3d.utility.Vector3dVector([])
                    pcd_cropped.colors = o3d.utility.Vector3dVector([])

                vis.update_geometry(pcd_raw)
                vis.update_geometry(pcd_cropped)

                frame_count += 1
                print(f"[Frame {frame_count}] "
                      f"Raw={len(raw):,} pts | "
                      f"Cropped={len(kitti_pts):,} pts | "
                      f"Livox Y: {raw[:,1].min():.2f} to {raw[:,1].max():.2f} | "
                      f"KITTI Z: {kitti_pts[:,2].min():.2f} to {kitti_pts[:,2].max():.2f}"
                      if len(kitti_pts) > 0 else
                      f"[Frame {frame_count}] Raw={len(raw):,} pts | Cropped=0 pts")

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
