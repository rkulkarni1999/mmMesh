#!/usr/bin/env python3
"""
Open3D visualizations for the mmWave dataset.

This file provides:
    create_dataset_visualizations(train, test, output_dir, frame_rate, max_frames_for_videos)

which is called by your existing dataset script. It expects:
    train, test : numpy arrays with shape either
                  (num_frames, pc_size, 6)  or
                  (num_files, num_frames, pc_size, 6)
                  where last dim is [x, y, z, velocity, snr, range]

Controls in the Open3D window:
    Space   : Play / Pause
    N       : Next frame
    P       : Previous frame
    Q / Esc : Quit
"""

import time
from typing import Optional

import numpy as np
import open3d as o3d


def _flatten_frames(arr: np.ndarray, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Make sure data is (num_frames, pc_size, 6).

    Handles:
        (F, P, 6)        -> unchanged
        (B, F, P, 6)     -> flattened to (B*F, P, 6)
    """
    if arr is None or arr.size == 0:
        return np.zeros((0, 0, 6), dtype=np.float32)

    if arr.ndim == 3:
        frames = arr
    elif arr.ndim == 4:
        # (B, F, P, 6) -> (B*F, P, 6)
        B, F, P, C = arr.shape
        frames = arr.reshape(B * F, P, C)
    else:
        raise ValueError(f"Unsupported array shape for visualization: {arr.shape}")

    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]

    return frames


def _visualize_frames_open3d(frames: np.ndarray, fps: float = 10.0, title: str = "PointCloud"):
    """
    Animate a list of frames in Open3D.

    frames: (num_frames, pc_size, 6)
            last dim is [x, y, z, velocity, snr, range]
    """
    num_frames = frames.shape[0]
    if num_frames == 0:
        print(f"[{title}] No frames to visualize.")
        return

    print(f"[{title}] Visualizing {num_frames} frame(s) with Open3D")

    # State for the viewer
    state = {
        "frames": frames,
        "idx": 0,
        "playing": False,
        "last_time": time.time(),
        "fps": max(0.1, float(fps)),
    }

    # Create Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)

    # Geometry: point cloud + coordinate frame
    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    state["pcd"] = pcd

    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------
    def update_pcd_for_frame(idx: int, vis_obj):
        frame = state["frames"][idx]  # (pc_size, 6)
        pts = frame[:, :3]            # x, y, z
        snr = frame[:, 4]             # snr

        # Points
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # Color by SNR (simple normalized colormap)
        if snr.size > 0:
            s_min = float(np.min(snr))
            s_max = float(np.max(snr))
            if s_max > s_min:
                s_norm = (snr - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(snr)

            # Map: low SNR -> blue, high SNR -> yellow
            colors = np.stack(
                [
                    s_norm,        # R
                    s_norm,        # G
                    1.0 - s_norm,  # B
                ],
                axis=1,
            )
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        else:
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((pts.shape[0], 3), dtype=np.float64)
            )

        vis_obj.update_geometry(pcd)
        print(f"Frame {idx+1}/{num_frames}  (points={pts.shape[0]})")

    def go_to_frame(new_idx: int, vis_obj):
        new_idx = new_idx % num_frames
        state["idx"] = new_idx
        update_pcd_for_frame(new_idx, vis_obj)
        return True

    # ------------------------------------------------
    # Key callbacks
    # ------------------------------------------------
    def cb_next(vis_obj):
        return go_to_frame(state["idx"] + 1, vis_obj)

    def cb_prev(vis_obj):
        return go_to_frame(state["idx"] - 1, vis_obj)

    def cb_toggle_play(vis_obj):
        state["playing"] = not state["playing"]
        mode = "PLAY" if state["playing"] else "PAUSE"
        print(f"[{title}] {mode}")
        state["last_time"] = time.time()
        return False

    def cb_quit(vis_obj):
        print(f"[{title}] QUIT")
        vis_obj.close()
        return False

    # ------------------------------------------------
    # Animation callback
    # ------------------------------------------------
    def animation_callback(vis_obj):
        if not state["playing"]:
            return False

        now = time.time()
        if now - state["last_time"] < 1.0 / state["fps"]:
            return False

        state["last_time"] = now
        go_to_frame(state["idx"] + 1, vis_obj)
        return True

    # ------------------------------------------------
    # Register & run
    # ------------------------------------------------
    vis.add_geometry(axis)
    update_pcd_for_frame(0, vis)
    vis.add_geometry(pcd)

    print("Controls: Space=Play/Pause, N=Next, P=Prev, Q/Esc=Quit")

    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("P"), cb_prev)
    vis.register_key_callback(ord(" "), cb_toggle_play)
    vis.register_key_callback(ord("Q"), cb_quit)
    vis.register_key_callback(256, cb_quit)  # ESC
    vis.register_animation_callback(animation_callback)

    vis.run()
    vis.destroy_window()


def create_dataset_visualizations(
    train: np.ndarray,
    test: np.ndarray,
    output_dir: str,
    frame_rate: float,
    max_frames_for_videos: Optional[int] = None,
):
    """
    Entry point called by your existing script.

    It will:
      1. Flatten train/test to (num_frames, pc_size, 6)
      2. Open an Open3D animated viewer for train
      3. Then (optionally) another viewer for test
    """
    print("\n[visualize] Preparing frames for visualization...")

    train_frames = _flatten_frames(train, max_frames_for_videos)
    test_frames = _flatten_frames(test, max_frames_for_videos)

    if train_frames.size > 0:
        _visualize_frames_open3d(
            train_frames,
            fps=frame_rate,
            title="mmWave Train Point Clouds (spherical -> cartesian)",
        )
    else:
        print("[visualize] No train frames to visualize.")

    if test_frames.size > 0:
        _visualize_frames_open3d(
            test_frames,
            fps=frame_rate,
            title="mmWave Test Point Clouds (spherical -> cartesian)",
        )
    else:
        print("[visualize] No test frames to visualize.")
