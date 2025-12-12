#!/usr/bin/env python3
"""
Animate mmWave radar frames (cartesian) from JSON using Open3D.

Supports:
- A single JSON object (one frame)
- A JSON list of frames
- NDJSON (one JSON object per line; each line is a frame)

Controls:
    Space   : Play / Pause
    N       : Next frame
    P       : Previous frame
    Q / Esc : Quit

Usage:
    python radar_open3d_anim.py radar_frames.json
    python radar_open3d_anim.py radar_frames.json --min-snr 1000 --fps 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import open3d as o3d


# -------------------------
#   File / data helpers
# -------------------------

def load_frames(path: Path) -> List[Dict[str, Any]]:
    """Load radar frames from JSON / NDJSON file."""
    text = path.read_text().strip()
    frames: List[Dict[str, Any]] = []

    # Try normal JSON first (dict or list)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            frames = [data]
        elif isinstance(data, list):
            frames = data
        else:
            raise ValueError("Top-level JSON must be dict or list.")
        return frames
    except Exception:
        # Fallback to NDJSON (one JSON object per line)
        frames = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frames.append(json.loads(line))
        return frames


def extract_points_and_snr_for_frame(
    frame: Dict[str, Any],
    min_snr: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (x, y, z) and SNR for a single frame.

    Returns:
        points: (N, 3) float32 array
        snr:    (N,) float32 array
    """
    pts = []
    snr_list = []

    for p in frame.get("points", []):
        cart = p.get("cartesian", None)
        sph = p.get("spherical", {})
        if cart is None:
            continue

        snr = float(sph.get("snr", 0.0))
        if min_snr is not None and snr < min_snr:
            continue

        x = float(cart["x_m"])
        y = float(cart["y_m"])
        z = float(cart["z_m"])

        pts.append([x, y, z])
        snr_list.append(snr)

    if not pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    points = np.asarray(pts, dtype=np.float32)
    snr = np.asarray(snr_list, dtype=np.float32)
    return points, snr


def update_point_cloud_for_frame(
    pcd: o3d.geometry.PointCloud,
    frame: Dict[str, Any],
    min_snr: float,
    color_by_snr: bool,
):
    """Modify an existing PointCloud to show the given frame."""
    points, snr = extract_points_and_snr_for_frame(frame, min_snr=min_snr)

    if points.shape[0] == 0:
        # Empty frame: clear geometry
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return

    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if color_by_snr:
        s_min = float(np.min(snr))
        s_max = float(np.max(snr))
        if s_max > s_min:
            snr_norm = (snr - s_min) / (s_max - s_min)
        else:
            snr_norm = np.zeros_like(snr)

        # Map SNR -> RGB (low: dark blue, high: yellow-ish)
        colors = np.stack(
            [
                snr_norm,          # R
                snr_norm,          # G
                1.0 - snr_norm,    # B
            ],
            axis=1,
        )
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    else:
        # All-white points
        colors = np.ones((points.shape[0], 3), dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)


# -------------------------
#   Main / visualization
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Animate mmWave JSON point clouds in Open3D.")
    parser.add_argument("json_file", type=str, help="Path to JSON / NDJSON file with radar frames.")
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        help="Optional SNR threshold: drop points with snr < MIN_SNR.",
    )
    parser.add_argument(
        "--no-snr-color",
        action="store_true",
        help="Disable coloring by SNR (points will be white).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Playback speed when in 'play' mode (frames per second).",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frames = load_frames(path)
    if not frames:
        raise RuntimeError("No frames found in file.")

    print(f"Loaded {len(frames)} frame(s) from {path}")

    # Visualization state shared with callbacks
    state = {
        "frames": frames,
        "idx": 0,
        "playing": False,
        "last_time": time.time(),
        "fps": max(0.1, float(args.fps)),
        "min_snr": args.min_snr,
        "color_by_snr": not args.no_snr_color,
        "pcd": o3d.geometry.PointCloud(),
    }

    # Initialize Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="mmWave Radar Animation")

    # Initialize point cloud for frame 0
    update_point_cloud_for_frame(
        state["pcd"],
        state["frames"][state["idx"]],
        min_snr=state["min_snr"],
        color_by_snr=state["color_by_snr"],
    )
    vis.add_geometry(state["pcd"])

    print("Controls: Space=Play/Pause, N=Next, P=Prev, Q/Esc=Quit")

    # ---------- Key callbacks ----------

    def go_to_frame(new_idx: int, vis_obj):
        new_idx = new_idx % len(state["frames"])
        state["idx"] = new_idx
        update_point_cloud_for_frame(
            state["pcd"],
            state["frames"][state["idx"]],
            min_snr=state["min_snr"],
            color_by_snr=state["color_by_snr"],
        )
        vis_obj.update_geometry(state["pcd"])
        print(f"Frame: {state['idx']+1}/{len(state['frames'])}")
        return True  # Request re-render

    def cb_next(vis_obj):
        return go_to_frame(state["idx"] + 1, vis_obj)

    def cb_prev(vis_obj):
        return go_to_frame(state["idx"] - 1, vis_obj)

    def cb_toggle_play(vis_obj):
        state["playing"] = not state["playing"]
        mode = "PLAY" if state["playing"] else "PAUSE"
        print(f"[{mode}]")
        state["last_time"] = time.time()
        return False  # No immediate geometry change

    def cb_quit(vis_obj):
        print("[QUIT]")
        vis_obj.close()
        return False

    # Register key callbacks
    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("P"), cb_prev)
    vis.register_key_callback(ord(" "), cb_toggle_play)
    vis.register_key_callback(ord("Q"), cb_quit)
    vis.register_key_callback(256, cb_quit)  # Esc key

    # ---------- Animation callback ----------

    def animation_callback(vis_obj):
        if not state["playing"]:
            return False

        now = time.time()
        dt = now - state["last_time"]
        if dt < 1.0 / state["fps"]:
            return False

        state["last_time"] = now
        go_to_frame(state["idx"] + 1, vis_obj)
        return True  # Geometry updated

    vis.register_animation_callback(animation_callback)

    # Start viewer
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
