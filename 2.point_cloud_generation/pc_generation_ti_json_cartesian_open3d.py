#!/usr/bin/env python3
"""
Open3D viewer for new-format mmWave JSON files (Cartesian pointCloud).

- Uses the same config fields as your dataset script:
  RANGE_CUT, RANGE_MIN/MAX, STATIC_CLUTTER, STATIC_CLUTTER_THRESH,
  FLIP_X/Y/Z, XYZ_PERMUTE, TOPK, TOPK_K (or TOPK_POINTS),
  MMWAVE_RADAR_LOC (shift).

Controls:
    Space   : Play / Pause
    N       : Next frame
    P       : Previous frame
    Q / Esc : Quit

Usage:
    python open3d_view_cartesian_json.py your_file.json --fps 5
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d

import configuration as cfg


# --------------------------------------------------------
#  Helper: same coordinate flips / permutation as script
# --------------------------------------------------------
def apply_coordinate_flips_and_permute(points_xyz: np.ndarray) -> np.ndarray:
    """Apply FLIP_X/Y/Z and XYZ_PERMUTE from configuration.py to the Nx3 XYZ array."""
    pts = points_xyz.copy()

    if getattr(cfg, "FLIP_X", False):
        pts[:, 0] *= -1.0
    if getattr(cfg, "FLIP_Y", False):
        pts[:, 1] *= -1.0
    if getattr(cfg, "FLIP_Z", False):
        pts[:, 2] *= -1.0

    perm = getattr(cfg, "XYZ_PERMUTE", (0, 1, 2))
    pts = pts[:, perm]
    return pts


# --------------------------------------------------------
#  Convert one frameData -> (points, energy)
# --------------------------------------------------------
def frame_to_points_from_cartesian(frame_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one JSON 'frameData' object with Cartesian pointCloud into:
        points: (N, 3) float32
        energy: (N,) float32

    Same semantics as frame_to_pointcloud_from_json_cartesian in your script,
    but we keep the *variable-length* cloud (no reg_data).
    """

    point_cloud = frame_data.get("pointCloud", [])

    # Configs (same names as in your main script)
    RANGE_CUT = getattr(cfg, "RANGE_CUT", False)
    RANGE_MIN = getattr(cfg, "RANGE_MIN", getattr(cfg, "RANGE_CUT_MIN", 0.0))
    RANGE_MAX = getattr(cfg, "RANGE_MAX", getattr(cfg, "RANGE_CUT_MAX", 1e9))

    TOPK = getattr(cfg, "TOPK", False)
    TOPK_K = getattr(cfg, "TOPK_K", getattr(cfg, "TOPK_POINTS", 256))

    STATIC_CLUTTER = getattr(cfg, "STATIC_CLUTTER", False)
    STATIC_V_THRESH = getattr(cfg, "STATIC_CLUTTER_THRESH", 0.05)

    # Radar location shift
    shift_arr = None
    if hasattr(cfg, "MMWAVE_RADAR_LOC"):
        shift_arr = np.array(cfg.MMWAVE_RADAR_LOC, dtype=np.float32)

    records = []

    for p in point_cloud:
        # Expect at least [x, y, z, velocity, energy]
        if len(p) < 5:
            continue

        x = float(p[0])
        y = float(p[1])
        z = float(p[2])
        v = float(p[3])
        energy = float(p[4])

        r = math.sqrt(x * x + y * y + z * z)

        # Optional range cut
        if RANGE_CUT and not (RANGE_MIN <= r <= RANGE_MAX):
            continue

        # Optional static clutter removal
        if STATIC_CLUTTER and abs(v) < STATIC_V_THRESH:
            continue

        records.append((x, y, z, v, energy, r))

    if not records:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    data = np.asarray(records, dtype=np.float32)  # (N, 6): x,y,z,v,energy,r

    # Shift XYZ by radar location if provided
    if shift_arr is not None:
        data[:, :3] += shift_arr

    # Apply flips + permutation
    data[:, :3] = apply_coordinate_flips_and_permute(data[:, :3])

    # Optional TOP-K selection by energy (column index 4)
    if TOPK and data.shape[0] > TOPK_K:
        idx = np.argpartition(data[:, 4], -TOPK_K)[-TOPK_K:]
        data = data[idx]

    points = data[:, :3].astype(np.float32)
    energy = data[:, 4].astype(np.float32)
    return points, energy


# --------------------------------------------------------
#  Load all frames from a new-format JSON file
# --------------------------------------------------------
def load_frames_from_json(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (points, energy) for each frame in the JSON.

    points: (Ni, 3)
    energy: (Ni,)
    """
    with path.open("r") as f:
        raw = json.load(f)

    data_entries = raw.get("data", [])
    frames: List[Tuple[np.ndarray, np.ndarray]] = []

    for entry in data_entries:
        frame_data = entry.get("frameData")
        if frame_data is None:
            continue
        pts, energy = frame_to_points_from_cartesian(frame_data)
        # We still keep frames with 0 points; we just show nothing for those
        frames.append((pts, energy))

    if not frames:
        raise ValueError(f"No frames parsed from {path}")

    return frames


# --------------------------------------------------------
#  Open3D visualization / animation
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for Cartesian mmWave JSON files.")
    parser.add_argument("json_file", type=str, help="Path to JSON file (new 'data' + 'frameData' format).")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed in frames per second.")
    parser.add_argument(
        "--no-energy-color",
        action="store_true",
        help="Disable coloring by energy (points will be white).",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    if not path.exists():
        raise FileNotFoundError(path)

    frames = load_frames_from_json(path)
    print(f"Loaded {len(frames)} frame(s) from {path}")

    state = {
        "frames": frames,
        "idx": 0,
        "playing": False,
        "last_time": time.time(),
        "fps": max(0.1, float(args.fps)),
        "color_by_energy": not args.no_energy_color,
        "pcd": o3d.geometry.PointCloud(),
    }

    # Initialize visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"mmWave Cartesian Viewer - {path.name}")

    # Helper to update pcd for given frame index
    def update_pcd_for_frame(idx: int, vis_obj):
        pts, energy = state["frames"][idx]

        if pts.shape[0] == 0:
            state["pcd"].points = o3d.utility.Vector3dVector(
                np.zeros((0, 3), dtype=np.float64)
            )
            state["pcd"].colors = o3d.utility.Vector3dVector(
                np.zeros((0, 3), dtype=np.float64)
            )
        else:
            state["pcd"].points = o3d.utility.Vector3dVector(pts.astype(np.float64))

            if state["color_by_energy"]:
                e_min = float(np.min(energy))
                e_max = float(np.max(energy))
                if e_max > e_min:
                    e_norm = (energy - e_min) / (e_max - e_min)
                else:
                    e_norm = np.zeros_like(energy)

                # Low energy: dark blue; high: yellow-ish
                colors = np.stack(
                    [
                        e_norm,        # R
                        e_norm,        # G
                        1.0 - e_norm,  # B
                    ],
                    axis=1,
                )
                state["pcd"].colors = o3d.utility.Vector3dVector(
                    colors.astype(np.float64)
                )
            else:
                colors = np.ones((pts.shape[0], 3), dtype=np.float64)
                state["pcd"].colors = o3d.utility.Vector3dVector(colors)

        vis_obj.update_geometry(state["pcd"])
        print(f"Frame: {state['idx']+1}/{len(state['frames'])}  (points={pts.shape[0]})")

    # Initialize with frame 0
    update_pcd_for_frame(0, vis)
    vis.add_geometry(state["pcd"])

    print("Controls: Space=Play/Pause, N=Next, P=Prev, Q/Esc=Quit")

    # --- key callbacks ---

    def go_to_frame(new_idx: int, vis_obj):
        new_idx = new_idx % len(state["frames"])
        state["idx"] = new_idx
        update_pcd_for_frame(new_idx, vis_obj)
        return True

    def cb_next(vis_obj):
        return go_to_frame(state["idx"] + 1, vis_obj)

    def cb_prev(vis_obj):
        return go_to_frame(state["idx"] - 1, vis_obj)

    def cb_toggle_play(vis_obj):
        state["playing"] = not state["playing"]
        mode = "PLAY" if state["playing"] else "PAUSE"
        print(f"[{mode}]")
        state["last_time"] = time.time()
        return False

    def cb_quit(vis_obj):
        print("[QUIT]")
        vis_obj.close()
        return False

    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("P"), cb_prev)
    vis.register_key_callback(ord(" "), cb_toggle_play)
    vis.register_key_callback(ord("Q"), cb_quit)
    vis.register_key_callback(256, cb_quit)  # ESC

    # --- animation callback ---

    def animation_callback(vis_obj):
        if not state["playing"]:
            return False

        now = time.time()
        if now - state["last_time"] < 1.0 / state["fps"]:
            return False

        state["last_time"] = now
        go_to_frame(state["idx"] + 1, vis_obj)
        return True

    vis.register_animation_callback(animation_callback)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
