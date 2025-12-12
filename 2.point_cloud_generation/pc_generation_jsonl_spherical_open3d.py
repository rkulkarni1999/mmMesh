#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import os
import glob
import json
import math
from tqdm import tqdm

import configuration as cfg
from visualize_open3d import create_dataset_visualizations


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


def reg_data(data: np.ndarray, pc_size: int) -> np.ndarray:
    """
    Regularize a variable-length point cloud to fixed pc_size points.

    data: (N, 6) float32 array
    returns: (pc_size, 6)
    """
    pc_tmp = np.zeros((pc_size, data.shape[1]), dtype=np.float32)
    pc_no = data.shape[0]

    if pc_no == 0:
        return pc_tmp

    if pc_no < pc_size:
        # Randomly choose slots in the output to fill with the original points
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        # Remaining slots are filled by randomly duplicating existing points
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        # If we have more than pc_size points, randomly downsample
        pc_list = np.random.choice(pc_no, size=pc_size, replace=False)
        pc_tmp = data[pc_list]

    return pc_tmp


def frame_to_pointcloud_from_json(frame_obj: dict, pc_size: int, shift_arr: np.ndarray) -> np.ndarray:
    """
    Convert one JSON 'frame' object into a fixed-size (pc_size, 6) point cloud.

    Uses only the 'spherical' fields:
      range_m, azimuth_deg, elevation_deg, doppler_m_s, snr

    Output features per point: [x, y, z, velocity, snr, range]
    where x,y,z are recomputed from spherical coords and then:
      - shifted by MMWAVE_RADAR_LOC (if provided)
      - flipped and permuted according to FLIP_X/Y/Z and XYZ_PERMUTE
      - optionally filtered by RANGE_CUT / STATIC_CLUTTER / TOPK
    """
    points = frame_obj.get("points", [])

    # Configs with safe fallbacks so script doesn't crash if not defined
    RANGE_CUT = getattr(cfg, "RANGE_CUT", False)
    RANGE_MIN = getattr(cfg, "RANGE_MIN", getattr(cfg, "RANGE_CUT_MIN", 0.0))
    RANGE_MAX = getattr(cfg, "RANGE_MAX", getattr(cfg, "RANGE_CUT_MAX", 1e9))

    TOPK = getattr(cfg, "TOPK", False)
    TOPK_K = getattr(cfg, "TOPK_K", getattr(cfg, "TOPK_POINTS", 256))

    STATIC_CLUTTER = getattr(cfg, "STATIC_CLUTTER", False)
    STATIC_V_THRESH = getattr(cfg, "STATIC_CLUTTER_THRESH", 0.05)

    records = []

    for p in points:
        sph = p.get("spherical", {})
        r = float(sph.get("range_m", 0.0))
        az_deg = float(sph.get("azimuth_deg", 0.0))
        el_deg = float(sph.get("elevation_deg", 0.0))
        v = float(sph.get("doppler_m_s", 0.0))
        snr = float(sph.get("snr", 0.0))

        # Optional range cut in meters
        if RANGE_CUT and not (RANGE_MIN <= r <= RANGE_MAX):
            continue

        # Optional static clutter removal based on small velocity
        if STATIC_CLUTTER and abs(v) < STATIC_V_THRESH:
            continue

        # Convert to radians
        az = math.radians(az_deg)
        el = math.radians(el_deg)

        # TI-style spherical -> Cartesian
        # x ~ cross-range, y ~ range, z ~ elevation
        # x = r * math.sin(az) * math.cos(el)
        # y = r * math.cos(az) * math.cos(el)
        # z = r * math.sin(el)
        x = r * math.cos(el) * math.sin(az)
        z = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)


        # NOTE: we are explicitly ignoring the 'cartesian' entry from JSON
        records.append((x, y, z, v, snr, r))

    if not records:
        # No valid points for this frame
        return np.zeros((pc_size, 6), dtype=np.float32)

    data = np.asarray(records, dtype=np.float32)  # (N, 6)

    # Shift XYZ by radar location if provided
    if shift_arr is not None:
        data[:, :3] += shift_arr

    # Apply flips and permutation
    data[:, :3] = apply_coordinate_flips_and_permute(data[:, :3])

    # Optional TOP-K selection by SNR (column 4)
    if TOPK and data.shape[0] > TOPK_K:
        idx = np.argpartition(data[:, 4], -TOPK_K)[-TOPK_K:]
        data = data[idx]

    # Finally, regularize to fixed pc_size
    return reg_data(data, pc_size)


def process_jsonl_file(jsonl_path: Path, pc_size: int, shift_arr: np.ndarray) -> np.ndarray:
    """Read one .jsonl file and convert it to (n_frames, pc_size, 6)."""
    frames = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_obj = json.loads(line)
            pc_frame = frame_to_pointcloud_from_json(frame_obj, pc_size, shift_arr)
            frames.append(pc_frame)

    if not frames:
        raise ValueError(f"No frames parsed from {jsonl_path}")
    return np.stack(frames, axis=0)  # (n_frames, pc_size, 6)


def split_random(dataset, test_ratio, seed=0):
    """Random sampling split: per-file split into train/test by frame."""
    rng = np.random.default_rng(seed)
    train_splits = []
    test_splits = []
    for arr in dataset:
        n = arr.shape[0]
        perm = rng.permutation(n)
        split_idx = int((1 - test_ratio) * n)
        train_splits.append(arr[perm[:split_idx]])
        test_splits.append(arr[perm[split_idx:]])
    return np.stack(train_splits, axis=0), np.stack(test_splits, axis=0)


def split_end(dataset, test_ratio):
    """End-of-file split: last fraction of frames per file goes to test set."""
    train_splits = []
    test_splits = []
    for idx, arr in enumerate(dataset):
        n = arr.shape[0]
        test_count = int(test_ratio * n)
        test_count = max(test_count, 1) if n > 1 else 0
        train_count = n - test_count

        print(f"Dataset[{idx}]: Total = {n}, train= {train_count}, test= {test_count}")

        train_splits.append(arr[:train_count])
        test_splits.append(arr[train_count:])

    min_train = min(a.shape[0] for a in train_splits) if train_splits else 0
    min_test = min(a.shape[0] for a in test_splits) if test_splits else 0

    train_array = (
        np.stack([a[:min_train] for a in train_splits], axis=0) if min_train > 0 else
        np.empty((0, dataset[0].shape[1], dataset[0].shape[2]), dtype=dataset[0].dtype)
    )
    test_array = (
        np.stack([a[-min_test:] for a in test_splits], axis=0) if min_test > 0 else
        np.empty((0, dataset[0].shape[1], dataset[0].shape[2]), dtype=dataset[0].dtype)
    )

    return train_array, test_array


def split_sequential(dataset, train_count, min_frames):
    """Sequential split: fill train sequentially, rest goes to test."""
    train_splits = []
    test_splits = []
    current_train_frames = 0

    for i, arr in enumerate(dataset):
        arr_trunc = arr[:min_frames]

        if current_train_frames + min_frames <= train_count:
            train_splits.append(arr_trunc)
            current_train_frames += min_frames
        else:
            if current_train_frames < train_count:
                remaining_train_needed = train_count - current_train_frames
                train_splits.append(arr_trunc[:remaining_train_needed])
                test_splits.append(arr_trunc[remaining_train_needed:])
                current_train_frames = train_count
            else:
                test_splits.append(arr_trunc)

    if train_splits:
        max_train_len = max(arr.shape[0] for arr in train_splits)
        train_padded = []
        for arr in train_splits:
            if arr.shape[0] < max_train_len:
                padded = np.zeros(
                    (max_train_len, arr.shape[1], arr.shape[2]), dtype=arr.dtype
                )
                padded[: arr.shape[0]] = arr
                train_padded.append(padded)
            else:
                train_padded.append(arr)
        train_array = np.concatenate(train_padded, axis=0).reshape(
            -1, train_splits[0].shape[1], train_splits[0].shape[2]
        )
    else:
        train_array = np.empty(
            (0, dataset[0].shape[1], dataset[0].shape[2]), dtype=dataset[0].dtype
        )

    if test_splits:
        test_array = np.concatenate(test_splits, axis=0).reshape(
            -1, test_splits[0].shape[1], test_splits[0].shape[2]
        )
    else:
        test_array = np.empty(
            (0, dataset[0].shape[1], dataset[0].shape[2]), dtype=dataset[0].dtype
        )

    return train_array, test_array


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, getattr(cfg, "INPUT_DIR", "data/input"))
    output_dir = os.path.join(base_dir, getattr(cfg, "OUTPUT_DIR", "data/output"))

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    pc_size = cfg.PC_SIZE
    test_ratio = cfg.TEST_RATIO
    split_method = cfg.SPLIT_METHOD

    # Radar location shift in meters, or None
    shift_arr = None
    if hasattr(cfg, "MMWAVE_RADAR_LOC"):
        shift_arr = np.array(cfg.MMWAVE_RADAR_LOC, dtype=np.float32)

    # Collect all .jsonl files
    jsonl_files = sorted(
        [
            Path(f)
            for f in glob.glob(os.path.join(input_dir, "**", "*.jsonl"), recursive=True)
        ]
    )

    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {input_dir}")
        return

    # Print configuration info
    RANGE_CUT = getattr(cfg, "RANGE_CUT", False)
    RANGE_MIN = getattr(cfg, "RANGE_MIN", getattr(cfg, "RANGE_CUT_MIN", 0.0))
    RANGE_MAX = getattr(cfg, "RANGE_MAX", getattr(cfg, "RANGE_CUT_MAX", 1e9))
    TOPK = getattr(cfg, "TOPK", False)
    TOPK_K = getattr(cfg, "TOPK_K", getattr(cfg, "TOPK_POINTS", 256))
    STATIC_CLUTTER = getattr(cfg, "STATIC_CLUTTER", True)
    STATIC_V_THRESH = getattr(cfg, "STATIC_CLUTTER_THRESH", 0.05)

    print("Config:")
    print(f"  PC_SIZE={pc_size}, TEST_RATIO={test_ratio}, SPLIT_METHOD={split_method}")
    print(f"  RANGE_CUT={RANGE_CUT} [{RANGE_MIN},{RANGE_MAX}]")
    print(f"  TOPK={TOPK} K={TOPK_K}")
    print(f"  STATIC_CLUTTER={STATIC_CLUTTER} thresh={STATIC_V_THRESH}")
    print(
        f"  FLIP_X/Y/Z={getattr(cfg,'FLIP_X',False)}/"
        f"{getattr(cfg,'FLIP_Y',False)}/"
        f"{getattr(cfg,'FLIP_Z',False)}, "
        f"XYZ_PERMUTE={getattr(cfg,'XYZ_PERMUTE',(0,1,2))}"
    )

    dataset = []
    for fp in tqdm(jsonl_files, desc="Processing .jsonl", unit="file"):
        arr = process_jsonl_file(fp, pc_size, shift_arr)
        dataset.append(arr)

    if not dataset:
        print("ERROR: No data processed; aborting.")
        return

    min_frames = min(arr.shape[0] for arr in dataset)
    train_count = int((1.0 - test_ratio) * min_frames)

    print(f"Minimum frames across all files: {min_frames}")
    print(f"Train count: {train_count}, Test count: {min_frames - train_count}")
    print(f"Total files processed: {len(dataset)}")
    print(f"Point cloud size: {pc_size}")

    # Split according to chosen method
    if split_method == "sequential":
        train, test = split_sequential(dataset, train_count, min_frames)
    elif split_method == "random":
        train, test = split_random(dataset, test_ratio, seed=getattr(cfg, "SPLIT_SEED", 0))
    elif split_method == "end":
        train, test = split_end(dataset, test_ratio)
    else:
        raise ValueError(f"Unknown SPLIT_METHOD: {split_method}")

    train_path = os.path.join(output_dir, "train.dat")
    test_path = os.path.join(output_dir, "test.dat")
    train.dump(train_path)
    print(f"Saved train.dat {train.shape} -> {train_path}")
    test.dump(test_path)
    print(f"Saved test.dat  {test.shape}  -> {test_path}")

    # Create visualizations
    if hasattr(cfg, "CREATE_VISUALIZATIONS") and cfg.CREATE_VISUALIZATIONS:
        frame_rate = getattr(cfg, "VIS_FRAME_RATE", 10)
        max_frames_for_videos = getattr(cfg, "VIS_MAX_FRAMES", None)
        create_dataset_visualizations(
            train, test, output_dir, frame_rate, max_frames_for_videos
        )
    else:
        print("\nTo enable visualizations, add to configuration.py:")
        print("CREATE_VISUALIZATIONS = True")
        print("VIS_FRAME_RATE = 10")
        print("VIS_MAX_FRAMES = 100")


if __name__ == "__main__":
    main()
