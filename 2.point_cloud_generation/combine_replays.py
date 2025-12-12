import json
import glob
import os
import re
import sys

def numeric_key(path):
    filename = os.path.basename(path)
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else 0

def main():
    # Use folder from command-line arg, default to current folder
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "2.point_cloud_generation/data_from_ti_visualizer"  # current directory

    # Normalize path
    input_dir = os.path.abspath(input_dir)

    print(f"Looking for JSON files in: {input_dir}")

    pattern = os.path.join(input_dir, "replay_*.json")
    replay_files = sorted(glob.glob(pattern), key=numeric_key)

    if not replay_files:
        print("No files matching replay_*.json found in that directory.")
        return

    print("Combining files in this order:")
    for f in replay_files:
        print("  -", os.path.basename(f))

    with open(replay_files[0], "r") as f:
        combined = json.load(f)

    if "data" not in combined or not isinstance(combined["data"], list):
        combined["data"] = []

    for path in replay_files[1:]:
        with open(path, "r") as f:
            current = json.load(f)

        if current.get("cfg") != combined.get("cfg"):
            print(f"WARNING: cfg mismatch in {os.path.basename(path)}")
        if current.get("demo") != combined.get("demo"):
            print(f"WARNING: demo mismatch in {os.path.basename(path)}")
        if current.get("device") != combined.get("device"):
            print(f"WARNING: device mismatch in {os.path.basename(path)}")

        frames = current.get("data", [])
        if not isinstance(frames, list):
            print(f"WARNING: 'data' in {os.path.basename(path)} is not a list, skipping.")
            continue

        combined["data"].extend(frames)

    output_path = os.path.join(input_dir, "replay_combined.json")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=4)

    print(f"\nDone! Wrote {output_path}")
    print(f"  - Total frames: {len(combined['data'])}")
    print(f"  - From {len(replay_files)} files")

if __name__ == "__main__":
    main()
