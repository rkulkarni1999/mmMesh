import numpy as np

# -----------------------------
# Dataset I/O
# -----------------------------
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

# Options: 'end', 'random', 'file'
SPLIT_METHOD = "end"
SPLIT_SEED = 42
TEST_RATIO = 0.1

PC_SIZE = 128
MAX_FRAMES_PER_FILE = None  # set e.g. 500 to debug fast

# -----------------------------
# “Old-script-like” processing knobs
# -----------------------------
# RangeCut analogue of:
#   dopplerResultInDB[:, :25] = -100
#   dopplerResultInDB[:, 125:] = -100
# With your old RANGE_RESOLUTION (~0.043m), this becomes about [1.07m, 5.37m].
ENABLE_RANGE_CUT = True
RANGE_MIN_M = 1.07
RANGE_MAX_M = 5.37

# EnergyTop256 analogue: keep top K points per frame by "energy"
# (then you still sample/duplicate to PC_SIZE using reg_data)
ENABLE_TOPK = True
TOPK = 256

# Static clutter removal analogue:
# Old code removed the “mean across doppler loops” (stationary stuff).
# With JSONL detections, the closest analogue is “remove near-zero doppler points”.
ENABLE_STATIC_CLUTTER_REMOVAL = False
STATIC_DOPPLER_THRESH_M_S = 0.05  # if enabled, drop |doppler| < thresh

# Use presence flag
FILTER_PRESENCE_ONLY = False

# -----------------------------
# Feature mapping to match (x,y,z,V,energy,R)
# -----------------------------
# JSONL gives snr, we convert to something like old log scale.
SNR_TO_DB = True  # energy = 10*log10(snr)
SNR_DB_EPS = 1e-9

# -----------------------------
# Coordinate handling (same as before)
# -----------------------------
# Hard-coded mmWave radar location (world offset)
MMWAVE_RADAR_LOC = np.array([[0.146517, -3.030810, 1.0371905]], dtype=np.float32)

FLIP_X = True
FLIP_Y = True
FLIP_Z = False

# Optional permute if axes look swapped
XYZ_PERMUTE = (0, 1, 2)

# -----------------------------
# Visualization
# -----------------------------
CREATE_VISUALIZATIONS = True
VIS_FRAME_RATE = 10
