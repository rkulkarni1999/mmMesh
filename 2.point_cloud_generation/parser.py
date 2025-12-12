#!/usr/bin/env python3
"""
Corrected IWR6843AOP UART Parser + CFG Sender (verbose/debug)

This is your original working parser with added optional improvements:
 - frame integrity checks (basic packet length validation)
 - merge side-info (TLV 7) with point clouds when available
 - timestamping (host + radar cycles)
 - optional SNR filtering
 - rolling point-cloud buffer (for smoothing / tracking)
 - JSONL logging (existing) + vitals CSV export
 - simple matplotlib realtime visualizer (3D scatter + vitals) when --viz is used
 - command line args (--no-log, --viz, --filter-snr, --window, --cfg)
 - callback hook `on_new_frame(packet)` for external integration
All original parsing functions and behavior are preserved; new features are opt-in.
"""

import serial
import struct
import time
import logging
import sys
import json
from collections import namedtuple, deque
import math
import argparse
import threading
import csv
import os

# --------------------- PARSER / USER CONFIG (defaults preserved) ---------------------
CLI_PORT  = "COM8"     # CLI UART (115200)
DATA_PORT = "COM9"     # DATA UART (921600)
CFG_FILE_PATH = r"D:\Radar_Sensors\Ti\68xx_vital_signs\Vital_Signs_With_People_Tracking\chirp_configs\vital_signs_AOP_6m.cfg"

CLI_BAUD  = 115200
DATA_BAUD = 921600

JSON_LOG_PATH = "radar_log.jsonl"  # log every frame as JSON-Lines (default)
VITALS_CSV_PATH = "radar_vitals.csv"  # csv time series for vitals

# --------------------- LOGGING --------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG
)
LOG = logging.getLogger("iwr6843")

# --------------------- CONSTANTS ------------------------
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

TLV_DETECTED_POINTS        = 1
TLV_RANGE_PROFILE          = 2
TLV_NOISE_PROFILE          = 3
TLV_AZIMUTH_STATIC_HEATMAP = 4
TLV_RANGE_DOPPLER          = 5
TLV_STATS                  = 6
TLV_SIDE_INFO              = 7
TLV_COMPRESSED_POINTS      = 1020
TLV_PRESENSE_INDICATION    = 1021
TLV_TARGET_LIST_3D         = 1010
TLV_TARGET_INDEX           = 1011
TLV_VITAL_SIGNS            = 1040

DetectedPoint = namedtuple("DetectedPoint", "x y z doppler")
CompressedPoint = namedtuple("CompressedPoint", "x y z doppler snr noise")

# --------------------- STRUCT FORMATS -------------------
FRAME_HEADER_AFTER_MAGIC_FMT = "<8I"
FRAME_HEADER_AFTER_MAGIC_LEN = 32

TLV_HEADER_FMT = "<2I"
TLV_HEADER_LEN = 8

# --------------------- GLOBAL RUNTIME STATE -------------------
# Shared latest frame for visualizer / callbacks
_latest_frame = None
_latest_frame_lock = threading.Lock()
POINT_BUFFER = deque()  # rolling buffer of frames (each is list of points)

# --------------------- ARGPARSE / OPTIONS -------------------
def get_args():
    p = argparse.ArgumentParser(description="IWR6843 UART Parser (enhanced)")
    p.add_argument("--no-log", action="store_true", help="Disable JSON logging to file")
    p.add_argument("--viz", action="store_true", help="Enable realtime matplotlib visualizer (3D + vitals)")
    p.add_argument("--filter-snr", type=float, default=0.0, help="Minimum SNR threshold (meters) for points; 0 to disable")
    p.add_argument("--window", type=int, default=1, help="Rolling frame window size (frames) for smoothing; default 1 (disabled)")
    p.add_argument("--cfg", type=str, default=None, help="Path to CFG to send (overrides built-in path)")
    p.add_argument("--enable-side-merge", action="store_true", help="Merge TLV7 side-info into points when indices match")
    return p.parse_args()

# --------------------- UTILITIES ----------------------------
def open_uart_ports(cli_port_name, data_port_name, cli_baud, data_baud):
    try:
        cli = serial.Serial(cli_port_name, cli_baud, timeout=0.5)
        data = serial.Serial(data_port_name, data_baud, timeout=0.5)
        LOG.info("Opened CLI @ %s and DATA @ %s", cli_port_name, data_port_name)
        return cli, data
    except Exception as e:
        LOG.exception("Error opening serial ports: %s", e)
        sys.exit(1)

def send_cfg_and_start(cli, cfg_path):
    LOG.info("Sending CFG: %s", cfg_path)
    try:
        with open(cfg_path, "r") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("%"):
                    continue
                cli.write((s + "\n").encode())
                LOG.debug(">> %s", s)
                time.sleep(0.02)
    except FileNotFoundError:
        LOG.error("CFG not found: %s", cfg_path)
        sys.exit(1)

    t0 = time.time()
    while time.time() - t0 < 8:
        line = cli.readline().decode(errors="ignore").strip()
        if not line:
            continue
        LOG.debug("<< %s", line)
        if "Done" in line or "sensorStart" in line:
            LOG.info("Radar ACK: %s", line)
            return True

    LOG.warning("No ACK to sensorStart within timeout.")
    return False

def wait_for_magic(data_port, debug_interval=0.5):
    buf = bytearray()
    last_log = time.time()
    while True:
        b = data_port.read(1)
        if not b:
            time.sleep(0.001)
            if time.time() - last_log > debug_interval:
                LOG.debug("SYNC buffer tail: %s", buf[-8:].hex() if buf else "")
                last_log = time.time()
            continue
        buf += b
        if len(buf) > 512:
            buf = buf[-512:]
        if buf[-8:] == MAGIC_WORD:
            LOG.info("MAGIC WORD FOUND: %s", buf[-8:].hex())
            return True

def log_json_frame(packet, json_log_path, enabled=True):
    """Store output frame to JSON log file (JSON-lines)."""
    if not enabled:
        return
    try:
        with open(json_log_path, "a") as f:
            f.write(json.dumps(packet) + "\n")
    except Exception as e:
        LOG.error("Failed to log JSON: %s", e)

def log_vitals_csv(pkt, csv_path):
    """Append vitals row to CSV (timestamp, HR, BR). Create header if needed."""
    if not pkt.get("vitals"):
        return
    v = pkt["vitals"]
    # ensure file exists and header present
    first = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(["timestamp_host", "timestamp_radar_cycles", "frame", "heart_rate", "breathing_rate", "rangebin", "breathingDeviation"])
            writer.writerow([pkt.get("timestamp_host"), pkt.get("timestamp_radar_cycles"), pkt["frame"], v["heart_rate"], v["breathing_rate"], v["rangebin"], v["breathingDeviation"]])
    except Exception as e:
        LOG.error("Failed to write vitals csv: %s", e)

# --------------------- TLV DECODERS (unchanged logic preserved) ---------------------
def parse_detected_points(payload):
    pts = []
    n = len(payload) // 16
    for i in range(n):
        x,y,z,doppler = struct.unpack_from("<4f", payload, i*16)
        pts.append({"x": x, "y": y, "z": z, "doppler": doppler})
    return pts

def parse_side_info(payload):
    arr = []
    n = len(payload) // 4
    for i in range(n):
        snr, noise = struct.unpack_from("<2H", payload, i*4)
        arr.append({"snr": snr/10.0, "noise": noise/10.0})
    return arr

def parse_compressed_points(payload):
    """
    TI 1020 TLV Compressed Spherical 3D Point Cloud
    Converts spherical -> Cartesian (meters)
    Stores both formats.
    NOTE: We implement according to your provided format:
      header: elevationUnit, azimuthUnit, dopplerUnit, rangeUnit, snrUnit (5 floats) at offset 0
      then two uint16 for count at offset 20 (per your earlier variations we try both)
      each point: elevation int8, azimuth int8, doppler int16, range uint16, snr uint16 => total 8 bytes
    """
    if len(payload) < 24:
        return []

    # unpack units - 5 floats (20 bytes)
    try:
        elevUnit, azimUnit, dopplerUnit, rangeUnit, snrUnit = struct.unpack_from("<5f", payload, 0)
    except struct.error:
        LOG.warning("Compressed points unit unpack failed")
        return []

    # read counts - the docs vary; some pack 2x uint16 starting at offset 20, some do uint32; handle both
    num_points = 0
    if len(payload) >= 24:
        try:
            num0, num1 = struct.unpack_from("<2H", payload, 20)
            num_points = num0 if num0 != 0 else num1
        except struct.error:
            num_points = 0

    pts = []
    idx = 24  # first point after header & count

    for i in range(num_points):
        if idx + 8 > len(payload):
            LOG.debug("Compressed points truncated at point %d", i)
            break
        # note format: int8, int8, int16, uint16, uint16
        elev_i = struct.unpack_from("<b", payload, idx)[0]
        azim_i = struct.unpack_from("<b", payload, idx+1)[0]
        dop_i = struct.unpack_from("<h", payload, idx+2)[0]
        range_i = struct.unpack_from("<H", payload, idx+4)[0]
        snr_i = struct.unpack_from("<H", payload, idx+6)[0]
        idx += 8

        elev_deg = elev_i * elevUnit
        azim_deg = azim_i * azimUnit
        r = range_i * rangeUnit
        doppler = dop_i * dopplerUnit
        snr = snr_i * snrUnit

        az = math.radians(azim_deg)
        el = math.radians(elev_deg)

        x = r * math.cos(el) * math.sin(az)
        y = r * math.cos(el) * math.cos(az)
        z = r * math.sin(el)

        pts.append({
            "spherical": {
                "range_m": r,
                "azimuth_deg": azim_deg,
                "elevation_deg": elev_deg,
                "doppler_m_s": doppler,
                "snr": snr
            },
            "cartesian": {
                "x_m": x,
                "y_m": y,
                "z_m": z
            }
        })
    return pts

def parse_vital(payload):
    if len(payload) < 136:
        return None
    id_val, rangebin = struct.unpack_from("<2H", payload, 0)
    breathingDeviation, heart_rate, breathing_rate = struct.unpack_from("<3f", payload, 4)
    heart_wave = struct.unpack_from("<15f", payload, 16)
    breath_wave = struct.unpack_from("<15f", payload, 76)
    return {
        "id": int(id_val),
        "rangebin": int(rangebin),
        "breathingDeviation": float(breathingDeviation),
        "heart_rate": float(heart_rate),
        "breathing_rate": float(breathing_rate),
        "heart_waveform": list(heart_wave),
        "breath_waveform": list(breath_wave)
    }

def parse_presence(payload):
    if len(payload) < 4:
        return None
    (presence,) = struct.unpack_from("<I", payload, 0)
    return bool(presence)

# --------------------- FRAME READ / PARSE BASE (keeps your structure) ---------------------
def read_n(data_port, n):
    b = bytearray()
    start = time.time()
    while len(b) < n:
        chunk = data_port.read(n - len(b))
        if not chunk:
            if time.time() - start > 1.0:
                break
            continue
        b.extend(chunk)
    return bytes(b)

def parse_frame(data_port, options):
    """
    Parse one frame from the data stream. Returns a packet dict like before,
    but enriched with timestamps, merged side-info (optional), filtering, etc.
    """
    global _latest_frame

    # 1) find and consume magic
    if not wait_for_magic(data_port):
        return None

    # 2) read header after magic
    LOG.debug("Reading frame header (after magic)...")
    header_bytes = read_n(data_port, FRAME_HEADER_AFTER_MAGIC_LEN)
    if len(header_bytes) != FRAME_HEADER_AFTER_MAGIC_LEN:
        LOG.error("Frame header truncated: %d", len(header_bytes))
        return None

    try:
        (version, totalPacketLen, platform, frameNumber,
         timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber) = struct.unpack(FRAME_HEADER_AFTER_MAGIC_FMT, header_bytes)
    except Exception as e:
        LOG.exception("Header unpack failed: %s", e)
        return None

    # Basic sanity checks
    if totalPacketLen < 48 or totalPacketLen > 200000:
        LOG.warning("Suspicious packet length: %d (skipping frame)", totalPacketLen)
        # attempt to skip tlvs by returning None; higher-level read will re-sync on next magic
        return None

    LOG.info("FRAME HDR: version=%x totalLen=%d platform=0x%X frame=%d cpu=%d detected=%d tlvs=%d subFrame=%d",
             version, totalPacketLen, platform, frameNumber, timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber)

    packet = {
        "frame": int(frameNumber),
        "points": [],
        "side_info": [],
        "vitals": None,
        "presence": None,
        "tlvs": [],
        "timestamp_host": time.time(),
        "timestamp_radar_cycles": int(timeCpuCycles),
        "version": int(version),
        "platform": int(platform)
    }

    # parse TLVs
    for tlv_i in range(numTLVs):
        tlv_hdr_bytes = read_n(data_port, TLV_HEADER_LEN)
        if len(tlv_hdr_bytes) != TLV_HEADER_LEN:
            LOG.error("TLV header truncated (got %d bytes)", len(tlv_hdr_bytes))
            break

        tlv_type, tlv_len = struct.unpack(TLV_HEADER_FMT, tlv_hdr_bytes)
        LOG.info("TLV #%d: type=%d len=%d", tlv_i+1, tlv_type, tlv_len)

        payload = read_n(data_port, tlv_len)
        if len(payload) != tlv_len:
            LOG.error("TLV payload truncated: expected %d got %d", tlv_len, len(payload))
            break

        # dispatch decode: preserve your original behavior
        if tlv_type == TLV_DETECTED_POINTS:
            pts = parse_detected_points(payload)
            # unify representation: detected points are cartesian dicts
            packet["points"].extend([{"cartesian": {"x_m": p["x"], "y_m": p["y"], "z_m": p["z"]}, "doppler": p["doppler"]} for p in pts])

        elif tlv_type == TLV_SIDE_INFO:
            si = parse_side_info(payload)
            packet["side_info"].extend(si)

        elif tlv_type == TLV_COMPRESSED_POINTS:
            cpts = parse_compressed_points(payload)
            packet["points"].extend(cpts)  # these are dicts with spherical+cartesian
            LOG.debug("Decoded compressed points: %d", len(cpts))

        elif tlv_type == TLV_VITAL_SIGNS:
            packet["vitals"] = parse_vital(payload)

        elif tlv_type == TLV_PRESENSE_INDICATION:
            packet["presence"] = parse_presence(payload)
            LOG.info("Presence = %s", packet["presence"])

        else:
            LOG.debug("TLV type %d not decoded specifically (payload length %d).", tlv_type, len(payload))

        packet["tlvs"].append({"type": tlv_type, "len": tlv_len})

    # ----------------- Post-processing enhancements (opt-in) -----------------
    # 1) Merge side-info into points when requested & lengths match
    if options.enable_side_merge and packet["side_info"] and packet["points"]:
        try:
            # If the points are a mix (some dicts have 'spherical'), align by index
            if len(packet["side_info"]) == len(packet["points"]):
                for i in range(len(packet["points"])):
                    si = packet["side_info"][i]
                    # attach if not present
                    packet["points"][i].setdefault("side_info", si)
            else:
                LOG.debug("Side-info length (%d) != points length (%d); skipping merge",
                          len(packet["side_info"]), len(packet["points"]))
        except Exception as e:
            LOG.debug("Side-merge error: %s", e)

    # 2) Optional filtering by SNR
    if options.filter_snr and packet["points"]:
        filtered = []
        for p in packet["points"]:
            # compressed points store snr in spherical.snr
            snr_val = None
            if "spherical" in p:
                snr_val = p["spherical"].get("snr")
            elif "side_info" in p:
                snr_val = p["side_info"].get("snr")
            # fallback: accept if snr missing
            if snr_val is None or snr_val >= options.filter_snr:
                filtered.append(p)
        packet["points"] = filtered

    # 3) Add host timestamp & radar cycles already set above

    # 4) Rolling buffer
    if options.window and options.window > 1:
        POINT_BUFFER.append(packet)
        if len(POINT_BUFFER) > options.window:
            POINT_BUFFER.popleft()
    else:
        POINT_BUFFER.clear()
        POINT_BUFFER.append(packet)

    # 5) Merge points from buffer if requested (simple concatenation)
    if options.window and options.window > 1:
        merged_points = []
        for fr in POINT_BUFFER:
            merged_points.extend(fr["points"])
        packet["points_merged_window"] = merged_points

    # update global latest frame for viz/callbacks
    with _latest_frame_lock:
        _latest_frame = packet

    # call user callback hook (no-op unless user overrides)
    try:
        on_new_frame(packet)
    except Exception:
        LOG.exception("on_new_frame callback error")

    return packet

# --------------------- USER CALLBACK (integration hook) ---------------------
def on_new_frame(packet):
    """
    Placeholder hook: external code can override this function to receive frames.
    For example:
        import my_module
        parser.on_new_frame = my_module.handle_frame
    """
    # default: do nothing (kept for extensibility)
    return

# --------------------- VISUALIZER (matplotlib) ---------------------
# --------------------- VISUALIZER (matplotlib) ---------------------
def visualizer_thread(options):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    plt.ion()
    fig = plt.figure(figsize=(12,6))

    # 3D scatter for point cloud
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.set_title("Radar Point Cloud")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")

    # vitals subplot
    ax_vitals = fig.add_subplot(122)
    ax_vitals.set_title("Vitals (Heart Rate & Breathing Rate)")
    ax_vitals.set_xlabel("Time (s)")
    ax_vitals.set_ylabel("BPM")
    hr_times, hr_vals, br_vals = [], [], []
    hr_plot, = ax_vitals.plot([], [], 'r-o', label="HR")
    br_plot, = ax_vitals.plot([], [], 'b-o', label="BR")
    vitals_text = ax_vitals.text(0.5, 1.05, "", transform=ax_vitals.transAxes,
                                 ha='center', va='bottom', fontsize=12)
    ax_vitals.legend()
    plt.tight_layout()
    start_time = time.time()

    # Helper: equal aspect scaling
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        mid_x = sum(x_limits)/2
        mid_y = sum(y_limits)/2
        mid_z = sum(z_limits)/2
        ax.set_xlim3d(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim3d(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim3d(mid_z - max_range/2, mid_z + max_range/2)

    # Optional Z-axis amplification (if vertical changes are tiny)
    Z_SCALE = 3.0  # increase to exaggerate chest movement

    while True:
        try:
            with _latest_frame_lock:
                pkt = _latest_frame

            if pkt is None:
                time.sleep(0.05)
                continue

            # --- Point cloud ---
            pts = pkt.get("points_merged_window") or pkt.get("points") or []
            xs, ys, zs, colors = [], [], [], []
            for p in pts:
                c = p.get("cartesian") or p.get("spherical")
                if not c:
                    continue
                x = c.get("x_m") or c.get("x")
                y = c.get("y_m") or c.get("y")
                z = c.get("z_m") or c.get("z")
                if x is None or y is None or z is None:
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(z * Z_SCALE)  # scale Z
                colors.append(c.get("snr", 0.0))

            ax3d.cla()
            ax3d.set_title(f"Frame {pkt['frame']} Points={len(xs)}")
            ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
            if xs and zs:
                ax3d.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', s=10)
                # dynamic limits with padding
                ax3d.set_xlim(min(xs)-0.1, max(xs)+0.1)
                ax3d.set_ylim(min(ys)-0.1, max(ys)+0.1)
                ax3d.set_zlim(min(zs)-0.05, max(zs)+0.05)
                set_axes_equal(ax3d)

            # --- Vitals ---
            if pkt.get("vitals"):
                t = time.time() - start_time
                hr = pkt["vitals"]["heart_rate"]
                br = pkt["vitals"]["breathing_rate"]

                hr_times.append(t)
                hr_vals.append(hr)
                br_vals.append(br)

                if len(hr_times) > 50:
                    hr_times.pop(0); hr_vals.pop(0); br_vals.pop(0)

                hr_plot.set_data(hr_times, hr_vals)
                br_plot.set_data(hr_times, br_vals)
                ax_vitals.relim()
                ax_vitals.autoscale_view()

                # Display instantaneous HR/BR as text above the plot
                vitals_text.set_text(f"Instant HR: {hr:.1f} BPM   |   BR: {br:.1f} BPM")

            fig.canvas.flush_events()
            time.sleep(0.05)

        except Exception:
            LOG.exception("Visualizer thread error")
            time.sleep(0.5)


# --------------------- MAIN LOOP -----------------------
def main():
    args = get_args()

    # apply CLI cfg override
    cfg_path = args.cfg if args.cfg else CFG_FILE_PATH

    # open ports
    cli, data = open_uart_ports(CLI_PORT, DATA_PORT, CLI_BAUD, DATA_BAUD)

    # optionally start visualizer thread
    viz_thread = None
    if args.viz:
        t = threading.Thread(target=visualizer_thread, args=(args,), daemon=True)
        t.start()
        viz_thread = t
        LOG.info("Visualizer thread started")

    # send cfg if user wants
    cfg_flag = None
    # If script interactive, ask; if run with CFG path passed, send automatically
    if args.cfg is not None:
        cfg_flag = 'y'
    else:
        try:
            # if stdin not a tty, default to sending cfg to preserve previous behavior
            if sys.stdin.isatty():
                cfg_flag = input("Send CFG and start sensor? (y/n): ").strip().lower()
            else:
                cfg_flag = 'y'
        except Exception:
            cfg_flag = 'y'

    if cfg_flag != 'y':
        LOG.info("Starting without sending CFG. Make sure sensor is already configured.")
    else:
        ok = send_cfg_and_start(cli, cfg_path)
        if not ok:
            LOG.error("Failed to start the sensor (no ACK). Exiting.")
            try:
                cli.close(); data.close()
            except: pass
            sys.exit(1)

    LOG.warning("Streaming radar data... Press Ctrl+C to stop.")
    try:
        while True:
            pkt = parse_frame(data, args)
            if pkt is None:
                continue

            # For backward-compatibility keep 'points' as simple list (your earlier code expects that)
            # and we also add 'points_merged_window' when window >1
            LOG.info("Frame %d: total_points=%d vitals=%s presence=%s tlvs=%s",
                     pkt["frame"], len(pkt["points"]),
                     "yes" if pkt["vitals"] else "no",
                     pkt["presence"], pkt["tlvs"])

            if pkt["vitals"]:
                v = pkt["vitals"]
                LOG.info("  VITALS: id=%d rangebin=%d brDev=%.4f HR=%.2f BR=%.2f",
                         v["id"], v["rangebin"], v["breathingDeviation"], v["heart_rate"], v["breathing_rate"])
                # write vitals CSV
                log_vitals_csv(pkt, VITALS_CSV_PATH)

            # JSON log
            log_json_frame(pkt, JSON_LOG_PATH, enabled=(not args.no_log))

            # tiny sleep to avoid busy loop if data pauses
            time.sleep(0.001)

    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        try:
            data.close()
            cli.close()
        except:
            pass


if __name__ == "__main__":
    main()
