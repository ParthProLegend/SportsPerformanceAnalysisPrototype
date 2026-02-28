"""
benchmark.py  –  Performance & Accuracy Evaluation for SportsPerformanceAnalysisPrototype

Generates the following research-paper-ready artefacts (plots + CSVs):
  1. Time-series of prediction error vs time
  2. Processing time per frame
  3. Latency histogram (input → output)
  4. Frame-to-frame variance / error stability
  5. Mean Per Joint Position Error (MPJPE)

Nothing in main.py, utils.py, or exercises/ is modified.
All outputs are saved to  ./benchmark_results/

Usage
-----
    cd src/SportsPerformance
    python benchmark.py                         # uses defaults from main.py
    python benchmark.py --video path/to/video   # override video source
    python benchmark.py --exercise pushup       # override exercise
    python benchmark.py --no-display            # headless (no cv2 window)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── make sure the exercise / utils imports resolve ──────────────────────────
# (same working-dir convention as main.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exercises.simple_squat import SimpleSquat
from exercises.advanced_squat import AdvancedSquat
from exercises.pushup import Pushup

# ── constants ───────────────────────────────────────────────────────────────
NUM_LANDMARKS = 33          # MediaPipe Pose produces 33 landmarks
EMA_ALPHA     = 0.3         # exponential-moving-average smoothing factor
OUTPUT_DIR    = _SCRIPT_DIR / "benchmark_results"

# Landmark names for readable output
LANDMARK_NAMES = [lm.name for lm in mp.solutions.pose.PoseLandmark]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(video_path: str, exercise_selection: str, show: bool = True):
    """
    Mirrors main.py's pipeline but records per-frame telemetry.

    Returns
    -------
    records : list[dict]   – one dict per successfully-processed frame
    """
    mp_pose    = mp.solutions.pose
    pose       = mp_pose.Pose(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # exercise selection (same as main.py)
    if exercise_selection == "simple_squat":
        exercise_analyzer = SimpleSquat()
    elif exercise_selection == "advanced_squat":
        exercise_analyzer = AdvancedSquat(side="left", angle_up=160, angle_down=90)
    elif exercise_selection == "pushup":
        exercise_analyzer = Pushup()
    else:
        raise ValueError(f"Unknown exercise: {exercise_selection}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[benchmark] Video FPS={fps:.1f}  Total frames≈{total_frames}")
    print(f"[benchmark] Exercise: {exercise_selection}")

    records: list[dict] = []
    frame_idx = 0
    prev_positions = None          # for frame-to-frame jitter
    ema_positions  = None          # exponential moving average (pseudo-GT)

    while cap.isOpened():
        t_read_start = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break                   # end of video (don't loop)

        t_pre = time.perf_counter()

        # ── pose estimation (same as main.py) ──
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t_infer_start = time.perf_counter()
        results = pose.process(frame_rgb)
        t_infer_end = time.perf_counter()

        frame.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # ── exercise analysis (same as main.py) ──
        detected = False
        try:
            landmarks = results.pose_landmarks.landmark
            frame_bgr = exercise_analyzer.process_frame(frame_bgr, landmarks, mp_pose)
            detected = True
        except Exception:
            pass

        t_post = time.perf_counter()

        # ── record metrics only when landmarks were detected ──
        if detected:
            # (x, y, z) for each of the 33 landmarks
            positions = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])   # (33, 3)
            visibility = np.array([lm.visibility for lm in landmarks])        # (33,)

            # EMA pseudo-ground-truth
            if ema_positions is None:
                ema_positions = positions.copy()
            else:
                ema_positions = EMA_ALPHA * positions + (1 - EMA_ALPHA) * ema_positions

            # per-joint error  = ||raw − EMA||  (proxy for prediction noise)
            per_joint_error = np.linalg.norm(positions - ema_positions, axis=1)  # (33,)
            mpjpe = per_joint_error.mean()

            # frame-to-frame displacement (jitter)
            if prev_positions is not None:
                f2f_displacement = np.linalg.norm(positions - prev_positions, axis=1)  # (33,)
                f2f_mean = f2f_displacement.mean()
                f2f_std  = f2f_displacement.std()
            else:
                f2f_displacement = np.zeros(NUM_LANDMARKS)
                f2f_mean = 0.0
                f2f_std  = 0.0

            prev_positions = positions.copy()

            record = {
                "frame_idx":          frame_idx,
                "timestamp_s":        frame_idx / fps,
                "processing_time_ms": (t_post - t_pre) * 1000,
                "inference_time_ms":  (t_infer_end - t_infer_start) * 1000,
                "total_latency_ms":   (t_post - t_read_start) * 1000,
                "mpjpe":              mpjpe,
                "mean_visibility":    visibility.mean(),
                "f2f_jitter_mean":    f2f_mean,
                "f2f_jitter_std":     f2f_std,
                "rep_count":          exercise_analyzer.rep_counter,
                "stage":              exercise_analyzer.stage,
            }
            # store per-joint errors & jitter for detailed CSV
            for j in range(NUM_LANDMARKS):
                record[f"err_{LANDMARK_NAMES[j]}"] = per_joint_error[j]
                record[f"jitter_{LANDMARK_NAMES[j]}"] = f2f_displacement[j]

            records.append(record)

        # ── optional display (same as main.py) ──
        if show:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )
            cv2.imshow("Benchmark – Exercise Performance Analyzer", frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  [benchmark] processed {frame_idx}/{total_frames} frames …")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"[benchmark] Done – {len(records)} frames with detections out of {frame_idx} total.")
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_csvs(df: pd.DataFrame, out: Path):
    """Save summary and per-joint detail CSVs."""
    summary_cols = [
        "frame_idx", "timestamp_s",
        "processing_time_ms", "inference_time_ms", "total_latency_ms",
        "mpjpe", "mean_visibility",
        "f2f_jitter_mean", "f2f_jitter_std",
        "rep_count", "stage",
    ]
    df[summary_cols].to_csv(out / "summary_metrics.csv", index=False)
    print(f"  → {out / 'summary_metrics.csv'}")

    err_cols    = [c for c in df.columns if c.startswith("err_")]
    jitter_cols = [c for c in df.columns if c.startswith("jitter_")]

    df[["frame_idx", "timestamp_s"] + err_cols].to_csv(
        out / "per_joint_errors.csv", index=False)
    print(f"  → {out / 'per_joint_errors.csv'}")

    df[["frame_idx", "timestamp_s"] + jitter_cols].to_csv(
        out / "per_joint_jitter.csv", index=False)
    print(f"  → {out / 'per_joint_jitter.csv'}")

    # Per-joint MPJPE summary (mean ± std over all frames)
    err_summary = pd.DataFrame({
        "joint":     [c.replace("err_", "") for c in err_cols],
        "mpjpe_mean": df[err_cols].mean().values,
        "mpjpe_std":  df[err_cols].std().values,
        "mpjpe_max":  df[err_cols].max().values,
    })
    err_summary.to_csv(out / "per_joint_mpjpe_summary.csv", index=False)
    print(f"  → {out / 'per_joint_mpjpe_summary.csv'}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING  (publication-quality, 300 dpi)
# ═══════════════════════════════════════════════════════════════════════════

def _style():
    """Apply a clean style for research-paper figures."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "legend.fontsize":   10,
        "figure.figsize":    (10, 5),
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "lines.linewidth":   1.2,
    })


def plot_prediction_error_timeseries(df: pd.DataFrame, out: Path):
    """1. Time-series of prediction error (MPJPE proxy) vs time."""
    fig, ax = plt.subplots()
    ax.plot(df["timestamp_s"], df["mpjpe"], color="steelblue", alpha=0.6, label="Raw MPJPE")
    # rolling average (window = 30 frames ≈ 1 s)
    window = min(30, len(df) // 3) if len(df) > 10 else 1
    rolling = df["mpjpe"].rolling(window=window, center=True).mean()
    ax.plot(df["timestamp_s"], rolling, color="darkred", linewidth=2, label=f"Rolling mean (w={window})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prediction Error (MPJPE, normalised coords)")
    ax.set_title("Time Series of Prediction Error vs Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "1_prediction_error_timeseries.png")
    plt.close(fig)
    print(f"  → 1_prediction_error_timeseries.png")


def plot_processing_time(df: pd.DataFrame, out: Path):
    """2. Processing time per frame."""
    fig, ax = plt.subplots()
    ax.plot(df["frame_idx"], df["processing_time_ms"], color="teal", alpha=0.5,
            label="Total processing")
    ax.plot(df["frame_idx"], df["inference_time_ms"], color="coral", alpha=0.5,
            label="Pose inference only")
    mean_proc  = df["processing_time_ms"].mean()
    mean_infer = df["inference_time_ms"].mean()
    ax.axhline(mean_proc,  color="teal",  linestyle="--", linewidth=1,
               label=f"Mean total = {mean_proc:.2f} ms")
    ax.axhline(mean_infer, color="coral", linestyle="--", linewidth=1,
               label=f"Mean inference = {mean_infer:.2f} ms")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Processing Time per Frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "2_processing_time_per_frame.png")
    plt.close(fig)
    print(f"  → 2_processing_time_per_frame.png")


def plot_latency_histogram(df: pd.DataFrame, out: Path):
    """3. Latency histogram (input → output)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title, color in [
        (axes[0], "total_latency_ms", "Total Latency (read → output)", "mediumpurple"),
        (axes[1], "inference_time_ms", "Pose-Inference Latency", "salmon"),
    ]:
        data = df[col]
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {data.mean():.2f} ms")
        ax.axvline(data.median(), color="grey", linestyle=":", linewidth=1.2,
                   label=f"Median = {data.median():.2f} ms")
        p95 = data.quantile(0.95)
        ax.axvline(p95, color="red", linestyle="-.", linewidth=1.2,
                   label=f"P95 = {p95:.2f} ms")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out / "3_latency_histogram.png")
    plt.close(fig)
    print(f"  → 3_latency_histogram.png")


def plot_f2f_stability(df: pd.DataFrame, out: Path):
    """4. Frame-to-frame variance / error stability."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 4a – jitter mean over time
    ax = axes[0]
    ax.plot(df["timestamp_s"], df["f2f_jitter_mean"], color="darkorange", alpha=0.6,
            label="Mean F2F displacement")
    window = min(30, len(df) // 3) if len(df) > 10 else 1
    rolling = df["f2f_jitter_mean"].rolling(window=window, center=True).mean()
    ax.plot(df["timestamp_s"], rolling, color="darkblue", linewidth=2,
            label=f"Rolling mean (w={window})")
    ax.set_ylabel("Mean F2F Displacement\n(normalised coords)")
    ax.set_title("Frame-to-Frame Jitter / Error Stability")
    ax.legend()

    # 4b – rolling std (stability indicator)
    ax = axes[1]
    rolling_std = df["f2f_jitter_mean"].rolling(window=window, center=True).std()
    ax.fill_between(df["timestamp_s"], 0, rolling_std, color="crimson", alpha=0.3)
    ax.plot(df["timestamp_s"], rolling_std, color="crimson", linewidth=1.5,
            label=f"Rolling σ of jitter (w={window})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("σ of Jitter")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "4_f2f_stability.png")
    plt.close(fig)
    print(f"  → 4_f2f_stability.png")


def plot_mpjpe(df: pd.DataFrame, out: Path):
    """5. Mean Per Joint Position Error (MPJPE) – bar chart + time series."""
    err_cols = [c for c in df.columns if c.startswith("err_")]
    joint_names = [c.replace("err_", "").replace("_", " ").title() for c in err_cols]
    means = df[err_cols].mean().values
    stds  = df[err_cols].std().values

    # sort by descending mean error
    order  = np.argsort(means)[::-1]
    means  = means[order]
    stds   = stds[order]
    labels = [joint_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, means, xerr=stds, color="steelblue", edgecolor="white",
                   capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MPJPE (normalised coords)")
    ax.set_title("Mean Per Joint Position Error (MPJPE) — All 33 Landmarks")
    overall = df["mpjpe"].mean()
    ax.axvline(overall, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall MPJPE = {overall:.5f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "5_mpjpe_per_joint.png")
    plt.close(fig)
    print(f"  → 5_mpjpe_per_joint.png")

    # 5b – MPJPE over time
    fig2, ax2 = plt.subplots()
    ax2.plot(df["timestamp_s"], df["mpjpe"], color="steelblue", alpha=0.5, label="MPJPE")
    window = min(30, len(df) // 3) if len(df) > 10 else 1
    ax2.plot(df["timestamp_s"],
             df["mpjpe"].rolling(window=window, center=True).mean(),
             color="darkred", linewidth=2, label=f"Rolling mean (w={window})")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("MPJPE (normalised coords)")
    ax2.set_title("MPJPE Over Time")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out / "5b_mpjpe_over_time.png")
    plt.close(fig2)
    print(f"  → 5b_mpjpe_over_time.png")


def plot_combined_dashboard(df: pd.DataFrame, out: Path):
    """Bonus: single figure with all key metrics for a paper overview."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1 – prediction error
    ax = axes[0, 0]
    ax.plot(df["timestamp_s"], df["mpjpe"], color="steelblue", alpha=0.5)
    window = min(30, len(df) // 3) if len(df) > 10 else 1
    ax.plot(df["timestamp_s"],
            df["mpjpe"].rolling(window=window, center=True).mean(),
            color="darkred", linewidth=2)
    ax.set_title("Prediction Error vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MPJPE")

    # 2 – processing time
    ax = axes[0, 1]
    ax.plot(df["frame_idx"], df["processing_time_ms"], alpha=0.4, color="teal")
    ax.axhline(df["processing_time_ms"].mean(), color="teal", ls="--")
    ax.set_title("Processing Time per Frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ms")

    # 3 – latency histogram
    ax = axes[0, 2]
    ax.hist(df["total_latency_ms"], bins=40, color="mediumpurple", edgecolor="white")
    ax.axvline(df["total_latency_ms"].mean(), color="black", ls="--")
    ax.set_title("Latency Histogram")
    ax.set_xlabel("Latency (ms)")

    # 4 – F2F stability
    ax = axes[1, 0]
    ax.plot(df["timestamp_s"], df["f2f_jitter_mean"], alpha=0.5, color="darkorange")
    ax.plot(df["timestamp_s"],
            df["f2f_jitter_mean"].rolling(window=window, center=True).mean(),
            color="darkblue", linewidth=2)
    ax.set_title("Frame-to-Frame Jitter")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean displacement")

    # 5 – top-10 MPJPE bar
    ax = axes[1, 1]
    err_cols = [c for c in df.columns if c.startswith("err_")]
    means = df[err_cols].mean()
    top10 = means.nlargest(10)
    labels = [n.replace("err_", "").replace("_", " ").title() for n in top10.index]
    ax.barh(labels[::-1], top10.values[::-1], color="steelblue")
    ax.set_title("Top-10 Joint MPJPE")
    ax.set_xlabel("MPJPE")

    # 6 – rep count over time
    ax = axes[1, 2]
    ax.step(df["timestamp_s"], df["rep_count"], color="green", linewidth=2)
    ax.set_title("Repetition Count Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reps")

    fig.suptitle("Sports Performance Analysis — Benchmark Dashboard", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "0_combined_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → 0_combined_dashboard.png")


def print_summary_table(df: pd.DataFrame):
    """Print a concise summary table to the console."""
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    n = len(df)
    duration = df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0] if n > 1 else 0
    print(f"  Frames analysed         : {n}")
    print(f"  Duration                : {duration:.2f} s")
    print(f"  Effective FPS           : {n / duration:.1f}" if duration > 0 else "")
    print(f"  ---")
    print(f"  Processing time (mean)  : {df['processing_time_ms'].mean():.2f} ms")
    print(f"  Processing time (std)   : {df['processing_time_ms'].std():.2f} ms")
    print(f"  Processing time (P95)   : {df['processing_time_ms'].quantile(0.95):.2f} ms")
    print(f"  Inference time  (mean)  : {df['inference_time_ms'].mean():.2f} ms")
    print(f"  Total latency   (mean)  : {df['total_latency_ms'].mean():.2f} ms")
    print(f"  Total latency   (P95)   : {df['total_latency_ms'].quantile(0.95):.2f} ms")
    print(f"  ---")
    print(f"  MPJPE (mean)            : {df['mpjpe'].mean():.6f}")
    print(f"  MPJPE (std)             : {df['mpjpe'].std():.6f}")
    print(f"  F2F jitter (mean)       : {df['f2f_jitter_mean'].mean():.6f}")
    print(f"  F2F jitter (std)        : {df['f2f_jitter_std'].mean():.6f}")
    print(f"  ---")
    print(f"  Total reps detected     : {df['rep_count'].iloc[-1]}")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark script for SportsPerformanceAnalysisPrototype")
    parser.add_argument("--video", type=str,
        default=r"C:\Users\reaper\Downloads\Chrome Downloads\How to Perform the Hindu Squat.mp4",
        help="Path to video file (or 0 for webcam)")
    parser.add_argument("--exercise", type=str, default="advanced_squat",
        choices=["simple_squat", "advanced_squat", "pushup"],
        help="Exercise to analyse")
    parser.add_argument("--no-display", action="store_true",
        help="Run headless without OpenCV display window")
    args = parser.parse_args()

    # handle webcam index
    video_source = int(args.video) if args.video.isdigit() else args.video

    print("=" * 70)
    print("  SportsPerformanceAnalysisPrototype — BENCHMARK")
    print("=" * 70)

    # ── run the pipeline ──
    records = run_pipeline(video_source, args.exercise, show=not args.no_display)

    if len(records) < 2:
        print("[benchmark] ERROR: Not enough frames with detections to produce metrics.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # ── create output directory ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[benchmark] Saving results to {OUTPUT_DIR}/")

    # ── export CSVs ──
    print("\n── CSV exports ──")
    export_csvs(df, OUTPUT_DIR)

    # ── generate plots ──
    _style()
    print("\n── Generating plots ──")
    plot_prediction_error_timeseries(df, OUTPUT_DIR)
    plot_processing_time(df, OUTPUT_DIR)
    plot_latency_histogram(df, OUTPUT_DIR)
    plot_f2f_stability(df, OUTPUT_DIR)
    plot_mpjpe(df, OUTPUT_DIR)
    plot_combined_dashboard(df, OUTPUT_DIR)

    # ── print summary to console ──
    print_summary_table(df)

    print(f"[benchmark] All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("[benchmark] Done ✓")


if __name__ == "__main__":
    main()
