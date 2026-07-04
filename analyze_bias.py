"""
analyze_bias.py — Analyze detection CSV and clip distribution for bias mitigation.

Usage:
    python analyze_bias.py
    python analyze_bias.py --csv path/to/detections.csv
"""

import os
import argparse
from collections import defaultdict, Counter
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. Clips-per-video analysis (from disk)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_clips_on_disk(clips_dir: str) -> dict[str, int]:
    """Return {video_id: clip_count} for all videos in clips_dir."""
    counts: dict[str, int] = {}
    if not os.path.isdir(clips_dir):
        print(f"[WARN] clips dir not found: {clips_dir}")
        return counts
    for video_id in sorted(os.listdir(clips_dir)):
        clip_dir = os.path.join(clips_dir, video_id)
        if os.path.isdir(clip_dir):
            clips = [f for f in os.listdir(clip_dir) if f.endswith(".mp4")]
            counts[video_id] = len(clips)
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# 2. Detection-CSV analysis
# ──────────────────────────────────────────────────────────────────────────────

def parse_csv_data(csv_path: str):
    """
    Parse frame_path, class_name, confidence from the detections CSV.
    Returns list of dicts.
    """
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_video_id_from_path(frame_path: str) -> str:
    """
    Handles both colab-style paths:
      /content/drive/.../filtered/<video_id>/<clip_id>/frame_XXXX.jpg
    and local-style paths.
    """
    parts = Path(frame_path).parts
    for i, p in enumerate(parts):
        if p == "filtered" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def analyze_detections(rows: list[dict]) -> None:
    """Print top-class and per-video-bias analysis."""

    # ── Top classes overall ──────────────────────────────────────────────────
    class_counter: Counter = Counter()
    video_class_counter: dict[str, Counter] = defaultdict(Counter)
    video_frame_counter: Counter = Counter()

    for row in rows:
        cls = row["class_name"].strip()
        fp  = row["frame_path"].strip()
        vid = extract_video_id_from_path(fp)
        class_counter[cls] += 1
        video_class_counter[vid][cls] += 1
        video_frame_counter[vid] += 1

    total_detections = sum(class_counter.values())
    print("\n" + "=" * 60)
    print("TOP 10 OBJECT CLASSES (all detections)")
    print("=" * 60)
    for cls, cnt in class_counter.most_common(10):
        pct = 100 * cnt / total_detections
        print(f"  {cls:<20} {cnt:>5}  ({pct:5.1f}%)")

    print(f"\nTotal detections : {total_detections}")
    print(f"Unique classes   : {len(class_counter)}")
    print(f"Unique videos    : {len(video_frame_counter)}")

    # ── Top-5 classes: source-video breakdown ────────────────────────────────
    top5 = [cls for cls, _ in class_counter.most_common(5)]

    print("\n" + "=" * 60)
    print("BIAS CHECK — Top-5 classes per source video")
    print("=" * 60)
    for cls in top5:
        total_cls = class_counter[cls]
        print(f"\n  Class: {cls}  (total detections: {total_cls})")
        # Which videos contribute to this class?
        contributors = sorted(
            [(vid, cnts[cls]) for vid, cnts in video_class_counter.items() if cls in cnts],
            key=lambda x: -x[1],
        )
        for vid, cnt in contributors[:5]:
            pct = 100 * cnt / total_cls
            print(f"    {vid:<25} {cnt:>4}  ({pct:5.1f}%)")
        if len(contributors) > 5:
            rest = sum(cnt for _, cnt in contributors[5:])
            print(f"    ... {len(contributors)-5} other videos  {rest:>4}  ({100*rest/total_cls:5.1f}%)")

    # ── Per-video detection counts ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DETECTIONS PER VIDEO (top 15 by volume)")
    print("=" * 60)
    for vid, cnt in video_frame_counter.most_common(15):
        pct = 100 * cnt / total_detections
        print(f"  {vid:<30} {cnt:>5}  ({pct:5.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Optimal max_clip recommendation
# ──────────────────────────────────────────────────────────────────────────────

def recommend_max_clip(
    video_clip_counts: dict[str, int],
    filter_log_csv: str | None = None,
    thresholds: list[int] | None = None,
) -> None:
    """
    Show the passed-clip distribution and simulate MAX_PASSED_CLIPS_PER_VIDEO
    at various thresholds.

    The cap is applied to *passed* clips (post-filter), not raw clips, so the
    simulation is done on passed counts when the filter log is available.
    """
    import csv

    if not video_clip_counts:
        return

    thresholds = thresholds or [10, 20, 30, 40, 50]

    # Build passed-clips-per-video from the filter log
    passed_per_video: dict[str, int] = {}
    if filter_log_csv and os.path.exists(filter_log_csv):
        with open(filter_log_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("pass", "").lower() in ("true", "1"):
                    vid = row.get("video_id", "unknown")
                    passed_per_video[vid] = passed_per_video.get(vid, 0) + 1

    have_passed = bool(passed_per_video)
    counts_for_sim = passed_per_video if have_passed else video_clip_counts
    label = "passed clips" if have_passed else "raw clips (no filter log found)"

    total_clips = sum(video_clip_counts.values())
    total_videos = len(video_clip_counts)

    print("\n" + "=" * 60)
    print("CLIP DISTRIBUTION ON DISK (raw segments)")
    print("=" * 60)
    sorted_counts = sorted(video_clip_counts.items(), key=lambda x: -x[1])
    for vid, cnt in sorted_counts:
        passed = passed_per_video.get(vid, "–")
        passed_str = f"  ({passed} passed)" if passed != "–" else ""
        print(f"  {vid:<30} {cnt:>4} raw clips{passed_str}")
    print(f"\n  Total: {total_clips} raw clips across {total_videos} videos")

    total_for_sim = sum(counts_for_sim.values())
    print(f"\n  Total {label}: {total_for_sim}")

    print("\n" + "=" * 60)
    print(f"MAX_PASSED_CLIPS_PER_VIDEO SIMULATION (on {label})")
    print("=" * 60)
    for cap in thresholds:
        kept = sum(min(cnt, cap) for cnt in counts_for_sim.values())
        pct_kept = 100 * kept / total_for_sim if total_for_sim else 0
        videos_capped = sum(1 for cnt in counts_for_sim.values() if cnt > cap)
        print(
            f"  cap={cap:>3}:  {kept:>4} clips exported  ({pct_kept:5.1f}%)  "
            f"  {videos_capped} video(s) would be capped"
        )

    sorted_sim_counts = sorted(counts_for_sim.values())
    median_clips = sorted_sim_counts[len(sorted_sim_counts) // 2]
    recommended = int(median_clips * 1.5)
    recommended = round(recommended / 5) * 5 or 5  # at least 5
    print(f"\n  Median {label}/video: {median_clips}")
    print(f"  Suggested cap (median × 1.5, rounded to 5): {recommended}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bias analysis for ExoBaby pipeline")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to merged frame-detections CSV (optional)",
    )
    parser.add_argument(
        "--clips-dir",
        default="data/clips",
        help="Path to clips directory (default: data/clips)",
    )
    parser.add_argument(
        "--filter-log",
        default="logs/filter_results.csv",
        help="Path to filter_results.csv (default: logs/filter_results.csv)",
    )
    args = parser.parse_args()

    # ── Disk analysis ────────────────────────────────────────────────────────
    video_clip_counts = analyze_clips_on_disk(args.clips_dir)

    # ── Detection-CSV analysis ───────────────────────────────────────────────
    if args.csv and os.path.exists(args.csv):
        rows = parse_csv_data(args.csv)
        analyze_detections(rows)
    else:
        if args.csv:
            print(f"[INFO] Detection CSV not found at '{args.csv}', skipping class analysis.")
        else:
            print("[INFO] No --csv provided; skipping class analysis.")

    # ── Max-clip recommendation ──────────────────────────────────────────────
    recommend_max_clip(
        video_clip_counts=video_clip_counts,
        filter_log_csv=args.filter_log,
        thresholds=[10, 20, 30, 40, 50],
    )


if __name__ == "__main__":
    main()
