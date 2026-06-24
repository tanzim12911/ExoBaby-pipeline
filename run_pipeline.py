# =============================================================================
# run_pipeline.py — Main orchestrator for the ExoBaby pipeline
# =============================================================================
# Usage:
#   python run_pipeline.py                  # run all steps
#   python run_pipeline.py --step download  # run only the download step
#   python run_pipeline.py --step segment
#   python run_pipeline.py --step filter
#   python run_pipeline.py --step export
#
# The pipeline is resumable: already-processed clips are skipped automatically.

import os
import csv
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import config
from pipeline.downloader import search_and_download, download_direct, get_all_videos
from pipeline.segmenter  import segment_video
from pipeline.sampler    import sample_frames
from pipeline.vlm_filter import filter_clip
from pipeline.exporter   import export_at_1fps

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
os.makedirs(config.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "pipeline.log")),
    ],
)
logger = logging.getLogger("run_pipeline")


# -----------------------------------------------------------------------------
# Resume helpers — read the CSV log to find already-processed clips
# -----------------------------------------------------------------------------
def load_processed_clips(log_path: str) -> dict[str, dict]:
    """
    Returns a dict mapping clip_path -> result row for all clips
    already recorded in the CSV log.
    """
    processed = {}
    if not os.path.exists(log_path):
        return processed
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed[row["clip_path"]] = row
    return processed


def append_to_csv(log_path: str, row: dict):
    """Append a single result row to the CSV log."""
    file_exists = os.path.exists(log_path)
    fieldnames = ["clip_path", "video_id", "clip_name", "pass", "reason", "error"]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(
    log_path: str,
    raw_videos: int,
    total_clips: int,
    passed_clips: int,
    failed_clips: int,
    sampled_frames: int,
    exported_frames: int,
    search_queries: list[str] | None = None,
):
    """
    Append one summary row to the run-summary CSV.
    Creates the file with a header if it does not yet exist.
    """
    file_exists = os.path.exists(log_path)
    fieldnames = [
        "timestamp",
        "search_queries",
        "raw_videos",
        "total_clips",
        "passed_clips",
        "failed_clips",
        "sampled_frames",
        "exported_frames",
    ]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "search_queries": " | ".join(search_queries) if search_queries else "",
            "raw_videos": raw_videos,
            "total_clips": total_clips,
            "passed_clips": passed_clips,
            "failed_clips": failed_clips,
            "sampled_frames": sampled_frames,
            "exported_frames": exported_frames,
        })


def append_to_json(log_path: str, row: dict):
    """
    Append a single result row to the JSON log.

    The file is kept as a JSON array. We read the existing array,
    append the new entry, and write it back. This is safe for the
    dataset sizes we expect (thousands of clips, not millions).
    """
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Normalise the 'pass' field to a proper bool for JSON
    entry = dict(row)
    if isinstance(entry.get("pass"), str):
        entry["pass"] = entry["pass"].lower() == "true"

    data.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Step 1 — Download
# -----------------------------------------------------------------------------
def run_download():
    logger.info("=== STEP 1: Downloading videos ===")
    for query in config.SEARCH_QUERIES:
        search_and_download(
            query=query,
            max_videos=config.MAX_VIDEOS_PER_QUERY,
            output_dir=config.RAW_VIDEO_DIR,
            video_format=config.VIDEO_FORMAT,
            merge_format=config.VIDEO_MERGE_FORMAT,
        )
    download_direct(
        urls_file=config.DIRECT_VIDEOS_FILE,
        output_dir=config.RAW_VIDEO_DIR,
        video_format=config.VIDEO_FORMAT,
        merge_format=config.VIDEO_MERGE_FORMAT,
    )
    videos = get_all_videos(config.RAW_VIDEO_DIR)
    logger.info(f"Total videos available: {len(videos)}")
    return videos


# -----------------------------------------------------------------------------
# Step 2 — Segment
# -----------------------------------------------------------------------------
def run_segment(videos: list[str]) -> list[tuple[str, str]]:
    """
    Returns a list of (video_id, clip_path) tuples for all clips.
    """
    logger.info("=== STEP 2: Segmenting videos into 30s clips ===")
    all_clips = []
    for video_path in videos:
        video_id = Path(video_path).stem
        clip_dir = os.path.join(config.CLIPS_DIR, video_id)

        # Skip if already segmented
        if os.path.isdir(clip_dir) and len(os.listdir(clip_dir)) > 0:
            logger.info(f"Already segmented, skipping: {video_id}")
            existing = [
                (video_id, os.path.join(clip_dir, f))
                for f in sorted(os.listdir(clip_dir))
                if f.endswith(".mp4")
            ]
            all_clips.extend(existing)
            continue

        clips = segment_video(video_path, clip_dir, config.CLIP_DURATION_SECONDS)
        all_clips.extend([(video_id, c) for c in clips])

    logger.info(f"Total clips: {len(all_clips)}")
    return all_clips


# -----------------------------------------------------------------------------
# Step 3+4+5 — Sample frames and filter via VLM
# -----------------------------------------------------------------------------
def run_filter(all_clips: list[tuple[str, str]]) -> tuple[list[str], int]:
    """
    For each clip, sample frames and send to Gemini for filtering.
    Skips clips already recorded in the CSV log.
    Returns (passed_clip_paths, total_sampled_frames).
    """
    logger.info("=== STEPS 3-5: Sampling frames and filtering via Gemini ===")

    processed = load_processed_clips(config.FILTER_LOG_CSV)
    passed_clips = [
        row["clip_path"]
        for row in processed.values()
        if row["pass"] == "True"
    ]

    remaining = [
        (vid_id, clip_path)
        for vid_id, clip_path in all_clips
        if clip_path not in processed
    ]
    logger.info(
        f"{len(processed)} clips already processed, "
        f"{len(remaining)} remaining, "
        f"{len(passed_clips)} passed so far."
    )

    total_sampled_frames = 0

    for vid_id, clip_path in remaining:
        clip_name = Path(clip_path).stem

        # Step 3: Sample frames
        frame_dir = os.path.join(config.FRAMES_DIR, vid_id, clip_name)
        frames = sample_frames(clip_path, frame_dir, config.FRAMES_PER_CLIP)

        if not frames:
            logger.warning(f"No frames sampled for {clip_path}, skipping.")
            failed_row = {
                "clip_path": clip_path,
                "video_id": vid_id,
                "clip_name": clip_name,
                "pass": False,
                "reason": "Frame sampling failed",
                "error": "no frames",
            }
            append_to_csv(config.FILTER_LOG_CSV, failed_row)
            append_to_json(config.FILTER_LOG_JSON, failed_row)
            continue

        total_sampled_frames += len(frames)

        # Steps 4+5: VLM filter
        result = filter_clip(
            frame_paths=frames,
            api_key=config.GEMINI_API_KEY,
            model_name=config.GEMINI_MODEL,
            rate_limit_seconds=config.GEMINI_RATE_LIMIT_SECONDS,
        )

        row = {
            "clip_path": clip_path,
            "video_id": vid_id,
            "clip_name": clip_name,
            "pass": result.get("pass", False),
            "reason": result.get("reason", ""),
            "error": result.get("error", ""),
        }
        append_to_csv(config.FILTER_LOG_CSV, row)
        append_to_json(config.FILTER_LOG_JSON, row)

        status = "PASS" if result.get("pass") else "FAIL"
        logger.info(f"[{status}] {clip_name}: {result.get('reason', '')}")

        if result.get("pass"):
            passed_clips.append(clip_path)

    failed_count = len(all_clips) - len(passed_clips)
    logger.info(
        f"Filtering complete. "
        f"Passed: {len(passed_clips)}, "
        f"Failed: {failed_count}, "
        f"Sampled frames: {total_sampled_frames}"
    )
    return passed_clips, total_sampled_frames


# -----------------------------------------------------------------------------
# Step 7 — Export at 1 FPS
# -----------------------------------------------------------------------------
def run_export(passed_clips: list[str]) -> int:
    """
    Export approved clips at 1 FPS.
    Returns the total number of exported frames.
    """
    logger.info("=== STEP 7: Exporting approved clips at 1 FPS ===")
    total_exported_frames = 0
    for clip_path in passed_clips:
        # Reconstruct output path from clip path
        # data/clips/<video_id>/<clip_name>.mp4
        #   -> data/filtered/<video_id>/<clip_name>/
        parts = Path(clip_path).parts
        try:
            clips_idx = parts.index("clips")
            video_id  = parts[clips_idx + 1]
            clip_name = Path(clip_path).stem
        except (ValueError, IndexError):
            video_id  = "unknown"
            clip_name = Path(clip_path).stem

        output_dir = os.path.join(config.FILTERED_DIR, video_id, clip_name)

        # Skip if already exported
        if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.info(f"Already exported, skipping: {clip_name}")
            total_exported_frames += len([
                f for f in os.listdir(output_dir) if f.endswith(".jpg")
            ])
            continue

        frames = export_at_1fps(clip_path, output_dir)
        total_exported_frames += len(frames)

    logger.info(f"Export complete. Total exported frames: {total_exported_frames}")
    return total_exported_frames


# -----------------------------------------------------------------------------
# Summary helpers — always scan disk so partial/resumed runs are accurate
# -----------------------------------------------------------------------------
def count_sampled_frames() -> int:
    """Count JPEG frames under data/frames/ (sampled by the VLM filter step)."""
    total = 0
    if not os.path.isdir(config.FRAMES_DIR):
        return total
    for vid_id in os.listdir(config.FRAMES_DIR):
        vid_dir = os.path.join(config.FRAMES_DIR, vid_id)
        if not os.path.isdir(vid_dir):
            continue
        for clip_name in os.listdir(vid_dir):
            clip_dir = os.path.join(vid_dir, clip_name)
            if os.path.isdir(clip_dir):
                total += sum(1 for f in os.listdir(clip_dir) if f.endswith(".jpg"))
    return total


def count_exported_frames() -> int:
    """Count JPEG frames under data/filtered/ (exported at 1 FPS)."""
    total = 0
    if not os.path.isdir(config.FILTERED_DIR):
        return total
    for vid_id in os.listdir(config.FILTERED_DIR):
        vid_dir = os.path.join(config.FILTERED_DIR, vid_id)
        if not os.path.isdir(vid_dir):
            continue
        for clip_name in os.listdir(vid_dir):
            clip_dir = os.path.join(vid_dir, clip_name)
            if os.path.isdir(clip_dir):
                total += sum(1 for f in os.listdir(clip_dir) if f.endswith(".jpg"))
    return total


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ExoBaby pipeline orchestrator")
    parser.add_argument(
        "--step",
        choices=["download", "segment", "filter", "export", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    if config.GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning(
            "GEMINI_API_KEY is not set. "
            "Set it in config.py or as the GEMINI_API_KEY environment variable."
        )

    step = args.step

    if step in ("download", "all"):
        videos = run_download()
    else:
        videos = get_all_videos(config.RAW_VIDEO_DIR)

    if step in ("segment", "all"):
        all_clips = run_segment(videos)
    else:
        # Reconstruct clip list from disk
        all_clips = []
        if os.path.isdir(config.CLIPS_DIR):
            for vid_id in os.listdir(config.CLIPS_DIR):
                clip_dir = os.path.join(config.CLIPS_DIR, vid_id)
                if os.path.isdir(clip_dir):
                    for f in sorted(os.listdir(clip_dir)):
                        if f.endswith(".mp4"):
                            all_clips.append((vid_id, os.path.join(clip_dir, f)))

    if step in ("filter", "all"):
        passed_clips, _ = run_filter(all_clips)
    else:
        processed = load_processed_clips(config.FILTER_LOG_CSV)
        passed_clips = [
            row["clip_path"]
            for row in processed.values()
            if row["pass"] == "True"
        ]

    if step in ("export", "all"):
        run_export(passed_clips)

    # ------------------------------------------------------------------
    # Write summary — totals are always read from disk/CSV so every run
    # (including partial or resumed ones) produces accurate numbers.
    # ------------------------------------------------------------------
    processed    = load_processed_clips(config.FILTER_LOG_CSV)
    raw_count    = len(get_all_videos(config.RAW_VIDEO_DIR))
    total_clips  = len(all_clips)
    passed_count = sum(1 for r in processed.values() if r["pass"] == "True")
    failed_count = sum(1 for r in processed.values() if r["pass"] != "True")
    sampled_frames  = count_sampled_frames()
    exported_frames = count_exported_frames()

    write_summary(
        log_path=config.SUMMARY_LOG_CSV,
        search_queries=config.SEARCH_QUERIES,
        raw_videos=raw_count,
        total_clips=total_clips,
        passed_clips=passed_count,
        failed_clips=failed_count,
        sampled_frames=sampled_frames,
        exported_frames=exported_frames,
    )
    logger.info(
        f"=== RUN SUMMARY === "
        f"raw_videos={raw_count}, "
        f"total_clips={total_clips}, "
        f"passed={passed_count}, "
        f"failed={failed_count}, "
        f"sampled_frames={sampled_frames}, "
        f"exported_frames={exported_frames}"
    )


if __name__ == "__main__":
    main()
