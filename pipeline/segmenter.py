# =============================================================================
# segmenter.py — Step 2: Split videos into 30-second clips
# =============================================================================

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def segment_video(
    video_path: str,
    output_dir: str,
    clip_duration: int = 30,
) -> list[str]:
    """
    Split a video into fixed-length clips using ffmpeg.

    Uses stream copy (-c copy) so there is no re-encoding — this is fast
    and lossless. The last clip may be shorter than clip_duration if the
    video length is not a perfect multiple.

    Returns a list of paths to the generated clip files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Naming pattern: 0000.mp4, 0001.mp4, ...
    output_pattern = os.path.join(output_dir, "%04d.mp4")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-c", "copy",               # no re-encode, very fast
        "-map", "0",
        "-segment_time", str(clip_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        "-y",                       # overwrite existing files
        output_pattern,
    ]

    logger.info(f"Segmenting: {video_path} -> {output_dir}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg error for {video_path}:\n{result.stderr}")
        return []

    clips = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".mp4")
    ])
    logger.info(f"Created {len(clips)} clips from {os.path.basename(video_path)}")
    return clips


def get_clip_dirs(clips_root: str) -> list[str]:
    """Return all subdirectory paths under clips_root (one per video)."""
    return [
        os.path.join(clips_root, d)
        for d in os.listdir(clips_root)
        if os.path.isdir(os.path.join(clips_root, d))
    ]
