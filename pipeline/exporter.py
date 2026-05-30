# =============================================================================
# exporter.py — Step 7: Extract 1 FPS frames from approved clips
# =============================================================================

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def export_at_1fps(clip_path: str, output_dir: str) -> list[str]:
    """
    Extract one frame per second from a clip using ffmpeg.

    Output frames are named frame_0001.jpg, frame_0002.jpg, etc.
    Returns a list of output frame paths, or empty list on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", clip_path,
        "-vf", "fps=1",
        "-q:v", "2",        # JPEG quality (2 = high quality, lower = smaller files)
        "-y",               # overwrite existing
        output_pattern,
    ]

    logger.info(f"Exporting 1FPS frames: {clip_path} -> {output_dir}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg export error for {clip_path}:\n{result.stderr}")
        return []

    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jpg")
    ])
    logger.info(f"Exported {len(frames)} frames from {os.path.basename(clip_path)}")
    return frames
