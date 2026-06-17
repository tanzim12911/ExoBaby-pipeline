# =============================================================================
# sampler.py — Step 3: Extract N evenly spaced frames from each clip
# =============================================================================

import os
import cv2
import logging

logger = logging.getLogger(__name__)


def sample_frames(
    clip_path: str,
    output_dir: str,
    n_frames: int = 4,
) -> list[str]:
    """
    Extract `n_frames` evenly spaced frames from a video clip.

    Frames are saved as JPEG files named frame_0000.jpg, frame_0001.jpg, etc.
    Returns a list of saved frame paths, or an empty list on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.error(f"Cannot open clip: {clip_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or width == 0 or height == 0:
        logger.warning(f"Invalid video stream (frames={total_frames}, {width}x{height}): {clip_path}")
        cap.release()
        return []

    # Calculate evenly spaced frame indices across the clip
    # e.g. for 900 frames and n_frames=4: [0, 225, 450, 675]
    indices = [
        int(total_frames * i / n_frames)
        for i in range(n_frames)
    ]

    saved_paths = []
    for seq, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {idx} from {clip_path}")
            continue
        out_path = os.path.join(output_dir, f"frame_{seq:04d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved_paths.append(out_path)

    cap.release()
    logger.debug(f"Sampled {len(saved_paths)} frames from {os.path.basename(clip_path)}")
    return saved_paths
