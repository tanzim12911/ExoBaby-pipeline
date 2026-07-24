# =============================================================================
# detection/craftx_detector.py — CraftX API detection backend for Qwen3.5-9B
# =============================================================================
# Uses CraftX's OpenAI-compatible API to perform closed-list object detection
# on video frames using Qwen3.5 9B.
#
# Features:
#   - Base64 image encoding with automatic resizing & quality compression
#   - Closed-list prompt with categories sorted by presence frequency DESC
#   - Multi-threaded parallel API requests (10x faster than serial calls)
#   - Automatic exponential backoff & retry handling for rate limits / network glitches
#   - Resumable CSV output matching the detector.py schema
#
# Public functions:
#   load_mining_exclusions(log_path) -> set[str]
#   run_craftx_detection(...) -> None

import base64
import csv
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default API configuration
DEFAULT_API_URL: str = "https://api.craftx.corecraftsolutions.com/api/v1/chat/completions"
DEFAULT_MODEL:   str = "Qwen3.5 9B"
_FIELDNAMES = ["frame_path", "class_name", "confidence", "clip_similarity"]

_SYSTEM_PROMPT = (
    "You are a strict object detection assistant. "
    "You only output objects from the provided list."
)


def _build_prompt(categories: list) -> str:
    """
    Build closed-list detection prompt sorted by category frequency.
    """
    vocab_str = ", ".join(categories)
    return (
        "From the list below, name only the objects you can clearly see in this image.\n"
        "Be conservative — if you are not confident an object is present, omit it.\n"
        "Reply with ONLY a comma-separated list of the visible object names, "
        "or 'none' if nothing from the list is visible.\n\n"
        f"Objects ({len(categories)} total, most frequent first):\n{vocab_str}"
    )


def load_mining_exclusions(log_path: str) -> set:
    """
    Parse the word-miner frame log and return a set of relative frame paths
    (video_id/chunk/frame_name.jpg) that were used to mine the CDI vocabulary.
    """
    if not os.path.exists(log_path):
        logger.warning("Mining log not found at '%s'. No frames excluded.", log_path)
        return set()

    exclusions = set()
    pattern = re.compile(r"Full path:\s*.+?/data/filtered/(.+?)\)")

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                exclusions.add(m.group(1).strip())

    logger.info("Loaded %d mining exclusions from %s", len(exclusions), log_path)
    return exclusions


def load_already_done(output_csv: str) -> set:
    """Return set of frame paths already saved in output_csv."""
    if not os.path.exists(output_csv):
        return set()
    import pandas as pd
    df = pd.read_csv(output_csv)
    if "frame_path" not in df.columns:
        return set()
    return set(df["frame_path"].dropna().unique())


def _encode_image_base64(image_path: str, max_dim: int = 768) -> str:
    """
    Loads, resizes (max 768px side to save bandwidth), and converts image to JPEG Base64 data string.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64_bytes = base64.b64encode(buffer.getvalue())
        return f"data:image/jpeg;base64,{b64_bytes.decode('utf-8')}"


def _parse_response(response_text: str, categories_set: set) -> list:
    """
    Parse comma-separated model output into valid category list.
    """
    raw = response_text.strip().rstrip(".")
    if not raw or raw.lower() in ("none", "null", "no objects"):
        return []
    tokens = [t.strip().lower().rstrip(".,;:") for t in raw.split(",")]
    return [t for t in tokens if t in categories_set]


def call_craftx_api(
    frame_path: str,
    prompt: str,
    categories_set: set,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    max_retries: int = 4,
) -> list:
    """
    Sends a single frame request to CraftX API with retries and exponential backoff.
    """
    img_b64 = _encode_image_base64(frame_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            },
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _parse_response(content, categories_set)
            elif resp.status_code in (429, 500, 502, 503, 504):
                logger.warning("CraftX API status %d (attempt %d/%d), retrying in %.1fs...", resp.status_code, attempt + 1, max_retries, delay)
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error("CraftX API error %d: %s", resp.status_code, resp.text[:200])
                break
        except Exception as exc:
            logger.warning("CraftX API request exception (%s), retrying in %.1fs...", exc, delay)
            time.sleep(delay)
            delay *= 2.0

    return []


def run_craftx_detection(
    frame_paths: list,
    included_categories: list,
    output_csv: str,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    max_workers: int = 8,
    prune_stale: bool = False,
) -> None:
    """
    Runs multi-threaded CraftX API detection across frame_paths and appends results to output_csv.
    """
    if not api_key:
        raise ValueError("CRAFTX_API_KEY is empty. Please set a valid API key.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    already_done = load_already_done(output_csv)
    frames_to_run = [f for f in frame_paths if f not in already_done]

    logger.info(
        "CraftX API detection: %d done, %d remaining (total %d frames). Workers: %d.",
        len(already_done), len(frames_to_run), len(frame_paths), max_workers
    )

    if not frames_to_run:
        logger.info("All frames already processed.")
        return

    categories_set = set(c.lower() for c in included_categories)
    prompt = _build_prompt(included_categories)
    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Multi-threaded execution for API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(
                    call_craftx_api,
                    frame, prompt, categories_set, api_key, model_name, api_url
                ): frame
                for frame in frames_to_run
            }

            for future in tqdm(as_completed(future_to_frame), total=len(frames_to_run), desc="CraftX API Detection"):
                frame_path = future_to_frame[future]
                try:
                    detected = future.result()
                except Exception as exc:
                    logger.error("Failed frame %s: %s", os.path.basename(frame_path), exc)
                    detected = []

                if not detected:
                    writer.writerow({
                        "frame_path": frame_path,
                        "class_name": "",
                        "confidence": "",
                        "clip_similarity": "",
                    })
                else:
                    for class_name in detected:
                        writer.writerow({
                            "frame_path": frame_path,
                            "class_name": class_name,
                            "confidence": 1.0,
                            "clip_similarity": 1.0,
                        })
                csvfile.flush()

    logger.info("CraftX detection complete!")
