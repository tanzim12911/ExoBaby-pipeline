# =============================================================================
# detection/craftx_detector.py — CraftX API detection backend for Qwen3.5-9B
# =============================================================================
# Uses CraftX's OpenAI-compatible API to perform closed-list object detection
# on video frames using Qwen3.5 9B.
#
# Key Features:
#   - Pre-flight diagnostic check before running batch inference
#   - Base64 image encoding with automatic resizing
#   - Multi-strategy response parser (handles comma, bullets, numbered lists, raw text)
#   - Multi-threaded parallel API requests (ThreadPoolExecutor)
#   - Automatic exponential backoff & retry handling for rate limits
#   - Resumable CSV output matching detector.py schema

import base64
import csv
import io
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_API_URL: str = "https://api.craftx.corecraftsolutions.com/api/v1/chat/completions"
DEFAULT_MODEL:   str = "Qwen3.5 9B"
_FIELDNAMES = ["frame_path", "class_name", "confidence", "clip_similarity"]


def _build_prompt(categories: list) -> str:
    """
    Build closed-list detection prompt sorted by category presence frequency.
    """
    vocab_str = ", ".join(categories)
    return (
        "You are a strict object detection assistant.\n"
        "Examine the image carefully.\n"
        "From the vocabulary list below, identify ONLY the objects you can clearly see in this image.\n"
        "Be CONSERVATIVE — if you are not confident an object is present, omit it.\n"
        "Reply with ONLY a comma-separated list of the visible object names from the vocabulary, "
        "or 'none' if no listed objects are visible.\n\n"
        f"Vocabulary ({len(categories)} objects, most frequent first):\n{vocab_str}"
    )


def load_mining_exclusions(log_path: str) -> set:
    """
    Parse the word-miner frame log and return set of relative frame paths
    (video_id/chunk/frame_name.jpg) used for word mining.
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
    Loads, resizes (max 768px side to balance quality & API speed), and converts to JPEG Base64 data string.
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
    Robust multi-strategy response parser.
    Handles comma-separated lists, numbered lists, bullet points, JSON, and raw text.
    """
    if not response_text or not isinstance(response_text, str):
        return []

    text = response_text.strip()
    if text.lower() in ("none", "null", "no objects", "no objects visible", "n/a"):
        return []

    # Clean list numbers, bullets, newlines, and semicolons
    text_clean = re.sub(r"^\s*[\d+\-\*•]\s*", "", text, flags=re.MULTILINE)
    text_clean = text_clean.replace("\n", ",").replace(";", ",")

    tokens = [t.strip().lower().rstrip(".,;:!\"'") for t in text_clean.split(",")]

    detected = []
    seen = set()
    for t in tokens:
        t_clean = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)
        if t_clean in categories_set and t_clean not in seen:
            detected.append(t_clean)
            seen.add(t_clean)

    # Fallback: substring regex search if line parsing produced nothing
    if not detected:
        text_lower = text.lower()
        for cat in categories_set:
            if re.search(rf"\b{re.escape(cat)}\b", text_lower) and cat not in seen:
                detected.append(cat)
                seen.add(cat)

    return detected


def call_craftx_api(
    frame_path: str,
    prompt: str,
    categories_set: set,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    max_retries: int = 4,
    debug: bool = False,
) -> tuple:
    """
    Sends a single frame request to CraftX API with exponential backoff retries.
    Returns (detected_categories_list, raw_response_text, status_code).
    """
    try:
        img_b64 = _encode_image_base64(frame_path)
    except Exception as exc:
        if debug:
            print(f"[Image Error] Could not read/encode image {frame_path}: {exc}")
        return [], "", 0

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Combined prompt in user message for maximum vision model compatibility
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            }
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }

    delay = 1.0
    last_status = 0
    last_text = ""

    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=40)
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _parse_response(content, categories_set)
                return parsed, content, 200
            elif resp.status_code in (429, 500, 502, 503, 504):
                last_text = resp.text[:300]
                if debug:
                    print(f"[CraftX API Retry] Status {resp.status_code} (attempt {attempt+1}/{max_retries}): {last_text}")
                time.sleep(delay)
                delay *= 2.0
            else:
                last_text = resp.text[:300]
                if debug:
                    print(f"[CraftX API Error {resp.status_code}] {last_text}")
                break
        except Exception as exc:
            last_text = str(exc)
            if debug:
                print(f"[CraftX Network Exception] {exc}")
            time.sleep(delay)
            delay *= 2.0

    return [], last_text, last_status


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
    Runs multi-threaded CraftX API detection across frame_paths with pre-flight diagnostics.
    """
    if not api_key or api_key == "YOUR_CRAFTX_API_KEY_HERE":
        raise ValueError("CRAFTX_API_KEY is invalid or not set. Please update CRAFTX_API_KEY in Step 1.")

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

    # -------------------------------------------------------------------------
    # Pre-flight Diagnostic Check on First Frame
    # -------------------------------------------------------------------------
    test_frame = frames_to_run[0]
    print("\n" + "=" * 70)
    print("🔍 RUNNING CRAFTX PRE-FLIGHT DIAGNOSTIC TEST...")
    print(f"Testing frame : {os.path.basename(test_frame)}")
    print(f"Model ID      : {model_name}")
    print(f"API Endpoint  : {api_url}")
    print("=" * 70)

    test_detected, test_raw, test_status = call_craftx_api(
        test_frame, prompt, categories_set, api_key, model_name, api_url, debug=True
    )

    print(f"HTTP Status Code : {test_status}")
    print(f"Raw API Response : {repr(test_raw)}")
    print(f"Parsed Detections: {test_detected}")
    print("=" * 70 + "\n")

    if test_status != 200:
        print(f"❌ PRE-FLIGHT TEST FAILED with HTTP Status {test_status}!")
        print(f"Error details: {test_raw}")
        print("Please check your CRAFTX_API_KEY, wallet balance, and Model Name in Step 1.")
        return

    print("✅ PRE-FLIGHT TEST PASSED! Starting batch detection...\n")

    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Multi-threaded execution for API requests
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
                    detected, _, status = future.result()
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
