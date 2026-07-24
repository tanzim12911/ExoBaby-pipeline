# =============================================================================
# detection/qwen35_detector.py — Qwen3.5-9B closed-list detection pipeline
# =============================================================================
# Uses the model and quantization config from qwen3_5_9b_word_miner.ipynb,
# but runs inference as a closed-list detector (ExoBaby_LongTailed_Colab style):
#   - Show the model the CDI vocabulary
#   - Ask which categories are clearly visible in the frame
#   - Parse the comma-separated response
#   - Write matched categories to CSV (same schema as detector.py)
#
# Model config (exact from word miner)
# -------------------------------------
#   Model      : Qwen/Qwen3.5-9B
#   Class      : AutoModelForImageTextToText (fallback: AutoModelForVision2Seq)
#   Quant      : 4-bit NF4, float16, double_quant=True
#   Resolution : min_pixels=256*256, max_pixels=512*512
#   Thinking   : enable_thinking=False  (critical — suppresses reasoning chain)
#   Generation : max_new_tokens=150, temperature=0, do_sample=False
#
# Frame exclusion
# ---------------
#   load_mining_exclusions(log_path) reads the mining log and returns a set of
#   relative frame paths to exclude from inference (frames already used to build
#   the CDI vocabulary, preventing circular evaluation).
#
# Output CSV schema (identical to detector.py for downstream compatibility)
# -------------------------------------------------------------------------
#   frame_path      — absolute path to the input frame
#   class_name      — detected CDI category; empty string = sentinel (no match)
#   confidence      — 1.0 for all detections (model gives no score)
#   clip_similarity — 1.0 (no CLIP stage; field preserved for downstream compat)
#
# Public functions
# ----------------
#   load_mining_exclusions(log_path) -> set[str]
#   load_qwen35_model(device) -> (model, processor)
#   run_qwen35_detection(
#       frame_paths, included_categories, output_csv,
#       model, processor, device, prune_stale=False
#   ) -> None

import csv
import logging
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants  (exact values from qwen3_5_9b_word_miner.ipynb)
# ---------------------------------------------------------------------------

QWEN35_MODEL_ID: str = "Qwen/Qwen3.5-9B"
MAX_PIXELS:      int = 512 * 512   # 262,144  — exact word miner value
MIN_PIXELS:      int = 256 * 256   # 65,536   — exact word miner value
MAX_NEW_TOKENS:  int = 150         # generous for closed-list comma-sep response

_FIELDNAMES = ["frame_path", "class_name", "confidence", "clip_similarity"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a strict object detection assistant. "
    "You only output objects from the provided list."
)

def _build_prompt(categories: list) -> str:
    """
    Build the closed-list detection prompt.

    ``categories`` must already be sorted by frequency descending — the caller
    is responsible for this ordering so the most important categories appear
    first and receive the model's strongest attention.
    """
    vocab_str = ", ".join(categories)
    return (
        "From the list below, name only the objects you can clearly see in this image.\n"
        "Be conservative — if you are not confident an object is present, omit it.\n"
        "Reply with ONLY a comma-separated list of the visible object names, "
        "or 'none' if nothing from the list is visible.\n\n"
        f"Objects ({len(categories)} total, most frequent first):\n{vocab_str}"
    )


# ---------------------------------------------------------------------------
# Mining frame exclusion helper
# ---------------------------------------------------------------------------

def load_mining_exclusions(log_path: str) -> set:
    """
    Parse the word-miner frame log and return a set of relative frame paths
    (``video_id/chunk/frame_name.jpg``) that should be excluded from inference.

    These are the frames that were already used to build the CDI vocabulary.
    Excluding them prevents circular evaluation.

    Log line format:
        1: frame_0026.jpg (Full path: /content/drive/MyDrive/data/filtered/VID/CHUNK/frame.jpg)

    Parameters
    ----------
    log_path : str
        Path to ``Qwen3_5_9B_mining_sampled_frames_log.txt``.

    Returns
    -------
    set of strings — relative paths in the form ``video_id/chunk/frame.jpg``.
    Returns an empty set (with a warning) if the file is not found.
    """
    if not os.path.exists(log_path):
        logger.warning(
            "Mining log not found at '%s'. No frames will be excluded.", log_path
        )
        return set()

    exclusions = set()
    pattern    = re.compile(r"Full path:\s*.+?/data/filtered/(.+?)\)")

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                exclusions.add(m.group(1).strip())

    logger.info("Loaded %d mining exclusions from %s", len(exclusions), log_path)
    return exclusions


# ---------------------------------------------------------------------------
# Model loading  (exact word miner config)
# ---------------------------------------------------------------------------

def load_qwen35_model(device: str, model_id: str = QWEN35_MODEL_ID):
    """
    Load Qwen3.5-9B in 4-bit NF4 quantization (float16).

    Configuration is exactly as used in qwen3_5_9b_word_miner.ipynb.

    Estimated VRAM on T4 (16 GB):
      - Weights (4-bit NF4, 9B): ~5.5 GB
      - KV cache + activations:  ~4–5 GB
      - Total:                   ~9–11 GB  (fits T4 with headroom)

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.
    model_id : str
        HuggingFace model ID (default: Qwen/Qwen3.5-9B).

    Returns
    -------
    (model, processor)
    """
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "transformers not found. "
            "Run: pip install git+https://github.com/huggingface/transformers.git"
        ) from exc

    try:
        from qwen_vl_utils import process_vision_info  # noqa: F401 — verify it exists
    except ImportError as exc:
        raise ImportError(
            "qwen-vl-utils not installed. Run: pip install qwen-vl-utils"
        ) from exc

    # Model class — try newest API first (exact word miner fallback chain)
    _QwenCls = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            import importlib
            _mod    = importlib.import_module("transformers")
            _QwenCls = getattr(_mod, cls_name)
            logger.info("Using model class: %s", cls_name)
            break
        except AttributeError:
            continue

    if _QwenCls is None:
        raise ImportError(
            "No compatible model class found. Upgrade transformers."
        )

    # Quantization config — exact word miner values
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # float16, matching word miner
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading %s (4-bit NF4, float16) ...", model_id)
    model = _QwenCls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,              # exact word miner arg
    )
    model.eval()
    logger.info("Model loaded.")

    # Processor — no pixel constraints here; they go in the message content
    logger.info("Loading processor ...")
    processor = AutoProcessor.from_pretrained(model_id)
    logger.info("Processor ready.")

    return model, processor


# ---------------------------------------------------------------------------
# Resume / stale-row helpers
# ---------------------------------------------------------------------------

def load_already_done(output_csv: str) -> set:
    """Return the set of frame paths already written to *output_csv*."""
    if not os.path.exists(output_csv):
        return set()
    import pandas as pd
    df = pd.read_csv(output_csv)
    if "frame_path" not in df.columns:
        return set()
    return set(df["frame_path"].dropna().unique())


def _check_stale_rows(output_csv: str, current_frame_paths: list) -> int:
    if not os.path.exists(output_csv):
        return 0
    import pandas as pd
    df          = pd.read_csv(output_csv)
    current_set = set(current_frame_paths)
    stale       = int((~df["frame_path"].isin(current_set)).sum())
    if stale > 0:
        logger.warning(
            "%d rows in the CSV reference frames not in current frame_paths. "
            "Set prune_stale=True to remove them.", stale
        )
    else:
        logger.info("CSV check passed — no stale rows found.")
    return stale


def _prune_stale_rows(output_csv: str, current_frame_paths: list) -> int:
    if not os.path.exists(output_csv):
        return 0
    import pandas as pd
    df          = pd.read_csv(output_csv)
    current_set = set(current_frame_paths)
    df_clean    = df[df["frame_path"].isin(current_set)]
    removed     = len(df) - len(df_clean)
    if removed > 0:
        df_clean.to_csv(output_csv, index=False)
        logger.info("Pruned %d stale rows.", removed)
    return removed


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(response_text: str, categories_set: set) -> list:
    """
    Parse the model's comma-separated closed-list response.

    Lowercases and strips each token, then keeps only tokens that appear
    in ``categories_set``.  Returns an empty list for 'none' or empty output.
    """
    raw = response_text.strip().rstrip(".")
    if not raw or raw.lower() == "none":
        return []
    tokens = [t.strip().lower().rstrip(".,;:") for t in raw.split(",")]
    return [t for t in tokens if t in categories_set]


# ---------------------------------------------------------------------------
# Core detection loop
# ---------------------------------------------------------------------------

def run_qwen35_detection(
    frame_paths: list,
    included_categories: list,
    output_csv: str,
    model,
    processor,
    device: str,
    prune_stale: bool = False,
) -> None:
    """
    Run Qwen3.5-9B closed-list detection on *frame_paths*.

    Parameters
    ----------
    frame_paths : list[str]
        Absolute paths to frames for inference (mining frames already excluded
        by the caller using load_mining_exclusions()).
    included_categories : list[str]
        CDI category names — MUST be sorted by frequency DESCENDING so the
        most important categories appear first in the prompt.
    output_csv : str
        Path to the output CSV (created or appended to).
    model, processor
        Objects returned by load_qwen35_model().
    device : str
        ``"cuda"`` or ``"cpu"``.
    prune_stale : bool
        If True, remove CSV rows for frames not in frame_paths before running.

    Notes
    -----
    - Resumable: frames already in output_csv are skipped.
    - One frame per inference call (matches word miner pattern).
    """
    from qwen_vl_utils import process_vision_info

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    if prune_stale:
        _prune_stale_rows(output_csv, frame_paths)
    else:
        _check_stale_rows(output_csv, frame_paths)

    already_done  = load_already_done(output_csv)
    frames_to_run = [f for f in frame_paths if f not in already_done]

    logger.info(
        "Qwen3.5-9B detection: %d already done, %d remaining (total %d).",
        len(already_done), len(frames_to_run), len(frame_paths),
    )

    if not frames_to_run:
        logger.info("All frames already processed — nothing to do.")
        return

    categories_set = set(c.lower() for c in included_categories)
    prompt         = _build_prompt(included_categories)

    write_header = not os.path.exists(output_csv)
    n_errors     = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for frame_path in tqdm(frames_to_run, desc="Qwen3.5-9B detection", unit="frame"):
            if not os.path.isfile(frame_path):
                logger.warning("Frame not found, skipping: %s", frame_path)
                continue
            try:
                _process_frame(
                    frame_path, prompt, categories_set,
                    model, processor, device,
                    process_vision_info, writer,
                )
            except Exception as exc:
                n_errors += 1
                if n_errors <= 5:
                    logger.warning(
                        "Error on frame %s: %s", os.path.basename(frame_path), exc
                    )
                _write_sentinel(writer, frame_path)

    logger.info("Detection complete. Errors: %d.", n_errors)
    if n_errors > 5:
        logger.warning("(%d additional errors suppressed)", n_errors - 5)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_sentinel(writer: csv.DictWriter, frame_path: str) -> None:
    """Write a sentinel row (no detection) so the frame is skipped on resume."""
    writer.writerow({
        "frame_path":      frame_path,
        "class_name":      "",
        "confidence":      "",
        "clip_similarity": "",
    })


def _process_frame(
    frame_path: str,
    prompt: str,
    categories_set: set,
    model,
    processor,
    device: str,
    process_vision_info,
    writer: csv.DictWriter,
) -> None:
    """Run one inference call on a single frame and write results."""

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type":       "image",
                    "image":      frame_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # apply_chat_template with enable_thinking=False — exact word miner flag
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move all tensors to model device — exact word miner pattern
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    logger.debug("Response: %.200s", response)

    detected = _parse_response(response, categories_set)

    if not detected:
        _write_sentinel(writer, frame_path)
    else:
        for class_name in detected:
            writer.writerow({
                "frame_path":      frame_path,
                "class_name":      class_name,
                "confidence":      1.0,
                "clip_similarity": 1.0,
            })
