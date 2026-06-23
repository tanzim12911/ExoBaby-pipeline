# =============================================================================
# detection/detector.py — YOLOE open-vocabulary detection + CLIP filtering
# =============================================================================
# Replicates the exact detection pipeline from Yang et al. (2026):
#   1. YOLOE-v8-L detects objects using the 129 CDI category names.
#   2. Each detection's bounding-box crop is compared to its label via CLIP.
#   3. Only detections with CLIP cosine similarity >= CLIP_SIM_THRESHOLD are kept.
#
# Results are written incrementally to a CSV so the run is fully resumable.
#
# Public functions
# ----------------
#   load_models(device)
#       -> (yoloe_model, clip_model, clip_preprocess, clip_tokenizer)
#
#   run_detection(frame_paths, included_categories, output_csv, ...)
#       -> None  (writes/appends rows to output_csv)

import csv
import logging
import os

import torch
from PIL import Image
from tqdm import tqdm

from detection.config import (
    YOLOE_CONF_THRESHOLD,
    CLIP_SIM_THRESHOLD,
    YOLOE_REPO,
    YOLOE_FILENAME,
    CLIP_MODEL,
    CLIP_PRETRAIN,
)

logger = logging.getLogger(__name__)

_FIELDNAMES = ["frame_path", "class_name", "confidence", "clip_similarity"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: str) -> tuple:
    """
    Download (if needed) and load YOLOE + CLIP models.

    Returns
    -------
    yoloe_model, clip_model, clip_preprocess, clip_tokenizer
    """
    # Lazy imports: ultralytics and open_clip are GPU-only / Colab dependencies
    try:
        import open_clip
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLOE
    except ImportError as exc:
        raise ImportError(
            "GPU detection dependencies missing. "
            "Run: pip install 'ultralytics>=8.3' open-clip-torch huggingface_hub"
        ) from exc

    logger.info("Loading YOLOE-v8-L weights from %s/%s ...", YOLOE_REPO, YOLOE_FILENAME)
    weights_path = hf_hub_download(repo_id=YOLOE_REPO, filename=YOLOE_FILENAME)
    yoloe_model  = YOLOE(weights_path)
    yoloe_model.eval()
    logger.info("YOLOE loaded.")

    logger.info("Loading CLIP %s (%s) ...", CLIP_MODEL, CLIP_PRETRAIN)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN
    )
    clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    logger.info("CLIP loaded.")

    return yoloe_model, clip_model, clip_preprocess, clip_tokenizer


def set_cdi_classes(yoloe_model, included_categories: list[str]) -> None:
    """Configure YOLOE for open-vocabulary detection on CDI class names."""
    logger.info("Setting %d CDI classes on YOLOE ...", len(included_categories))
    yoloe_model.set_classes(
        included_categories,
        yoloe_model.get_text_pe(included_categories),
    )


# ---------------------------------------------------------------------------
# Stale row checker
# ---------------------------------------------------------------------------

def check_stale_rows(output_csv: str, current_frame_paths: list[str]) -> int:
    """
    Compare the detection CSV against the current frame list and report
    how many rows reference frames that no longer exist on disk.

    Does NOT modify the CSV — only reports the count.
    Returns the number of stale rows found.
    """
    if not os.path.exists(output_csv):
        return 0

    import pandas as pd

    df          = pd.read_csv(output_csv)
    current_set = set(current_frame_paths)
    stale_mask  = ~df["frame_path"].isin(current_set)
    stale_count = int(stale_mask.sum())

    if stale_count > 0:
        stale_frames = df.loc[stale_mask, "frame_path"].nunique()
        logger.warning(
            "\n"
            "  ╔══════════════════════════════════════════════════════════╗\n"
            "  ║  WARNING: Stale detections found in CSV                 ║\n"
            "  ╠══════════════════════════════════════════════════════════╣\n"
            "  ║  %d detection rows reference %d frame(s) not on disk.   \n"
            "  ║                                                          ║\n"
            "  ║  This could mean:                                        ║\n"
            "  ║    a) You intentionally deleted those frames             ║\n"
            "  ║       → safe to prune (set PRUNE_STALE = True)          ║\n"
            "  ║    b) Drive has not fully synced yet                     ║\n"
            "  ║       → do NOT prune, wait and re-run                   ║\n"
            "  ║                                                          ║\n"
            "  ║  Analysis will include these stale rows until pruned.   ║\n"
            "  ╚══════════════════════════════════════════════════════════╝",
            stale_count, stale_frames,
        )
    else:
        logger.info("CSV check passed — no stale rows found.")

    return stale_count


def prune_stale_rows(output_csv: str, current_frame_paths: list[str]) -> int:
    """
    Remove rows from the detection CSV whose frame_path no longer exists
    on disk. Rewrites the CSV in place.

    Returns the number of rows removed.
    """
    if not os.path.exists(output_csv):
        return 0

    import pandas as pd

    df          = pd.read_csv(output_csv)
    current_set = set(current_frame_paths)
    df_clean    = df[df["frame_path"].isin(current_set)]
    removed     = len(df) - len(df_clean)

    if removed > 0:
        df_clean.to_csv(output_csv, index=False)
        logger.info("Pruned %d stale rows. %d rows remaining.", removed, len(df_clean))
    else:
        logger.info("No stale rows to prune.")

    return removed


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def load_already_done(output_csv: str) -> set[str]:
    """Return the set of frame paths already written to *output_csv*."""
    if not os.path.exists(output_csv):
        return set()
    import pandas as pd
    df = pd.read_csv(output_csv)
    if "frame_path" not in df.columns:
        return set()
    return set(df["frame_path"].unique())


# ---------------------------------------------------------------------------
# Detection + CLIP filtering loop
# ---------------------------------------------------------------------------

def run_detection(
    frame_paths: list[str],
    included_categories: list[str],
    output_csv: str,
    yoloe_model,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    device: str,
    batch_size: int = 32,
    yoloe_conf: float = YOLOE_CONF_THRESHOLD,
    clip_sim:   float = CLIP_SIM_THRESHOLD,
    prune_stale: bool = False,
) -> None:
    """
    Run YOLOE detection + CLIP filtering on *frame_paths* and append
    passing detections to *output_csv*.

    Already-processed frames are skipped automatically (resumable).

    Before processing, the CSV is checked for rows referencing frames that
    no longer exist on disk:
      - prune_stale=False (default): prints a warning, does not modify CSV.
      - prune_stale=True: removes stale rows from the CSV before proceeding.

    Parameters
    ----------
    frame_paths : list[str]
        Absolute paths to input JPEG/PNG frames.
    included_categories : list[str]
        Ordered list of CDI class names (index must match YOLOE cls IDs).
    output_csv : str
        Path to the output CSV file (created or appended to).
    yoloe_model, clip_model, clip_preprocess, clip_tokenizer
        Loaded model objects returned by load_models().
    device : str
        ``"cuda"`` or ``"cpu"``.
    batch_size : int
        Number of frames per YOLOE inference call.
    yoloe_conf : float
        YOLOE confidence threshold (default: from config).
    clip_sim : float
        CLIP cosine similarity threshold (default: from config).
    prune_stale : bool
        If True, remove CSV rows for frames not in frame_paths before running.
        If False (default), warn about stale rows but leave CSV untouched.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # --- Stale row check / prune ---
    if prune_stale:
        prune_stale_rows(output_csv, frame_paths)
    else:
        check_stale_rows(output_csv, frame_paths)

    already_done  = load_already_done(output_csv)
    frames_to_run = [f for f in frame_paths if f not in already_done]

    logger.info(
        "Detection: %d already done, %d remaining (total %d).",
        len(already_done), len(frames_to_run), len(frame_paths),
    )

    if not frames_to_run:
        logger.info("All frames already processed — nothing to do.")
        return

    write_header = not os.path.exists(output_csv)
    n_errors     = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i in tqdm(
            range(0, len(frames_to_run), batch_size),
            desc="YOLOE + CLIP pipeline",
            unit="batch",
        ):
            batch = frames_to_run[i : i + batch_size]
            try:
                _process_batch(
                    batch,
                    included_categories,
                    yoloe_model,
                    clip_model,
                    clip_preprocess,
                    clip_tokenizer,
                    device,
                    yoloe_conf,
                    clip_sim,
                    writer,
                )
            except Exception as exc:
                n_errors += 1
                if n_errors <= 5:
                    logger.warning("Batch %d error: %s", i, exc)

    logger.info("Detection complete. Errors: %d.", n_errors)
    if n_errors > 5:
        logger.warning("(%d additional batch errors suppressed)", n_errors - 5)


# ---------------------------------------------------------------------------
# Internal batch processor
# ---------------------------------------------------------------------------

def _process_batch(
    batch: list[str],
    included_categories: list[str],
    yoloe_model,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    device: str,
    yoloe_conf: float,
    clip_sim_thresh: float,
    writer: csv.DictWriter,
) -> None:
    """Detect objects in one batch and write passing detections to *writer*."""
    results = yoloe_model.predict(
        batch,
        conf=yoloe_conf,
        verbose=False,
        device=device,
        imgsz=640,
    )

    det_records: list[dict] = []
    crops: list[torch.Tensor] = []

    for frame_path, result in zip(batch, results):
        if result.boxes is None or not len(result.boxes):
            continue

        img = Image.open(frame_path).convert("RGB")
        w, h = img.size

        for box in result.boxes:
            cls_id     = int(box.cls[0])
            class_name = included_categories[cls_id]
            conf       = float(box.conf[0])
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

            # Clamp to image boundaries
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(w, int(xmax))
            ymax = min(h, int(ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            crop_tensor = clip_preprocess(img.crop((xmin, ymin, xmax, ymax)))
            crops.append(crop_tensor)
            det_records.append(
                {"frame_path": frame_path, "class_name": class_name, "confidence": round(conf, 4)}
            )

    if not crops:
        return

    crops_stacked = torch.stack(crops).to(device)
    labels        = [r["class_name"] for r in det_records]
    text_tokens   = clip_tokenizer(labels).to(device)

    autocast_ctx = (
        torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad()
    )
    with torch.no_grad(), autocast_ctx:
        img_feat  = clip_model.encode_image(crops_stacked)
        text_feat = clip_model.encode_text(text_tokens)
        img_feat  /= img_feat.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat * text_feat).sum(dim=-1).cpu().numpy()

    for record, sim in zip(det_records, sims):
        if sim >= clip_sim_thresh:
            writer.writerow({
                **record,
                "clip_similarity": round(float(sim), 4),
            })
