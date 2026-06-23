# =============================================================================
# run_detection.py — Orchestrator for the YOLOE+CLIP detection pipeline
# =============================================================================
# Usage (local, with GPU):
#   python run_detection.py --frames data/filtered --output ExoBaby-results
#
# Usage (single steps):
#   python run_detection.py --frames data/filtered --output ExoBaby-results --step detect
#   python run_detection.py --frames data/filtered --output ExoBaby-results --step analyse
#
# The detection step is resumable: already-processed frames are skipped.
# All outputs (CSVs, figures) land under --output.

import argparse
import logging
import os
import sys

import torch

import detection.config as cfg
from detection.analysis    import build_category_summary, fit_power_law, print_summary
from detection.cdi_loader  import load_cdi_data
from detection.detector    import load_models, run_detection, set_cdi_classes
from detection.visualizer  import plot_loglog, plot_per_domain, plot_rank_bar

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("run_detection")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_output_dirs(output_base: str) -> dict[str, str]:
    dirs = {
        "data":       os.path.join(output_base, cfg.DATA_SUBDIR),
        "frame_data": os.path.join(output_base, cfg.FRAME_DATA_SUBDIR),
        "results":    os.path.join(output_base, cfg.RESULTS_SUBDIR),
        "figures":    os.path.join(output_base, cfg.FIGURES_SUBDIR),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def collect_frames(frames_dir: str) -> list[str]:
    """Walk *frames_dir* and return all JPEG/PNG paths."""
    paths = []
    for root, _, files in os.walk(frames_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    return paths


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_detect(frames_dir: str, dirs: dict[str, str]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning(
            "No GPU detected. Detection on a large dataset will be very slow. "
            "Consider running on a machine with a CUDA-capable GPU or on Colab (T4)."
        )

    batch_size = 32 if device == "cuda" else 4
    detection_csv = os.path.join(dirs["frame_data"], cfg.DETECTION_CSV_NAME)

    frame_paths = collect_frames(frames_dir)
    logger.info("Found %d frames in %s", len(frame_paths), frames_dir)

    if not frame_paths:
        logger.error("No frames found. Check --frames path.")
        sys.exit(1)

    included_categories, _ = load_cdi_data(dirs["data"])

    yoloe_model, clip_model, clip_preprocess, clip_tokenizer = load_models(device)
    set_cdi_classes(yoloe_model, included_categories)

    run_detection(
        frame_paths=frame_paths,
        included_categories=included_categories,
        output_csv=detection_csv,
        yoloe_model=yoloe_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        device=device,
        batch_size=batch_size,
    )
    logger.info("Detection CSV: %s", detection_csv)


def run_analyse(frames_dir: str, dirs: dict[str, str]) -> None:
    import pandas as pd

    detection_csv    = os.path.join(dirs["frame_data"], cfg.DETECTION_CSV_NAME)
    intermediate_csv = os.path.join(dirs["results"],    cfg.INTERMEDIATE_CSV_NAME)

    if not os.path.exists(detection_csv):
        logger.error(
            "Detection CSV not found: %s\nRun --step detect first.", detection_csv
        )
        sys.exit(1)

    frame_paths = collect_frames(frames_dir)
    total_frames = len(frame_paths)
    if total_frames == 0:
        logger.error("No frames found. Check --frames path.")
        sys.exit(1)

    included_categories, lemma_to_semantic = load_cdi_data(dirs["data"])

    # Build category summary
    df_cat = build_category_summary(
        detection_csv=detection_csv,
        included_categories=included_categories,
        lemma_to_semantic=lemma_to_semantic,
        total_frames=total_frames,
        output_csv=intermediate_csv,
    )

    # Power-law fit
    fit = fit_power_law(df_cat)

    # Total detections (for titles / summary)
    df_raw = pd.read_csv(detection_csv)
    total_detected = len(df_raw)

    # Figures
    plot_rank_bar(df_cat, fit, total_frames, total_detected, dirs["figures"])
    plot_loglog(fit["nonzero_df"], fit["counts_arr"], fit["alpha"], dirs["figures"])
    plot_per_domain(df_cat, dirs["figures"])

    # Console summary
    print_summary(df_cat, fit, total_frames, total_detected)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExoBaby YOLOE+CLIP detection pipeline"
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Path to the directory containing filtered frames (data/filtered).",
    )
    parser.add_argument(
        "--output",
        default="ExoBaby-results",
        help="Root directory for all outputs (default: ExoBaby-results).",
    )
    parser.add_argument(
        "--step",
        choices=["detect", "analyse", "all"],
        default="all",
        help="Which step to run (default: all).",
    )
    args = parser.parse_args()

    dirs = resolve_output_dirs(args.output)

    if args.step in ("detect", "all"):
        run_detect(args.frames, dirs)

    if args.step in ("analyse", "all"):
        run_analyse(args.frames, dirs)


if __name__ == "__main__":
    main()
