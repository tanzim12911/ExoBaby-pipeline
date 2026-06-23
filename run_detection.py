# =============================================================================
# run_detection.py — Orchestrator for the YOLOE+CLIP detection pipeline
# =============================================================================
# Usage (local, with GPU):
#   python run_detection.py --frames data/filtered --output ExoBaby-results --tag v1_42videos
#
# Usage (single steps):
#   python run_detection.py --frames data/filtered --output ExoBaby-results --tag v1_42videos --step detect
#   python run_detection.py --frames data/filtered --output ExoBaby-results --tag v1_42videos --step analyse
#
# Output layout
# -------------
#   ExoBaby-results/
#   ├── data/                        ← CDI reference files (shared)
#   ├── frame_data/                  ← detection CSV (shared, resumable)
#   └── analysis/
#       ├── v1_42videos/             ← figures + summary CSV (versioned per tag)
#       └── v2_60videos/
#
# The detection CSV is shared across all runs — new frames are appended,
# already-processed frames are skipped. Only the analysis outputs are versioned.

import argparse
import logging
import os
import sys

import torch

import detection.config as cfg
from detection.analysis   import build_category_summary, fit_power_law, print_summary
from detection.cdi_loader import load_cdi_data
from detection.detector   import load_models, run_detection, set_cdi_classes, check_stale_rows
from detection.visualizer import plot_loglog, plot_per_domain, plot_rank_bar

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

def resolve_shared_dirs(output_base: str) -> dict[str, str]:
    """
    Shared directories — same location for every run.
    The detection CSV lives here so resume always works.
    """
    dirs = {
        "data":       os.path.join(output_base, cfg.DATA_SUBDIR),
        "frame_data": os.path.join(output_base, cfg.FRAME_DATA_SUBDIR),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def resolve_analysis_dirs(output_base: str, run_tag: str) -> dict[str, str]:
    """
    Versioned directories — one subfolder per run tag.
    Each run tag gets its own results/ and figures/ so nothing is overwritten.
    """
    analysis_root = os.path.join(output_base, cfg.ANALYSIS_SUBDIR, run_tag)
    dirs = {
        "results": os.path.join(analysis_root, "results"),
        "figures": os.path.join(analysis_root, "figures"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    logger.info("Analysis output: %s", analysis_root)
    return dirs


def collect_frames(frames_dir: str) -> list[str]:
    """Walk *frames_dir* recursively and return all JPEG/PNG paths."""
    paths = []
    for root, _, files in os.walk(frames_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    return paths


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_detect(frames_dir: str, shared_dirs: dict[str, str], prune_stale: bool = False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning(
            "No GPU detected. Detection will be very slow. "
            "Consider running on Colab (T4 GPU)."
        )

    batch_size    = 32 if device == "cuda" else 4
    detection_csv = os.path.join(shared_dirs["frame_data"], cfg.DETECTION_CSV_NAME)

    frame_paths = collect_frames(frames_dir)
    logger.info("Found %d frames in %s", len(frame_paths), frames_dir)
    if not frame_paths:
        logger.error("No frames found. Check --frames path.")
        sys.exit(1)

    included_categories, _ = load_cdi_data(shared_dirs["data"])

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
        prune_stale=prune_stale,
    )
    logger.info("Detection CSV: %s", detection_csv)


def run_analyse(
    frames_dir: str,
    shared_dirs: dict[str, str],
    analysis_dirs: dict[str, str],
) -> None:
    import pandas as pd

    detection_csv    = os.path.join(shared_dirs["frame_data"], cfg.DETECTION_CSV_NAME)
    intermediate_csv = os.path.join(analysis_dirs["results"], cfg.INTERMEDIATE_CSV_NAME)

    if not os.path.exists(detection_csv):
        logger.error(
            "Detection CSV not found: %s\nRun --step detect first.", detection_csv
        )
        sys.exit(1)

    frame_paths  = collect_frames(frames_dir)
    total_frames = len(frame_paths)
    if total_frames == 0:
        logger.error("No frames found. Check --frames path.")
        sys.exit(1)

    included_categories, lemma_to_semantic = load_cdi_data(shared_dirs["data"])

    df_cat = build_category_summary(
        detection_csv=detection_csv,
        included_categories=included_categories,
        lemma_to_semantic=lemma_to_semantic,
        total_frames=total_frames,
        output_csv=intermediate_csv,
    )

    fit            = fit_power_law(df_cat)
    total_detected = len(pd.read_csv(detection_csv))

    plot_rank_bar(df_cat, fit, total_frames, total_detected, analysis_dirs["figures"])
    plot_loglog(fit["nonzero_df"], fit["counts_arr"], fit["alpha"], analysis_dirs["figures"])
    plot_per_domain(df_cat, analysis_dirs["figures"])

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
        help="Path to filtered frames directory (e.g. data/filtered).",
    )
    parser.add_argument(
        "--output",
        default="ExoBaby-results",
        help="Root output directory (default: ExoBaby-results).",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help=(
            "Run tag — describes what is different about this run. "
            "Used as the analysis subfolder name. "
            "Examples: v1_42videos, v2_60videos, v3_threshold-0.25"
        ),
    )
    parser.add_argument(
        "--step",
        choices=["detect", "analyse", "all"],
        default="all",
        help="Which step to run (default: all).",
    )
    parser.add_argument(
        "--prune-stale",
        action="store_true",
        default=False,
        help=(
            "Remove rows from the detection CSV whose frame_path no longer "
            "exists on disk before running. Use only when you have intentionally "
            "deleted frames. Default: False (warn but do not modify CSV)."
        ),
    )
    args = parser.parse_args()

    shared_dirs   = resolve_shared_dirs(args.output)
    analysis_dirs = resolve_analysis_dirs(args.output, args.tag)

    if args.step in ("detect", "all"):
        run_detect(args.frames, shared_dirs, prune_stale=args.prune_stale)

    if args.step in ("analyse", "all"):
        run_analyse(args.frames, shared_dirs, analysis_dirs)


if __name__ == "__main__":
    main()
