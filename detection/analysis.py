# =============================================================================
# detection/analysis.py — Long-tailed distribution analysis & power-law fit
# =============================================================================
# Replicates Notebook 01 from Yang et al. (2026).
#
# Public functions
# ----------------
#   build_category_summary(detection_csv, included_categories,
#                          lemma_to_semantic, total_frames, output_csv)
#       -> pd.DataFrame
#
#   fit_power_law(df_cat)
#       -> dict  {"alpha", "xmin", "r_vs_lognormal", "p_vs_lognormal",
#                 "nonzero_df", "counts_arr"}
#
#   print_summary(df_cat, fit, total_frames, total_detected)
#       -> None
#
#   build_provenance(detection_csv, df_cat, output_csv, top_n)
#       -> pd.DataFrame

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category-level summary
# ---------------------------------------------------------------------------

def build_category_summary(
    detection_csv: str,
    included_categories: list[str],
    lemma_to_semantic: dict[str, str],
    total_frames: int,
    output_csv: str,
) -> pd.DataFrame:
    """
    Aggregate per-frame binary presence for each CDI category.

    Mirrors the method from the BabyView paper:
    - Proportion = (frames containing ≥1 detection of category) / total_frames
    - Zero-count categories are included (proportion = 0).

    Parameters
    ----------
    detection_csv : str
        Path to the YOLOE+CLIP output CSV (from detector.py).
    included_categories : list[str]
        Ordered list of valid129 CDI categories.
    lemma_to_semantic : dict[str, str]
        Mapping of lemma -> CDI semantic domain (from cdi_loader.py).
    total_frames : int
        Total number of frames in the dataset (used as denominator).
    output_csv : str
        Where to save the per-category summary CSV.

    Returns
    -------
    pd.DataFrame with columns:
        category, count_detected, count_frames, proportion, cdi_semantic
    (sorted descending by proportion)
    """
    included_set = set(included_categories)

    df_det = pd.read_csv(detection_csv, usecols=["frame_path", "class_name"])
    df_det["class_name"] = df_det["class_name"].astype(str).str.strip().str.lower()
    df_det["frame_path"] = df_det["frame_path"].astype(str).str.strip()

    # Keep only valid129 categories
    df_det = df_det[df_det["class_name"].isin(included_set)].copy()

    # Raw detection counts (for display)
    counts_raw = df_det.groupby("class_name").size().to_dict()

    # Frame-level binary presence (deduplicated per frame × category)
    counts_frames = (
        df_det[["frame_path", "class_name"]]
        .drop_duplicates()
        .groupby("class_name")
        .size()
        .to_dict()
    )

    total_detected = int(sum(counts_raw.values()))

    df_cat = pd.DataFrame({"category": included_categories})
    df_cat["count_detected"] = df_cat["category"].map(lambda c: int(counts_raw.get(c, 0)))
    df_cat["count_frames"]   = df_cat["category"].map(lambda c: int(counts_frames.get(c, 0)))
    df_cat["proportion"]     = df_cat["count_frames"] / total_frames if total_frames else 0.0
    df_cat["cdi_semantic"]   = df_cat["category"].map(
        lambda x: lemma_to_semantic.get(x, "other")
    )
    df_cat = df_cat.sort_values("proportion", ascending=False).reset_index(drop=True)

    df_cat[["category", "proportion", "cdi_semantic"]].to_csv(output_csv, index=False)
    logger.info("Category summary saved to %s", output_csv)

    logger.info(
        "Summary: %d total detections, %d/%d categories with ≥1 detection.",
        total_detected,
        (df_cat["count_frames"] > 0).sum(),
        len(included_categories),
    )
    return df_cat


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_power_law(df_cat: pd.DataFrame) -> dict:
    """
    Fit a power-law to the frame-count distribution (categories with ≥1 hit).

    Uses the ``powerlaw`` package (Alstott et al., 2014) with discrete=True,
    matching the paper's methodology.

    Parameters
    ----------
    df_cat : pd.DataFrame
        Output of build_category_summary().

    Returns
    -------
    dict with keys:
        alpha          (float) power-law exponent
        xmin           (float) fitted x_min
        r_vs_lognormal (float) log-likelihood ratio vs lognormal
        p_vs_lognormal (float) p-value of that comparison
        nonzero_df     (pd.DataFrame) subset with count_frames > 0, desc sorted
        counts_arr     (np.ndarray) frame counts for nonzero categories
    """
    try:
        import powerlaw as pl_pkg
    except ImportError as exc:
        raise ImportError(
            "powerlaw package not found. Run: pip install powerlaw"
        ) from exc

    nonzero    = df_cat[df_cat["count_frames"] > 0].sort_values(
        "count_frames", ascending=False
    )
    counts_arr = nonzero["count_frames"].values.astype(float)

    fit          = pl_pkg.Fit(counts_arr, discrete=True, verbose=False)
    alpha_val    = fit.power_law.alpha
    xmin_val     = fit.xmin
    r_vs_ln, p_vs_ln = fit.distribution_compare("power_law", "lognormal")

    logger.info(
        "Power-law fit: α=%.3f, xmin=%.1f, R(PL vs LN)=%.3f (p=%.4f)",
        alpha_val, xmin_val, r_vs_ln, p_vs_ln,
    )
    return {
        "alpha":          alpha_val,
        "xmin":           xmin_val,
        "r_vs_lognormal": r_vs_ln,
        "p_vs_lognormal": p_vs_ln,
        "nonzero_df":     nonzero,
        "counts_arr":     counts_arr,
    }


# ---------------------------------------------------------------------------
# Summary comparison with the paper
# ---------------------------------------------------------------------------

def print_summary(
    df_cat: pd.DataFrame,
    fit: dict,
    total_frames: int,
    total_detected: int,
) -> None:
    """Print a formatted comparison table against the BabyView paper values."""
    from detection.config import (
        BABYVIEW_ALPHA,
        BABYVIEW_FRAMES,
        BABYVIEW_DETECTIONS,
        BABYVIEW_TOP5,
    )

    alpha_val = fit["alpha"]
    r_vs_ln   = fit["r_vs_lognormal"]

    sep = "=" * 62
    print(sep)
    print("  RESULTS SUMMARY — ExoBaby vs. BabyView Paper")
    print(sep)
    print(f"  {'Metric':<30} {'ExoBaby':>12}  {'BabyView':>12}")
    print(f"  {'-' * 58}")
    print(f"  {'View type':<30} {'Exocentric':>12}  {'Egocentric':>12}")
    print(f"  {'Total frames':<30} {total_frames:>12,}  {BABYVIEW_FRAMES:>12}")
    print(f"  {'Total detections':<30} {total_detected:>12,}  {BABYVIEW_DETECTIONS:>12}")
    print(f"  {'Categories detected':<30} {(df_cat['count_frames'] > 0).sum():>12}  {'129':>12}")
    print(f"  {'Power-law exponent α':<30} {alpha_val:>12.3f}  {BABYVIEW_ALPHA:>12.3f}")
    print(f"  {'PL preferred over LN?':<30} {str(r_vs_ln > 0):>12}  {'True':>12}")
    print()
    print("  Top-5 ExoBaby categories:")
    for _, row in df_cat.head(5).iterrows():
        print(
            f"    {row['category']:<18} prop={row['proportion']:.4f}  "
            f"(frames={row['count_frames']})  [{row['cdi_semantic']}]"
        )
    print()
    print("  Top-5 BabyView categories (from paper):")
    for cat, prop, sem in BABYVIEW_TOP5:
        print(f"    {cat:<18} prop={prop:.4f}  [{sem}]")
    print()
    print(sep)
    if abs(alpha_val - BABYVIEW_ALPHA) < 0.30:
        print("  ✅ α is close to the paper — long-tail shape is similar across view types!")
    elif alpha_val < BABYVIEW_ALPHA:
        print("  📊 α < paper value — distribution is LESS steep (more uniform) than egocentric.")
    else:
        print("  📊 α > paper value — distribution is MORE steep (more concentrated) than egocentric.")
    print(sep)


# ---------------------------------------------------------------------------
# Provenance analysis
# ---------------------------------------------------------------------------

def build_provenance(
    detection_csv: str,
    df_cat: pd.DataFrame,
    output_csv: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For each CDI semantic domain, identify the top-N categories by frame
    count, then break down each category's detections by source video.

    The video ID is parsed from the frame path, which follows the structure:
        .../filtered/<video_id>/<clip_name>/frame_XXXX.jpg

    Parameters
    ----------
    detection_csv : str
        Path to the YOLOE+CLIP output CSV (from detector.py).
    df_cat : pd.DataFrame
        Output of build_category_summary() — provides domain labels and
        category ranks.
    output_csv : str
        Where to save the provenance CSV.
    top_n : int
        How many top categories to include per domain (default: 5).

    Returns
    -------
    pd.DataFrame with columns:
        domain, category, video_id, detection_count, pct_of_category
    Sorted by domain → category rank → detection_count descending.
    """
    import os

    df_det = pd.read_csv(detection_csv, usecols=["frame_path", "class_name"])
    df_det["class_name"] = df_det["class_name"].astype(str).str.strip().str.lower()
    df_det["frame_path"] = df_det["frame_path"].astype(str).str.strip()

    # Parse video_id from frame path:
    # Works for both Unix (/filtered/VIDEO_ID/clip/frame.jpg)
    # and Windows (\filtered\VIDEO_ID\clip\frame.jpg) paths stored in the CSV
    def _extract_video_id(path: str) -> str:
        parts = path.replace("\\", "/").split("/")
        try:
            idx = next(
                i for i, p in enumerate(parts)
                if p.lower() == "filtered"
            )
            return parts[idx + 1]
        except (StopIteration, IndexError):
            return "unknown"

    df_det["video_id"] = df_det["frame_path"].map(_extract_video_id)

    # Build per-domain top-N category list (ranked by count_frames from df_cat)
    from detection.config import CDI_SEMANTIC_ORDER

    rows = []
    for domain in CDI_SEMANTIC_ORDER:
        domain_cats = df_cat[df_cat["cdi_semantic"] == domain].head(top_n)
        if domain_cats.empty:
            continue

        for _, cat_row in domain_cats.iterrows():
            category = cat_row["category"]

            # Detections for this category across all videos
            cat_det = df_det[df_det["class_name"] == category]
            total_cat_detections = len(cat_det)

            if total_cat_detections == 0:
                continue

            # Per-video breakdown
            per_video = (
                cat_det.groupby("video_id")
                .size()
                .reset_index(name="detection_count")
                .sort_values("detection_count", ascending=False)
            )
            per_video["pct_of_category"] = (
                per_video["detection_count"] / total_cat_detections * 100
            ).round(1)
            per_video["domain"]   = domain
            per_video["category"] = category

            rows.append(per_video)

    if not rows:
        logger.warning("Provenance: no detections found — returning empty DataFrame.")
        return pd.DataFrame(
            columns=["domain", "category", "video_id", "detection_count", "pct_of_category"]
        )

    df_prov = pd.concat(rows, ignore_index=True)[
        ["domain", "category", "video_id", "detection_count", "pct_of_category"]
    ]

    df_prov.to_csv(output_csv, index=False)
    logger.info(
        "Provenance saved to %s  (%d rows, %d unique videos)",
        output_csv,
        len(df_prov),
        df_prov["video_id"].nunique(),
    )
    return df_prov
