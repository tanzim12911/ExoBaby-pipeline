# =============================================================================
# detection/visualizer.py — Figures for the long-tailed distribution analysis
# =============================================================================
# All figures match the style of Yang et al. (2026) Figure 2.
#
# Public functions
# ----------------
#   plot_rank_bar(df_cat, fit, total_frames, total_detected, figures_dir)
#       -> str  (saved PNG path)
#
#   plot_loglog(nonzero_df, counts_arr, alpha, figures_dir)
#       -> str
#
#   plot_per_domain(df_cat, figures_dir)
#       -> str
#
#   plot_provenance(df_prov, top_n, figures_dir)
#       -> str

import logging
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from detection.config import (
    CDI_SEMANTIC_COLORS,
    CDI_SEMANTIC_ORDER,
)

logger = logging.getLogger(__name__)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})

_CATEGORY_SET = "valid129"


# ---------------------------------------------------------------------------
# Shared style helper
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", labelsize=12, width=1.2)


# ---------------------------------------------------------------------------
# Figure 1-A: Rank-ordered bar chart
# ---------------------------------------------------------------------------

def plot_rank_bar(
    df_cat,
    fit: dict,
    total_frames: int,
    total_detected: int,
    figures_dir: str,
) -> str:
    """
    Rank-ordered bar chart coloured by CDI semantic domain.

    Matches Figure 2 in the BabyView paper.
    Saves both PNG (150 dpi) and PDF.

    Returns
    -------
    str — path to the saved PNG.
    """
    alpha_val   = fit["alpha"]
    bar_colors  = [CDI_SEMANTIC_COLORS.get(s, "#8B9A9E") for s in df_cat["cdi_semantic"]]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(range(len(df_cat)), df_cat["proportion"],
           color=bar_colors, width=0.85, linewidth=0)
    _style_ax(ax)

    ax.set_xlabel("Object categories (rank-ordered by frequency)", fontsize=13)
    ax.set_ylabel("Proportion of frames with object detection", fontsize=13)
    ax.set_title(
        f"ExoBaby (Exocentric Frames) — Long-Tailed Object Distribution\n"
        f"{total_frames:,} total frames  ·  {total_detected:,} detections  ·  "
        f"Power-law α = {alpha_val:.2f}",
        fontsize=13, pad=12,
    )
    ax.set_xlim(-1, len(df_cat))

    present_domains = df_cat["cdi_semantic"].unique()
    patches = [
        mpatches.Patch(
            color=CDI_SEMANTIC_COLORS[s],
            label=s.replace("_", " ").title(),
        )
        for s in CDI_SEMANTIC_ORDER if s in present_domains
    ]
    ax.legend(
        handles=patches, loc="upper right", fontsize=10,
        framealpha=0.9, title="CDI Semantic Domain", title_fontsize=10,
    )

    fig.tight_layout()
    out_path = _save_figure(fig, figures_dir, f"fig1a_long_tailed_bar_{_CATEGORY_SET}.png")
    return out_path


# ---------------------------------------------------------------------------
# Figure 1-B: Log-log rank vs frequency
# ---------------------------------------------------------------------------

def plot_loglog(
    nonzero_df,
    counts_arr: np.ndarray,
    alpha: float,
    figures_dir: str,
) -> str:
    """
    Log-log scatter of rank vs frame count with power-law fit line.

    Returns
    -------
    str — path to the saved PNG.
    """
    ranks     = np.arange(1, len(nonzero_df) + 1)
    pt_colors = [CDI_SEMANTIC_COLORS.get(s, "#8B9A9E") for s in nonzero_df["cdi_semantic"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ranks, counts_arr, c=pt_colors, s=65, alpha=0.85, linewidths=0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_ax(ax)

    ax.set_xlabel("Rank (log scale)", fontsize=13)
    ax.set_ylabel("Frame count (log scale)", fontsize=13)
    ax.set_title(f"Rank–Frequency (log–log)\nPower-law fit: α = {alpha:.2f}", fontsize=13)

    x_line = np.linspace(1, len(nonzero_df), 400)
    y_line = counts_arr[0] * (x_line ** -(alpha - 1))
    ax.plot(x_line, y_line, "k--", lw=2, label=f"Power law  α = {alpha:.2f}", zorder=4)
    ax.legend(fontsize=11, framealpha=0.85)

    fig.tight_layout()
    out_path = _save_figure(fig, figures_dir, f"fig1b_loglog_{_CATEGORY_SET}.png")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Per-domain bar charts
# ---------------------------------------------------------------------------

def plot_per_domain(df_cat, figures_dir: str) -> str:
    """
    One bar chart per CDI semantic domain showing within-domain proportions.

    Returns
    -------
    str — path to the saved PNG.
    """
    present = [d for d in CDI_SEMANTIC_ORDER if d in df_cat["cdi_semantic"].values]
    ncols   = 4
    nrows   = -(-len(present) // ncols)   # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    for ax, domain in zip(axes_flat, present):
        sub   = df_cat[df_cat["cdi_semantic"] == domain].sort_values("proportion", ascending=False)
        color = CDI_SEMANTIC_COLORS.get(domain, "#8B9A9E")
        ax.bar(range(len(sub)), sub["proportion"], color=color, width=0.85, linewidth=0)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["category"].tolist(), rotation=60, ha="right", fontsize=8)
        ax.set_title(
            domain.replace("_", " ").title(),
            fontsize=11, color=color, fontweight="bold",
        )
        _style_ax(ax)

    for ax in axes_flat[len(present):]:
        ax.set_visible(False)

    fig.suptitle("Per-Domain Object Distribution (ExoBaby Exocentric)", fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = _save_figure(fig, figures_dir, f"fig2_per_domain_{_CATEGORY_SET}.png")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Provenance — top-N categories per domain × video breakdown
# ---------------------------------------------------------------------------

def plot_provenance(
    df_prov,
    top_n: int,
    figures_dir: str,
) -> str:
    """
    For each CDI semantic domain, draw a stacked horizontal bar chart showing
    the top-N categories with each bar segment representing one source video.

    Highlights dataset bias: if one video dominates a category, that
    category's frequency reflects that video's setting, not a general pattern.

    Parameters
    ----------
    df_prov : pd.DataFrame
        Output of build_provenance() — columns: domain, category,
        video_id, detection_count, pct_of_category.
    top_n : int
        Number of categories shown per domain (used in the figure title).
    figures_dir : str
        Directory where figures are saved.

    Returns
    -------
    str — path to the saved PNG, or empty string if no data.
    """
    present_domains = [d for d in CDI_SEMANTIC_ORDER if d in df_prov["domain"].values]
    if not present_domains:
        logger.warning("plot_provenance: no domains in provenance data, skipping.")
        return ""

    # Consistent color per video ID across all subplots
    all_videos   = sorted(df_prov["video_id"].unique())
    palette      = plt.cm.get_cmap("tab20", max(len(all_videos), 1))
    video_colors = {vid: palette(i) for i, vid in enumerate(all_videos)}

    ncols     = 3
    nrows     = -(-len(present_domains) // ncols)
    row_height = max(0.5 * top_n, 2.5)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, row_height * nrows),
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, domain in zip(axes_flat, present_domains):
        domain_data = df_prov[df_prov["domain"] == domain]

        # Categories ordered by total detections, largest at top
        cat_totals = (
            domain_data.groupby("category")["detection_count"]
            .sum()
            .sort_values(ascending=True)
        )
        categories  = cat_totals.index.tolist()
        y_positions = np.arange(len(categories))

        for y, category in zip(y_positions, categories):
            cat_data = (
                domain_data[domain_data["category"] == category]
                .sort_values("detection_count", ascending=False)
            )
            left = 0.0
            for _, vrow in cat_data.iterrows():
                vid   = vrow["video_id"]
                count = vrow["detection_count"]
                ax.barh(y, count, left=left,
                        color=video_colors[vid], height=0.6, linewidth=0)
                # Label segment if wide enough to read (>=8% of bar)
                if vrow["pct_of_category"] >= 8:
                    ax.text(
                        left + count / 2, y,
                        f"{vrow['pct_of_category']:.0f}%",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )
                left += count

        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlabel("Detection count", fontsize=10)
        ax.set_title(
            domain.replace("_", " ").title(),
            fontsize=11,
            color=CDI_SEMANTIC_COLORS.get(domain, "#8B9A9E"),
            fontweight="bold",
        )
        _style_ax(ax)

    for ax in axes_flat[len(present_domains):]:
        ax.set_visible(False)

    # Global legend — one patch per video, below the figure
    legend_patches = [
        mpatches.Patch(color=video_colors[vid], label=vid)
        for vid in all_videos
    ]
    fig.legend(
        handles=legend_patches,
        title="Source video ID",
        loc="lower center",
        ncol=min(len(all_videos), 6),
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9,
    )

    fig.suptitle(
        f"Provenance: Top-{top_n} Categories per Domain — Source Video Breakdown",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out_path = _save_figure(
        fig, figures_dir, f"fig3_provenance_top{top_n}_{_CATEGORY_SET}.png"
    )
    return out_path


# ---------------------------------------------------------------------------
# Internal save helper
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, figures_dir: str, filename: str) -> str:
    """Save *fig* as PNG + PDF, return the PNG path."""
    os.makedirs(figures_dir, exist_ok=True)
    png_path = os.path.join(figures_dir, filename)
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", png_path)
    return png_path
