# =============================================================================
# merge_detections.py
# =============================================================================
# Merges per-language detection CSVs into a single union detection CSV.
#
# Mode 1 — language subfolders (original):
#   python merge_detections.py --langs English Persian --results-dir results
#   python merge_detections.py                             # all 3 languages
#
# Mode 2 — explicit file paths:
#   python merge_detections.py \
#       --files results/English/English_merged_frame_detections_with_metadata_filtered-0.27.csv \
#               results/BanglaProtha_PreDisjoint_merged_frame_detections_with_metadata_filtered-0.27.csv \
#       --out-name results/union_en_bangla_merged_frame_detections_with_metadata_filtered-0.27.csv
#
# Output (Mode 1):
#   <results-dir>/union_<langs>_merged_frame_detections_with_metadata_filtered-0.27.csv
# Output (Mode 2):
#   path given by --out-name
# =============================================================================

import os
import glob
import argparse
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DETECTION_PATTERN  = "*merged_frame_detections_with_metadata_filtered-0.27.csv"
DETECTION_FILENAME = "merged_frame_detections_with_metadata_filtered-0.27.csv"
ALL_LANGUAGES      = ["English", "Persian", "Turkish"]


def find_detection_csv(lang: str, results_dir: str) -> str:
    """Find the detection CSV inside results_dir/<lang>/ using a glob pattern."""
    folder = os.path.join(results_dir, lang)
    matches = glob.glob(os.path.join(folder, DETECTION_PATTERN))
    if not matches:
        raise FileNotFoundError(
            f"No detection CSV found for '{lang}' in: {folder}\n"
            f"Expected a file matching: {DETECTION_PATTERN}\n"
            f"Files present: {os.listdir(folder) if os.path.isdir(folder) else 'folder missing'}"
        )
    if len(matches) > 1:
        print(f"  [WARN] Multiple CSVs found for '{lang}', using: {matches[0]}")
    return matches[0]


def _dedup_and_save(dfs: list, labels: list[str], out_path: str) -> str:
    """
    Shared dedup + save logic used by both merge modes.
    dfs    : list of DataFrames to merge
    labels : human-readable label for each df (for printing)
    out_path: where to write the output CSV
    """
    for df, label in zip(dfs, labels):
        n_det = int(df["class_name"].notna().sum())
        print(f"  {label:<45}: {len(df):>6,} rows  ({n_det:>6,} with detections)")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal before dedup : {len(df_all):,}")

    # Split detection rows vs empty-frame markers
    has_class = df_all["class_name"].notna() & (
        df_all["class_name"].astype(str).str.strip() != ""
    )
    df_det   = df_all[has_class].copy()
    df_nodet = df_all[~has_class].copy()

    # Normalise class names to lowercase
    df_det["class_name"] = df_det["class_name"].astype(str).str.strip().str.lower()

    # Dedup: same frame + same class → keep highest clip_similarity
    df_det = (
        df_det
        .sort_values("clip_similarity", ascending=False)
        .drop_duplicates(subset=["frame_path", "class_name"])
        .reset_index(drop=True)
    )

    # One empty-frame marker per frame
    df_nodet = (
        df_nodet
        .drop_duplicates(subset=["frame_path"])
        .reset_index(drop=True)
    )

    df_union = pd.concat([df_det, df_nodet], ignore_index=True)
    df_union.to_csv(out_path, index=False)

    print(f"\nAfter dedup")
    print(f"  Total rows       : {len(df_union):,}")
    print(f"  Detection rows   : {len(df_det):,}")
    print(f"  Empty-frame rows : {len(df_nodet):,}")
    print(f"  Unique frames    : {df_union['frame_path'].nunique():,}")
    print(f"  Unique classes   : {df_det['class_name'].nunique():,}")
    print(f"\nSaved -> {out_path}")
    return out_path


def merge_detections(langs: list[str], results_dir: str) -> str:
    """Mode 1: merge by language subfolder names."""
    print(f"Languages to merge : {langs}")
    print(f"Results directory  : {results_dir}\n")

    dfs, labels = [], []
    for lang in langs:
        path = find_detection_csv(lang, results_dir)
        dfs.append(pd.read_csv(path))
        labels.append(lang)

    lang_tag = "_".join(l[:2].lower() for l in langs)
    out_path = os.path.join(results_dir, f"union_{lang_tag}_{DETECTION_FILENAME}")
    return _dedup_and_save(dfs, labels, out_path)


def merge_files(file_paths: list[str], out_path: str) -> str:
    """Mode 2: merge arbitrary explicit file paths."""
    print(f"Files to merge:")
    dfs, labels = [], []
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        dfs.append(pd.read_csv(path))
        labels.append(os.path.basename(path))
    print()
    return _dedup_and_save(dfs, labels, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge YOLOE detection CSVs into a union CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Mode 1 — language subfolders:\n"
            "  python merge_detections.py --langs English Persian --results-dir results\n\n"
            "Mode 2 — explicit file paths:\n"
            "  python merge_detections.py \\\n"
            "      --files results/English/English_merged_frame_detections_with_metadata_filtered-0.27.csv \\\n"
            "              results/BanglaProtha_PreDisjoint_merged_frame_detections_with_metadata_filtered-0.27.csv \\\n"
            "      --out-name results/union_en_bangla_merged_frame_detections_with_metadata_filtered-0.27.csv"
        ),
    )

    # Mode 1 args
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=ALL_LANGUAGES,
        metavar="LANG",
        help="Language subfolder names to merge (Mode 1).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root folder containing per-language subfolders (default: results/).",
    )

    # Mode 2 args
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="PATH",
        help="Explicit CSV file paths to merge (Mode 2). Use with --out-name.",
    )
    parser.add_argument(
        "--out-name",
        metavar="PATH",
        help="Output file path when using --files (Mode 2).",
    )

    args = parser.parse_args()

    if args.files:
        # Mode 2: explicit paths
        if not args.out_name:
            parser.error("--out-name is required when using --files")
        merge_files(file_paths=args.files, out_path=args.out_name)
    else:
        # Mode 1: language subfolders
        langs = args.langs or ALL_LANGUAGES
        merge_detections(langs=langs, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
