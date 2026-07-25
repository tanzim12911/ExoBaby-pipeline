# =============================================================================
# merge_shards.py — Standalone utility for merging sharded detection CSVs
# =============================================================================
# Merges multiple frame-range sharded detection CSVs (e.g., shard_0_3000.csv,
# shard_3000_6000.csv) into a single unified detection CSV.
#
# Usage (directory + glob pattern):
#   python merge_shards.py --input-dir ExoBaby-results/frame_data \
#                          --pattern "detections_*_shard_*.csv" \
#                          --out ExoBaby-results/frame_data/detections_UNIFIED.csv
#
# Usage (explicit files):
#   python merge_shards.py --files path/to/shard_0_3000.csv path/to/shard_3000_6000.csv \
#                          --out path/to/unified.csv
# =============================================================================

import argparse
import glob
import os
import sys
import pandas as pd


def merge_shards(
    file_paths: list[str],
    output_path: str,
    verbose: bool = True
) -> str:
    """
    Merges a list of sharded detection CSV files into output_path,
    deduplicating identical (frame_path, class_name) entries.
    """
    if not file_paths:
        raise ValueError("No input CSV files provided to merge.")

    dfs = []
    labels = []

    for path in file_paths:
        if not os.path.exists(path):
            if verbose:
                print(f"  [WARN] File not found, skipping: {path}")
            continue
        try:
            df = pd.read_csv(path)
            if "frame_path" not in df.columns or "class_name" not in df.columns:
                if verbose:
                    print(f"  [WARN] CSV missing required columns, skipping: {path}")
                continue
            dfs.append(df)
            labels.append(os.path.basename(path))
            if verbose:
                print(f"  Loaded {os.path.basename(path):<45}: {len(df):>6,} rows ({df['frame_path'].nunique():>5,} frames)")
        except Exception as exc:
            if verbose:
                print(f"  [ERROR] Could not read {path}: {exc}")

    if not dfs:
        raise RuntimeError("No valid dataframes loaded from specified shard files.")

    df_concat = pd.concat(dfs, ignore_index=True)

    # Separate detection rows vs empty frame markers
    has_class = df_concat["class_name"].notna() & (
        df_concat["class_name"].astype(str).str.strip() != ""
    )
    df_det = df_concat[has_class].copy()
    df_nodet = df_concat[~has_class].copy()

    # Normalise class names
    df_det["class_name"] = df_det["class_name"].astype(str).str.strip().str.lower()

    # Deduplicate detections per frame + class
    df_det = df_det.drop_duplicates(subset=["frame_path", "class_name"]).reset_index(drop=True)

    # Deduplicate empty frame markers per frame
    df_nodet = df_nodet.drop_duplicates(subset=["frame_path"]).reset_index(drop=True)

    # Keep empty frame markers only for frames that have no detections
    detected_frames = set(df_det["frame_path"].unique())
    df_nodet = df_nodet[~df_nodet["frame_path"].isin(detected_frames)].reset_index(drop=True)

    df_unified = pd.concat([df_det, df_nodet], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df_unified.to_csv(output_path, index=False)

    if verbose:
        print("\n" + "=" * 60)
        print("MERGE COMPLETE:")
        print(f"  Total input rows       : {len(df_concat):,}")
        print(f"  Unified rows           : {len(df_unified):,}")
        print(f"  Detection rows         : {len(df_det):,}")
        print(f"  Empty-frame marker rows: {len(df_nodet):,}")
        print(f"  Total unique frames    : {df_unified['frame_path'].nunique():,}")
        print(f"  Total unique classes   : {df_det['class_name'].nunique():,}")
        print(f"  Saved -> {output_path}")
        print("=" * 60 + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge sharded detection CSV files.")
    parser.add_argument("--input-dir", type=str, help="Directory containing shard CSV files")
    parser.add_argument("--pattern", type=str, default="detections_*_shard_*.csv", help="Glob pattern for finding shard CSVs")
    parser.add_argument("--files", nargs="+", type=str, help="Explicit list of shard CSV file paths")
    parser.add_argument("--out", type=str, required=True, help="Output path for the merged CSV file")

    args = parser.parse_args()

    files_to_merge = []
    if args.files:
        files_to_merge.extend(args.files)

    if args.input_dir:
        pattern_path = os.path.join(args.input_dir, args.pattern)
        found = glob.glob(pattern_path)
        files_to_merge.extend(found)

    # Deduplicate file list
    files_to_merge = sorted(list(set(files_to_merge)))

    if not files_to_merge:
        print("Error: No input shard files found. Check --input-dir / --pattern or --files arguments.")
        sys.exit(1)

    print(f"Merging {len(files_to_merge)} shard files...")
    merge_shards(files_to_merge, args.out, verbose=True)


if __name__ == "__main__":
    main()
