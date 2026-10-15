# =============================================================================
# merge_cdi.py
# =============================================================================
# Merges per-language CDI stage4_concrete_objects.csv files into a union CSV.
#
# Usage:
#   python merge_cdi.py                          # merge all 3 languages
#   python merge_cdi.py --langs English Persian  # merge only EN + FA
#   python merge_cdi.py --langs Turkish          # single language
#   python merge_cdi.py --cdi-dir /path/to/CDI  # custom CDI root
#
# Expected input structure:
#   <cdi-dir>/
#     English/   stage4_concrete_objects.csv
#     Persian/   stage4_concrete_objects.csv
#     Turkish/   stage4_concrete_objects.csv
#
# Output (saved in <cdi-dir>/):
#   union_<langs>_stage4_concrete_objects.csv
#
# Output columns:
#   concept        — normalised lowercase English label (deduplicated key)
#   category       — CDI semantic domain (food_drink, toys, animals, ...)
#   lang_source    — which language first introduced this concept
#   item_definition— native-language word(s) (semicolon-joined if conflict)
#   AoA            — age of acquisition (first-seen value kept)
# =============================================================================

import os
import glob
import argparse
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CDI_FILENAME  = "stage4_concrete_objects.csv"
ALL_LANGUAGES = ["English", "Persian", "Turkish"]

# Which column carries the English concept label in each CDI
LABEL_COL = {
    "English": "uni_lemma_norm",
    "Persian": "english_label",
    "Turkish": "english_label",
}


def find_cdi_csv(lang: str, cdi_dir: str) -> str:
    """Find stage4_concrete_objects.csv inside cdi_dir/<lang>/."""
    folder  = os.path.join(cdi_dir, lang)
    pattern = os.path.join(folder, CDI_FILENAME)
    if os.path.exists(pattern):
        return pattern

    # Fallback: glob for any CSV whose name ends with the filename
    matches = glob.glob(os.path.join(folder, f"*{CDI_FILENAME}"))
    if not matches:
        files = os.listdir(folder) if os.path.isdir(folder) else "folder missing"
        raise FileNotFoundError(
            f"CDI CSV not found for '{lang}' in: {folder}\n"
            f"Expected: {CDI_FILENAME}\n"
            f"Files present: {files}"
        )
    return matches[0]


def normalise_concept(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Add a 'concept' column with the lowercase English label."""
    col = LABEL_COL[lang]
    if col not in df.columns:
        # Graceful fallback
        fallbacks = ["english_label", "uni_lemma_norm", "uni_lemma", "english_gloss"]
        col = next((c for c in fallbacks if c in df.columns), None)
        if col is None:
            raise ValueError(
                f"No usable English-label column found in {lang} CDI. "
                f"Columns present: {list(df.columns)}"
            )
        print(f"  [{lang}] Column '{LABEL_COL[lang]}' not found, using '{col}' instead.")
    df = df.copy()
    df["concept"]     = df[col].astype(str).str.strip().str.lower()
    df["lang_source"] = lang
    return df


def merge_cdi(langs: list[str], cdi_dir: str) -> str:
    """
    Merge CDI files for the given languages and save the union CSV.

    Deduplication key: 'concept' (normalised English label).
    Priority order follows the order of `langs` — first occurrence wins.

    Parameters
    ----------
    langs    : list of language names matching CDI subfolder names
    cdi_dir  : root folder containing per-language CDI subfolders

    Returns
    -------
    Path to the saved union CSV.
    """
    print(f"Languages to merge : {langs}")
    print(f"CDI directory      : {cdi_dir}\n")

    # ── 1. Load and normalise each CDI ────────────────────────────────────────
    frames = []
    for lang in langs:
        path = find_cdi_csv(lang, cdi_dir)
        df   = pd.read_csv(path)
        df   = normalise_concept(df, lang)
        frames.append(df)
        print(f"  {lang:<10}: {len(df):>3} entries  (from {os.path.basename(path)})")

    # ── 2. Concatenate ────────────────────────────────────────────────────────
    df_all = pd.concat(frames, ignore_index=True)

    # Drop rows where concept is missing or literally 'nan'
    df_all = df_all[df_all["concept"].notna() & (df_all["concept"] != "nan")]

    print(f"\nTotal before dedup : {len(df_all)}")

    # ── 3. Deduplicate on concept ─────────────────────────────────────────────
    # Keep the first occurrence (priority = order of `langs`).
    # This preserves the category, AoA, and item_definition of the
    # language that introduced each concept.
    df_union = df_all.drop_duplicates(subset=["concept"]).reset_index(drop=True)

    print(f"After dedup        : {len(df_union)} unique concepts")

    # ── 4. Build a clean output table ─────────────────────────────────────────
    # Always output these core columns; add AoA and item_definition if present.
    core_cols = ["concept", "category", "lang_source"]

    optional = ["AoA", "item_definition"]
    extra    = [c for c in optional if c in df_union.columns]

    out_df = df_union[core_cols + extra].copy()

    # ── 5. Per-language breakdown ─────────────────────────────────────────────
    print()
    for lang in langs:
        n = (df_union["lang_source"] == lang).sum()
        print(f"  {lang:<10}: {n:>3} concepts in union")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    lang_tag = "_".join(l[:2].lower() for l in langs)   # e.g. "en_fa_tr"
    out_name = f"union_{lang_tag}_stage4_concrete_objects.csv"
    out_path = os.path.join(cdi_dir, out_name)

    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge per-language CDI stage4_concrete_objects CSVs into a union CSV."
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=ALL_LANGUAGES,
        choices=ALL_LANGUAGES,
        metavar="LANG",
        help=(
            "Languages to include. Choices: English Persian Turkish. "
            "Default: all three. Example: --langs English Persian"
        ),
    )
    parser.add_argument(
        "--cdi-dir",
        default="CDI",
        help="Root folder containing per-language CDI subfolders (default: CDI/)",
    )
    args = parser.parse_args()

    merge_cdi(langs=args.langs, cdi_dir=args.cdi_dir)


if __name__ == "__main__":
    main()
