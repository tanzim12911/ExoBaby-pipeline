# =============================================================================
# detection/cdi_loader.py — Download and parse BabyView CDI reference data
# =============================================================================
# Downloads two files from the BabyView public repository (once, then cached):
#   • cdi_words.csv          — maps CDI noun lemmas to semantic domains
#   • included_categories_valid129.txt — the 129 validated CDI categories
#
# Public functions
# ----------------
#   load_cdi_data(data_dir)
#       -> (included_categories: list[str], lemma_to_semantic: dict[str, str])

import os
import logging
import requests
import pandas as pd

from detection.config import CDI_WORDS_URL, VALID129_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest_path: str, force: bool = False) -> None:
    """Download *url* to *dest_path*, skipping if already cached."""
    if os.path.exists(dest_path) and not force:
        logger.info("Already cached: %s", os.path.basename(dest_path))
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info("Downloading %s ...", os.path.basename(dest_path))
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest_path, "wb") as fh:
        fh.write(r.content)
    logger.info("Saved: %s", dest_path)


def _load_valid129(txt_path: str) -> list[str]:
    """Return the list of valid129 category names (lowercased, stripped)."""
    with open(txt_path, "r", encoding="utf-8") as fh:
        return [line.strip().lower() for line in fh if line.strip()]


def _build_semantic_map(csv_path: str) -> dict[str, str]:
    """Return a dict mapping uni_lemma -> CDI semantic domain."""
    df = pd.read_csv(csv_path, usecols=["uni_lemma", "category"])
    df["uni_lemma"] = df["uni_lemma"].astype(str).str.strip().str.lower()
    df["category"]  = df["category"].astype(str).str.strip().str.lower()
    return (
        df.drop_duplicates(subset=["uni_lemma"], keep="first")
        .set_index("uni_lemma")["category"]
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cdi_data(
    data_dir: str,
    force_download: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """
    Ensure CDI reference files are present, then return:
      - included_categories : list of 129 CDI noun category strings
      - lemma_to_semantic   : dict mapping each lemma to its CDI domain

    Parameters
    ----------
    data_dir : str
        Directory where CDI files are cached (created if absent).
    force_download : bool
        Re-download even if files already exist.
    """
    cdi_csv_path   = os.path.join(data_dir, "cdi_words.csv")
    valid129_path  = os.path.join(data_dir, "included_categories_valid129.txt")

    _download_file(CDI_WORDS_URL,  cdi_csv_path,  force=force_download)
    _download_file(VALID129_URL,   valid129_path,  force=force_download)

    included_categories = _load_valid129(valid129_path)
    lemma_to_semantic   = _build_semantic_map(cdi_csv_path)

    logger.info(
        "CDI data ready: %d valid129 categories, %d semantic mappings",
        len(included_categories),
        len(lemma_to_semantic),
    )
    return included_categories, lemma_to_semantic
