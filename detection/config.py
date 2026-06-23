# =============================================================================
# detection/config.py — Configuration for the YOLOE+CLIP detection pipeline
# =============================================================================
# All paths below are relative to OUTPUT_BASE, which is set at runtime by
# run_detection.py (or overridden in the Colab notebook).
#
# Edit YOLOE_CONF_THRESHOLD and CLIP_SIM_THRESHOLD to tune recall/precision.
# Edit CDI_WORDS_URL / VALID129_URL if the upstream repo moves.

import os

# ---------------------------------------------------------------------------
# Detection thresholds (match the BabyView paper defaults)
# ---------------------------------------------------------------------------
YOLOE_CONF_THRESHOLD: float = 0.25   # YOLOE minimum confidence score
CLIP_SIM_THRESHOLD:   float = 0.27   # CLIP cosine similarity minimum

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
YOLOE_REPO:     str = "jameslahm/yoloe"
YOLOE_FILENAME: str = "yoloe-v8l-seg.pt"
CLIP_MODEL:     str = "ViT-B-32"
CLIP_PRETRAIN:  str = "openai"

# ---------------------------------------------------------------------------
# BabyView reference data URLs
# ---------------------------------------------------------------------------
CDI_WORDS_URL: str = (
    "https://raw.githubusercontent.com/babyview-project/object-detection"
    "/main/data/cdi_words.csv"
)
VALID129_URL: str = (
    "https://raw.githubusercontent.com/babyview-project/object-detection"
    "/main/data/shared_data_manuscript_2026/category_lists"
    "/included_categories_valid129.txt"
)

# ---------------------------------------------------------------------------
# CDI semantic domain colour palette (used by visualizer.py)
# ---------------------------------------------------------------------------
CDI_SEMANTIC_ORDER: list[str] = [
    "animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
    "household", "outside", "people", "toys", "vehicles", "other",
]

CDI_SEMANTIC_COLORS: dict[str, str] = {
    "animals":         "#4DB8A8",
    "body_parts":      "#E87A5F",
    "clothing":        "#9B7EC8",
    "food_drink":      "#E8A54C",
    "furniture_rooms": "#6BAB7A",
    "household":       "#D97B9E",
    "outside":         "#5B9BD5",
    "people":          "#E8C44C",
    "toys":            "#B07CC8",
    "vehicles":        "#6BA3D5",
    "other":           "#8B9A9E",
}

# ---------------------------------------------------------------------------
# BabyView paper reference values (for the comparison summary)
# ---------------------------------------------------------------------------
BABYVIEW_ALPHA:   float = 1.93
BABYVIEW_FRAMES:  str   = "3,680,000"
BABYVIEW_DETECTIONS: str = "~2,994,667"
BABYVIEW_TOP5: list[tuple] = [
    ("chair",  0.1310, "furniture_rooms"),
    ("lamp",   0.1057, "household"),
    ("table",  0.0943, "furniture_rooms"),
    ("couch",  0.0867, "furniture_rooms"),
    ("pillow", 0.0683, "household"),
]

# ---------------------------------------------------------------------------
# Output sub-folder names (resolved at runtime relative to OUTPUT_BASE)
# ---------------------------------------------------------------------------
DATA_SUBDIR:        str = "data"
FRAME_DATA_SUBDIR:  str = "frame_data"
RESULTS_SUBDIR:     str = os.path.join("analysis", "results")
FIGURES_SUBDIR:     str = os.path.join("analysis", "figures")

DETECTION_CSV_NAME: str = (
    f"merged_frame_detections_with_metadata_filtered-{CLIP_SIM_THRESHOLD}.csv"
)
INTERMEDIATE_CSV_NAME: str = (
    f"long_tailed_dist_prop_included_categories_"
    f"filtered-{CLIP_SIM_THRESHOLD}_valid129.csv"
)
