# =============================================================================
# config.py — Central configuration for the ExoBaby pipeline
# =============================================================================
# Copy this file and fill in your values. Never commit your API key to git.
# You can also set GEMINI_API_KEY as an environment variable instead.

import os
from dotenv import load_dotenv

# Load .env file from the project root (if it exists)
load_dotenv()

# --- Gemini API ---
# Get your free key at https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

# Gemini free tier: 15 RPM, 1500 req/day
# 4 seconds between calls keeps us safely under 15 RPM
GEMINI_RATE_LIMIT_SECONDS = 4

# Model to use — 3.1 Flash Lite is fast and free-tier compatible
GEMINI_MODEL = "gemini-3.1-flash-lite"

# --- YouTube Search ---
# Path to a plain-text file listing search queries to run.
# One entry per line. Three formats are accepted (mix freely):
#
#   https://www.youtube.com/hashtag/bangladeshibabyvlogger
#   https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx
#   Bangladeshi baby vlog   ← plain text search
#
# Blank lines and lines starting with # are ignored.
# If the file does not exist the step is silently skipped.
SEARCH_QUERIES_FILE = "data/search_queries.txt"

# Number of videos to download per search query
MAX_VIDEOS_PER_QUERY = 10

# Video download quality — merges best video+audio into mp4, falls back to best available
VIDEO_FORMAT = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"

# Merge output into mp4 regardless of source format
VIDEO_MERGE_FORMAT = "mp4"

# --- Segmentation ---
CLIP_DURATION_SECONDS = 30

# --- Bias Mitigation: max *passed* clips exported per video ---
# After VLM filtering, this limits how many approved clips from any single
# video are actually exported to the final 1-FPS frame dataset.
#
# The cap is applied at EXPORT time (Step 7), not at segmentation.
# This is intentional: every clip is still seen by the VLM filter so we
# don't blindly discard content before knowing whether it is useful.
# Only the final export is capped, keeping the per-video contribution
# to the dataset balanced.
#
# Analysis of the current 43-video corpus (320 passed clips total):
#   Median passed clips/video : 9
#   Worst offender            : zuowKeRoWl4 (50 passed) → pillow-heavy bias
#
# Simulation on passed clips (total 320):
#   cap=10 → 223 exported  (70 %),  8 videos capped
#   cap=20 → 296 exported  (93 %),  2 videos capped   ← recommended
#   cap=30 → 316 exported  (99 %),  1 video  capped
#
# Set to None to disable the cap entirely.
MAX_PASSED_CLIPS_PER_VIDEO: int | None = 20

# --- Frame Sampling ---
FRAMES_PER_CLIP = 4

# --- Direct Video Downloads ---
# Path to a plain-text file listing specific videos to download.
# One entry per line. Accepted formats (mix freely):
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   VIDEO_ID   ← bare 11-character YouTube ID
#
# Blank lines and lines starting with # are ignored.
# If the file does not exist the step is silently skipped.
DIRECT_VIDEOS_FILE = "data/direct_videos.txt"

# --- Paths ---
RAW_VIDEO_DIR   = "data/raw_videos"
CLIPS_DIR       = "data/clips"
FRAMES_DIR      = "data/frames"
FILTERED_DIR    = "data/filtered"
LOGS_DIR        = "logs"
FILTER_LOG_CSV  = "logs/filter_results.csv"
FILTER_LOG_JSON = "logs/filter_results.json"
SUMMARY_LOG_CSV = "logs/summary.csv"
