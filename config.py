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
