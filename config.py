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

# Model to use — Flash is faster and cheaper on the free tier
GEMINI_MODEL = "gemini-1.5-flash"

# --- YouTube Search ---
# You can use two formats here:
#
# 1. Plain text search (less precise):
#    "Bangladeshi baby daily routine vlog"
#
# 2. Hashtag URL (more precise — scrapes the actual hashtag feed):
#    "https://www.youtube.com/hashtag/bangladeshibabyvlog"
#
# Mix both as needed. Hashtag URLs tend to surface more relevant content.
SEARCH_QUERIES = [
    "https://www.youtube.com/hashtag/bangladeshibabyvlog",
    "https://www.youtube.com/hashtag/babyvlogbangladesh",
    "https://www.youtube.com/hashtag/dhakababylife",
    "https://www.youtube.com/hashtag/babyvlog",
    "https://www.youtube.com/hashtag/toddlervlog",
]

# Number of videos to download per search query
MAX_VIDEOS_PER_QUERY = 10

# Video download quality — 720p balances quality and disk space
VIDEO_FORMAT = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best"

# --- Segmentation ---
CLIP_DURATION_SECONDS = 30

# --- Frame Sampling ---
FRAMES_PER_CLIP = 4

# --- Paths ---
RAW_VIDEO_DIR   = "data/raw_videos"
CLIPS_DIR       = "data/clips"
FRAMES_DIR      = "data/frames"
FILTERED_DIR    = "data/filtered"
LOGS_DIR        = "logs"
FILTER_LOG_CSV  = "logs/filter_results.csv"
FILTER_LOG_JSON = "logs/filter_results.json"
