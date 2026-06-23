# ExoBaby Pipeline

A video filtering pipeline that downloads YouTube baby vlogs, segments them into 30-second clips, and uses the Gemini VLM to identify clips suitable for studying toddler cognitive development in South Asian contexts.

## Pipeline Overview

```
YouTube Search
     │
     ▼
[1] Download videos (yt-dlp)
     │
     ▼
[2] Segment into 30s clips (ffmpeg)
     │
     ▼
[3] Sample 4 frames per clip (OpenCV)
     │
     ▼
[4+5] Filter via Gemini VLM (Gemini 3.1 Flash Lite)
     │
     ├── FAIL → logged to filter_results.csv/.json, discarded
     │
     └── PASS ▼
[6] Manual validation (human review of logs)
     │
     ▼
[7] Export at 1 FPS (ffmpeg) → final dataset frames
```

## Project Structure

```
ExoBaby-pipeline/
├── .env                        # Your Gemini API key — never commit this
├── config.py                   # Search terms, thresholds, paths (pipeline)
├── requirements.txt
│
├── run_pipeline.py             # Orchestrator: download → segment → filter → export
├── pipeline/
│   ├── downloader.py           # Step 1: YouTube search + download
│   ├── segmenter.py            # Step 2: Split into 30s clips
│   ├── sampler.py              # Step 3: Extract 4 frames per clip
│   ├── vlm_filter.py           # Steps 4+5: Gemini VLM filtering
│   └── exporter.py             # Step 7: 1 FPS extraction
│
├── run_detection.py            # Orchestrator: detect → analyse (local GPU)
├── detection/
│   ├── config.py               # Thresholds, model IDs, colour palette, URLs
│   ├── cdi_loader.py           # Download + parse BabyView CDI reference data
│   ├── detector.py             # YOLOE + CLIP detection pipeline (resumable)
│   ├── analysis.py             # Power-law fit + category summary
│   └── visualizer.py           # Rank-bar, log-log, per-domain figures
│
├── ExoBaby_LongTailed_Colab.ipynb  # Colab notebook (thin — calls detection/)
│
├── data/
│   ├── raw_videos/             # Downloaded full videos
│   ├── clips/                  # 30s segments (organised by video ID)
│   ├── frames/                 # Sampled frames per clip
│   └── filtered/               # 1 FPS frames from approved clips
└── logs/
    ├── pipeline.log            # Full run log
    ├── filter_results.csv      # Per-clip VLM decisions (open in Excel/Sheets)
    ├── filter_results.json     # Same data as JSON array (for scripting)
    └── summary.csv             # Per-run totals
```

## Setup

### 1. Install Python 3.10+

Check your version:
```cmd
python --version
```
If needed, download from https://python.org. During install, check **"Add Python to PATH"**.

### 2. Install ffmpeg

Download from https://ffmpeg.org/download.html (Windows → gyan.dev essentials build), extract to `C:\ffmpeg`, and add `C:\ffmpeg\bin` to your system PATH.

Verify:
```cmd
ffmpeg -version
```

### 3. Install Python dependencies

For the local pipeline (download → segment → filter → export):
```cmd
pip install -r requirements.txt
```

For the detection pipeline on a local GPU machine:
```cmd
pip install -r requirements-detection.txt
```

> **Colab users:** the first cell in `ExoBaby_LongTailed_Colab.ipynb` runs
> `pip install -r requirements-detection.txt` automatically — nothing to do manually.

### 4. Set your Gemini API key

Get a free key at https://aistudio.google.com/app/apikey, then open `.env` in the project root and paste it in:

```
GEMINI_API_KEY=your_key_here
```

The pipeline loads this automatically on every run — no need to set environment variables manually.

### 5. Configure search queries

Open `config.py` and edit `SEARCH_QUERIES`. You can use two formats:

**Hashtag URLs** (recommended — scrapes the actual hashtag feed, more relevant results):
```python
"https://www.youtube.com/hashtag/bangladeshibabyvlog"
```

**Text search** (broader, more noise):
```python
"Bangladeshi baby daily routine vlog"
```

Mix both as needed. The defaults use hashtag URLs targeting Bangladeshi and South Asian baby content.

## Usage

### Run the full pipeline

```cmd
python run_pipeline.py
```

### Run individual steps

```cmd
python run_pipeline.py --step download   # Step 1 only
python run_pipeline.py --step segment    # Step 2 only
python run_pipeline.py --step filter     # Steps 3-5 only
python run_pipeline.py --step export     # Step 7 only
```

The pipeline is **resumable**. If it stops mid-run (crash, daily quota hit, etc.), re-running it skips already-processed clips and picks up where it left off.

## Manual Validation (Step 6)

After running the filter step, review the results before exporting:

- Open `logs/filter_results.csv` in Excel or Google Sheets for visual review
- Or load `logs/filter_results.json` programmatically:
  ```python
  import json
  with open("logs/filter_results.json") as f:
      results = json.load(f)
  passed = [r for r in results if r["pass"] is True]
  ```
- Watch a random sample of ~20 clips from each group (pass and fail)
- If Gemini is rejecting too many valid clips or accepting bad ones, edit `FILTER_PROMPT` in `pipeline/vlm_filter.py` and re-run `--step filter`

Already-processed clips are skipped automatically, so re-running the filter step only processes new or unreviewed clips.

## Gemini Free Tier

The pipeline is designed to stay within the free tier limits:

| Limit | Free Tier | Pipeline behaviour |
|---|---|---|
| Requests per minute | 15 RPM | 4s sleep between calls |
| Requests per day | 1,500 | ~75 videos/day (20 clips each) |

If you hit the daily quota, wait until the next day and re-run — no progress is lost.

## Output

The final dataset lives in `data/filtered/`. Each approved clip has its own subfolder of JPEG frames at 1 FPS, ready to feed into the detection pipeline.

```
data/filtered/
└── <video_id>/
    └── <clip_name>/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        └── ...
```

---

## Detection Pipeline — Long-Tailed Distribution Analysis

Replicates Finding 1 from Yang et al. (2026): runs YOLOE open-vocabulary detection
on the 129 CDI noun categories, filters with CLIP ViT-B/32, fits a power-law, and
compares the exponent α to the BabyView paper baseline (~1.93).

### Run on a local GPU machine

```cmd
python run_detection.py --frames data/filtered --output ExoBaby-results
```

Individual steps:

```cmd
python run_detection.py --frames data/filtered --output ExoBaby-results --step detect
python run_detection.py --frames data/filtered --output ExoBaby-results --step analyse
```

The `--step detect` run is **resumable** — already-processed frames are skipped.

### Run on Google Colab (recommended — free T4 GPU)

Open `ExoBaby_LongTailed_Colab.ipynb` in Colab. The notebook is thin: each step
calls into the `detection/` package. To change a threshold or tweak a figure, edit
the relevant module rather than the notebook.

| Module | Responsibility |
|---|---|
| `detection/config.py` | All thresholds, model IDs, CDI colour palette |
| `detection/cdi_loader.py` | Download + parse BabyView CDI reference files |
| `detection/detector.py` | YOLOE + CLIP batch detection loop (resumable) |
| `detection/analysis.py` | Category summary, power-law fit, comparison table |
| `detection/visualizer.py` | Rank-bar, log-log, and per-domain figures |

### Detection outputs

```
ExoBaby-results/
├── data/
│   ├── cdi_words.csv
│   └── included_categories_valid129.txt
├── frame_data/
│   └── merged_frame_detections_with_metadata_filtered-0.27.csv
└── analysis/
    ├── results/
    │   └── long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv
    └── figures/
        ├── fig1a_long_tailed_bar_valid129.png  (.pdf)
        ├── fig1b_loglog_valid129.png  (.pdf)
        └── fig2_per_domain_valid129.png
```

### Tuning the thresholds

Edit `detection/config.py`:

```python
YOLOE_CONF_THRESHOLD = 0.25   # lower = more detections, more noise
CLIP_SIM_THRESHOLD   = 0.27   # lower = more detections, lower precision
```

Re-run `--step detect` — it will only process frames not yet in the CSV.
To re-run from scratch, delete the detection CSV first.

---

## Common Errors

| Error | Fix |
|---|---|
| `ffmpeg is not recognized` | ffmpeg not on PATH — redo setup step 2 |
| `GEMINI_API_KEY is not set` | Add your key to `.env` |
| `429 Too Many Requests` | Daily quota hit — wait until tomorrow and re-run |
| `No module named 'cv2'` | Run `pip install -r requirements.txt` again |
| `No module named 'ultralytics'` | Run the pip cell in the Colab notebook, or `pip install 'ultralytics>=8.3'` |
| `No GPU found` in Colab | Runtime > Change runtime type > T4 GPU, then re-run all cells |
| Detection CSV missing | Run `--step detect` (or the Colab Step 3 cell) before `--step analyse` |
