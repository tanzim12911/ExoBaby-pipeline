# =============================================================================
# downloader.py — Step 1: Search YouTube and download videos
# =============================================================================

import os
import yt_dlp
import logging

logger = logging.getLogger(__name__)


def search_and_download(
    query: str,
    max_videos: int,
    output_dir: str,
    video_format: str,
    merge_format: str = "mp4",
) -> list[str]:
    """
    Download videos from a YouTube hashtag URL or a text search query.

    - Hashtag URL (e.g. "https://www.youtube.com/hashtag/babyvlog"):
      Scrapes the hashtag feed directly, much more precise than text search.
    - Text query (e.g. "Bangladeshi baby vlog"):
      Falls back to YouTube search (ytsearchN).

    Returns a list of file paths for successfully downloaded videos.
    Skips videos already downloaded via the archive file.
    """
    os.makedirs(output_dir, exist_ok=True)
    archive_file = os.path.join(output_dir, "downloaded.txt")

    downloaded_paths = []

    def progress_hook(d):
        if d["status"] == "finished":
            downloaded_paths.append(d["filename"])
            logger.info(f"Downloaded: {d['filename']}")

    is_url = query.startswith("http://") or query.startswith("https://")

    ydl_opts = {
        "format": video_format,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "noplaylist": False,        # allow playlist/feed for hashtag URLs
        "playlistend": max_videos,  # limit to max_videos from the feed
        "download_archive": archive_file,
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        # Skip entries that error (e.g. images, deleted videos, Shorts
        # with no downloadable video stream) instead of crashing the run
        "ignoreerrors": True,
        # Force merge output into mp4 so ffmpeg can process it downstream
        "merge_output_format": merge_format,
    }

    if not is_url:
        # Text search — use ytsearchN prefix
        ydl_opts["noplaylist"] = True
        ydl_opts["default_search"] = f"ytsearch{max_videos}"

    logger.info(
        f"{'Hashtag feed' if is_url else 'Text search'}: '{query}' "
        f"(top {max_videos})"
    )
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])

    return downloaded_paths


def get_all_videos(output_dir: str) -> list[str]:
    """Return paths of all video files (.mp4, .webm, .mkv) in output_dir."""
    extensions = (".mp4", ".webm", ".mkv")
    return [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith(extensions)
    ]
