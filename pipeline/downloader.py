# =============================================================================
# downloader.py — Step 1: Search YouTube and download videos
# =============================================================================

import os
import yt_dlp
import logging

logger = logging.getLogger(__name__)


def _load_archive(archive_file: str) -> set[str]:
    """Return the set of video IDs already in the download archive."""
    if not os.path.exists(archive_file):
        return set()
    ids = set()
    with open(archive_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])  # format: "youtube <video_id>"
    return ids


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

    For hashtag/playlist feeds, iterates through the feed in batches,
    skipping already-archived videos, until max_videos new ones are
    downloaded or the feed is exhausted — no hardcoded playlist offset needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    archive_file = os.path.join(output_dir, "downloaded.txt")

    is_url = query.startswith("http://") or query.startswith("https://")

    # For plain text search, yt-dlp handles the cap natively via ytsearchN.
    if not is_url:
        return _download_text_search(
            query, max_videos, output_dir, archive_file, video_format, merge_format
        )

    # For feeds/hashtags: walk the feed in batches, stop once we have enough.
    return _download_feed(
        query, max_videos, output_dir, archive_file, video_format, merge_format
    )


def _download_text_search(
    query: str,
    max_videos: int,
    output_dir: str,
    archive_file: str,
    video_format: str,
    merge_format: str,
) -> list[str]:
    downloaded_paths = []

    def progress_hook(d):
        if d["status"] == "finished":
            downloaded_paths.append(d["filename"])
            logger.info(f"Downloaded: {d['filename']}")

    ydl_opts = {
        "format": video_format,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "default_search": f"ytsearch{max_videos}",
        "download_archive": archive_file,
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "merge_output_format": merge_format,
    }

    logger.info(f"Text search: '{query}' (top {max_videos})")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])

    return downloaded_paths


def _download_feed(
    url: str,
    max_videos: int,
    output_dir: str,
    archive_file: str,
    video_format: str,
    merge_format: str,
    batch_size: int = 20,
) -> list[str]:
    """
    Walk a hashtag/playlist feed in batches of `batch_size`.

    Strategy:
      1. Extract the flat entry list for the current batch window (no download).
      2. Skip any IDs already in the archive.
      3. Download only the new ones, stop once max_videos reached.
      4. If the feed window returned fewer entries than batch_size, we've hit
         the end of the feed — stop.
    """
    archived = _load_archive(archive_file)
    downloaded_paths = []
    position = 1

    logger.info(f"Hashtag feed: '{url}' (fetching up to {max_videos} new videos)")

    while len(downloaded_paths) < max_videos:
        batch_end = position + batch_size - 1

        # Step 1: list entries in this window without downloading
        logger.info(f"  Scanning feed positions {position}–{batch_end} ...")
        with yt_dlp.YoutubeDL({
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "noplaylist": False,
            "playliststart": position,
            "playlistend": batch_end,
            "extract_flat": True,
        }) as ydl:
            info = ydl.extract_info(url, download=False)

        entries = info.get("entries", []) if info else []
        if not entries:
            logger.info("  Feed exhausted — no entries in window.")
            break

        # Step 2: filter out already-archived IDs
        new_ids = [
            e["id"] for e in entries
            if e and e.get("id") and e["id"] not in archived
        ]
        needed = max_videos - len(downloaded_paths)
        ids_to_fetch = new_ids[:needed]

        logger.info(
            f"  {len(entries)} entries in window, "
            f"{len(new_ids)} new, "
            f"downloading {len(ids_to_fetch)}"
        )

        # Step 3: download the new IDs
        if ids_to_fetch:
            batch_downloaded = []

            def progress_hook(d):
                if d["status"] == "finished":
                    batch_downloaded.append(d["filename"])
                    logger.info(f"Downloaded: {d['filename']}")

            ydl_opts = {
                "format": video_format,
                "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
                "noplaylist": True,   # download individual IDs directly
                "download_archive": archive_file,
                "progress_hooks": [progress_hook],
                "quiet": True,
                "no_warnings": True,
                "ignoreerrors": True,
                "merge_output_format": merge_format,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={vid_id}" for vid_id in ids_to_fetch])

            downloaded_paths.extend(batch_downloaded)
            archived.update(ids_to_fetch)

        # Step 4: if the window was shorter than batch_size, feed is exhausted
        if len(entries) < batch_size:
            logger.info("  Reached end of feed.")
            break

        position = batch_end + 1

    logger.info(f"  Done. {len(downloaded_paths)} new video(s) downloaded.")
    return downloaded_paths

    logger.info(f"  Done. {len(downloaded_paths)} new video(s) downloaded.")
    return downloaded_paths


def download_direct(
    urls_file: str,
    output_dir: str,
    video_format: str,
    merge_format: str = "mp4",
) -> list[str]:
    """
    Download a specific list of YouTube videos from a plain-text file.

    Each line in the file can be:
      - A full URL: https://www.youtube.com/watch?v=VIDEO_ID
      - A short URL: https://youtu.be/VIDEO_ID
      - A bare video ID (11 characters)

    Blank lines and lines starting with '#' are ignored.
    Videos already in the download archive are skipped.
    Returns a list of file paths for successfully downloaded videos.
    """
    if not os.path.exists(urls_file):
        logger.info(f"No direct-videos file found at '{urls_file}', skipping.")
        return []

    with open(urls_file, encoding="utf-8") as f:
        entries = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not entries:
        logger.info(f"Direct-videos file '{urls_file}' is empty, skipping.")
        return []

    # Normalise bare IDs to full watch URLs
    urls = [
        e if e.startswith("http") else f"https://www.youtube.com/watch?v={e}"
        for e in entries
    ]

    os.makedirs(output_dir, exist_ok=True)
    archive_file = os.path.join(output_dir, "downloaded.txt")
    downloaded_paths = []

    def progress_hook(d):
        if d["status"] == "finished":
            downloaded_paths.append(d["filename"])
            logger.info(f"Downloaded: {d['filename']}")

    ydl_opts = {
        "format": video_format,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "download_archive": archive_file,
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "merge_output_format": merge_format,
    }

    logger.info(f"Downloading {len(urls)} direct video(s) from '{urls_file}'")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

    logger.info(f"Direct downloads complete. {len(downloaded_paths)} new video(s) downloaded.")
    return downloaded_paths


def get_all_videos(output_dir: str) -> list[str]:
    """Return paths of all video files (.mp4, .webm, .mkv) in output_dir."""
    extensions = (".mp4", ".webm", ".mkv")
    return [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith(extensions)
    ]
