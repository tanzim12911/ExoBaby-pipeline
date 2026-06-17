# =============================================================================
# vlm_filter.py — Steps 4 & 5: Filter clips using Gemini VLM
# =============================================================================
# Free tier limits: 15 RPM, 1500 requests/day
# We enforce a 4-second sleep between calls to stay safely under 15 RPM.

import time
import json
import re
import logging
import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prompt — edit this to tune false positive / false negative rates (Step 6)
# -----------------------------------------------------------------------------
FILTER_PROMPT = """
You are a strict data annotator for a child development research dataset.

You will be shown 4 frames sampled from a 30-second video clip.
Your job is to decide whether this clip is USEFUL for studying how toddlers
interact with their environment.

A clip is USEFUL (pass: true) if ALL of the following are true:
  1. A toddler aged approximately 6 months to 3 years is clearly visible.
  2. The toddler's surrounding environment is visible — the camera is NOT
     zoomed in tightly on just the baby's face.
  3. At least one other person, animal, or object that the toddler is
     interacting with (or could interact with) is visible in the scene.

A clip should be REJECTED (pass: false) if ANY of the following are true:
  - No toddler is visible, or the person is clearly older than 3 years.
  - The shot is a tight close-up of only the baby's face with no context.
  - The video is animated, a cartoon, or a screen recording.
  - The frames are too dark, blurry, or corrupted to assess.

Respond ONLY with a valid JSON object. Do not include any other text.
Format:
{"pass": true, "reason": "brief one-sentence explanation"}
or
{"pass": false, "reason": "brief one-sentence explanation"}
"""


def filter_clip(
    frame_paths: list[str],
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    rate_limit_seconds: float = 4.0,
) -> dict:
    """
    Send sampled frames to Gemini and return a pass/fail decision.

    Returns a dict with keys:
      - pass (bool): whether the clip meets the criteria
      - reason (str): short explanation from the model
      - raw (str): raw model response text (for debugging)
      - error (str | None): error message if the call failed
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Load frames as PIL images
    images = []
    for path in frame_paths:
        try:
            images.append(Image.open(path))
        except Exception as e:
            logger.warning(f"Could not open frame {path}: {e}")

    if not images:
        return {"pass": False, "reason": "No valid frames to send", "raw": "", "error": "no frames"}

    try:
        response = model.generate_content(
            [FILTER_PROMPT] + images,
            request_options={"timeout": 60},  # fail fast if API hangs
        )
        raw_text = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Return a safe failure so the pipeline can continue
        return {"pass": False, "reason": "API error", "raw": "", "error": str(e)}
    finally:
        # Always sleep to respect rate limits, even on error
        time.sleep(rate_limit_seconds)

    # Parse JSON from the response
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not match:
        logger.warning(f"Could not parse JSON from response: {raw_text}")
        return {"pass": False, "reason": "Parse error", "raw": raw_text, "error": "json parse failed"}

    try:
        result = json.loads(match.group())
        result["raw"] = raw_text
        result["error"] = None
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e} | raw: {raw_text}")
        return {"pass": False, "reason": "JSON decode error", "raw": raw_text, "error": str(e)}
