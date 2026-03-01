"""
logo_utils.py

Utilities for fetching team logos from Wikimedia.

Responsibilities:
- Convert raw upload.wikimedia.org URLs to thumb.php REST endpoint URLs
- Download and return PIL Images, with retry logic for rate limiting
"""

import io
import re
import time
from urllib.request import Request, urlopen

from PIL import Image


def _wikimedia_thumbnail_url(upload_url, width=320):
    """Convert any upload.wikimedia.org URL to a Wikimedia REST thumbnail URL.

    The thumb.php endpoint does not enforce step-size restrictions and has
    a more generous rate-limit policy than the raw upload CDN.

    Examples
    --------
    SVG  : .../wikipedia/en/c/cf/LogoASMonacoFC2021.svg
        -> https://en.wikipedia.org/w/thumb.php?f=LogoASMonacoFC2021.svg&w=320
    PNG  : .../wikipedia/en/e/e9/Maccabi_Tel_Aviv.png
        -> https://en.wikipedia.org/w/thumb.php?f=Maccabi_Tel_Aviv.png&w=320
    commons: .../wikipedia/commons/1/1b/FC_Bayern_logo.svg
        -> https://commons.wikimedia.org/w/thumb.php?f=FC_Bayern_logo.svg&w=320
    """
    if not upload_url:
        return upload_url

    pattern = (
        r"https://upload\.wikimedia\.org/wikipedia/([^/]+)"
        r"/(?:thumb/)?[a-f0-9]/[a-f0-9]{2}/([^/]+?)(?:/\d+px-.+)?$"
    )
    match = re.match(pattern, upload_url)
    if not match:
        return upload_url

    wiki, filename = match.group(1), match.group(2)
    host = "commons.wikimedia.org" if wiki == "commons" else f"{wiki}.wikipedia.org"
    return f"https://{host}/w/thumb.php?f={filename}&w={width}"


def _fetch_logo_pil(url, size=(80, 80)):
    """Download a logo from a WikipediaLogoUrl and return a PIL Image.

    Uses the Wikimedia thumb.php REST endpoint. Retries up to 3 times on
    HTTP 429. Returns None on any failure or if Pillow is not installed.
    """
    thumb_url = _wikimedia_thumbnail_url(url, width=320)
    headers = {
        "User-Agent": "football-team-analyser/1.0 (educational project; python-urllib)",
        "Accept": "image/png,image/*",
    }

    for attempt in range(3):
        try:
            req = Request(thumb_url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGBA").resize(
                size, Image.Resampling.LANCZOS
            )
        except (OSError, ValueError, RuntimeError) as exc:
            if "429" in str(exc) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            break

    return None
