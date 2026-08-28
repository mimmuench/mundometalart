"""Fetch listing photographs, cached on disk and kept small.

Etsy serves every photo at several widths from the same path, and the
fingerprints are computed on a 512px working copy anyway, so pulling the
full-resolution original would cost bandwidth we never use.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

USER_AGENT = "mundometalart-guardian/0.1 (catalog self-monitoring)"
ETSY_VARIANT = "il_794xN"
TIMEOUT = 30
RETRIES = 4


def etsy_downscaled(url: str, variant: str = ETSY_VARIANT) -> str:
    """Point an Etsy image URL at a narrower rendition of the same photo."""
    if "il_fullxfull." in url:
        return url.replace("il_fullxfull.", f"{variant}.")
    return url


def cache_path(url: str, root: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    return root / digest[:2] / f"{digest}{suffix}"


def fetch(url: str, root: Path, *, force: bool = False) -> Path | None:
    """Download one image, or return the cached copy. None if unreachable."""
    target = cache_path(url, root)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = response.read()
            if not payload:
                raise OSError("empty response")
            target.write_bytes(payload)
            return target
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == RETRIES - 1:
                return None
            time.sleep(2**attempt)
    return None


def fetch_many(
    urls: list[str], root: Path, *, workers: int = 8, force: bool = False
) -> dict[str, Path | None]:
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda u: fetch(u, root, force=force), urls))
    return dict(zip(urls, results))
