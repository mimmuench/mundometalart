"""Fetch listing photographs, cached on disk and kept small.

Etsy serves every photo at several widths from the same path, and the
fingerprints are computed on a 512px working copy anyway, so pulling the
full-resolution original would cost bandwidth we never use.
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

USER_AGENT = "mundometalart-guardian/0.1 (catalog self-monitoring)"
ETSY_VARIANT = "il_794xN"
TIMEOUT = 15
RETRIES = 3
# If this many downloads fail before a single one succeeds, the host is
# refusing us and grinding through the rest just burns an hour to learn it.
PROBE_FAILURES = 25


class Unreachable(RuntimeError):
    """The image host rejected everything we asked it for."""


def etsy_downscaled(url: str, variant: str = ETSY_VARIANT) -> str:
    """Point an Etsy image URL at a narrower rendition of the same photo."""
    if "il_fullxfull." in url:
        return url.replace("il_fullxfull.", f"{variant}.")
    return url


def cache_path(url: str, root: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    return root / digest[:2] / f"{digest}{suffix}"


def fetch(url: str, root: Path, *, force: bool = False) -> tuple[Path | None, str]:
    """Download one image, or return the cached copy.

    Returns the path and an empty reason on success, or None and the reason
    it gave up — callers report those reasons rather than a bare count, since
    "403 from the CDN" and "timed out" need completely different responses.
    """
    target = cache_path(url, root)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target, ""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    reason = "unknown"
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = response.read()
            if not payload:
                raise OSError("empty response")
            target.write_bytes(payload)
            return target, ""
        except urllib.error.HTTPError as exc:
            reason = f"HTTP {exc.code}"
            if exc.code in (401, 403, 404, 410):
                return None, reason  # a refusal will not change on retry
        except urllib.error.URLError as exc:
            reason = f"{type(exc.reason).__name__ if exc.reason else 'URLError'}"
        except (TimeoutError, OSError) as exc:
            reason = type(exc).__name__
        if attempt < RETRIES - 1:
            time.sleep(2**attempt)
    return None, reason


def fetch_many(
    urls: list[str],
    root: Path,
    *,
    workers: int = 8,
    force: bool = False,
    progress_every: int = 50,
) -> dict[str, Path | None]:
    """Download a batch, reporting as it goes and bailing out if it is futile."""
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | None] = {}
    reasons: Counter[str] = Counter()
    done = succeeded = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, url, root, force=force): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            path, reason = future.result()
            results[url] = path
            done += 1
            if path is not None:
                succeeded += 1
            else:
                reasons[reason] += 1

            if done % progress_every == 0 or done == len(urls):
                print(f"  {done}/{len(urls)} fetched ({succeeded} ok)", flush=True)

            if succeeded == 0 and done >= PROBE_FAILURES:
                for pending in futures:
                    pending.cancel()
                top = ", ".join(f"{r} x{n}" for r, n in reasons.most_common(3))
                raise Unreachable(
                    f"{done} downloads attempted, none succeeded ({top}). "
                    "The image host is refusing this network."
                )

    if reasons:
        top = ", ".join(f"{r} x{n}" for r, n in reasons.most_common(3))
        print(f"  {len(urls) - succeeded} failed: {top}", file=sys.stderr, flush=True)
    return results
