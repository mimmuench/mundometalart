"""Our own clean design files, which beat photographs as a reference.

A photograph of a piece on a wall is the design plus a wall, a shadow, a
camera and a crop, and every one of those has to be undone before the shape
can be compared. The design file has none of them, so when we have it, we
use it and skip the guessing.

Naming is deliberately undemanding: the file name is the design's name, and
it is matched to an Etsy listing when the words line up. A file nobody can
match still gets watched — it just shows up in reports under its own name.
"""

from __future__ import annotations

import re
from pathlib import Path

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
WORD_RE = re.compile(r"[a-z0-9]+")

# Words that appear in nearly every file name and listing title, and so tell
# us nothing about which listing a file belongs to.
NOISE = {
    "final", "copy", "new", "v1", "v2", "v3", "design", "designs", "file",
    "files", "dxf", "svg", "cut", "laser", "metal", "steel", "art", "wall",
    "decor", "sign", "mockup", "photo", "image", "img", "untitled", "the",
    "and", "for", "with",
}


def design_name(path: Path) -> str:
    """A human-readable name from the file name, kept as the seller wrote it."""
    cleaned = re.sub(r"[_\-]+", " ", path.stem).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or path.stem


def _words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if len(w) > 2 and w not in NOISE}


def find_designs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES)


def link_to_listing(
    path: Path, listings: list[dict], min_overlap: int = 2
) -> tuple[str | None, float]:
    """Guess which listing a design file belongs to, from its name alone.

    Two shared meaningful words is the bar. Below that the guess is worse
    than no guess: a wrongly linked design would send a takedown pointing at
    the wrong listing of ours.
    """
    file_words = _words(path.stem)
    if not file_words:
        return None, 0.0
    best_id, best_score, best_hits = None, 0.0, 0
    for listing in listings:
        title_words = _words(listing["title"])
        if not title_words:
            continue
        hits = len(file_words & title_words)
        score = hits / len(file_words | title_words)
        if hits > best_hits or (hits == best_hits and score > best_score):
            best_id, best_score, best_hits = listing["id"], score, hits
    if best_hits < min_overlap:
        return None, 0.0
    return best_id, best_score
