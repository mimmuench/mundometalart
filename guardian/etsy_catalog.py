"""Parse the Etsy 'Currently for Sale Listings' CSV into a catalog.

Etsy's export gives us TITLE, DESCRIPTION, TAGS, MATERIALS and up to ten
IMAGE columns, but no listing id and no listing URL, so listings are keyed
by a slug derived from the title.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_COLUMNS = [f"IMAGE{i}" for i in range(1, 11)]

# Words that say "metal wall art shop" rather than "this particular design".
# Data-driven scoring below does most of the work; this only pins down terms
# that are generic in the trade even when our own catalog uses them rarely.
TRADE_WORDS = {
    "art", "artwork", "wall", "decor", "decoration", "decorative", "metal",
    "steel", "iron", "sign", "signs", "plaque", "gift", "gifts", "home",
    "house", "custom", "customized", "personalized", "large", "small", "big",
    "modern", "unique", "handmade", "new", "set", "piece", "panel", "panels",
    "hanging", "indoor", "outdoor", "living", "room", "theme", "themed",
    "style", "styled", "design", "designed", "quality", "premium", "perfect",
    "beautiful", "lovely", "cut", "laser", "powder", "coated", "inch", "inches",
}

# Ordinary English words are rare in a 229-title corpus purely by chance, so
# IDF alone promotes them ("said", "no"). They carry no design meaning.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "get", "had", "has", "have", "he", "her", "him", "his",
    "how", "i", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on",
    "one", "or", "our", "out", "over", "said", "say", "says", "she", "so",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "too", "up", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your", "yours",
}

# Etsy sellers title a listing "<what it is>, <who it's for>, <occasion>", so
# the opening clause names the design and later clauses name the market.
CLAUSE_SPLIT_RE = re.compile(r"\s*[,|\u2013\u2014]\s*|\s+-\s+")

# Longest match wins, so more specific phrases come first.
CATEGORY_TERMS = [
    "garden stake", "yard stake", "yard art", "garden decor",
    "address sign", "welcome sign", "house number", "door hanger",
    "wall sculpture", "wall art", "wall decor", "wall panel",
    "sculpture", "sign",
]
DEFAULT_CATEGORY = "metal wall art"

WORD_RE = re.compile(r"[a-z0-9']+")


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "listing"


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


@dataclass
class Listing:
    id: str
    title: str
    price: str
    currency: str
    tags: list[str]
    materials: list[str]
    image_urls: list[str]
    # Filled in by build_catalog once the whole corpus is known.
    keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def read_listings(csv_path: Path) -> list[Listing]:
    csv.field_size_limit(10**7)
    listings: list[Listing] = []
    seen: Counter[str] = Counter()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            title = (row.get("TITLE") or "").strip()
            if not title:
                continue
            slug = slugify(title)
            seen[slug] += 1
            if seen[slug] > 1:  # Etsy allows duplicate titles.
                slug = f"{slug}-{seen[slug]}"
            urls = [
                (row.get(col) or "").strip()
                for col in IMAGE_COLUMNS
                if (row.get(col) or "").strip()
            ]
            listings.append(
                Listing(
                    id=slug,
                    title=title,
                    price=(row.get("PRICE") or "").strip(),
                    currency=(row.get("CURRENCY_CODE") or "").strip(),
                    tags=[t.replace("_", " ") for t in _split_list(row.get("TAGS", ""))],
                    materials=_split_list(row.get("MATERIALS", "")),
                    image_urls=urls,
                )
            )
    return listings


def _document_frequency(listings: list[Listing]) -> Counter[str]:
    df: Counter[str] = Counter()
    for listing in listings:
        df.update(set(tokenize(listing.title)))
    return df


def clauses(title: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT_RE.split(title) if c.strip()]


def category_term(title: str) -> str:
    """The noun a copycat would also have to use to be found in search."""
    lowered = title.lower()
    for term in CATEGORY_TERMS:
        if term in lowered:
            return term if term.startswith("metal") else f"metal {term}"
    return DEFAULT_CATEGORY


def distinctive_keywords(
    title: str, df: Counter[str], corpus_size: int, limit: int = 6
) -> list[str]:
    """Rank a title's words by how rare they are in our own catalog.

    A word we use on 60 listings ("housewarming") describes our shop; a word
    we use on two ("nutcracker") describes the design, and is what a copy of
    that design is likely to be titled with too. Rarity alone is not enough,
    so words in the opening clause — where the design is named — outrank
    equally rare words from the gift-occasion tail of the title.
    """
    parts = clauses(title)
    head_words = set(tokenize(parts[0])) if parts else set()
    scored: list[tuple[float, str]] = []
    for word in dict.fromkeys(tokenize(title)):  # de-dupe, keep title order
        if len(word) < 3 or word.isdigit():
            continue
        if word in TRADE_WORDS or word in STOP_WORDS:
            continue
        idf = math.log(corpus_size / (1 + df.get(word, 0)))
        weight = 1.6 if word in head_words else 1.0
        scored.append((idf * weight, word))
    scored.sort(key=lambda pair: -pair[0])
    return [word for _, word in scored[:limit]]


def build_search_queries(title: str, keywords: list[str], limit: int = 3) -> list[str]:
    """Queries to run against marketplace search in the keyword lane.

    Marketplace search engines reward short phrases, and a copy is usually
    titled with the design's own nouns plus the category, so we pair the two
    rarest words with the category term rather than replaying the full title.
    """
    category = category_term(title)
    queries: list[str] = []
    if keywords:
        queries.append(" ".join(keywords[:2] + [category]))
    if len(keywords) >= 3:
        queries.append(" ".join(keywords[:3]))
    parts = clauses(title)
    if parts and parts[0].lower() not in {q.lower() for q in queries}:
        queries.append(parts[0])
    seen: set[str] = set()
    unique = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]
    return unique[:limit]


def build_catalog(csv_path: Path) -> dict:
    listings = read_listings(csv_path)
    df = _document_frequency(listings)
    corpus_size = max(len(listings), 1)
    for listing in listings:
        listing.keywords = distinctive_keywords(listing.title, df, corpus_size)
        listing.search_queries = build_search_queries(listing.title, listing.keywords)
    return {
        "source": csv_path.name,
        "listing_count": len(listings),
        "image_count": sum(len(l.image_urls) for l in listings),
        "listings": [asdict(l) for l in listings],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m guardian.etsy_catalog <etsy.csv> <catalog.json>")
        return 2
    catalog = build_catalog(Path(argv[1]))
    out = Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{catalog['listing_count']} listings, {catalog['image_count']} images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
