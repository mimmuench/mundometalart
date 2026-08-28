"""Command line entry points for the monitoring pipeline.

    python -m guardian.cli catalog  <etsy.csv>          -> data/catalog.json
    python -m guardian.cli index                        -> data/index.json
    python -m guardian.cli check    <image> [image ...] -> match against index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from guardian.etsy_catalog import build_catalog
from guardian.fingerprint import fingerprint, load_image
from guardian.imagery import Unreachable, etsy_downscaled, fetch_many
from guardian.matching import CatalogIndex

DATA = Path(__file__).resolve().parent / "data"
CATALOG_PATH = DATA / "catalog.json"
INDEX_PATH = DATA / "index.json"
IMAGE_CACHE = DATA / "images"


def cmd_catalog(args: argparse.Namespace) -> int:
    catalog = build_catalog(Path(args.csv))
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{catalog['listing_count']} listings, {catalog['image_count']} images "
        f"-> {CATALOG_PATH}"
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    if not CATALOG_PATH.exists():
        print(f"no catalog at {CATALOG_PATH}; run `catalog` first", file=sys.stderr)
        return 1
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    # The hero shots carry the design; later slots hold size charts, packaging
    # and lifestyle scenes that would match other shops' equivalents and
    # generate noise rather than evidence.
    wanted: list[tuple[str, int, str]] = []
    for listing in catalog["listings"]:
        urls = listing["image_urls"]
        if args.images_per_listing > 0:
            urls = urls[: args.images_per_listing]
        for position, url in enumerate(urls):
            wanted.append((listing["id"], position, etsy_downscaled(url)))

    if args.limit:
        wanted = wanted[: args.limit]

    print(f"fetching {len(wanted)} images for {catalog['listing_count']} listings...")
    try:
        downloaded = fetch_many(
            [url for _, _, url in wanted], IMAGE_CACHE, workers=args.workers
        )
    except Unreachable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries = []
    failures = 0
    for listing_id, position, url in wanted:
        path = downloaded.get(url)
        if path is None:
            failures += 1
            continue
        try:
            entries.append((listing_id, position, fingerprint(load_image(path))))
        except Exception as exc:  # a corrupt download should not stop the build
            print(f"  skipped {url}: {exc}", file=sys.stderr)
            failures += 1

    index = CatalogIndex(entries)
    INDEX_PATH.write_text(json.dumps(index.to_json()), encoding="utf-8")
    size_kb = INDEX_PATH.stat().st_size / 1024
    print(f"indexed {len(index)} images ({failures} failed) -> {INDEX_PATH} [{size_kb:.0f} KB]")
    return 0 if len(index) else 1


def cmd_check(args: argparse.Namespace) -> int:
    if not INDEX_PATH.exists():
        print(f"no index at {INDEX_PATH}; run `index` first", file=sys.stderr)
        return 1
    index = CatalogIndex.from_json(json.loads(INDEX_PATH.read_text(encoding="utf-8")))
    print(f"index holds {len(index)} images\n")
    for image_path in args.images:
        probe = fingerprint(load_image(image_path))
        match = index.best_match(probe)
        if match is None:
            print(f"{image_path}: index empty")
            continue
        print(
            f"{image_path}\n  -> {match.verdict:13s} {match.listing_id} "
            f"(image {match.image_index}) shape={match.shape_px:.2f}px "
            f"phash={match.phash_bits}b dhash={match.dhash_bits}b"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guardian", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_catalog = sub.add_parser("catalog", help="parse the Etsy listings CSV")
    p_catalog.add_argument("csv")
    p_catalog.set_defaults(func=cmd_catalog)

    p_index = sub.add_parser("index", help="download images and fingerprint them")
    p_index.add_argument("--images-per-listing", type=int, default=3,
                         help="0 for every image (default: 3)")
    p_index.add_argument("--workers", type=int, default=8)
    p_index.add_argument("--limit", type=int, default=0,
                         help="stop after this many images; use a small value to "
                              "check the host is reachable before a full run")
    p_index.set_defaults(func=cmd_index)

    p_check = sub.add_parser("check", help="match images against the index")
    p_check.add_argument("images", nargs="+")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
