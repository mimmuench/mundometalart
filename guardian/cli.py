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

from guardian.designs import design_name, find_designs, link_to_listing
from guardian.etsy_catalog import build_catalog
from guardian.fingerprint import fingerprint, load_image
from guardian.imagery import Unreachable, etsy_downscaled, fetch_many
from guardian.matching import (
    DESIGN_MATCH_PX,
    REVIEW_PX,
    CatalogIndex,
    find_boilerplate,
)

DATA = Path(__file__).resolve().parent / "data"
DESIGNS = DATA / "designs"
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
        # Our own design files are the better reference anyway, so a blocked
        # CDN degrades the index rather than failing the build.
        print(f"warning: {exc}", file=sys.stderr)
        downloaded = {}

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

    # Clean design files, when we have them, are the better reference and are
    # indexed alongside the photographs rather than instead of them: a copy
    # may resemble either our artwork or our particular photograph of it.
    designs = find_designs(Path(args.designs) if args.designs else DESIGNS)
    linked = 0
    for position, path in enumerate(designs):
        listing_id, _ = link_to_listing(path, catalog["listings"])
        if listing_id is not None:
            linked += 1
        try:
            entries.append(
                (listing_id or f"design:{design_name(path)}", 100 + position,
                 fingerprint(load_image(path)))
            )
        except Exception as exc:
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
    if designs:
        print(f"added {len(designs)} design files ({linked} matched to a listing "
              f"by name, {len(designs) - linked} kept under their own name)")

    boilerplate = find_boilerplate(entries)
    if boilerplate:
        dropped = sorted({entries[i][0] for i in boilerplate})
        entries = [e for i, e in enumerate(entries) if i not in boilerplate]
        print(f"dropped {len(boilerplate)} boilerplate images (size charts and "
              f"the like, recurring across {len(dropped)} listings)")

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


def _render(mask, width: int = 46, height: int = 20) -> list[str]:
    """Draw a stored silhouette as text, so a build log can show what the
    extractor actually found in a photo we cannot open from here."""
    import numpy as np
    from PIL import Image as PILImage

    small = np.asarray(
        PILImage.fromarray((mask * 255).astype("uint8"), "L").resize(
            (width, height), PILImage.BILINEAR
        )
    ) > 127
    return ["".join("#" if v else "." for v in row) for row in small]


def _explain(index, catalog: dict, pairs: list, count: int = 4) -> None:
    """Show the silhouettes behind the closest collisions, side by side.

    A distance alone cannot say whether two listings share a design or the
    extractor grabbed the same sofa out of two lifestyle shots.
    """
    urls = {
        listing["id"]: listing["image_urls"] for listing in catalog["listings"]
    }
    by_key = {(lid, idx): fp for lid, idx, fp in index.entries}
    print("\nwhat the extractor found in the closest pairs:")
    for shape_px, mine, mine_idx, other, other_idx in pairs[:count]:
        left = by_key.get((mine, mine_idx))
        right = by_key.get((other, other_idx))
        if left is None or right is None:
            continue
        print(f"\n  {shape_px:.2f}px apart")
        print(f"    L {mine[:40]} [image {mine_idx}]")
        print(f"      {(urls.get(mine) or [''])[mine_idx] if mine_idx < len(urls.get(mine, [])) else ''}")
        print(f"    R {other[:40]} [image {other_idx}]")
        print(f"      {(urls.get(other) or [''])[other_idx] if other_idx < len(urls.get(other, [])) else ''}")
        for row_l, row_r in zip(_render(left.area_mask), _render(right.area_mask)):
            print(f"    {row_l}   {row_r}")


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Measure how alike our own designs are.

    The thresholds were calibrated on rendered shapes. This asks the only
    question those renders cannot: across the real catalog, how close does
    one of our designs come to a *different* one of our designs? Anything
    under the match threshold here is a pair the scanner will confuse, and
    it is better to learn which pairs those are now than from a takedown
    aimed at the wrong seller.
    """
    if not INDEX_PATH.exists():
        print(f"no index at {INDEX_PATH}; run `index` first", file=sys.stderr)
        return 1
    index = CatalogIndex.from_json(json.loads(INDEX_PATH.read_text(encoding="utf-8")))
    print(f"index holds {len(index)} images across "
          f"{len({e[0] for e in index.entries})} listings\n")

    # One image per listing keeps this to a single pass over the catalog.
    firsts: dict[str, int] = {}
    for position, (listing_id, _, _) in enumerate(index.entries):
        firsts.setdefault(listing_id, position)

    collisions = []
    for listing_id, position in firsts.items():
        probe = index.entries[position][2]
        match = index.best_match(probe, exclude_listing=listing_id)
        if match is not None and match.shape_px <= REVIEW_PX:
            collisions.append(
                (match.shape_px, listing_id, index.entries[position][1],
                 match.listing_id, match.image_index, match.verdict)
            )

    collisions.sort()
    confusable = [c for c in collisions if c[0] <= DESIGN_MATCH_PX]
    print(f"{len(confusable)} of {len(firsts)} designs sit within "
          f"{DESIGN_MATCH_PX}px of a different design of ours "
          f"({len(confusable) / max(len(firsts), 1):.1%})")
    print(f"{len(collisions)} sit within the {REVIEW_PX}px review band\n")
    for shape_px, mine, mine_idx, other, other_idx, verdict in collisions[:15]:
        print(f"  {shape_px:5.2f}px  {verdict:13s} {mine[:32]:32s} [{mine_idx}] "
              f"~ {other[:32]} [{other_idx}]")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    _explain(index, catalog, [(c[0], c[1], c[2], c[3], c[4]) for c in collisions])
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
    p_index.add_argument("--designs", default=None,
                         help=f"folder of clean design files (default: {DESIGNS})")
    p_index.add_argument("--limit", type=int, default=0,
                         help="stop after this many images; use a small value to "
                              "check the host is reachable before a full run")
    p_index.set_defaults(func=cmd_index)

    p_self = sub.add_parser("selfcheck", help="how alike are our own designs?")
    p_self.set_defaults(func=cmd_selfcheck)

    p_check = sub.add_parser("check", help="match images against the index")
    p_check.add_argument("images", nargs="+")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
