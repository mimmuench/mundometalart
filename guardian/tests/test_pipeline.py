"""Fast regression tests. Run: python -m guardian.tests.test_pipeline

The calibration sweep (guardian/tests/calibrate.py) is the thorough one; this
keeps a small version of each property it established so a refactor cannot
quietly undo them.
"""

from __future__ import annotations

import json
import sys

from guardian.etsy_catalog import build_search_queries, category_term, distinctive_keywords
from guardian.fingerprint import Fingerprint, fingerprint, shape_distance
from guardian.matching import (
    DESIGN_MATCH_PX,
    CatalogIndex,
    VERDICT_CLEAR,
)
from guardian.tests.synth import design_mask, photograph

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        FAILURES.append(message)


def catalog_index(seeds=range(1, 9)) -> CatalogIndex:
    entries = []
    for family in ("mountain", "creature", "botanical"):
        for seed in seeds:
            entries.append(
                (
                    f"{family}-{seed}",
                    0,
                    fingerprint(photograph(design_mask(family, seed), seed=seed)),
                )
            )
    return CatalogIndex(entries)


def test_reshot_copy_is_found() -> None:
    print("a copy photographed by someone else is traced to the right listing")
    index = catalog_index()
    for family in ("mountain", "creature", "botanical"):
        probe = fingerprint(
            photograph(
                design_mask(family, 5),
                seed=500,
                wall=(54, 54, 60),   # their wall, not ours
                rotate=2.5,          # hung by hand
                mirror=True,         # flipped to dodge image matching
                crop=0.05,
                jpeg_quality=60,
            )
        )
        match = index.best_match(probe)
        check(match.listing_id == f"{family}-5", f"{family}: identified {match.listing_id}")
        check(match.shape_px <= DESIGN_MATCH_PX, f"{family}: {match.shape_px:.2f}px is a match")


def test_unrelated_design_is_cleared() -> None:
    print("a design we do not own is not reported")
    index = catalog_index(seeds=range(1, 6))
    for family in ("mountain", "creature", "botanical"):
        probe = fingerprint(photograph(design_mask(family, 40), seed=940))
        match = index.best_match(probe)
        check(
            match.verdict == VERDICT_CLEAR,
            f"{family}: verdict {match.verdict} at {match.shape_px:.2f}px",
        )


def test_watermark_does_not_hide_a_copy() -> None:
    print("a watermark stamped on a stolen photo does not hide it")
    index = catalog_index(seeds=range(1, 6))
    probe = fingerprint(
        photograph(design_mask("creature", 3), seed=3, watermark="HotDeals", jpeg_quality=55)
    )
    match = index.best_match(probe)
    check(match.listing_id == "creature-3", f"identified {match.listing_id}")
    check(match.is_hit, f"verdict {match.verdict} at {match.shape_px:.2f}px")


def test_index_survives_a_round_trip() -> None:
    print("the index reloads from disk unchanged")
    index = catalog_index(seeds=range(1, 4))
    reloaded = CatalogIndex.from_json(json.loads(json.dumps(index.to_json())))
    check(len(reloaded) == len(index), f"{len(reloaded)} entries")
    probe = fingerprint(photograph(design_mask("mountain", 2), seed=2))
    check(
        reloaded.best_match(probe).listing_id == index.best_match(probe).listing_id,
        "same verdict before and after reload",
    )


def test_search_queries_name_the_design() -> None:
    print("search queries carry the design's own words, not the shop's")
    from collections import Counter

    titles = [
        "Mountain Metal Wall Art, Hiking Couple Triptych, Adventure Anniversary Gift",
        "Black Cat Garden Stake, \"The Cat Said No\" Yard Decor, Whimsical Steel Art",
        "Personalized Rooster Garden Stake Metal Sign | Outdoor Farm Sign",
        "Ocean Wave Metal Wall Art - Coastal Home Decor",
    ] + ["Housewarming Metal Wall Art Gift"] * 20
    df: Counter[str] = Counter()
    for title in titles:
        df.update(set(title.lower().replace(",", " ").split()))
    keywords = distinctive_keywords(titles[1], df, len(titles))
    check("cat" in keywords, f"'cat' survives, 'said' does not: {keywords[:3]}")
    check("said" not in keywords, "stop words are dropped")
    check(
        category_term(titles[2]) == "metal garden stake",
        f"garden stakes are not called wall art: {category_term(titles[2])}",
    )
    query = build_search_queries(titles[0], distinctive_keywords(titles[0], df, len(titles)))[0]
    check("metal wall art" in query, f"query keeps the category: {query!r}")


def main() -> int:
    for test in (
        test_reshot_copy_is_found,
        test_unrelated_design_is_cleared,
        test_watermark_does_not_hide_a_copy,
        test_index_survives_a_round_trip,
        test_search_queries_name_the_design,
    ):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
