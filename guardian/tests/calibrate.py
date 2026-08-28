"""Measure where the copy/not-a-copy line falls, instead of guessing it.

Positives are one design photographed two different ways — the case that
matters, because a copycat shoots their own picture. Negatives are two
genuinely different designs from the same family, verified as different
against their clean masks first, so that a generator that happens to emit
one design as another's mirror image cannot quietly poison the negatives.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
from PIL import Image

from guardian.fingerprint import (
    Fingerprint,
    contour_distance,
    contour_points,
    fingerprint,
    hamming,
    shape_distance,
    shape_iou,
)
from guardian.tests.synth import FAMILIES, design_mask, photograph

# How a copy tends to differ from the original photograph.
RESHOOT_STYLES = [
    dict(seed=11, scale=0.85),
    dict(seed=12, wall=(58, 58, 62), scale=1.1),          # dark wall
    dict(seed=13, mirror=True, scale=0.95),                # mirrored to dodge matching
    dict(seed=14, rotate=3.0, crop=0.06),                  # hand-held, cropped
    dict(seed=15, wall=(30, 32, 38), rotate=-2.5, mirror=True),
    dict(seed=16, scale=1.15, jpeg_quality=55, noise=12),  # heavy recompression
]
# How a stolen photograph tends to be altered before reposting.
REPOST_STYLES = [
    dict(jpeg_quality=45),
    dict(crop=0.05, jpeg_quality=70),
    dict(watermark="BestDeals", jpeg_quality=60),
]

SEEDS = list(range(1, 13))
DISTINCT_MIN = 3.0  # clean-mask px distance below which two designs are the same design


def clean_contour(family: str, seed: int) -> np.ndarray:
    mask = np.asarray(design_mask(family, seed)) > 127
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    tile = Image.fromarray((mask * 255).astype(np.uint8), "L").resize((128, 128), Image.BILINEAR)
    return contour_points(np.asarray(tile) > 127)


def _mirror(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 1] = 127 - out[:, 1]
    return out


def build_pairs() -> tuple[list[dict], list[dict]]:
    rng = random.Random(0)
    originals: dict[tuple[str, int], Fingerprint] = {}
    reshoots: dict[tuple[str, int], list[Fingerprint]] = {}
    for family in FAMILIES:
        for seed in SEEDS:
            mask = design_mask(family, seed)
            originals[(family, seed)] = fingerprint(photograph(mask, seed=seed))
            reshoots[(family, seed)] = [
                fingerprint(photograph(mask, **style)) for style in RESHOOT_STYLES
            ]

    positives = [
        {"kind": "reshoot", "a": originals[key], "b": shot}
        for key, shots in reshoots.items()
        for shot in shots
    ]
    for family in FAMILIES:
        for seed in SEEDS:
            mask = design_mask(family, seed)
            for style in REPOST_STYLES:
                positives.append(
                    {
                        "kind": "repost",
                        "a": originals[(family, seed)],
                        "b": fingerprint(photograph(mask, seed=seed, **style)),
                    }
                )

    negatives = []
    skipped = 0
    for family in FAMILIES:
        clean = {seed: clean_contour(family, seed) for seed in SEEDS}
        for s1, s2 in itertools.combinations(SEEDS, 2):
            truth = min(
                contour_distance(clean[s1], clean[s2]),
                contour_distance(clean[s1], _mirror(clean[s2])),
            )
            if truth < DISTINCT_MIN:  # the generator emitted the same design twice
                skipped += 1
                continue
            negatives.append(
                {"kind": "different", "a": originals[(family, s1)], "b": originals[(family, s2)]}
            )
    rng.shuffle(negatives)
    print(f"pairs: {len(positives)} positive, {len(negatives)} negative "
          f"({skipped} near-duplicate design pairs excluded from negatives)")
    return positives, negatives


def summarise(name: str, values: list[float]) -> None:
    arr = np.asarray(values)
    pct = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"  {name:22s} min={arr.min():6.2f}  p5={pct[0]:6.2f}  median={pct[2]:6.2f}  "
          f"p95={pct[4]:6.2f}  max={arr.max():6.2f}")


def main() -> int:
    positives, negatives = build_pairs()

    reshoot = [p for p in positives if p["kind"] == "reshoot"]
    repost = [p for p in positives if p["kind"] == "repost"]

    print("\ncontour distance (px on a 128 grid; lower = more alike)")
    summarise("same design, reshot", [shape_distance(p["a"], p["b"]) for p in reshoot])
    summarise("same photo, reposted", [shape_distance(p["a"], p["b"]) for p in repost])
    summarise("different designs", [shape_distance(p["a"], p["b"]) for p in negatives])

    print("\npHash hamming distance (bits of 64; lower = more alike)")
    summarise("same design, reshot", [float(hamming(p["a"].phash, p["b"].phash)) for p in reshoot])
    summarise("same photo, reposted", [float(hamming(p["a"].phash, p["b"].phash)) for p in repost])
    summarise("different designs", [float(hamming(p["a"].phash, p["b"].phash)) for p in negatives])

    print("\narea IoU (screening signal; higher = more alike)")
    summarise("same design, reshot", [shape_iou(p["a"].area_mask, p["b"].area_mask) for p in reshoot])
    summarise("different designs", [shape_iou(p["a"].area_mask, p["b"].area_mask) for p in negatives])

    pos_d = np.asarray([shape_distance(p["a"], p["b"]) for p in positives])
    neg_d = np.asarray([shape_distance(p["a"], p["b"]) for p in negatives])
    print("\nthreshold sweep on contour distance")
    print("  cutoff   recall   precision   flagged-per-1000-scanned")
    for cutoff in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        tp = int((pos_d <= cutoff).sum())
        fp = int((neg_d <= cutoff).sum())
        recall = tp / len(pos_d)
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(f"  {cutoff:5.1f}   {recall:6.1%}   {precision:9.1%}   "
              f"{1000 * fp / len(neg_d):22.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
