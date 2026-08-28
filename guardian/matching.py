"""Decide whether a marketplace listing is showing one of our designs.

Thresholds come from guardian/tests/calibrate.py rather than from intuition.
On its synthetic set, a contour distance of 2.5px caught 89% of copies with
no false positives, and 3.0px caught 95% at 99.7% precision — so 2.5 is the
line for "this is ours" and everything out to 3.5 is worth a human glance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guardian.fingerprint import (
    AREA_SIZE,
    Fingerprint,
    contour_distance,
    hamming,
    shape_distance,
)

DESIGN_MATCH_PX = 2.5   # calibrated: 89% recall, 100% precision
REVIEW_PX = 3.5         # calibrated: 98% recall, 99% precision
IMAGE_REUSE_BITS = 10   # pHash/dHash distance that survives recompression
IMAGE_REUSE_SHAPE_PX = 4.0

SHORTLIST_AREA = 200    # cheap stage keeps this many, mirror-aware
SHORTLIST_CONTOUR = 10  # rotation search only runs on this many

VERDICT_IMAGE_REUSE = "image_reuse"
VERDICT_DESIGN_MATCH = "design_match"
VERDICT_REVIEW = "review"
VERDICT_CLEAR = "clear"


@dataclass
class Match:
    listing_id: str
    image_index: int
    verdict: str
    shape_px: float
    phash_bits: int
    dhash_bits: int

    @property
    def is_hit(self) -> bool:
        return self.verdict != VERDICT_CLEAR


def classify(shape_px: float, phash_bits: int, dhash_bits: int) -> str:
    """Rank the evidence: a reused photograph is the strongest claim we have.

    Hash agreement is checked together with shape because unrelated designs
    collided at four bits in calibration, and a hash alone would call that a
    theft.
    """
    hash_bits = min(phash_bits, dhash_bits)
    if hash_bits <= IMAGE_REUSE_BITS and shape_px <= IMAGE_REUSE_SHAPE_PX:
        return VERDICT_IMAGE_REUSE
    if shape_px <= DESIGN_MATCH_PX:
        return VERDICT_DESIGN_MATCH
    if shape_px <= REVIEW_PX:
        return VERDICT_REVIEW
    return VERDICT_CLEAR


class CatalogIndex:
    """Our own designs, held ready to compare a stranger's photo against."""

    def __init__(self, entries: list[tuple[str, int, Fingerprint]]):
        self.entries = entries
        if entries:
            masks = np.stack([e[2].area_mask.ravel() for e in entries]).astype(np.float32)
        else:
            masks = np.zeros((0, AREA_SIZE * AREA_SIZE), dtype=np.float32)
        self._masks = masks
        self._areas = masks.sum(axis=1)
        self._phash = np.array([e[2].phash for e in entries], dtype=object)
        self._dhash = np.array([e[2].dhash for e in entries], dtype=object)

    def __len__(self) -> int:
        return len(self.entries)

    def _area_scores(self, probe: Fingerprint) -> np.ndarray:
        """Vectorised IoU against every catalog image, better of two mirrorings.

        This stage only has to keep the real match somewhere in the shortlist;
        the contour stages below do the deciding.
        """
        best = np.zeros(len(self.entries), dtype=np.float32)
        for candidate in (probe.area_mask, np.fliplr(probe.area_mask)):
            vec = candidate.ravel().astype(np.float32)
            intersection = self._masks @ vec
            union = self._areas + vec.sum() - intersection
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = np.where(union > 0, intersection / union, 0.0)
            best = np.maximum(best, iou)
        return best

    def best_match(self, probe: Fingerprint) -> Match | None:
        """The closest design we own, with the verdict for it."""
        if not self.entries:
            return None

        order = np.argsort(-self._area_scores(probe))[:SHORTLIST_AREA]

        # Contour distance without the rotation search: cheap, and enough to
        # rank. The expensive aligned comparison then runs on the survivors.
        coarse = sorted(
            (
                (
                    min(
                        contour_distance(probe.contour, self.entries[i][2].contour),
                        contour_distance(probe.contour, self.entries[i][2].contour_mirrored),
                    ),
                    int(i),
                )
                for i in order
            )
        )[:SHORTLIST_CONTOUR]

        best: Match | None = None
        for _, idx in coarse:
            listing_id, image_index, fp = self.entries[idx]
            shape_px = shape_distance(probe, fp)
            phash_bits = hamming(probe.phash, fp.phash)
            dhash_bits = hamming(probe.dhash, fp.dhash)
            match = Match(
                listing_id=listing_id,
                image_index=image_index,
                verdict=classify(shape_px, phash_bits, dhash_bits),
                shape_px=shape_px,
                phash_bits=phash_bits,
                dhash_bits=dhash_bits,
            )
            if best is None or match.shape_px < best.shape_px:
                best = match
        return best

    def to_json(self) -> dict:
        return {
            "version": 1,
            "entries": [
                {"listing_id": lid, "image_index": idx, **fp.to_json()}
                for lid, idx, fp in self.entries
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "CatalogIndex":
        return cls(
            [
                (row["listing_id"], row["image_index"], Fingerprint.from_json(row))
                for row in data["entries"]
            ]
        )
