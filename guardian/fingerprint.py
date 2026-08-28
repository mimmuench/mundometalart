"""Visual fingerprints for laser-cut metal wall art.

Two different thefts need two different signals:

* Someone reposts our photograph. Perceptual hashes (pHash/dHash) catch that
  through re-compression, resizing, mild recolouring and watermarking.
* Someone rebuilds our design and shoots their own photograph. No hash of the
  photograph survives that, because every pixel is different — but the design
  itself is a silhouette, so we normalise the artwork's shape and compare
  shapes instead. That is the signal that matters for this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass

import imagehash
import numpy as np
from PIL import Image, ImageOps

SILHOUETTE_SIZE = 128  # shape grid; the contour detail lives here
AREA_SIZE = 64       # coarser grid, used only for cheap screening
MAX_CONTOUR_POINTS = 256
HASH_SIZE = 8  # 64-bit pHash/dHash, the size the usual distance thresholds assume
WORK_SIZE = 512


def load_image(path_or_file) -> Image.Image:
    """Open an image, honour EXIF rotation, and bound its working size."""
    img = Image.open(path_or_file)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    return img


def otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu: the 0-255 cut that best separates the histogram."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    weight_bg = np.cumsum(hist) / total
    mean_cum = np.cumsum(hist * np.arange(256)) / total
    mean_total = mean_cum[-1]
    denom = weight_bg * (1.0 - weight_bg)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean_total * weight_bg - mean_cum) ** 2 / denom
    between[~np.isfinite(between)] = -1.0
    return int(np.argmax(between))


def _ink_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of the artwork, separated from whatever it hangs on.

    Polarity is decided by which side owns the frame's border rather than by
    which side is darker or smaller. Black steel on a white wall and the same
    piece shot against a charcoal wall are both common, and brightness- or
    area-based rules pick the wall in one of those two cases.
    """
    gray = np.asarray(ImageOps.autocontrast(img.convert("L")), dtype=np.uint8)
    threshold = otsu_threshold(gray)
    dark = gray <= threshold

    border = np.concatenate([dark[0, :], dark[-1, :], dark[:, 0], dark[:, -1]])
    mask = ~dark if border.mean() > 0.5 else dark
    if not mask.any() or mask.all():
        mask = dark if dark.mean() <= 0.5 else ~dark
    return mask


def largest_component(mask: np.ndarray, work: int = 256) -> np.ndarray:
    """Keep the biggest connected blob and drop everything else.

    Sellers stamp watermarks and shop logos onto stolen photos, and those
    marks land far from the artwork. Left in, they widen the bounding box we
    normalise against and wreck the shape — a watermark alone was enough to
    hide a copy in calibration. Labelling runs on a reduced grid, which is
    ample for deciding which blob is the piece.
    """
    if not mask.any():
        return mask
    small = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8), "L").resize(
            (work, work), Image.NEAREST
        )
    ) > 127
    remaining = small.copy()
    best = None
    best_size = 0
    while remaining.any():
        ys, xs = np.nonzero(remaining)
        blob = np.zeros_like(remaining)
        blob[ys[0], xs[0]] = True
        while True:
            grown = blob.copy()
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                grown |= np.roll(blob, shift, axis=axis)
            grown &= remaining
            if grown.sum() == blob.sum():
                break
            blob = grown
        size = int(blob.sum())
        if size > best_size:
            best_size, best = size, blob
        remaining &= ~blob
    if best is None:
        return mask
    full = np.asarray(
        Image.fromarray((best * 255).astype(np.uint8), "L").resize(
            (mask.shape[1], mask.shape[0]), Image.NEAREST
        )
    ) > 127
    return mask & full


def silhouette(img: Image.Image, size: int = SILHOUETTE_SIZE) -> np.ndarray:
    """Scale- and crop-invariant shape of the artwork, as a size x size mask.

    The mask is cropped to the artwork's bounding box before rescaling, so a
    copy photographed closer, further away or cropped differently normalises
    onto the same grid. Aspect ratio is deliberately discarded here and kept
    as a separate weak signal.
    """
    mask = largest_component(_ink_mask(img))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((size, size), dtype=bool)
    cropped = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    tile = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")
    tile = tile.resize((size, size), Image.BILINEAR)
    return np.asarray(tile) > 127


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """Seven log-scaled Hu invariants: stable under scale and rotation."""
    binary = mask.astype(np.float64)
    total = binary.sum()
    if total == 0:
        return np.zeros(7)
    ys, xs = np.nonzero(binary)
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy

    def mu(p: int, q: int) -> float:
        return float(np.sum((dx**p) * (dy**q)))

    def nu(p: int, q: int) -> float:
        return mu(p, q) / (total ** (1.0 + (p + q) / 2.0))

    n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
    n30, n03, n21, n12 = nu(3, 0), nu(0, 3), nu(2, 1), nu(1, 2)
    h = np.empty(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11**2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * (
        (n30 + n12) ** 2 - 3 * (n21 + n03) ** 2
    ) + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (
        n30 + n12
    ) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * (
        (n30 + n12) ** 2 - 3 * (n21 + n03) ** 2
    ) - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    return np.sign(h) * np.log10(np.abs(h) + 1e-30)


def boundary(mask: np.ndarray) -> np.ndarray:
    """The design's outline: mask pixels that touch a non-mask neighbour."""
    interior = mask.copy()
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        interior &= np.roll(mask, shift, axis=axis)
    return mask & ~interior


def contour_points(mask: np.ndarray, limit: int = MAX_CONTOUR_POINTS) -> np.ndarray:
    """Outline pixels as (y, x) coordinates, evenly thinned to `limit`.

    Area overlap is the wrong measure for this catalog: two unrelated filled
    silhouettes of similar bulk overlap almost perfectly, so the peaks that
    tell one mountain design from another get outvoted by the mass beneath
    them. The outline is where a design's identity actually is.
    """
    ys, xs = np.nonzero(boundary(mask))
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if ys.size > limit:
        idx = np.linspace(0, ys.size - 1, limit).astype(int)
        ys, xs = ys[idx], xs[idx]
    return np.stack([ys, xs], axis=1).astype(np.float64)


def contour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean chamfer distance between two outlines, in grid pixels."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float(SILHOUETTE_SIZE)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


@dataclass(frozen=True)
class Fingerprint:
    """Everything we keep about one image. No pixels are retained."""

    phash: int
    dhash: int
    area_bits: bytes
    contour: np.ndarray
    hu: tuple[float, ...]
    aspect: float
    ink_ratio: float

    @property
    def area_mask(self) -> np.ndarray:
        flat = np.unpackbits(np.frombuffer(self.area_bits, dtype=np.uint8))
        return flat.reshape(AREA_SIZE, AREA_SIZE).astype(bool)

    @property
    def contour_mirrored(self) -> np.ndarray:
        """The same outline seen in a mirror; copies are routinely flipped."""
        if self.contour.shape[0] == 0:
            return self.contour
        flipped = self.contour.copy()
        flipped[:, 1] = (SILHOUETTE_SIZE - 1) - flipped[:, 1]
        return flipped

    def to_json(self) -> dict:
        return {
            "phash": f"{self.phash:016x}",
            "dhash": f"{self.dhash:016x}",
            "area": self.area_bits.hex(),
            "contour": self.contour.astype(np.uint8).tobytes().hex(),
            "hu": list(self.hu),
            "aspect": self.aspect,
            "ink_ratio": self.ink_ratio,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Fingerprint":
        return cls(
            phash=int(data["phash"], 16),
            dhash=int(data["dhash"], 16),
            area_bits=bytes.fromhex(data["area"]),
            contour=np.frombuffer(bytes.fromhex(data["contour"]), dtype=np.uint8)
            .reshape(-1, 2)
            .astype(np.float64),
            hu=tuple(data["hu"]),
            aspect=float(data["aspect"]),
            ink_ratio=float(data["ink_ratio"]),
        )


def fingerprint(img: Image.Image) -> Fingerprint:
    mask = silhouette(img, SILHOUETTE_SIZE)
    coarse = silhouette(img, AREA_SIZE)
    return Fingerprint(
        phash=int(str(imagehash.phash(img, hash_size=HASH_SIZE)), 16),
        dhash=int(str(imagehash.dhash(img, hash_size=HASH_SIZE)), 16),
        area_bits=np.packbits(coarse).tobytes(),
        contour=contour_points(mask),
        hu=tuple(hu_moments(mask)),
        aspect=img.width / img.height,
        ink_ratio=float(mask.mean()),
    )


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def shape_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Area overlap, better of the two mirrorings. Screening only — see
    contour_points() for why this is not the deciding signal."""
    best = 0.0
    for candidate in (b, np.fliplr(b)):
        union = np.logical_or(a, candidate).sum()
        if union == 0:
            continue
        best = max(best, float(np.logical_and(a, candidate).sum()) / float(union))
    return best


def rotate_contour(points: np.ndarray, degrees: float, size: int = SILHOUETTE_SIZE) -> np.ndarray:
    """Spin an outline about its own centre, for the alignment search."""
    if points.shape[0] == 0 or degrees == 0.0:
        return points
    theta = np.radians(degrees)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = points.mean(axis=0)
    return (points - centre) @ rot.T + centre


def shape_distance(
    a: "Fingerprint", b: "Fingerprint", angles: tuple[float, ...] = (-6, -4, -2, 0, 2, 4, 6)
) -> float:
    """Best contour distance over mirroring and a small rotation search.

    A copy is photographed freehand, so it hangs a couple of degrees off
    square; without the search a three-degree tilt looks as different as an
    unrelated design.
    """
    best = float(SILHOUETTE_SIZE)
    for reference in (b.contour, b.contour_mirrored):
        for angle in angles:
            best = min(best, contour_distance(a.contour, rotate_contour(reference, angle)))
    return best


def hu_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())
