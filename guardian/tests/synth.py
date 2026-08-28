"""Synthetic stand-ins for catalog photos and for copies of them.

Real calibration needs real photographs, but the decisions the matcher has to
get right are reproducible without them: the same design shot against a
different wall must score high, and a *different* design from the same family
must not. These renderers make both cases on demand.
"""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw, ImageFilter

CANVAS = 900


def _mountain(rng: random.Random) -> list[list[tuple[float, float]]]:
    peaks = rng.randint(2, 4)
    pts: list[tuple[float, float]] = [(0.0, 1.0)]
    for i in range(peaks):
        base = (i + 0.5) / peaks
        pts.append((base - 0.5 / peaks, 1.0 - rng.uniform(0.15, 0.35)))
        pts.append((base, 1.0 - rng.uniform(0.55, 0.95)))
    pts.append((1.0, 1.0 - rng.uniform(0.15, 0.3)))
    pts.append((1.0, 1.0))
    return [pts]


def _creature(rng: random.Random) -> list[list[tuple[float, float]]]:
    lobes = rng.randint(5, 9)
    radii = [rng.uniform(0.25, 0.5) for _ in range(lobes)]
    pts = []
    for i in range(lobes * 6):
        angle = 2 * math.pi * i / (lobes * 6)
        radius = radii[i % lobes] * (1 + 0.25 * math.sin(lobes * angle))
        pts.append((0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)))
    return [pts]


def _botanical(rng: random.Random) -> list[list[tuple[float, float]]]:
    shapes = []
    stems = rng.randint(3, 5)
    for i in range(stems):
        x = 0.5 + (i - stems / 2) * rng.uniform(0.09, 0.14)
        top = rng.uniform(0.12, 0.35)
        width = rng.uniform(0.02, 0.045)
        shapes.append([(x - width, 0.95), (x - width * 0.4, top), (x + width * 0.4, top), (x + width, 0.95)])
        for leaf in range(rng.randint(1, 3)):
            ly = top + (0.95 - top) * (leaf + 1) / 4
            side = 1 if (leaf + i) % 2 else -1
            span = rng.uniform(0.07, 0.13)
            shapes.append(
                [(x, ly), (x + side * span, ly - 0.05), (x + side * span * 0.7, ly + 0.05)]
            )
    return shapes


FAMILIES = {"mountain": _mountain, "creature": _creature, "botanical": _botanical}


def design_mask(family: str, seed: int, size: int = CANVAS) -> Image.Image:
    """The design itself: white artwork on black, no photographic context."""
    rng = random.Random(seed)
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for shape in FAMILIES[family](rng):
        draw.polygon([(x * size, y * size) for x, y in shape], fill=255)
    return canvas


def photograph(
    mask: Image.Image,
    *,
    seed: int = 0,
    wall: tuple[int, int, int] | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
    mirror: bool = False,
    crop: float = 0.0,
    blur: float = 0.6,
    noise: int = 6,
    jpeg_quality: int | None = None,
    watermark: str | None = None,
) -> Image.Image:
    """Render a design as if photographed hanging on somebody's wall."""
    rng = random.Random(seed)
    wall = wall or (rng.randint(205, 245),) * 3
    out_size = 700
    scene = Image.new("RGB", (out_size, out_size), wall)

    art = mask
    if mirror:
        art = art.transpose(Image.FLIP_LEFT_RIGHT)
    if rotate:
        art = art.rotate(rotate, resample=Image.BICUBIC, expand=True, fillcolor=0)
    target = max(40, int(out_size * 0.72 * scale))
    art = art.resize((target, target), Image.LANCZOS)

    offset = ((out_size - target) // 2, (out_size - target) // 2)
    ink = rng.randint(12, 45)
    # Shading scales the wall's own luminance; a cast shadow is never darker
    # than the black steel throwing it.
    shade = max(ink + 8, int(wall[0] * 0.78))
    shadow = Image.new("RGB", art.size, (shade,) * 3)
    scene.paste(shadow, (offset[0] + 6, offset[1] + 8), art.filter(ImageFilter.GaussianBlur(4)))
    scene.paste(Image.new("RGB", art.size, (ink, ink, ink + rng.randint(0, 6))), offset, art)

    if blur:
        scene = scene.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        px = scene.load()
        for _ in range(out_size * out_size // 40):
            x, y = rng.randrange(out_size), rng.randrange(out_size)
            r, g, b = px[x, y]
            d = rng.randint(-noise, noise)
            px[x, y] = (max(0, min(255, r + d)), max(0, min(255, g + d)), max(0, min(255, b + d)))
    if crop:
        inset = int(out_size * crop)
        scene = scene.crop((inset, inset, out_size - inset, out_size - inset))
    if watermark:
        draw = ImageDraw.Draw(scene)
        draw.text((12, scene.height - 24), watermark, fill=(255, 90, 90))
    if jpeg_quality:
        import io

        buf = io.BytesIO()
        scene.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        scene = Image.open(buf).convert("RGB")
    return scene
