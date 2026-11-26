"""Photos -> page recipes -> finished pages.

`analyze` runs every automatic step on one photo and records the decisions
in a PageRecipe. `render` turns a recipe (automatic or hand-edited) back
into the page content, and `layout` places all pages on paper. The CLI uses
analyze + layout; the GUI lets you edit recipes between the two; a saved
project is rendered with render + layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .core import (
    _page_geometry,
    boost_contrast,
    find_content_box,
    find_skew,
    fit_to_page,
    flatten_lighting,
    staff_spacing,
    to_gray,
    whiten_dark_edges,
)
from .recipe import PageRecipe, Settings
from .warp import apply_guides, auto_guides


# ---------------------------------------------------------------- frames
def box_to_rel(box, size) -> list[float]:
    """Absolute pixel box in an image of `size` -> centre-relative."""
    cx, cy = size[0] / 2, size[1] / 2
    return [box[0] - cx, box[1] - cy, box[2] - cx, box[3] - cy]


def box_to_abs(box, size) -> tuple[int, int, int, int]:
    cx, cy = size[0] / 2, size[1] / 2
    return (max(0, round(box[0] + cx)), max(0, round(box[1] + cy)),
            min(size[0], round(box[2] + cx)), min(size[1], round(box[3] + cy)))


def _shift_guides(guides, dx: float, dy: float):
    for g in guides:
        g.points = [[x + dx, y + dy] for x, y in g.points]
    return guides


# ---------------------------------------------------------------- stages
@dataclass
class Source:
    """One photo, loaded once and cleaned once (the slow, rarely-changed part).

    The GUI keeps these around so editing later steps is quick.
    """
    path: str
    rotate90: int
    rgb: Image.Image
    clean: Image.Image                     # cleaned gray, not yet straightened
    _key: tuple = field(default=(), repr=False)


def load_source(path: str | Path, rotate90: int, settings: Settings) -> Source:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    if rotate90 % 4:
        rgb = rgb.rotate(-90 * (rotate90 % 4), expand=True)
    gray = to_gray(rgb, settings.keep_color_marks)
    raw = gray
    if settings.flatten_lighting:
        gray = flatten_lighting(gray)
    if settings.clean_edges:
        gray = whiten_dark_edges(gray, raw)
    key = (rotate90 % 4, settings.keep_color_marks, settings.flatten_lighting, settings.clean_edges)
    return Source(str(path), rotate90 % 4, rgb, gray, key)


def straighten(img: Image.Image, angle: float) -> Image.Image:
    if not angle:
        return img
    fill = 255 if img.mode == "L" else (255, 255, 255)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)


def flattened(src: Source, recipe: PageRecipe) -> Image.Image:
    """Cleaned, straightened and flattened, before cropping."""
    img = straighten(src.clean, recipe.angle)
    img, _ = apply_guides(img, recipe.guides, recipe.flatten_strength)
    return img


# ---------------------------------------------------------------- analyze
def analyze(path: str | Path, settings: Settings, rotate90: int = 0,
            src: Source | None = None) -> tuple[PageRecipe, Image.Image | None, Source]:
    """Run every automatic step. Returns (recipe, finished content, source).

    Content is None when the page is skipped (non-music with non_music=skip).
    """
    src = src or load_source(path, rotate90, settings)
    r = PageRecipe(source=str(path), rotate90=src.rotate90)

    r.angle = find_skew(src.clean, settings.max_angle)
    straight = straighten(src.clean, r.angle)
    box = find_content_box(straight)
    content = straight.crop(box)

    spacing = staff_spacing(content)
    if spacing is None and settings.non_music != "music":
        r.kind = "photo"
        r.angle = 0.0
        if settings.non_music == "skip":
            r.include = False
            return r, None, src
        return r, render(r, settings, src), src

    r.guides = auto_flatten_guides(straight, box)
    flat, _ = apply_guides(straight, r.guides)
    r.crop = auto_crop(flat)
    final = flat.crop(box_to_abs(r.crop, flat.size))
    r.staff_spacing = staff_spacing(final) or spacing
    return r, _contrast(final, settings), src


def auto_flatten_guides(straight: Image.Image, box=None) -> list:
    """Automatic guides for a straightened page, in its centre-relative frame.

    Measured on the content box (found if not given), where the staves are.
    """
    box = box or find_content_box(straight)
    guides = auto_guides(straight.crop(box))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return _shift_guides(guides, cx - straight.width / 2, cy - straight.height / 2)


def auto_crop(flat: Image.Image) -> list[float]:
    """Automatic crop box for a flattened page, centre-relative."""
    return box_to_rel(find_content_box(flat), flat.size)


# ---------------------------------------------------------------- render
def _contrast(img: Image.Image, settings: Settings) -> Image.Image:
    return boost_contrast(img, settings.white_point, settings.darken)


def render(recipe: PageRecipe, settings: Settings, src: Source | None = None,
           contrast: bool = True) -> Image.Image:
    """Page content (cropped, contrast applied) from a recipe."""
    if src is None or src.rotate90 != recipe.rotate90 % 4:
        src = load_source(recipe.source, recipe.rotate90, settings)
    if recipe.kind == "photo":
        img = straighten(src.rgb, recipe.angle)
        if recipe.crop:
            img = img.crop(box_to_abs(recipe.crop, img.size))
        return ImageOps.autocontrast(img.convert("L"), cutoff=0.5)
    img = flattened(src, recipe)
    if recipe.crop:
        img = img.crop(box_to_abs(recipe.crop, img.size))
    return _contrast(img, settings) if contrast else img


# ---------------------------------------------------------------- layout
def auto_scales(images: list[Image.Image], spacings: list[float | None],
                settings: Settings) -> list[float | None]:
    """Per-page scale so staff lines print at the same size on every page.

    Each music page could be scaled up to fit_i; its printed staff spacing
    would then be fit_i * spacing_i. Use the smallest of those for everyone
    so every page fits. Pages whose spacing looks misdetected (far smaller
    than the rest) don't drag the others down; they just fit themselves.
    """
    if not settings.uniform_scale:
        return [None] * len(images)
    fits = [_page_geometry(im, settings.dpi, settings.margin, settings.orientation,
                           settings.paper)[3] for im in images]
    printed = [f * s for f, s in zip(fits, spacings) if s]
    if len(printed) < 2:
        return [None] * len(images)
    target = min(v for v in printed if v >= 0.75 * float(np.median(printed)))
    return [target / s if s else None for s in spacings]


def page_scale(img: Image.Image, auto: float | None, recipe: PageRecipe,
               settings: Settings) -> float:
    """Output px per content px for one page, including the manual tweak."""
    fit = _page_geometry(img, settings.dpi, settings.margin, settings.orientation,
                         settings.paper)[3]
    return (fit if auto is None else auto) * recipe.scale_adjust


def layout(recipes: list[PageRecipe], images: list[Image.Image],
           settings: Settings) -> list[Image.Image]:
    """Place each page's content on paper."""
    spacings = [r.staff_spacing if r.kind == "music" else None for r in recipes]
    autos = auto_scales(images, spacings, settings)
    pages = []
    for r, img, auto in zip(recipes, images, autos):
        page = fit_to_page(img, settings.dpi, settings.margin, settings.orientation,
                           settings.align, settings.paper,
                           page_scale(img, auto, r, settings), tuple(r.offset_in))
        if settings.bw:
            t = settings.bw_threshold
            page = page.point(lambda v: 255 if v > t else 0).convert("1")
        pages.append(page)
    return pages
