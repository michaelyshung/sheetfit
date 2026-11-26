"""Image operations for sheetfit (no state, no I/O).

The pipeline (see pipeline.py) runs, per image:
  1. load, apply EXIF orientation, convert to gray by taking the brightest
     colour channel (drops tinted paper, highlighter, coloured banners)
  2. flatten lighting: divide by an estimated paper background, removing
     phone-photo shadows and gradients
  3. blank page-edge shadows, neighbouring pages, curled corners and tables
  4. deskew: find the rotation that makes staff lines most horizontal
  5. crop to the music: largest block of ink, ignoring specks
  6. flatten page curl along guide lines that follow the staves (warp.py)
  7. boost contrast for monochrome printing: background to pure white,
     ink and faded lines to black
  8. measure staff-line spacing on every page (none = not music: kept as a
     plain photo) and scale pages so staff lines print the same size
  9. fit to the page (Letter, A4 or Legal) within margins
 10. write a multi-page PDF and/or one PNG per page
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


PAPER_IN = {"letter": (8.5, 11.0), "a4": (8.27, 11.69), "legal": (8.5, 14.0)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}

try:  # iPhone photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_OK = True
except ImportError:  # optional: everything else still works
    HEIF_OK = False


# ---------------------------------------------------------------- helpers
def otsu(a: np.ndarray) -> int:
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    w = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w - mu * total) ** 2 / (w * (total - w))
    return int(np.nanargmax(between))


def flatten_lighting(gray: Image.Image) -> Image.Image:
    """Divide out uneven illumination so paper becomes ~white everywhere."""
    w, h = gray.size
    f = max(1, max(w, h) // 600)
    small = gray.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR)
    # Max filter erases dark ink, leaving the paper; blur smooths it.
    bg = small.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(15))
    bg = np.asarray(bg.resize((w, h), Image.BILINEAR), dtype=np.float32)
    g = np.asarray(gray, dtype=np.float32)
    norm = np.clip(g / np.maximum(bg, 1.0) * 255.0, 0, 255)
    return Image.fromarray(norm.astype(np.uint8))


def ink_mask(gray: Image.Image) -> np.ndarray:
    a = np.asarray(gray)
    t = min(otsu(a), 200)
    return a < t


def working_copy(img: Image.Image, max_dim: int = 1000) -> tuple[Image.Image, float]:
    s = min(1.0, max_dim / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    return img, s


def to_gray(img: Image.Image, keep_color_marks: bool) -> Image.Image:
    """Grayscale that drops coloured backgrounds.

    Default takes the brightest RGB channel per pixel: black ink is dark in
    every channel and stays black, while tinted paper, highlighter, coloured
    title banners and pastel boxes are bright in at least one channel and go
    to white. With keep_color_marks, use normal luminance so coloured pencil
    or pen annotations survive as grey.
    """
    rgb = img.convert("RGB")
    if keep_color_marks:
        return rgb.convert("L")
    a = np.asarray(rgb)
    return Image.fromarray(a.max(axis=2))


def boost_contrast(gray: Image.Image, white_point: int, darken: float) -> Image.Image:
    """Levels + curve for monochrome printing.

    Everything at/above white_point becomes pure white (kills paper texture
    and leftover background). The darkest ink (2nd percentile of ink pixels)
    is pinned to pure black, and a gamma curve (darken > 1) pulls grey,
    faded or thin strokes toward black.
    """
    a = np.asarray(gray, dtype=np.float32)
    ink = a[a < otsu(a.astype(np.uint8))]
    black = float(np.percentile(ink, 2)) if ink.size else 0.0
    black = min(black, white_point - 40)
    x = np.clip((a - black) / (white_point - black), 0.0, 1.0)
    x = x ** darken
    return Image.fromarray((x * 255 + 0.5).astype(np.uint8))


# ---------------------------------------------------------------- deskew
def _profile_score(mask_img: Image.Image, angle: float) -> float:
    r = mask_img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=0)
    prof = np.asarray(r, dtype=np.float32).sum(axis=1)
    return float(np.dot(prof, prof))


def find_skew(gray: Image.Image, max_angle: float = 10.0) -> float:
    """Return the CCW rotation in degrees that levels the staff lines."""
    if max_angle <= 0:
        return 0.0
    small, _ = working_copy(gray, 1200)
    mask = Image.fromarray((ink_mask(small) * 255).astype(np.uint8))
    best = 0.0
    for step, span in ((0.5, max_angle), (0.1, 0.5), (0.02, 0.1)):
        center = best
        angles = np.arange(center - span, center + span + 1e-9, step)
        best = max(angles, key=lambda a: _profile_score(mask, float(a)))
    return float(round(best, 2))


# ---------------------------------------------------------------- crop
def _main_span(active: np.ndarray, weight: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """Merge runs of active indices separated by < max_gap; return heaviest span."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > max_gap)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    masses = [weight[s:e + 1].sum() for s, e in zip(starts, ends)]
    k = int(np.argmax(masses))
    return int(starts[k]), int(ends[k])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    d = np.diff(np.r_[0, flags.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _edge_mask(sm: np.ndarray, thresh: float, edge_frac: float,
               grey_band: bool, grow: int, flood_thresh: float | None = None) -> np.ndarray:
    """Margin pixels (at working scale) that are not paper. See whiten_dark_edges."""
    h, w = sm.shape
    mask = np.zeros((h, w), dtype=bool)
    for axis in (0, 1):  # 0: columns, 1: rows
        med = np.median(sm, axis=axis)
        n = med.size
        e = max(1, int(n * edge_frac))
        blank = np.zeros(n, dtype=bool)
        dark = med < thresh
        for idx in (np.arange(e), np.arange(n - 1, n - 1 - e, -1)):
            k = int(np.argmin(dark[idx])) if not dark[idx].all() else e
            blank[idx[:k]] = True
        if grey_band and axis == 0:
            grey = (med >= 120) & (med < thresh)
            grey[e:n - e] = False
            for r0, r1 in _runs(grey):
                if r1 - r0 >= 3:
                    blank[r0:r1] = True
        for r0, r1 in _runs(blank):
            sl = slice(max(0, r0 - 2), r1 + 2)
            if axis == 0:
                mask[:, sl] = True
            else:
                mask[sl] = True

    # Thick dark blobs connected to the photo border. Opening first removes
    # thin lines, so a staff running off the edge can't drag the fill into
    # the music.
    sm2 = np.where(mask, 0, sm)  # already-blanked areas conduct the fill
    ft = thresh if flood_thresh is None else flood_thresh
    dark = Image.fromarray(((sm2 < ft) * 255).astype(np.uint8))
    dark = dark.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    framed = ImageOps.expand(dark, 1, fill=255)
    ImageDraw.floodfill(framed, (0, 0), 128)
    fl = np.asarray(framed)[1:-1, 1:-1] == 128
    eh, ew = int(h * edge_frac), int(w * edge_frac)
    fl[eh:h - eh, ew:w - ew] = False
    mask |= fl
    if grow > 1 and mask.any():
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                          .filter(ImageFilter.MaxFilter(grow))) > 0
    return mask


def whiten_dark_edges(flat: Image.Image, raw: Image.Image | None = None,
                      edge_frac: float = 0.15, thresh: int = 212) -> Image.Image:
    """Paint page-edge shadows, neighbouring pages, page curl and table white.

    Only the outer `edge_frac` of each side is touched. Tests, using the
    *median* of each column/row (music is mostly white between staff lines,
    so music never has a dark median):
      * dark bands touching the image border;
      * columns only: bands of >= 3 consecutive mid-grey columns -- the soft
        shadow along a page edge (rows skip this: a row along a staff line
        has a grey median);
      * thick dark blobs connected to the border (curled page corners).
    Run on the flattened image, and on the raw image with a threshold
    relative to paper brightness, because flattening turns a big uniform
    table nearly white and only the raw image still shows it as dark.
    """
    a = np.asarray(flat).copy()
    small, _ = working_copy(flat, 1200)
    mask = _edge_mask(np.asarray(small), thresh, edge_frac, True, 1)
    if raw is not None:
        rs = np.asarray(working_copy(raw, 1200)[0])
        paper = float(np.percentile(rs, 90))
        # bands: < 60% of paper brightness; blobs: < 45% (only a real
        # table/background is that dark -- page shadows are ~55-60%)
        mask |= _edge_mask(rs, 0.6 * paper, edge_frac, False, 9, 0.45 * paper)
    if mask.any():
        full = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                          .resize(flat.size, Image.NEAREST)) > 0
        a[full] = 255
    return Image.fromarray(a)


def find_content_box(gray: Image.Image, gap_frac: float = 0.06,
                     border_frac: float = 0.015) -> tuple[int, int, int, int]:
    small, s = working_copy(gray, 1200)
    m = ink_mask(small).astype(np.float32)
    h, w = m.shape
    bh, bw = int(h * border_frac), int(w * border_frac)
    m[:bh] = m[h - bh:] = 0
    m[:, :bw] = m[:, w - bw:] = 0

    rows = m.mean(axis=1)
    rspan = _main_span(rows > 0.003, rows, int(h * gap_frac))
    if rspan is None:
        return (0, 0, gray.width, gray.height)
    r0, r1 = rspan
    # Pull in small, nearby lines of ink: page numbers and footers below the
    # music; above it only text-height lines (a title, a header), so a stray
    # speck or dash near the top edge isn't mistaken for content.
    near, small_h, text_h = int(h * 0.12), int(h * 0.04), max(3, int(h * 0.008))
    for a0, a1 in _runs(rows > 0.0005):
        a1 -= 1
        if a1 - a0 > small_h:
            continue
        if 0 < r0 - a1 <= near and a1 - a0 + 1 >= text_h:
            r0 = a0
        elif 0 < a0 - r1 <= near:
            r1 = a1
    cols = m[r0:r1 + 1].mean(axis=0)
    cspan = _main_span(cols > 0.003, cols, int(w * gap_frac))
    c0, c1 = cspan if cspan else (0, w - 1)

    pad = int(0.01 * max(h, w))
    box = (max(0, c0 - pad), max(0, r0 - pad), min(w, c1 + 1 + pad), min(h, r1 + 1 + pad))
    return tuple(round(v / s) for v in box)  # type: ignore[return-value]


# ---------------------------------------------------------------- staff size
def staff_spacing(gray: Image.Image) -> float | None:
    """Distance between neighbouring staff lines in pixels, or None if the
    page doesn't look like music.

    Staff lines make the row-darkness profile strongly periodic. Remove the
    slow trend, autocorrelate, and look for a clear peak at a plausible lag.
    """
    a = 255.0 - np.asarray(gray, dtype=np.float32)
    h, w = a.shape
    a = a[:, w // 10: w - w // 10]
    prof = (a > 100).mean(axis=1)
    lo, hi = max(3, int(h * 0.003)), max(8, int(h * 0.03))
    if prof.size < 4 * hi:
        return None
    k = np.ones(3 * hi) / (3 * hi)
    hp = prof - np.convolve(prof, k, mode="same")
    ac = np.array([np.dot(hp[:-L], hp[L:]) for L in range(1, hi + 2)])
    ac0 = float(np.dot(hp, hp))
    if ac0 <= 0:
        return None
    ac /= ac0  # ac[i] is lag i + 1
    best = None
    for L in range(lo, hi + 1):
        v = ac[L - 1]
        if v > ac[L - 2] and v >= ac[L] and v > 0.3:
            if best is None or v > ac[best - 1]:
                best = L
    if best is None:
        return None
    l, c, r = ac[best - 2], ac[best - 1], ac[best]
    den = l - 2 * c + r
    return best + (0.5 * (l - r) / den if den < 0 else 0.0)


# ---------------------------------------------------------------- fit
def _page_geometry(img: Image.Image, dpi: int, margin_in: float, orientation: str,
                   paper: str) -> tuple[int, int, int, float]:
    """(page_w, page_h, margin, largest scale that fits) in output pixels."""
    w_in, h_in = PAPER_IN[paper]
    pw, ph = round(w_in * dpi), round(h_in * dpi)
    if orientation == "landscape" or (orientation == "auto" and img.width > img.height):
        pw, ph = ph, pw
    m = round(margin_in * dpi)
    return pw, ph, m, min((pw - 2 * m) / img.width, (ph - 2 * m) / img.height)


def fit_to_page(img: Image.Image, dpi: int, margin_in: float, orientation: str,
                align: str, paper: str = "letter", scale: float | None = None,
                offset_in: tuple[float, float] = (0.0, 0.0)) -> Image.Image:
    """Place `img` on a blank page. Default: largest size that fits the
    margins, centred horizontally, top- or centre-aligned. `scale` (output
    px per image px) and `offset_in` (inches) override size and position.
    """
    pw, ph, m, fit = _page_geometry(img, dpi, margin_in, orientation, paper)
    scale = fit if scale is None else scale
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    fitted = img.resize((nw, nh), Image.LANCZOS)
    page = Image.new("L", (pw, ph), 255)
    x = (pw - nw) // 2 + round(offset_in[0] * dpi)
    y = (m if align == "top" else (ph - nh) // 2) + round(offset_in[1] * dpi)
    page.paste(fitted, (x, y))
    return page
