"""sheetfit — straighten, clean, crop and fit photos/scans of sheet music to printable pages.

Pipeline per image:
  1. load, apply EXIF orientation, convert to gray by taking the brightest
     colour channel (drops tinted paper, highlighter, coloured banners)
  2. flatten lighting: divide by an estimated paper background, removing
     phone-photo shadows and gradients
  3. blank page-edge shadows, neighbouring pages, curled corners and tables
  4. deskew: find the rotation that makes staff lines most horizontal
  5. crop to the music: largest block of ink, ignoring specks
  6. dewarp: measure how staff lines bend across the page (page curl near a
     book's spine) and shift each pixel column so they run straight
  7. boost contrast for monochrome printing: background to pure white,
     ink and faded lines to black
  8. fit to the page (Letter, A4 or Legal) within margins
  9. write a multi-page PDF and/or one PNG per page

Only needs numpy + Pillow.

Usage:
  sheetfit IMG [IMG ...] -o out.pdf
  sheetfit photos/ -o book.pdf --png-dir pages/ --bw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from . import __version__

PAPER_IN = {"letter": (8.5, 11.0), "a4": (8.27, 11.69), "legal": (8.5, 14.0)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


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
    cols = m[r0:r1 + 1].mean(axis=0)
    cspan = _main_span(cols > 0.003, cols, int(w * gap_frac))
    c0, c1 = cspan if cspan else (0, w - 1)

    pad = int(0.01 * max(h, w))
    box = (max(0, c0 - pad), max(0, r0 - pad), min(w, c1 + 1 + pad), min(h, r1 + 1 + pad))
    return tuple(round(v / s) for v in box)  # type: ignore[return-value]


# ---------------------------------------------------------------- dewarp
def _band_shift(p: np.ndarray, q: np.ndarray, S: int) -> tuple[float, float]:
    """Vertical shift s (subpixel) such that q[y + s] best matches p[y].

    Returns (shift, confidence). Confidence is 0 when either profile is
    nearly empty (no staff in this strip/band).
    """
    n = p.size
    energy = min(p.std(), q.std())
    if energy < 4.0:
        return 0.0, 0.0
    p = p - p.mean()
    q = q - q.mean()
    scores = np.array([np.dot(p[max(0, -s):n - max(0, s)], q[max(0, s):n - max(0, -s)])
                       for s in range(-S, S + 1)])
    k = int(np.argmax(scores))
    sub = 0.0
    if 0 < k < 2 * S:  # parabola through the peak for subpixel accuracy
        l, c, r = scores[k - 1], scores[k], scores[k + 1]
        den = l - 2 * c + r
        if den < 0:
            sub = 0.5 * (l - r) / den
    elif k in (0, 2 * S):  # peak at the search limit: unreliable
        return 0.0, 0.0
    return k - S + sub, float(energy)


def dewarp(gray: Image.Image, n_strips: int = 40, n_bands: int = 10,
           deg_x: int = 4, deg_y: int = 2) -> tuple[Image.Image, float]:
    """Straighten curved staff lines (page curl near a book's spine).

    1. Cut the page into `n_strips` vertical strips and `n_bands` overlapping
       horizontal bands; in each cell take the row-darkness profile.
    2. Walking outward from the middle strip, find how far each strip's
       profile is shifted vertically versus its neighbour (cross-correlation;
       small steps, so staff lines can't be confused with their neighbours)
       and accumulate: D(strip, band) = how far the music has drifted.
    3. Fit a smooth surface d(x, y) = sum a_pq x^p y^q (deg 4 across the page,
       quadratic down it, so top and bottom can bend differently -- the
       "pinch").
    4. Resample every pixel column: out(x, y) = in(x, y + d(x, y)).

    Returns the new image and the largest displacement applied (pixels).
    """
    small, s = working_copy(gray, 1600)
    a = 255.0 - np.asarray(small, dtype=np.float32)
    a[a < 60] = 0  # ignore paper texture
    h, w = a.shape

    ink_cols = np.flatnonzero((a > 0).mean(axis=0) > 0.01)
    if ink_cols.size < 0.3 * w:
        return gray, 0.0
    x0, x1 = int(ink_cols[0]), int(ink_cols[-1]) + 1
    xe = np.linspace(x0, x1, n_strips + 1).astype(int)
    prof = np.stack([a[:, xe[i]:xe[i + 1]].mean(axis=1) for i in range(n_strips)])

    half = h / n_bands  # bands overlap by 50%
    centers = np.linspace(half, h - half, n_bands)
    S = max(2, round(h * 0.006))  # max drift between neighbouring strips

    D = np.zeros((n_strips, n_bands))
    W = np.zeros((n_strips, n_bands))
    mid = n_strips // 2
    for k, c in enumerate(centers):
        y0, y1 = int(c - half), int(c + half)
        ref_w = min(prof[mid, y0:y1].std(), 1e9)
        W[mid, k] = ref_w if ref_w >= 4.0 else 0.0
        for order in (range(mid + 1, n_strips), range(mid - 1, -1, -1)):
            for j in order:
                i = j - 1 if j > mid else j + 1  # neighbour closer to the middle
                # follow the drift: compare at the neighbour's shifted window
                off = int(round(D[i, k]))
                ya, yb = max(0, y0 + off), min(h, y1 + off)
                st, conf = _band_shift(prof[i, ya:yb], prof[j, ya:yb], S)
                D[j, k] = D[i, k] + st
                W[j, k] = min(W[i, k], conf) if conf else 0.0

    if (W > 0).sum() < deg_x * deg_y * 3:
        return gray, 0.0

    # Fit the smooth surface. Coordinates normalised to [-1, 1].
    xc = ((xe[:-1] + xe[1:]) / 2 - x0) / max(1, x1 - x0) * 2 - 1
    yc = centers / h * 2 - 1
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    terms = [(p, q) for p in range(deg_x + 1) for q in range(deg_y + 1)]
    A = np.stack([XX.ravel() ** p * YY.ravel() ** q for p, q in terms], axis=1)
    sw = np.sqrt(W.ravel())
    coef, *_ = np.linalg.lstsq(A * sw[:, None], D.ravel() * sw, rcond=None)

    # Residual check: if the model doesn't explain the measurements, skip.
    good = W.ravel() > 0
    resid = np.abs(A[good] @ coef - D.ravel()[good])
    if np.median(resid) > 1.5:
        return gray, 0.0

    # Evaluate at full resolution, column by column.
    g = np.asarray(gray, dtype=np.float32)
    H, Wf = g.shape
    Y = np.arange(H, dtype=np.float32)
    yn = (Y * s) / h * 2 - 1
    xs = np.clip(((np.arange(Wf) * s) - x0) / max(1, x1 - x0) * 2 - 1, -1, 1)
    out = np.empty_like(g)
    max_d = 0.0
    for x in range(Wf):
        xp = xs[x]
        d = np.zeros(H, dtype=np.float32)
        for (p, q), cf in zip(terms, coef):
            d += cf * xp ** p * yn ** q
        d /= s  # working px -> full-res px
        max_d = max(max_d, float(np.abs(d).max()))
        out[:, x] = np.interp(Y + d, Y, g[:, x], left=255, right=255)
    if max_d > 0.08 * H:  # implausible -- don't trust it
        return gray, 0.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), max_d


# ---------------------------------------------------------------- fit
def fit_to_page(img: Image.Image, dpi: int, margin_in: float, orientation: str,
                align: str, paper: str = "letter") -> Image.Image:
    w_in, h_in = PAPER_IN[paper]
    pw, ph = round(w_in * dpi), round(h_in * dpi)
    if orientation == "landscape" or (orientation == "auto" and img.width > img.height):
        pw, ph = ph, pw
    m = round(margin_in * dpi)
    aw, ah = pw - 2 * m, ph - 2 * m
    scale = min(aw / img.width, ah / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    fitted = img.resize((nw, nh), Image.LANCZOS)
    page = Image.new("L", (pw, ph), 255)
    x = (pw - nw) // 2
    y = m if align == "top" else (ph - nh) // 2
    page.paste(fitted, (x, y))
    return page


# ---------------------------------------------------------------- pipeline
def process(path: Path, args: argparse.Namespace) -> Image.Image:
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im)
        gray = to_gray(img, args.keep_color_marks)
    if args.rotate:
        gray = gray.rotate(-args.rotate, expand=True, fillcolor=255)

    raw = gray
    if not args.no_flatten:
        gray = flatten_lighting(gray)

    if not args.no_crop:
        gray = whiten_dark_edges(gray, raw)

    angle = 0.0 if args.no_deskew else find_skew(gray, args.max_angle)
    if angle:
        gray = gray.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    box = (0, 0, gray.width, gray.height) if args.no_crop else find_content_box(gray)
    gray = gray.crop(box)

    warp = 0.0
    if not args.no_dewarp:
        gray, warp = dewarp(gray)
        if warp and not args.no_crop:
            gray = gray.crop(find_content_box(gray))

    gray = boost_contrast(gray, args.white_point, args.darken)

    page = fit_to_page(gray, args.dpi, args.margin, args.orientation, args.align, args.paper)
    if args.bw:
        page = page.point(lambda v: 255 if v > args.bw_threshold else 0).convert("1")

    print(f"  {path.name}: skew {angle:+.2f}°, crop {box}, dewarp {warp:.1f}px", file=sys.stderr)
    return page


def collect_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders (processed in name order)")
    ap.add_argument("-o", "--output", help="combined PDF to write (default: sheets.pdf unless --png-dir only)")
    ap.add_argument("--png-dir", help="also write one PNG per page here")
    ap.add_argument("--version", action="version", version=f"sheetfit {__version__}")
    ap.add_argument("--paper", choices=sorted(PAPER_IN), default="letter")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.4, help="page margin in inches (default 0.4)")
    ap.add_argument("--orientation", choices=["portrait", "landscape", "auto"], default="portrait")
    ap.add_argument("--align", choices=["top", "center"], default="top")
    ap.add_argument("--max-angle", type=float, default=10.0, help="largest skew to search, degrees")
    ap.add_argument("--rotate", type=float, default=0.0, help="pre-rotate clockwise by N degrees (e.g. 90)")
    ap.add_argument("--white-point", type=int, default=215,
                    help="grey level at/above which pixels become pure white (default 215; lower = cleaner background)")
    ap.add_argument("--darken", type=float, default=1.8,
                    help="gamma that pushes greys toward black (1 = off, default 1.8)")
    ap.add_argument("--keep-color-marks", action="store_true",
                    help="keep coloured annotations as grey instead of removing coloured backgrounds")
    ap.add_argument("--bw", action="store_true",help="output pure black/white (smaller, crisper prints)")
    ap.add_argument("--bw-threshold", type=int, default=170)
    ap.add_argument("--no-flatten", action="store_true", help="skip lighting correction")
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--no-dewarp", action="store_true", help="don't straighten curved staff lines")
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args(argv)

    paths = collect_inputs(args.inputs)
    if not paths:
        ap.error("no input images found")
    if not args.output and not args.png_dir:
        args.output = "sheets.pdf"

    pages = []
    for p in paths:
        try:
            pages.append(process(p, args))
        except Exception as e:  # keep going on a bad file
            print(f"  {p}: FAILED ({e})", file=sys.stderr)
    if not pages:
        return 1

    if args.png_dir:
        d = Path(args.png_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, (p, page) in enumerate(zip(paths, pages), 1):
            page.save(d / f"{i:03d}_{p.stem}.png", dpi=(args.dpi, args.dpi))
    if args.output:
        pages[0].save(args.output, "PDF", resolution=args.dpi, save_all=True, append_images=pages[1:])
        print(f"wrote {args.output} ({len(pages)} pages)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
