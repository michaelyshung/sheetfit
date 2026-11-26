"""Page-curl flattening driven by guide lines.

A guide is a line that should be straight: it traces a staff line where it
actually sits on the curled page. Flattening shifts every pixel column
vertically so each guide becomes horizontal at its target height; between
guides the shift is interpolated, beyond the outermost guides it is held
constant. Each guide point carries a strength, so the pull can be tuned
along the line.

`auto_guides` measures the curl from the staff lines and produces guides,
so the automatic result and a hand-edited one go through the same code.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from .core import staff_spacing, working_copy
from .recipe import Guide


# ---------------------------------------------------------------- curves
def spline(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Smooth curve through the points (cubic Hermite, finite-difference
    tangents). Constant beyond the end points. Same curve the GUI draws."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    n = xs.size
    if n == 1:
        return np.full_like(xq, ys[0])
    m = np.empty(n)
    m[0] = (ys[1] - ys[0]) / (xs[1] - xs[0] or 1)
    m[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2] or 1)
    if n > 2:
        m[1:-1] = (ys[2:] - ys[:-2]) / np.maximum(xs[2:] - xs[:-2], 1e-9)
    xc = np.clip(xq, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xc) - 1, 0, n - 2)
    h = np.maximum(xs[i + 1] - xs[i], 1e-9)
    t = (xc - xs[i]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i]
            + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1])


# ---------------------------------------------------------------- Bezier
def _cubic(p0, c1, c2, p1, t):
    t = np.asarray(t, dtype=np.float64)[:, None]
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t ** 3 * p1


def bezier_sample(ctrl, per_segment: int = 16) -> np.ndarray:
    """Points along a Bezier path [P0, C1, C2, P1, C3, C4, P2, ...]."""
    c = np.asarray(ctrl, dtype=np.float64)
    out = []
    for k in range(0, len(c) - 1, 3):
        ts = np.linspace(0, 1, per_segment, endpoint=False)
        out.append(_cubic(c[k], c[k + 1], c[k + 2], c[k + 3], ts))
    out.append(c[-1:])
    return np.vstack(out)


def _fit_cubic(pts: np.ndarray):
    """Least-squares cubic through pts with the end points fixed.
    Returns ([P0, C1, C2, P1], max error)."""
    p0, p1 = pts[0], pts[-1]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    t = d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(pts))
    u = 1 - t
    b0, b1, b2, b3 = u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3
    A = np.stack([b1, b2], axis=1)
    rhs = pts - b0[:, None] * p0 - b3[:, None] * p1
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    c1, c2 = sol
    err = np.hypot(*(A @ sol + b0[:, None] * p0 + b3[:, None] * p1 - pts).T).max()
    return [p0, c1, c2, p1], float(err)


def fit_bezier(points, tol: float, max_segments: int = 4) -> list[list[float]]:
    """Fewest-segment smooth Bezier path through `points` (sorted by x)
    within `tol`. Interior anchors get collinear handles (no kinks)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return [list(p) for p in pts]
    segs = 1
    while True:
        cuts = np.linspace(0, len(pts) - 1, segs + 1).round().astype(int)
        parts, worst = [], 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            seg, err = _fit_cubic(pts[a:b + 1]) if b - a >= 3 else (
                [pts[a], pts[a] + (pts[b] - pts[a]) / 3, pts[a] + 2 * (pts[b] - pts[a]) / 3, pts[b]], 0.0)
            parts.append(seg)
            worst = max(worst, err)
        if worst <= tol or segs >= max_segments:
            break
        segs += 1
    ctrl = [parts[0][0]]
    for seg in parts:
        ctrl += [seg[1], seg[2], seg[3]]
    ctrl = np.asarray(ctrl)
    for k in range(3, len(ctrl) - 1, 3):  # smooth the joints
        a, ci, co = ctrl[k], ctrl[k - 1], ctrl[k + 1]
        v = (co - a) - (ci - a)
        n = np.hypot(*v)
        if n > 0:
            v /= n
            ctrl[k - 1] = a - v * np.hypot(*(ci - a))
            ctrl[k + 1] = a + v * np.hypot(*(co - a))
    return ctrl.tolist()


def split_bezier(ctrl, seg: int, t: float) -> list[list[float]]:
    """Insert an anchor in segment `seg` at parameter t (de Casteljau);
    the curve's shape doesn't change."""
    c = [np.asarray(p, dtype=np.float64) for p in ctrl]
    p0, c1, c2, p1 = c[3 * seg:3 * seg + 4]
    a, b, cc = p0 + (c1 - p0) * t, c1 + (c2 - c1) * t, c2 + (p1 - c2) * t
    d, e = a + (b - a) * t, b + (cc - b) * t
    m = d + (e - d) * t
    new = c[:3 * seg] + [p0, a, d, m, e, cc, p1] + c[3 * seg + 4:]
    return [list(map(float, p)) for p in new]


def resolve(g: Guide, per_segment: int = 16) -> Guide:
    """Fill g.points / g.strengths from its Bezier form (in place)."""
    if not g.bezier:
        return g
    pts = bezier_sample(g.bezier, per_segment)
    ax = np.asarray(g.bezier, dtype=np.float64)[0::3, 0]
    order = np.argsort(ax)
    s = np.interp(pts[:, 0], ax[order], np.asarray(g.anchor_strengths, dtype=np.float64)[order])
    k = np.argsort(pts[:, 0], kind="stable")
    g.points = pts[k].tolist()
    g.strengths = s[k].tolist()
    return g


def ensure_bezier(g: Guide, tol: float = 1.0) -> Guide:
    """Give a points-only guide (older projects) a Bezier form."""
    if not g.bezier and len(g.points) >= 2:
        g.bezier = fit_bezier(g.points, tol)
        ax = np.asarray(g.bezier)[0::3, 0]
        p = np.asarray(g.points)
        g.anchor_strengths = np.interp(ax, p[:, 0], g.strengths).tolist()
    return g


def guide_target(g: Guide) -> float:
    """Height the guide is flattened to: where it sits at the middle of its span."""
    p = np.asarray(g.points)
    return float(spline(p[:, 0], p[:, 1], np.array([(p[0, 0] + p[-1, 0]) / 2]))[0])


# ---------------------------------------------------------------- apply
def displacement(guides: list[Guide], width: int, height: int, cx: float, cy: float,
                 strength: float = 1.0) -> np.ndarray:
    """Vertical shift d(x, y) (height x width): out(x, y) = in(x, y + d).

    Guide coordinates are relative to (cx, cy) in this image.
    """
    if not guides or strength == 0:
        return np.zeros((height, width), dtype=np.float32)
    X = np.arange(width, dtype=np.float64) - cx
    rows = []
    for g in guides:
        p = np.asarray(g.points, dtype=np.float64)
        tgt = guide_target(g)
        y = spline(p[:, 0], p[:, 1], X)
        s = np.interp(X, p[:, 0], np.asarray(g.strengths, dtype=np.float64))
        rows.append((tgt + cy, (y - tgt) * s * strength))
    rows.sort(key=lambda r: r[0])
    T = np.array([r[0] for r in rows])          # target rows (image px)
    D = np.stack([r[1] for r in rows])          # (G, W)
    Y = np.arange(height, dtype=np.float64)
    if len(rows) == 1:
        return np.broadcast_to(D[0], (height, width)).astype(np.float32)
    j = np.clip(np.searchsorted(T, Y) - 1, 0, len(T) - 2)
    f = np.clip((Y - T[j]) / np.maximum(T[j + 1] - T[j], 1e-9), 0.0, 1.0)
    return ((1 - f)[:, None] * D[j] + f[:, None] * D[j + 1]).astype(np.float32)


def apply_guides(gray: Image.Image, guides: list[Guide], strength: float = 1.0,
                 pad_frac: float = 0.08) -> tuple[Image.Image, float]:
    """Flatten `gray` along the guides (coordinates relative to its centre).

    The result is padded top and bottom with white by `pad_frac` of the
    height so content pushed towards an edge stays in frame; the padding is
    symmetric, so centre-relative coordinates stay valid. Returns the image
    and the largest shift applied, in pixels.
    """
    if not guides or strength == 0:
        return gray, 0.0
    pad = int(pad_frac * gray.height)
    g = np.asarray(ImageOps.expand(gray, (0, pad, 0, pad), fill=255), dtype=np.float32)
    H, W = g.shape
    d = displacement(guides, W, H, W / 2, H / 2, strength)
    ys = np.arange(H, dtype=np.float32)[:, None] + d
    y0 = np.floor(ys)
    fr = ys - y0
    y0 = y0.astype(np.int64)
    valid0 = (y0 >= 0) & (y0 < H)
    valid1 = (y0 + 1 >= 0) & (y0 + 1 < H)
    cols = np.arange(W)[None, :]
    a = np.where(valid0, g[np.clip(y0, 0, H - 1), cols], 255.0)
    b = np.where(valid1, g[np.clip(y0 + 1, 0, H - 1), cols], 255.0)
    out = a * (1 - fr) + b * fr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), float(np.abs(d).max())


# ---------------------------------------------------------------- measure
def _band_shift(p: np.ndarray, q: np.ndarray, S: int) -> tuple[float, float]:
    """Vertical shift s (subpixel) such that q[y + s] best matches p[y].

    Returns (shift, confidence); confidence 0 when either profile is nearly
    empty or the peak sits at the search limit.
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
    if k in (0, 2 * S):
        return 0.0, 0.0
    l, c, r = scores[k - 1], scores[k], scores[k + 1]
    den = l - 2 * c + r
    sub = 0.5 * (l - r) / den if den < 0 else 0.0
    return k - S + sub, float(energy)


def _staff_rows(profile: np.ndarray, spacing: float) -> list[float]:
    """Centres of staves in a row-darkness profile (working-scale px)."""
    win = max(3, int(round(4 * spacing)) | 1)
    sm = np.convolve(profile, np.ones(win) / win, mode="same")
    if sm.max() <= 0:
        return []
    thr = 0.3 * sm.max()
    sep = 6 * spacing
    cand = [i for i in range(1, sm.size - 1)
            if sm[i] >= thr and sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]
    cand.sort(key=lambda i: -sm[i])
    picked: list[int] = []
    for i in cand:
        if all(abs(i - j) >= sep for j in picked):
            picked.append(i)
    return sorted(float(i) for i in picked)


def auto_guides(gray: Image.Image, n_strips: int = 40, n_bands: int = 10,
                deg_x: int = 4, deg_y: int = 2, n_points: int = 33) -> list[Guide]:
    """Measure page curl from the staff lines and return guide lines.

    1. Cut the page into vertical strips and overlapping horizontal bands.
    2. Walking outward from the middle strip, cross-correlate each strip's
       row profile with its neighbour's (small steps, so staff lines can't
       be mistaken for their neighbours) and accumulate the drift.
    3. Fit a smooth surface to the drift (deg 4 across, quadratic down).
    4. Find the staves in the middle strip and lay one guide along each,
       following the surface, as the fewest Bezier segments that stay within
       a third of a pixel (usually 1-2 segments: easy to edit by hand).

    Coordinates are relative to the centre of `gray`. Returns [] when the
    page is flat or the measurement can't be trusted.
    """
    small, s = working_copy(gray, 1600)
    a = 255.0 - np.asarray(small, dtype=np.float32)
    a[a < 60] = 0
    h, w = a.shape

    ink_cols = np.flatnonzero((a > 0).mean(axis=0) > 0.01)
    if ink_cols.size < 0.3 * w:
        return []
    x0, x1 = int(ink_cols[0]), int(ink_cols[-1]) + 1
    xe = np.linspace(x0, x1, n_strips + 1).astype(int)
    prof = np.stack([a[:, xe[i]:xe[i + 1]].mean(axis=1) for i in range(n_strips)])

    half = h / n_bands
    centers = np.linspace(half, h - half, n_bands)
    S = max(2, round(h * 0.006))
    D = np.zeros((n_strips, n_bands))
    Wt = np.zeros((n_strips, n_bands))
    mid = n_strips // 2
    for k, c in enumerate(centers):
        y0, y1 = int(c - half), int(c + half)
        ref = prof[mid, y0:y1].std()
        Wt[mid, k] = ref if ref >= 4.0 else 0.0
        for order in (range(mid + 1, n_strips), range(mid - 1, -1, -1)):
            for j in order:
                i = j - 1 if j > mid else j + 1
                off = int(round(D[i, k]))
                ya, yb = max(0, y0 + off), min(h, y1 + off)
                st, conf = _band_shift(prof[i, ya:yb], prof[j, ya:yb], S)
                D[j, k] = D[i, k] + st
                Wt[j, k] = min(Wt[i, k], conf) if conf else 0.0
    if (Wt > 0).sum() < deg_x * deg_y * 3:
        return []

    xc = ((xe[:-1] + xe[1:]) / 2 - x0) / max(1, x1 - x0) * 2 - 1
    yc = centers / h * 2 - 1
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    terms = [(p, q) for p in range(deg_x + 1) for q in range(deg_y + 1)]
    A = np.stack([XX.ravel() ** p * YY.ravel() ** q for p, q in terms], axis=1)
    sw = np.sqrt(Wt.ravel())
    coef, *_ = np.linalg.lstsq(A * sw[:, None], D.ravel() * sw, rcond=None)
    good = Wt.ravel() > 0
    if np.median(np.abs(A[good] @ coef - D.ravel()[good])) > 1.5:
        return []

    def surface(xn: np.ndarray, yn: float) -> np.ndarray:
        return sum(cf * xn ** p * yn ** q for (p, q), cf in zip(terms, coef))

    # where the staves are, measured in the middle strip (the reference)
    spacing = staff_spacing(small)
    mid_prof = (a[:, xe[mid]:xe[mid + 1]] > 100).mean(axis=1)
    rows = _staff_rows(mid_prof, spacing) if spacing else []
    measured = centers[Wt.max(axis=0) > 0]
    lo, hi = measured.min() - half, measured.max() + half
    rows = [r for r in rows if lo <= r <= hi] or list(measured)

    xs_n = np.linspace(-1, 1, n_points)
    xs_px = x0 + (xs_n + 1) / 2 * (x1 - x0)
    guides = []
    max_shift = 0.0
    for r in rows:
        dy = surface(xs_n, r / h * 2 - 1)
        dy = dy - surface(np.array([0.0]), r / h * 2 - 1)[0]  # flat at the middle
        max_shift = max(max_shift, float(np.abs(dy).max()))
        pts = [[(x / s) - gray.width / 2, ((r + d) / s) - gray.height / 2]
               for x, d in zip(xs_px, dy)]
        g = Guide(points=[], bezier=fit_bezier(pts, tol=0.3 / s))
        guides.append(resolve(g))
    if max_shift < 0.5 or max_shift / s > 0.08 * gray.height:
        return []  # flat page, or implausibly strong curl
    return guides


def flatten(gray: Image.Image) -> tuple[Image.Image, float, list[Guide]]:
    """Automatic flatten: measure guides and apply them."""
    guides = auto_guides(gray)
    out, moved = apply_guides(gray, guides)
    return out, moved, guides
