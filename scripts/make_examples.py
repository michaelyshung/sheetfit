"""Make the README example images from scratch, with no copyrighted music.

1. Engrave an original, randomly composed piano page (a little waltz).
2. "Photograph" it: book curl near the spine, keystone, tilt, uneven
   warm lighting, spine shadow, the facing page and a wooden table,
   camera blur and noise, JPEG.
3. Run the real sheetfit CLI on the photos and compose before/after images.

    python scripts/make_examples.py            # writes docs/example_*.png

Needs the Noto Music font for the clefs (fonts-noto on Debian/Ubuntu);
falls back to drawn shapes without it.
"""
from __future__ import annotations

import io
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

PW, PH = 2550, 3300          # Letter at 300 dpi
SP = 22                      # staff line gap


def font(names, size):
    for n in names:
        for d in ("/usr/share/fonts/truetype", "/usr/share/fonts", "/Library/Fonts",
                  "C:/Windows/Fonts"):
            for p in Path(d).rglob(n) if Path(d).exists() else []:
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


SERIF = ["DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"]
SERIF_B = ["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf"]
SERIF_BI = ["DejaVuSerif-BoldItalic.ttf", "LiberationSerif-BoldItalic.ttf"]
SERIF_I = ["DejaVuSerif-Italic.ttf", "LiberationSerif-Italic.ttf"]
MUSIC = ["NotoMusic-Regular.ttf", "Symbola_hint.ttf", "Symbola.ttf"]


# ------------------------------------------------------------------ engraving
class Engraver:
    def __init__(self, seed: int):
        self.rnd = random.Random(seed)
        self.img = Image.new("L", (PW, PH), 255)
        self.d = ImageDraw.Draw(self.img)

    # staff position p: 0 = bottom line, each step = half a space
    @staticmethod
    def y_of(bottom: float, p: int) -> float:
        return bottom - p * SP / 2

    def staff(self, x0, x1, top):
        for k in range(5):
            y = top + k * SP
            self.d.line([x0, y, x1, y], fill=0, width=2)

    def glyph(self, ch, size, x, top, height):
        """Draw a music glyph scaled so its ink spans `height` from `top`."""
        f = font(MUSIC, size)
        bb = f.getbbox(ch)
        if bb[3] - bb[1] <= 0:
            return False
        s = height / (bb[3] - bb[1])
        f = font(MUSIC, max(8, int(size * s)))
        bb = f.getbbox(ch)
        self.d.text((x - bb[0], top - bb[1]), ch, font=f, fill=0)
        return True

    def head(self, x, y, filled=True):
        w, h, t = SP * 0.66, SP * 0.47, math.radians(-22)
        pts = []
        for k in range(40):
            a = 2 * math.pi * k / 40
            ex, ey = w * math.cos(a), h * math.sin(a)
            pts.append((x + ex * math.cos(t) - ey * math.sin(t),
                        y + ex * math.sin(t) + ey * math.cos(t)))
        self.d.polygon(pts, fill=0)
        if not filled:
            inner = []
            for k in range(40):
                a = 2 * math.pi * k / 40
                ex, ey = w * 0.72 * math.cos(a), h * 0.42 * math.sin(a)
                t2 = math.radians(-35)
                inner.append((x + ex * math.cos(t2) - ey * math.sin(t2),
                              y + ex * math.sin(t2) + ey * math.cos(t2)))
            self.d.polygon(inner, fill=255)

    def ledgers(self, x, bottom, p):
        for q in range(-2, p - 1, -2) if p <= -2 else range(10, p + 1, 2):
            y = self.y_of(bottom, q)
            self.d.line([x - SP * 1.0, y, x + SP * 1.0, y], fill=0, width=2)

    def note(self, x, bottom, ps, dur, stem_dir=None, stem_len=3.4):
        """Chord of staff positions `ps`; dur in quarters (1, 2, 3, 0.5)."""
        filled = dur not in (2, 3)
        for p in ps:
            if p <= -2 or p >= 10:
                self.ledgers(x, bottom, p)
        lo, hi = min(ps), max(ps)
        up = stem_dir if stem_dir is not None else (lo + hi) / 2 < 4
        for p in ps:
            self.head(x, self.y_of(bottom, p), filled)
            if dur == 3:
                self.d.ellipse([x + SP * 1.0 - 4, self.y_of(bottom, p | 1) - 4,
                                x + SP * 1.0 + 4, self.y_of(bottom, p | 1) + 4], fill=0)
        sx = x + SP * 0.6 if up else x - SP * 0.6
        if up:
            y0, y1 = self.y_of(bottom, lo), self.y_of(bottom, hi) - stem_len * SP
        else:
            y0, y1 = self.y_of(bottom, hi), self.y_of(bottom, lo) + stem_len * SP
        self.d.line([sx, y0, sx, y1], fill=0, width=3)
        return sx, y1, up

    def beam(self, a, b):
        (x0, y0, _), (x1, y1, up) = a, b
        t = SP * 0.5 * (1 if up else -1)
        self.d.polygon([(x0 - 1, y0), (x1 + 1, y1), (x1 + 1, y1 + t), (x0 - 1, y0 + t)], fill=0)

    def slur(self, x0, x1, y, h):
        pts = []
        for k in range(41):
            t = k / 40
            pts.append((x0 + (x1 - x0) * t, y - h * 4 * t * (1 - t)))
        self.d.line(pts, fill=0, width=3)
        self.d.line([(px, py + 2) for px, py in pts[4:-4]], fill=0, width=2)

    def text(self, xy, s, names, size, anchor="la"):
        self.d.text(xy, s, font=font(names, size), fill=0, anchor=anchor)

    # ------------------------------------------------------------ the page
    def page(self, number: int, first: bool, title: str):
        rnd, d = self.rnd, self.d
        x0, x1 = 200, PW - 190
        if first:
            self.text((PW / 2, 170), title, SERIF_B, 84, "ma")
            self.text((PW / 2, 290), "for piano", SERIF_I, 44, "ma")
            self.text((x1, 360), "sheetfit sample", SERIF, 38, "ra")
            self.text((x0 + 60, 405), "Tempo di valse", SERIF_B, 42, "la")
            top0, systems = 520, 6
        else:
            top0, systems = 330, 6
        gap_staves, sys_h = 5.5 * SP, 14 * SP
        pitch = rnd.randint(3, 7)
        measure = 1 + (number - 1) * 24
        for s in range(systems):
            ttop = top0 + s * (sys_h + 6.2 * SP)
            btop = ttop + 4 * SP + gap_staves
            tbot, bbot = ttop + 4 * SP, btop + 4 * SP
            sx0 = x0 + (60 if s == 0 and first else 0)
            self.staff(sx0, x1, ttop)
            self.staff(sx0, x1, btop)
            # brace + system line
            d.line([sx0, ttop, sx0, bbot], fill=0, width=4)
            bx = sx0 - 26
            d.line([bx, ttop, bx, bbot], fill=0, width=9)
            d.line([bx, ttop, bx + 14, ttop - 10], fill=0, width=5)
            d.line([bx, bbot, bx + 14, bbot + 10], fill=0, width=5)
            # clefs
            if not self.glyph("\U0001D11E", 160, sx0 + 14, ttop - 1.7 * SP, 7.4 * SP):
                d.ellipse([sx0 + 20, ttop, sx0 + 60, tbot], outline=0, width=5)
            if not self.glyph("\U0001D122", 120, sx0 + 18, btop - 0.05 * SP, 3.2 * SP):
                d.ellipse([sx0 + 20, btop, sx0 + 50, btop + 2 * SP], outline=0, width=5)
            x = sx0 + 4.4 * SP
            if s == 0 and first:
                for top in (ttop, btop):
                    for k, ch in enumerate("34"):
                        self.text((x, top + k * 2 * SP + SP), ch, SERIF_B, int(2.9 * SP), "lm")
                x += 2.6 * SP
            else:
                self.text((sx0 + 2, ttop - 1.6 * SP), str(measure), SERIF_I, 30, "ls")
            n_meas = 4
            mw = (x1 - x - SP) / n_meas
            dyn_done = False
            for m in range(n_meas):
                mx = x + SP * 1.4 + m * mw
                step = (mw - SP * 1.6) / 3
                # right hand: a melody in 3/4
                pat = rnd.choice([[1, 1, 1], [2, 1], [0.5, 0.5, 1, 1], [3], [1, 0.5, 0.5, 1]])
                t, beam_prev, first_x = 0.0, None, None
                for dur in pat:
                    pitch = max(-1, min(11, pitch + rnd.choice([-2, -1, -1, 1, 1, 2, 3, -3])))
                    ps = [pitch] + ([pitch - 2] if rnd.random() < 0.18 else [])
                    nx = mx + t * step
                    first_x = first_x or nx
                    if rnd.random() < 0.07:
                        self.text((nx - SP * 1.7, self.y_of(tbot, pitch)), "\u266F", MUSIC, int(2.6 * SP), "mm")
                    if dur == 0.5:
                        st = self.note(nx, tbot, ps, 1, stem_dir=pitch < 4)
                        if beam_prev is None:
                            beam_prev = st
                        else:
                            up = beam_prev[2]
                            a = beam_prev
                            b = self.note(nx, tbot, ps, 1, stem_dir=up)
                            self.beam(a, (b[0], b[1], up))
                            beam_prev = None
                    else:
                        self.note(nx, tbot, ps, dur)
                    t += dur
                if rnd.random() < 0.35:
                    self.slur(first_x, mx + 2.6 * step, self.y_of(tbot, 12), SP * 1.1)
                # left hand: oom-pah-pah
                root = rnd.choice([-2, -1, 0, 1, 2])
                self.note(mx, bbot, [root], 1, stem_dir=False)
                ch = [root + 4, root + 6, root + 8] if rnd.random() < 0.5 else [root + 5, root + 7]
                for k in (1, 2):
                    self.note(mx + k * step, bbot, ch, 1, stem_dir=True, stem_len=3.0)
                bar = x + (m + 1) * mw
                d.line([bar, ttop, bar, tbot], fill=0, width=3)
                d.line([bar, btop, bar, bbot], fill=0, width=3)
                if not dyn_done and rnd.random() < 0.3:
                    self.text((mx, tbot + 2.6 * SP), rnd.choice(["p", "mf", "mp", "f"]),
                              SERIF_BI, int(2.2 * SP), "lm")
                    dyn_done = True
                elif rnd.random() < 0.12:
                    self.text((mx, tbot + 2.6 * SP), "cresc.", SERIF_I, int(1.6 * SP), "lm")
            measure += n_meas
        self.text((PW / 2, PH - 150), str(number), SERIF, 40, "ma")
        return self.img


# ------------------------------------------------------------------ photo
def bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray, fill: float) -> np.ndarray:
    h, w = a.shape
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    xc, yc = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    x0, y0 = xc.astype(np.int32), yc.astype(np.int32)
    fx, fy = xc - x0, yc - y0
    v = (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
         + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inside, v, fill)


def wood(h, w, rnd):
    """Dark table boards: long soft grain, built small and scaled up (no aliasing)."""
    rng = np.random.default_rng(rnd.randint(0, 999))
    sh, sw = h // 8, w // 8
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    wob = np.cumsum(rng.normal(0, 0.35, sw)).astype(np.float32)
    grain = 0.5 + 0.5 * np.sin(yy * 0.55 + wob[None, :] * 0.6 + 2.5 * np.sin(xx * 0.01))
    small = 0.6 + 0.22 * grain ** 2 + rng.normal(0, 0.03, (sh, sw))
    small[:, (np.arange(sw) % (sw // 3)) == 0] *= 0.55      # board seams
    base = np.asarray(Image.fromarray((small * 255 / 1.2).clip(0, 255).astype(np.uint8))
                      .resize((w, h), Image.BICUBIC), dtype=np.float32) / 255 * 1.2
    return np.stack([base * 92, base * 62, base * 40], -1)


def photograph(page: Image.Image, seed: int, spine: str = "left",
               curl_px: float = 70, tilt: float = 2.4) -> Image.Image:
    rnd = random.Random(seed)
    W, H = 3024, 4032
    ink = 1 - np.asarray(page, dtype=np.float32) / 255
    if spine == "right":
        ink = ink[:, ::-1]
    ph, pw = ink.shape
    s = 0.84 * W / pw                      # page size in the photo
    cx, cy = W / 2 + 40, H / 2 + 30
    th = math.radians(tilt)
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    # photo -> flat page coordinates (rotation + scale, mild keystone)
    du, dv = u - cx, v - cy
    xr = (du * math.cos(th) + dv * math.sin(th)) / s
    yr = (-du * math.sin(th) + dv * math.cos(th)) / s
    yn = yr / (ph / 2)
    xr = xr * (1 + 0.035 * yn)             # top a little narrower than the bottom
    x, y = xr + pw / 2, yr + ph / 2
    # book curl: near the spine the page lifts toward the camera, so lines fan
    # out and droop; a gentle overall bulge bends lines in the middle too
    g = np.exp(-np.clip(x, -0.15 * pw, None) / (0.16 * pw))
    bulge = np.sin(np.clip(x / pw, 0, 1) * math.pi)
    ys = y + curl_px * g * (0.5 + 0.9 * (y - ph / 2) / ph) \
        - 22 * bulge * (y - ph / 2) / (ph / 2) - 14 * bulge
    xs = x - 45 * g                        # foreshortened near the spine
    sampled = bilinear(ink, xs, ys, 0.0)
    on_page = (xs >= 0) & (xs <= pw) & (ys >= -10) & (ys <= ph + 10)
    facing = (xs < 0) & (xs > -0.09 * pw) & (ys >= -10) & (ys <= ph + 18)
    # light: warm, brighter top-left, spine shadow, vignette
    paper = np.array([241, 233, 214], np.float32)
    inkc = np.array([38, 34, 32], np.float32)
    light = 1.0 - 0.18 * (u / W) - 0.12 * (v / H)
    light *= 1 - 0.20 * (((u - W / 2) / W) ** 2 + ((v - H / 2) / H) ** 2) * 2
    spine_sh = 1 - 0.5 * np.exp(-np.abs(xs) / (0.045 * pw))
    rgb = paper * (1 - sampled[..., None]) + inkc * sampled[..., None]
    rgb = rgb * (light * spine_sh)[..., None] * np.array([1.0, 0.97, 0.9], np.float32)
    bg = wood(H, W, rnd) * light[..., None]
    out = np.where(on_page[..., None], rgb, bg)
    fpaper = paper * (light * spine_sh * 0.85)[..., None] * np.array([1.0, 0.96, 0.88], np.float32)
    out = np.where((facing & ~on_page)[..., None], fpaper, out)
    if spine == "right":
        out = out[:, ::-1]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(1.1))
    a = np.asarray(img, dtype=np.float32)
    a += np.random.default_rng(seed).normal(0, 3.5, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=87)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


# ------------------------------------------------------------------ figures
def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 34), "white")
    out.paste(img, (0, 34))
    ImageDraw.Draw(out).text((4, 8), text, font=font(SERIF, 17), fill=(40, 40, 40))
    return out


NOTICE = ("Generated example: original, randomly composed music engraved and \"photographed\" by "
          "scripts/make_examples.py, then processed by the real sheetfit pipeline. No copyrighted score is shown.")


def add_notice(img: Image.Image, text: str = None) -> Image.Image:
    """Append the disclaimer strip, so the image carries it wherever it's shown."""
    text = text or NOTICE
    f = font(SERIF_I, 15)
    d = ImageDraw.Draw(img)
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > img.width - 24:
            lines.append(cur)
            cur = w
        else:
            cur = t
    lines.append(cur)
    out = Image.new("RGB", (img.width, img.height + 16 + 22 * len(lines)), "white")
    out.paste(img.convert("RGB"), (0, 0))
    od = ImageDraw.Draw(out)
    od.line([0, img.height + 6, img.width, img.height + 6], fill=(200, 200, 200))
    for k, line in enumerate(lines):
        od.text((8, img.height + 12 + 22 * k), line, font=f, fill=(90, 90, 90))
    return out


def side_by_side(before: Image.Image, after: Image.Image, h: int = 700) -> Image.Image:
    b = before.resize((round(before.width * h / before.height), h), Image.LANCZOS)
    a = after.convert("RGB").resize((round(after.width * h / after.height), h), Image.LANCZOS)
    ImageDraw.Draw(a).rectangle([0, 0, a.width - 1, a.height - 1], outline=(190, 190, 190))
    b = label(b, "phone photo of a book page")
    a = label(a, "sheetfit output (Letter page)")
    gap = 70
    out = Image.new("RGB", (b.width + gap + a.width, b.height), "white")
    out.paste(b, (0, 0))
    out.paste(a, (b.width + gap, 0))
    d = ImageDraw.Draw(out)
    mx, my = b.width + gap // 2, 34 + h // 2
    d.polygon([(mx - 14, my - 14), (mx + 16, my), (mx - 14, my + 14)], fill=(30, 30, 30))
    return add_notice(out)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        photos = tmp / "photos"
        photos.mkdir()
        pages = [Engraver(11).page(1, True, "Little Waltz"),
                 Engraver(12).page(2, False, ""),
                 Engraver(13).page(3, False, "")]
        shots = [photograph(pages[0], 1, "left", curl_px=110, tilt=2.4),
                 photograph(pages[1], 2, "right", curl_px=60, tilt=-1.8),
                 photograph(pages[2], 3, "left", curl_px=85, tilt=0.9)]
        for k, im in enumerate(shots, 1):
            im.save(photos / f"page{k}.jpg", quality=92)
        out = tmp / "out"
        subprocess.run([sys.executable, "-m", "sheetfit", str(photos), "--png-dir", str(out),
                        "-o", str(tmp / "out.pdf")], check=True)
        results = sorted(out.glob("*.png"))
        side_by_side(shots[0], Image.open(results[0])).save(DOCS / "example_page1.png",
                                                            optimize=True)
        side_by_side(shots[1], Image.open(results[1])).save(DOCS / "example_page2.png", optimize=True)
        if "--keep" in sys.argv:
            keep = ROOT / "build" / "examples"
            keep.mkdir(parents=True, exist_ok=True)
            for p in list(photos.iterdir()) + results:
                Image.open(p).save(keep / p.name)
        try:
            app_screenshot(photos, DOCS / "app_flatten.png")
        except ImportError:
            print("PySide6 not installed: skipped the app screenshot")
    print("wrote", DOCS / "example_page1.png", DOCS / "example_page2.png")
    return 0


def app_screenshot(photos: Path, out: Path) -> None:
    """The app's Flatten editor on the generated photos (headless)."""
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication
    from sheetfit.gui.app import MainWindow
    from sheetfit.gui.model import STEPS

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.resize(1440, 880)
    win.show()
    win.add_paths([str(photos)])

    def settle(sec=0.5):
        t = time.time()
        while win.doc.working or time.time() - t < sec:
            app.processEvents()
            time.sleep(0.02)

    settle()
    win.strip.setCurrentRow(0)
    win.show_step(STEPS.index("flatten"))
    ed = win.editors[STEPS.index("flatten")]
    settle()
    ed.select(2, None)          # show one curve's anchors and handles
    settle(1.5)                 # thumbnails and the result preview
    shot = win.grab().toImage()
    from PySide6.QtCore import QBuffer, QIODevice
    qb = QBuffer()
    qb.open(QIODevice.WriteOnly)
    shot.save(qb, "PNG")
    img = Image.open(io.BytesIO(bytes(qb.data())))
    add_notice(img, "The app on generated example pages (original, randomly composed music; "
               "see scripts/make_examples.py). No copyrighted score is shown.").save(out, optimize=True)
    QThreadPool.globalInstance().waitForDone()
    win.deleteLater()
    app.processEvents()


if __name__ == "__main__":
    sys.exit(main())
