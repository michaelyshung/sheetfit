"""End-to-end and unit tests on synthetic sheet-music photos.

Run with:  python -m unittest discover tests     (or: pytest)
"""
from __future__ import annotations

import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sheetfit.core import (
    PAPER_IN,
    dewarp,
    find_content_box,
    find_skew,
    flatten_lighting,
    main,
    to_gray,
    whiten_dark_edges,
)

W, H = 1200, 1550


def music_page(seed: int = 0, paper=(242, 232, 205)) -> Image.Image:
    """A fake page: 6 systems of 2 staves with note heads and stems."""
    rnd = random.Random(seed)
    pg = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(pg)
    d.rectangle([150, 60, 1050, 140], fill=(255, 235, 110))  # coloured banner
    y = 220
    for _ in range(6):
        for staff in range(2):
            for line in range(5):
                yy = y + staff * 110 + line * 14
                d.line([110, yy, 1090, yy], fill=(80, 80, 80), width=2)
            for n in range(30):
                x = 140 + n * 31
                yy = y + staff * 110 + rnd.randint(0, 4) * 14
                d.ellipse([x, yy - 6, x + 15, yy + 6], fill=(40, 40, 40))
                d.line([x + 14, yy, x + 14, yy - 45], fill=(40, 40, 40), width=2)
        d.line([110, y, 110, y + 110 + 56], fill=(40, 40, 40), width=3)
        y += 215
    return pg


def staff_sharpness(gray: Image.Image) -> float:
    """How concentrated the row-darkness profile is: high = straight staff lines."""
    a = 255.0 - np.asarray(gray.convert("L"), dtype=np.float64)
    prof = a.sum(axis=1)
    return float((prof ** 2).sum() / (prof.sum() ** 2 + 1e-9) * prof.size)


def curl(img: Image.Image, amount: float = 35.0) -> Image.Image:
    """Book-spine style droop near the right edge, stronger at the top."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    Y = np.arange(h, dtype=np.float32)
    out = np.empty_like(a)
    for x in range(w):
        t = max(0.0, (x / w - 0.55) / 0.45)
        d = amount * t * t * (1.0 - 0.5 * Y / h)
        out[:, x] = np.interp(Y - d, Y, a[:, x], left=245, right=245)
    return Image.fromarray(out.astype(np.uint8))


class TestSteps(unittest.TestCase):
    def test_colour_background_removed(self):
        g = np.asarray(to_gray(music_page(), keep_color_marks=False))
        self.assertGreater(g[100, 600], 230)  # yellow banner -> near white
        self.assertGreater(g[5, 5], 230)      # cream paper -> near white

    def test_finds_skew(self):
        for angle in (-4.3, 2.7):
            g = to_gray(music_page().rotate(angle, expand=True, fillcolor=(242, 232, 205)), False)
            found = find_skew(flatten_lighting(g))
            self.assertAlmostEqual(found, -angle, delta=0.15)

    def test_table_removed_and_crop_inside_paper(self):
        page = music_page().rotate(-3, expand=True, fillcolor=(70, 55, 40))
        photo = Image.new("RGB", (page.width + 200, page.height + 200), (70, 55, 40))
        photo.paste(page, (100, 100))
        raw = to_gray(photo, False)
        flat = whiten_dark_edges(flatten_lighting(raw), raw)
        a = np.asarray(flat)
        self.assertGreater(a[:40].mean(), 250)       # top table strip blanked
        self.assertGreater(a[:, :40].mean(), 250)    # left table strip blanked
        x0, y0, x1, y1 = find_content_box(flat)
        self.assertGreater(x1 - x0, 0.7 * W)         # music kept

    def test_dewarp_straightens_staves(self):
        flat = flatten_lighting(to_gray(music_page(), False))
        curled = curl(flat)
        fixed, moved = dewarp(curled)
        self.assertGreater(moved, 5)
        self.assertGreater(staff_sharpness(fixed), staff_sharpness(curled) * 1.2)

    def test_dewarp_leaves_flat_page_alone(self):
        flat = flatten_lighting(to_gray(music_page(), False))
        _, moved = dewarp(flat)
        self.assertLess(moved, 3)


class TestCLI(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for i in range(2):
                music_page(seed=i).rotate(2 + i, expand=True, fillcolor=(242, 232, 205)) \
                    .save(td / f"p{i}.jpg", quality=90)
            pdf, pngs = td / "out.pdf", td / "pages"
            rc = main([str(td), "-o", str(pdf), "--png-dir", str(pngs), "--dpi", "150"])
            self.assertEqual(rc, 0)
            self.assertTrue(pdf.exists() and pdf.stat().st_size > 10_000)
            files = sorted(pngs.glob("*.png"))
            self.assertEqual(len(files), 2)
            with Image.open(files[0]) as im:
                self.assertEqual(im.size, (round(8.5 * 150), round(11 * 150)))

    def test_a4_bw(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            music_page().save(td / "p.png")
            main([str(td / "p.png"), "--png-dir", str(td / "o"), "--paper", "a4", "--bw", "--dpi", "100"])
            w_in, h_in = PAPER_IN["a4"]
            with Image.open(next((td / "o").glob("*.png"))) as im:
                self.assertEqual(im.size, (round(w_in * 100), round(h_in * 100)))
                self.assertEqual(im.mode, "1")


if __name__ == "__main__":
    unittest.main()
