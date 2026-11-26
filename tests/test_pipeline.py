"""End-to-end and unit tests on synthetic sheet-music photos.

Run with:  python -m unittest discover tests     (or: pytest)
"""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sheetfit.core import (
    HEIF_OK,
    PAPER_IN,
    dewarp,
    find_content_box,
    find_skew,
    flatten_lighting,
    main,
    staff_spacing,
    to_gray,
    whiten_dark_edges,
)

W, H = 1200, 1550


def music_page(seed: int = 0, paper=(242, 232, 205), page_no: bool = False) -> Image.Image:
    """A fake page: 6 systems of 2 staves (line gap 14 px) with notes."""
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
    if page_no:
        d.rectangle([1050, 1500, 1062, 1518], fill=(30, 30, 30))  # a small "page number"
    return pg


def photo_page(seed: int = 0) -> Image.Image:
    """A non-music page: smooth blobs and a face-ish shape, no staff lines."""
    rnd = random.Random(seed)
    pg = Image.new("RGB", (W, H), (200, 180, 150))
    d = ImageDraw.Draw(pg)
    for _ in range(25):
        x, y, r = rnd.randint(0, W), rnd.randint(0, H), rnd.randint(40, 250)
        c = tuple(rnd.randint(30, 230) for _ in range(3))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
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


class TestStaffSize(unittest.TestCase):
    def test_measures_staff_spacing(self):
        g = flatten_lighting(to_gray(music_page(), False))
        self.assertAlmostEqual(staff_spacing(g), 14.0, delta=0.5)

    def test_spacing_scales_with_zoom(self):
        g = flatten_lighting(to_gray(music_page(), False))
        g2 = g.resize((int(W * 0.7), int(H * 0.7)), Image.LANCZOS)
        self.assertAlmostEqual(staff_spacing(g2), 14.0 * 0.7, delta=0.5)

    def test_non_music_has_no_spacing(self):
        g = flatten_lighting(to_gray(photo_page(), False))
        self.assertIsNone(staff_spacing(g))


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

    def test_same_staff_size_on_every_page(self):
        """Photos taken from different distances print at the same staff size."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            music_page(seed=1).save(td / "a.png")
            near = music_page(seed=2)
            near.resize((int(W * 1.3), int(H * 1.3)), Image.LANCZOS).save(td / "b.png")
            # c is a small crop, so a plain "fill the page" would enlarge it a lot
            music_page(seed=3).crop((0, 0, 800, 800)).save(td / "c.png")
            main([str(td), "--png-dir", str(td / "o"), "--dpi", "150"])
            sizes = []
            for f in sorted((td / "o").glob("*.png")):
                with Image.open(f) as im:
                    sizes.append(staff_spacing(im.convert("L")))
            self.assertEqual(len(sizes), 3)
            self.assertLess(max(sizes) / min(sizes), 1.03)

    def test_page_number_kept(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            music_page(page_no=True).save(td / "p.png")
            main([str(td / "p.png"), "--png-dir", str(td / "o"), "--dpi", "100"])
            with Image.open(next((td / "o").glob("*.png"))) as im:
                a = np.asarray(im.convert("L"))
            rows = np.flatnonzero((a < 100).any(axis=1))
            # the page number sits ~60 px (at 100 dpi) below the last staff
            last_staff = np.flatnonzero((a < 100).mean(axis=1) > 0.3).max()
            self.assertGreater(rows.max() - last_staff, 25)

    def test_non_music_photo_and_skip(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            photo_page().save(td / "0cover.png")
            music_page().save(td / "1music.png")
            main([str(td / "0cover.png"), str(td / "1music.png"), "--png-dir", str(td / "keep"), "--dpi", "80"])
            self.assertEqual(len(list((td / "keep").glob("*.png"))), 2)
            main([str(td / "0cover.png"), str(td / "1music.png"), "--png-dir", str(td / "skip"),
                  "--dpi", "80", "--non-music", "skip"])
            self.assertEqual([f.name for f in (td / "skip").glob("*.png")], ["001_1music.png"])

    @unittest.skipUnless(HEIF_OK, "pillow-heif not installed")
    def test_heic_input(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            music_page().save(td / "p.heic", quality=90)
            rc = main([str(td), "-o", str(td / "o.pdf"), "--dpi", "80"])
            self.assertEqual(rc, 0)
            self.assertTrue((td / "o.pdf").exists())


if __name__ == "__main__":
    unittest.main()
