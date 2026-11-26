"""GUI tests: drive each editor the way a user would and check the recipe
and the exported pages. Run headless (QT_QPA_PLATFORM=offscreen).
Skipped when PySide6 isn't installed (pip install sheetfit[gui])."""
from __future__ import annotations

import math
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

from test_pipeline import curl, music_page
from sheetfit.core import flatten_lighting, to_gray


def wait(app, doc, timeout=60):
    t = time.time()
    while doc.working and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sheetfit.gui.app import MainWindow
        cls.app = QApplication.instance() or QApplication([])
        cls.td = tempfile.TemporaryDirectory()
        d = Path(cls.td.name)
        flat = flatten_lighting(to_gray(music_page(seed=11), False))
        curl(flat, 40).rotate(2.5, expand=True, fillcolor=245).save(d / "p1.png")
        music_page(seed=12).save(d / "p2.png")
        cls.dir = d
        cls.win = MainWindow()
        cls.win.resize(1200, 800)
        cls.win.show()
        cls.win.add_paths([str(d)])
        wait(cls.app, cls.win.doc)
        cls.doc = cls.win.doc

    @classmethod
    def tearDownClass(cls):
        # Tear Qt down while Python is still fully alive: on Python < 3.12,
        # PySide objects destroyed during interpreter shutdown can crash it.
        from PySide6.QtCore import QThreadPool
        cls.win.doc.dirty = False
        QThreadPool.globalInstance().waitForDone()
        cls.win.close()
        cls.win.deleteLater()
        cls.app.processEvents()
        del cls.win, cls.doc
        cls.app.processEvents()
        cls.td.cleanup()

    def editor(self, step: str, page: int = 0):
        from sheetfit.gui.model import STEPS
        k = STEPS.index(step)
        self.win.strip.setCurrentRow(page)
        self.win.show_step(k)
        self.app.processEvents()
        return self.win.editors[k]

    def test_1_pages_analyzed(self):
        self.assertEqual([p.status for p in self.doc.pages], ["ready", "ready"])
        r = self.doc.pages[0].recipe
        self.assertAlmostEqual(r.angle, -2.5, delta=0.2)
        self.assertGreaterEqual(len(r.guides), 10)
        for step in ("clean", "straighten", "flatten", "crop", "size", "contrast", "output"):
            ed = self.editor(step)
            self.assertIn("Page 1 of 2", ed.status.text(), step)

    def test_2_straighten_guide_sets_angle(self):
        ed = self.editor("straighten")
        before = self.doc.pages[0].recipe.angle
        # tilt the guide by 2 degrees and let go
        a, b = ed.a.pos(), ed.b.pos()
        dx = b.x() - a.x()
        ed.b.setPos(QPointF(b.x(), a.y() + dx * math.tan(math.radians(2))))
        ed._released(ed.b)
        self.app.processEvents()
        self.assertAlmostEqual(self.doc.pages[0].recipe.angle, before + 2, delta=0.05)
        self.assertIn("straighten", self.doc.pages[0].recipe.edited)
        self.doc.reset_step(0, "straighten")
        self.assertAlmostEqual(self.doc.pages[0].recipe.angle, before, delta=0.05)
        self.assertNotIn("straighten", self.doc.pages[0].recipe.edited)

    def test_3_flatten_point_drag_and_strength(self):
        ed = self.editor("flatten")
        st = self.doc.pages[0]
        g0 = [list(p) for p in st.recipe.guides[0].points]
        h = ed.items[0][2][4]          # middle point of the first guide
        h.setPos(h.pos() + QPointF(0, 10))
        ed._commit()
        moved = st.recipe.guides[0].points[4]
        self.assertAlmostEqual(moved[1] - g0[4][1], 10 / st.f, delta=0.6)
        # strength of the selected point
        ed.items[0][2][4].setSelected(True)
        self.app.processEvents()
        ed.point.set(40)
        ed._point_strength(40)
        ed._commit()
        self.assertAlmostEqual(st.recipe.guides[0].strengths[4], 0.4)
        # add a line, then delete it
        n = len(st.recipe.guides)
        ed.add.setChecked(True)
        ed._clicked(QPointF(100, 30))
        self.assertEqual(len(st.recipe.guides), n + 1)
        ed.items[-1][2][0].setSelected(True)
        self.app.processEvents()
        ed._delete()
        self.assertEqual(len(st.recipe.guides), n)
        self.doc.reset_step(0, "flatten")

    def test_4_crop_corner(self):
        ed = self.editor("crop")
        st = self.doc.pages[0]
        c0 = list(st.recipe.crop)
        h = ed.h[0]                     # top-left corner, 20 px right
        h.setPos(h.pos() + QPointF(20, 0))
        ed._released(h)
        self.assertAlmostEqual(st.recipe.crop[0] - c0[0], 20 / st.f, delta=1.5)
        self.assertAlmostEqual(st.recipe.crop[2], c0[2], delta=1.5)
        self.doc.reset_step(0, "crop")

    def test_5_size_resize_keeps_corner(self):
        ed = self.editor("size", page=1)
        before = self.doc.paper_layout(ed.DPI)[1]
        ed._resize_to(0.8)
        after = self.doc.paper_layout(ed.DPI)[1]
        r = self.doc.pages[1].recipe
        self.assertAlmostEqual(r.scale_adjust, 0.8, delta=0.01)
        self.assertAlmostEqual(after[5] / before[5], 0.8, delta=0.02)       # width
        self.assertLessEqual(abs(after[3] - before[3]), 1)                   # same x
        self.assertLessEqual(abs(after[4] - before[4]), 1)                   # same y
        self.doc.reset_step(1, "size")

    def test_6_export_matches_cli_project(self):
        from sheetfit.cli import main
        self.doc.update(1, "size", scale_adjust=0.9, offset_in=[0.2, 0.5])
        self.doc.set_settings(dpi=100)
        out = {}
        self.doc.export(lambda pages: out.setdefault("p", pages), lambda m: out.setdefault("err", m))
        wait(self.app, self.doc)
        self.assertNotIn("err", out)
        proj = self.dir / "t.sheetfit"
        self.doc.save_project(str(proj))
        main(["--project", str(proj), "--png-dir", str(self.dir / "cli")])
        cli = sorted((self.dir / "cli").glob("*.png"))
        self.assertEqual(len(cli), len(out["p"]))
        for f, page in zip(cli, out["p"]):
            with Image.open(f) as im:
                a = np.asarray(im.convert("L"), dtype=int)
            self.assertLess(np.abs(a - np.asarray(page.convert("L"), dtype=int)).mean(), 0.5)
        self.doc.reset_step(1, "size")
        self.doc.set_settings(dpi=300)


if __name__ == "__main__":
    unittest.main()
