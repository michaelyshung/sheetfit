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
    from PySide6.QtCore import QPointF, Qt
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

    def test_3_flatten_bezier_editing(self):
        ed = self.editor("flatten")
        st = self.doc.pages[0]
        g0 = st.recipe.guides[0]
        self.assertTrue(g0.bezier)                      # auto guides are Bezier curves
        self.assertLessEqual(len(g0.anchor_strengths), 4)  # ...with few anchors
        # click the first line: its handles appear
        ed.select(0, None)
        self.app.processEvents()
        anchors = [h for h, k in ed.handles if k % 3 == 0]
        self.assertEqual(len(anchors), len(g0.anchor_strengths))
        # drag the last anchor down 12 px: its handle comes along
        h, k = [(h, k) for h, k in ed.handles if k % 3 == 0][-1]
        before = np.array(ed.bz[0][k - 1])
        h.setPos(h.pos() + QPointF(0, 12))
        self.assertAlmostEqual(ed.bz[0][k - 1][1] - before[1], 12, delta=0.01)
        ed._commit()
        self.app.processEvents()
        g = st.recipe.guides[0]
        end_before = g0.bezier[-1][1]
        self.assertAlmostEqual(g.bezier[-1][1] - end_before, 12 / st.f, delta=0.5)
        self.assertAlmostEqual(g.points[-1][1], g.bezier[-1][1], delta=0.01)  # resolved
        # double-click the line: a new anchor, same shape
        n = len(g.anchor_strengths)
        pts = np.asarray(g.points)
        mid = ed.to_scene(ed.img, *pts[len(pts) // 3])
        ed.split_at(0, mid)
        self.app.processEvents()
        g2 = st.recipe.guides[0]
        self.assertEqual(len(g2.anchor_strengths), n + 1)
        p2 = np.asarray(g2.points)
        xs = np.linspace(pts[0, 0], pts[-1, 0], 50)
        same = np.abs(np.interp(xs, p2[:, 0], p2[:, 1]) - np.interp(xs, pts[:, 0], pts[:, 1]))
        self.assertLess(same.max(), 0.5)       # shape unchanged by the split
        # its strength, then delete it again
        ed.select(0, ed.sa)
        ed._anchor_strength(40)
        ed._commit()
        self.app.processEvents()
        self.assertAlmostEqual(st.recipe.guides[0].anchor_strengths[ed.sa], 0.4)
        ed._delete()
        self.app.processEvents()
        self.assertEqual(len(st.recipe.guides[0].anchor_strengths), n)
        # + Line, then delete the whole line
        total = len(st.recipe.guides)
        ed.add.setChecked(True)
        ed._clicked(QPointF(100, 30))
        self.app.processEvents()
        self.assertEqual(len(st.recipe.guides), total + 1)
        ed.select(len(st.recipe.guides) - 1, None)
        ed._delete()
        self.app.processEvents()
        self.assertEqual(len(st.recipe.guides), total)
        self.doc.reset_step(0, "flatten")
        self.app.processEvents()

    def test_3b_straighten_slider_is_live(self):
        ed = self.editor("straighten")
        a0 = self.doc.pages[0].recipe.angle
        ed.angle.s.setSliderDown(True)          # mid-drag
        ed.angle.s.setValue(ed.angle.s.value() + 100)   # +1 degree
        self.app.processEvents()
        self.assertAlmostEqual(self.doc.pages[0].recipe.angle, a0 + 1, delta=0.01)
        ed.angle.s.setSliderDown(False)
        self.doc.reset_step(0, "straighten")
        self.app.processEvents()

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

    def test_5b_size_grab_anywhere_and_ghost_list(self):
        from PySide6.QtWidgets import QGraphicsPixmapItem
        ed = self.editor("size", page=1)
        self.assertEqual(ed.item.shapeMode(), QGraphicsPixmapItem.BoundingRectShape)
        # default ghost: the previous music page
        checked = [ed.ghost_list.item(k).data(256) for k in range(ed.ghost_list.count())
                   if ed.ghost_list.item(k).checkState() == Qt.Checked]
        self.assertEqual(checked, [0])
        ed._ghost_preset("none")
        self.app.processEvents()
        self.assertEqual(ed._ghost_set(), set())
        ed._ghost_preset("prev")

    def test_5c_result_preview_everywhere(self):
        for step in ("clean", "straighten", "flatten", "crop", "size", "contrast"):
            ed = self.editor(step)
            ed._render_preview()
            self.assertFalse(ed.result_view.pix.pixmap().isNull(), step)

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
