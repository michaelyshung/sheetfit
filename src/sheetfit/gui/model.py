"""The open project as the app sees it.

Editing happens on a preview copy of each photo (at most PREVIEW_PX on the
long side) so every change redraws instantly; recipes always store full
resolution coordinates, and export re-renders from the original photos.
Slow work (analysing photos, re-cleaning, export) runs on a thread pool.
"""
from __future__ import annotations

import copy
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ..core import boost_contrast, find_skew, place
from ..pipeline import (
    Source,
    analyze,
    auto_crop,
    auto_flatten_guides,
    auto_scales,
    box_to_abs,
    layout,
    load_source,
    page_scale,
    render,
    straighten,
)
from ..recipe import Guide, PageRecipe, Project, Settings
from ..warp import apply_guides

PREVIEW_PX = 1600
STEPS = ["clean", "straighten", "flatten", "crop", "size", "contrast", "output"]
STEP_NAMES = {"clean": "Clean", "straighten": "Straighten", "flatten": "Flatten",
              "crop": "Crop", "size": "Size and position", "contrast": "Contrast",
              "output": "Output"}
SETTING_NAMES = {"white_point": "Clean background", "darken": "Darken lines",
                 "bw": "Black and white", "bw_threshold": "Black and white",
                 "paper": "Paper size", "margin": "Margins", "orientation": "Orientation",
                 "align": "Placement", "uniform_scale": "Same staff size",
                 "keep_color_marks": "Coloured markings", "flatten_lighting": "Even out lighting",
                 "clean_edges": "Page edges", "dpi": "Resolution", "non_music": "Non-music pages"}


def _preview_of(src: Source) -> tuple[Source, float]:
    f = min(1.0, PREVIEW_PX / max(src.clean.size))
    if f >= 1.0:
        return src, 1.0
    size = (round(src.clean.width * f), round(src.clean.height * f))
    return Source(src.path, src.rotate90, src.rgb.resize(size, Image.LANCZOS),
                  src.clean.resize(size, Image.LANCZOS), src._key), f


def scale_recipe(r: PageRecipe, f: float) -> PageRecipe:
    """Copy of a recipe with its geometry scaled by f (full res -> preview)."""
    s = copy.deepcopy(r)
    if s.crop:
        s.crop = [v * f for v in s.crop]
    for g in s.guides:
        g.map(lambda x, y: (x * f, y * f))
    if s.staff_spacing:
        s.staff_spacing *= f
    return s


# ---------------------------------------------------------------- threads
class _Signals(QObject):
    done = Signal(object, object)    # (token, result)
    failed = Signal(object, str)     # (token, message)


class Job(QRunnable):
    def __init__(self, token, fn, *args):
        super().__init__()
        self.token, self.fn, self.args = token, fn, args
        self.signals = _Signals()

    def run(self):
        try:
            res = self.fn(*self.args)
        except Exception as e:  # report, don't crash the app
            traceback.print_exc()
            self.signals.failed.emit(self.token, str(e))
            return
        self.signals.done.emit(self.token, res)


def _analyze_job(path: str, settings: Settings, rotate90: int):
    r, _, src = analyze(path, settings, rotate90=rotate90)
    prev, f = _preview_of(src)
    return r, prev, f


def _reclean_job(path: str, rotate90: int, settings: Settings):
    return _preview_of(load_source(path, rotate90, settings))


# ---------------------------------------------------------------- pages
class PageState:
    def __init__(self, recipe: PageRecipe):
        self.recipe = recipe
        self.preview: Source | None = None
        self.f = 1.0
        self.status = "pending"   # pending, working, ready, photo, skipped, check, error
        self.error = ""
        self.cache: dict[str, Image.Image] = {}

    @property
    def name(self) -> str:
        return Path(self.recipe.source).name

    def invalidate(self, frm: str) -> None:
        """Drop cached images for step `frm` and everything after it."""
        order = ["straight", "flat", "content", "final"]
        start = {"clean": 0, "straighten": 0, "flatten": 1, "crop": 2,
                 "contrast": 3}.get(frm, 0)
        for k in order[start:]:
            self.cache.pop(k, None)


class Document(QObject):
    page_changed = Signal(int)       # one page's recipe or images changed
    pages_changed = Signal()         # pages added, removed or reordered
    settings_changed = Signal()
    busy = Signal(str)               # status text ("" when idle)
    history_changed = Signal()       # undo/redo availability changed

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.pages: list[PageState] = []
        self.path: Path | None = None
        self.dirty = False
        self.pool = QThreadPool.globalInstance()
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self._last_tag = None
        self._last_time = 0.0
        self._quiet_history = 0
        self._jumping = False
        self._origin = "Start"
        self._jobs = 0
        self._token = 0
        self._callbacks: dict[int, tuple] = {}

    # ---- basic access
    @property
    def settings(self) -> Settings:
        return self.project.settings

    def _sync(self) -> None:
        self.project.pages = [p.recipe for p in self.pages]

    def _run(self, fn, args, done, failed=None) -> None:
        self._token += 1
        job = Job(self._token, fn, *args)
        job.setAutoDelete(False)  # we hold it until its result is delivered
        self._callbacks[self._token] = (done, failed, job)
        # bound methods of this QObject: Qt queues the call to the UI thread
        job.signals.done.connect(self._job_done)
        job.signals.failed.connect(self._job_failed)
        self._jobs += 1
        self._busy()
        self.pool.start(job)

    def _job_done(self, token, result) -> None:
        done, _, _ = self._callbacks.pop(token, (None, None, None))
        self._jobs -= 1
        if done:
            done(result)
        self._busy()

    def _job_failed(self, token, msg) -> None:
        _, failed, _ = self._callbacks.pop(token, (None, None, None))
        self._jobs -= 1
        if failed:
            failed(msg)
        self._busy()

    def _busy(self) -> None:
        self.busy.emit(f"Working on {self._jobs} page{'s' if self._jobs != 1 else ''}…" if self._jobs else "")

    @property
    def working(self) -> bool:
        return self._jobs > 0

    # ---- undo / redo
    HISTORY_LIMIT = 100
    RECLEAN_KEYS = ("keep_color_marks", "flatten_lighting", "clean_edges")

    def _snapshot(self) -> tuple:
        return ([(st, copy.deepcopy(st.recipe)) for st in self.pages],
                copy.deepcopy(self.settings))

    def record(self, tag=None, label: str = "Edit") -> None:
        """Save the current state before a change named `label`. Repeated
        changes with the same tag in quick succession (a slider drag) count
        as one step."""
        if self._quiet_history:
            return
        now = time.monotonic()
        if tag is not None and tag == self._last_tag and now - self._last_time < 1.5:
            self._last_time = now
            return
        self._undo.append((self._snapshot(), label))
        del self._undo[:-self.HISTORY_LIMIT]
        self._redo.clear()
        self._last_tag, self._last_time = tag, now
        self.history_changed.emit()

    @contextmanager
    def batch(self, label: str = "Edit", tag=None):
        """Several changes that undo as one step."""
        self.record(tag, label)
        self._quiet_history += 1
        try:
            yield
        finally:
            self._quiet_history -= 1

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if self._undo:
            snap, label = self._undo.pop()
            self._redo.append((self._snapshot(), label))
            self._restore(snap)

    def redo(self) -> None:
        if self._redo:
            snap, label = self._redo.pop()
            self._undo.append((self._snapshot(), label))
            self._restore(snap)

    def history(self) -> tuple[str, list[str], list[str]]:
        """(name of the starting state, steps done oldest first, steps undone
        in the order they'd be redone)."""
        return (self._origin, [lb for _, lb in self._undo], [lb for _, lb in reversed(self._redo)])

    def jump(self, k: int) -> None:
        """Go to the state after the first k steps (0 = the starting state)."""
        if k == len(self._undo):
            return
        self._jumping = True
        try:
            while len(self._undo) > k and self._undo:
                self.undo()
            while len(self._undo) < k and self._redo:
                self.redo()
        finally:
            self._jumping = False
        self.settings_changed.emit()
        self.pages_changed.emit()
        self.history_changed.emit()

    def _restore(self, snap) -> None:
        pages, settings = snap
        old_pages = list(self.pages)
        old = self.settings
        reclean = any(getattr(old, k) != getattr(settings, k) for k in self.RECLEAN_KEYS)
        settings_changed = old != settings
        self.project.settings = copy.deepcopy(settings)
        changed = []
        for st, r in pages:
            if st.recipe != r:
                st.recipe = copy.deepcopy(r)
                st.cache.clear()
                changed.append(st)
                if st.preview is not None and st.preview.rotate90 != r.rotate90 % 4:
                    self._load_preview(st)   # a 90-degree turn was undone
            st.status = (("photo" if st.recipe.kind == "photo" else "ready")
                         if st.recipe.include else "skipped") if st.preview is not None else st.status
        self.pages = [st for st, _ in pages]
        self._sync()
        self.dirty = True
        self._last_tag = None                # the next edit starts a new step
        if settings_changed:
            for st in self.pages:
                st.invalidate("clean" if reclean else "contrast")
                if reclean:
                    self._load_preview(st)
        if self._jumping:                    # jump() signals once at the end
            return
        if settings_changed:
            self.settings_changed.emit()
        if [id(p) for p in old_pages] != [id(p) for p in self.pages]:
            self.pages_changed.emit()
        else:
            for st in (self.pages if settings_changed else changed):
                self.page_changed.emit(self.pages.index(st))
        self.history_changed.emit()

    # ---- adding and analysing
    def add_files(self, paths: list[str]) -> None:
        self.record(label=f"Add {len(paths)} photo{'s' if len(paths) != 1 else ''}")
        for p in paths:
            st = PageState(PageRecipe(source=str(Path(p).resolve())))
            self.pages.append(st)
            self._analyze(st)
        self._sync()
        self.dirty = True
        self.pages_changed.emit()

    def _analyze(self, st: PageState, rotate90: int | None = None) -> None:
        st.status = "working"
        rot = st.recipe.rotate90 if rotate90 is None else rotate90

        def done(res, st=st):
            r, prev, f = res
            r.source = st.recipe.source
            st.recipe, st.preview, st.f = r, prev, f
            st.cache.clear()
            st.status = ("photo" if r.kind == "photo" else "ready") if r.include else "skipped"
            if r.kind == "music" and not r.staff_spacing:
                st.status = "check"
            self._sync()
            self._emit_page(st)

        def failed(msg, st=st):
            st.status, st.error = "error", msg
            self._emit_page(st)

        self._run(_analyze_job, (st.recipe.source, self.settings, rot), done, failed)

    def reanalyze(self, i: int, rotate90: int | None = None) -> None:
        self.record(label=f"Turn page {i + 1}" if rotate90 is not None
                    else f"Redo page {i + 1} automatically")
        self._analyze(self.pages[i], rotate90)

    def _emit_page(self, st: PageState) -> None:
        if st in self.pages:
            self.page_changed.emit(self.pages.index(st))

    def remove(self, i: int) -> None:
        self.record(label=f"Remove page {i + 1}")
        del self.pages[i]
        self._sync()
        self.dirty = True
        self.pages_changed.emit()

    def move(self, i: int, j: int) -> None:
        self.record(label=f"Move page {i + 1} to {j + 1}")
        st = self.pages.pop(i)
        self.pages.insert(j, st)
        self._sync()
        self.dirty = True
        self.pages_changed.emit()

    # ---- projects
    def open_project(self, path: str) -> None:
        self.project = Project.load(path)
        self.path = Path(path)
        self._undo.clear()
        self._redo.clear()
        self._origin = f"Opened {self.path.name}"
        self.history_changed.emit()
        self.pages = [PageState(r) for r in self.project.pages]
        for st in self.pages:
            self._load_preview(st)
        self.dirty = False
        self.pages_changed.emit()
        self.settings_changed.emit()

    def save_project(self, path: str | None = None) -> None:
        self._sync()
        self.path = Path(path) if path else self.path
        self.project.save(self.path)
        self.dirty = False

    def _load_preview(self, st: PageState) -> None:
        st.status = "working"

        def done(res, st=st):
            st.preview, st.f = res
            st.cache.clear()
            r = st.recipe
            st.status = ("photo" if r.kind == "photo" else "ready") if r.include else "skipped"
            self._emit_page(st)

        def failed(msg, st=st):
            st.status, st.error = "error", msg
            self._emit_page(st)

        self._run(_reclean_job, (st.recipe.source, st.recipe.rotate90, self.settings), done, failed)

    # ---- editing
    def update(self, i: int, step: str, **changes) -> None:
        """Change recipe fields for page i, made by hand in editor `step`."""
        st = self.pages[i]
        if "include" in changes:
            label = f"{'Include' if changes['include'] else 'Leave out'} page {i + 1}"
        else:
            label = f"{STEP_NAMES.get(step, step)} · page {i + 1}"
        self.record(("update", id(st), step, tuple(sorted(changes))), label)
        for k, v in changes.items():
            setattr(st.recipe, k, v)
        if step not in st.recipe.edited:
            st.recipe.edited.append(step)
        st.invalidate(step)
        self.dirty = True
        self.page_changed.emit(i)

    def reset_step(self, i: int, step: str) -> None:
        """Redo one step automatically (the editor's Auto button)."""
        st = self.pages[i]
        if st.preview is None:
            return
        with self.batch(f"Auto {STEP_NAMES.get(step, step).lower()} · page {i + 1}"):
            self._reset_step(i, st, step)

    def _reset_step(self, i: int, st: PageState, step: str) -> None:
        f = st.f
        if step == "straighten":
            angle = find_skew(st.preview.clean, self.settings.max_angle)
            self.update(i, step, angle=angle)
        elif step == "flatten":
            g = auto_flatten_guides(self.stage(i, "straight"))
            for gd in g:
                gd.map(lambda x, y: (x / f, y / f))
            self.update(i, step, guides=g, flatten_strength=1.0)
        elif step == "crop":
            c = auto_crop(self.stage(i, "flat"))
            self.update(i, step, crop=[v / f for v in c])
        elif step == "size":
            self.update(i, step, scale_adjust=1.0, offset_in=[0.0, 0.0])
        if step in st.recipe.edited:
            st.recipe.edited.remove(step)
        self.page_changed.emit(i)

    def set_settings(self, reclean: bool = False, **changes) -> None:
        names = sorted({SETTING_NAMES.get(k, k) for k in changes})
        self.record(("settings", tuple(sorted(changes))), ", ".join(names))
        self.project.settings = replace(self.settings, **changes)
        self.dirty = True
        for st in self.pages:
            st.invalidate("clean" if reclean else "contrast")
            if reclean:
                self._load_preview(st)
        self.settings_changed.emit()
        for i in range(len(self.pages)):
            self.page_changed.emit(i)

    # ---- preview images
    def stage(self, i: int, name: str) -> Image.Image | None:
        """Preview image after a step: straight, flat, content, final."""
        st = self.pages[i]
        if st.preview is None:
            return None
        if name in st.cache:
            return st.cache[name]
        r = scale_recipe(st.recipe, st.f)
        if name == "straight":
            img = straighten(st.preview.clean, r.angle)
        elif name == "flat":
            img, _ = apply_guides(self.stage(i, "straight"), r.guides, r.flatten_strength)
        elif name == "content":
            flat = self.stage(i, "flat")
            img = flat.crop(box_to_abs(r.crop, flat.size)) if r.crop else flat
        elif name == "final":
            if r.kind == "photo":
                img = render(r, self.settings, st.preview)
            else:
                img = boost_contrast(self.stage(i, "content"), self.settings.white_point,
                                     self.settings.darken)
        else:
            raise KeyError(name)
        st.cache[name] = img
        return img

    def to_preview(self, i: int, v: float) -> float:
        return v * self.pages[i].f

    def to_full(self, i: int, v: float) -> float:
        return v / self.pages[i].f

    def included(self) -> list[int]:
        return [i for i, p in enumerate(self.pages)
                if p.recipe.include and p.preview is not None and p.status != "error"]

    def paper_layout(self, dpi: int) -> dict[int, tuple]:
        """Where each included page lands on paper at `dpi`, using previews:
        {i: (final_img, page_w, page_h, x, y, w, h)}."""
        idx = self.included()
        if not idx:
            return {}
        s = replace(self.settings, dpi=dpi)
        imgs = [self.stage(i, "final") for i in idx]
        sp = [self.pages[i].recipe.staff_spacing * self.pages[i].f
              if self.pages[i].recipe.kind == "music" and self.pages[i].recipe.staff_spacing else None
              for i in idx]
        autos = auto_scales(imgs, sp, s)
        out = {}
        for i, img, auto in zip(idx, imgs, autos):
            r = self.pages[i].recipe
            sc = page_scale(img, auto, r, s)
            out[i] = (img,) + place(img.size, dpi, s.margin, s.orientation, s.align, s.paper,
                                    sc, tuple(r.offset_in))
        return out

    # ---- export
    def export(self, done, failed) -> None:
        """Render every included page at full resolution in the background;
        calls done(list of page images)."""
        self._sync()
        recipes = [self.pages[i].recipe for i in self.included()]
        settings = self.settings

        def job():
            imgs = [render(r, settings) for r in recipes]
            return layout(recipes, imgs, settings)

        self._run(job, (), done, failed)


def new_guide_at(y: float, x0: float, x1: float, n: int = 9) -> Guide:
    """A fresh, flat guide across [x0, x1] at height y (frame coordinates)."""
    return Guide(points=[[x0 + (x1 - x0) * k / (n - 1), y] for k in range(n)])
