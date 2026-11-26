"""Per-page recipes and the project file.

A recipe records every decision made for one page -- the automatic ones and
any manual edits -- so a page can be re-rendered exactly. The GUI edits
recipes; the CLI can render a saved project with `sheetfit --project`.

Geometry (crop box, flatten guide lines) lives in the *straightened frame*:
pixel coordinates of the source photo after the 90-degree turn and the
straighten rotation, measured from the image centre. Because rotation keeps
the centre fixed, changing the angle doesn't invalidate the other edits.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

FORMAT_VERSION = 1


@dataclass
class Guide:
    """A line that should be straight after flattening (normally a staff).

    points: [[x, y], ...] in the straightened frame, sorted by x, tracing
    where the line actually is on the (curled) page.
    strengths: one per point; 1 = pull fully flat, 0 = leave alone, >1 =
    overcorrect. Interpolated between points.
    """
    points: list[list[float]]
    strengths: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = sorted([[float(x), float(y)] for x, y in self.points])
        if len(self.strengths) != len(self.points):
            self.strengths = [1.0] * len(self.points)


@dataclass
class PageRecipe:
    source: str
    include: bool = True
    kind: str = "music"            # "music" or "photo"
    rotate90: int = 0              # clockwise quarter turns before anything else
    angle: float = 0.0             # straighten rotation, degrees CCW
    guides: list[Guide] = field(default_factory=list)
    flatten_strength: float = 1.0
    crop: list[float] | None = None    # [x0, y0, x1, y1] in the straightened frame
    staff_spacing: float | None = None  # px in the straightened frame
    scale_adjust: float = 1.0      # manual size tweak on top of the automatic fit
    offset_in: list[float] = field(default_factory=lambda: [0.0, 0.0])  # inches on paper
    edited: list[str] = field(default_factory=list)  # steps changed by hand

    @classmethod
    def from_dict(cls, d: dict) -> "PageRecipe":
        d = dict(d)
        d["guides"] = [Guide(**g) for g in d.get("guides", [])]
        return cls(**d)


@dataclass
class Settings:
    """Project-wide settings (mirror the CLI options)."""
    paper: str = "letter"
    dpi: int = 300
    margin: float = 0.4
    orientation: str = "portrait"
    align: str = "top"
    white_point: int = 215
    darken: float = 1.8
    bw: bool = False
    bw_threshold: int = 170
    keep_color_marks: bool = False
    flatten_lighting: bool = True
    clean_edges: bool = True
    uniform_scale: bool = True
    non_music: str = "photo"       # photo / skip / music
    max_angle: float = 10.0


@dataclass
class Project:
    settings: Settings = field(default_factory=Settings)
    pages: list[PageRecipe] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        base = path.parent.resolve()
        data = {"format": "sheetfit-project", "version": FORMAT_VERSION,
                "settings": asdict(self.settings), "pages": []}
        for p in self.pages:
            d = asdict(p)
            src = Path(p.source).resolve()
            try:  # keep projects portable: store paths relative to the project
                d["source"] = str(src.relative_to(base))
            except ValueError:
                d["source"] = str(src)
            data["pages"].append(d)
        path.write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        data = json.loads(path.read_text())
        if data.get("format") != "sheetfit-project":
            raise ValueError(f"{path} is not a sheetfit project")
        if data.get("version", 0) > FORMAT_VERSION:
            raise ValueError(f"{path} was made by a newer sheetfit; please upgrade")
        pages = []
        for d in data["pages"]:
            p = PageRecipe.from_dict(d)
            src = Path(p.source)
            if not src.is_absolute():
                p.source = str((path.parent / src).resolve())
            pages.append(p)
        return cls(settings=Settings(**data.get("settings", {})), pages=pages)
