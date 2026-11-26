"""sheetfit — straighten, clean, crop and fit photos of sheet music to printable pages.

Usage:
  sheetfit IMG [IMG ...] -o out.pdf
  sheetfit photos/ -o book.pdf --png-dir pages/ --bw
  sheetfit photos/ --save-project book.sheetfit      # automatic recipes, editable
  sheetfit --project book.sheetfit -o book.pdf        # render a saved project
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import __version__
from .core import HEIF_OK, IMAGE_EXTS, PAPER_IN
from .pipeline import analyze, layout, render
from .recipe import PageRecipe, Project, Settings


def collect_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:
            out.append(p)
    return out


def _settings_from_args(a: argparse.Namespace) -> Settings:
    return Settings(paper=a.paper, dpi=a.dpi, margin=a.margin, orientation=a.orientation,
                    align=a.align, white_point=a.white_point, darken=a.darken, bw=a.bw,
                    bw_threshold=a.bw_threshold, keep_color_marks=a.keep_color_marks,
                    flatten_lighting=not a.no_flatten, clean_edges=not a.no_crop,
                    uniform_scale=a.uniform_scale, non_music=a.non_music,
                    max_angle=0.0 if a.no_deskew else a.max_angle)


def _describe(name: str, r: PageRecipe) -> str:
    if r.kind == "photo":
        return f"  {name}: no staff lines found -- {'kept as a photo' if r.include else 'skipped'}"
    moved = max((abs(y - g.points[len(g.points) // 2][1]) for g in r.guides
                 for _, y in g.points), default=0.0)
    sp = f"{r.staff_spacing:.1f}px" if r.staff_spacing else "-"
    return (f"  {name}: skew {r.angle:+.2f}°, {len(r.guides)} flatten guides "
            f"(curl {moved:.0f}px), staff spacing {sp}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="image files or folders (processed in name order)")
    ap.add_argument("-o", "--output", help="combined PDF to write (default: sheets.pdf unless --png-dir only)")
    ap.add_argument("--png-dir", help="also write one PNG per page here")
    ap.add_argument("--project", help="render this saved project (from --save-project or the app)")
    ap.add_argument("--save-project", metavar="FILE", help="save the automatic recipes as an editable project")
    ap.add_argument("--version", action="version", version=f"sheetfit {__version__}")
    ap.add_argument("--paper", choices=sorted(PAPER_IN), default="letter")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.4, help="page margin in inches (default 0.4)")
    ap.add_argument("--orientation", choices=["portrait", "landscape", "auto"], default="portrait")
    ap.add_argument("--align", choices=["top", "center"], default="top")
    ap.add_argument("--max-angle", type=float, default=10.0, help="largest skew to search, degrees")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="turn photos clockwise first (for sideways photos without EXIF)")
    ap.add_argument("--white-point", type=int, default=215,
                    help="grey level at/above which pixels become pure white (default 215; lower = cleaner background)")
    ap.add_argument("--darken", type=float, default=1.8,
                    help="gamma that pushes greys toward black (1 = off, default 1.8)")
    ap.add_argument("--keep-color-marks", action="store_true",
                    help="keep coloured markings (pencil, pen) as grey instead of removing all colour")
    ap.add_argument("--bw", action="store_true", help="output pure black/white (smaller, crisper prints)")
    ap.add_argument("--bw-threshold", type=int, default=170)
    ap.add_argument("--no-flatten", action="store_true", help="skip lighting correction")
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--no-dewarp", action="store_true", help="don't flatten page curl")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-uniform-scale", dest="uniform_scale", action="store_false",
                    help="scale each page to fill the paper instead of matching staff size across pages")
    ap.add_argument("--non-music", choices=["photo", "skip", "music"], default="photo",
                    help="pages without staff lines (e.g. a cover): keep as a plain photo (default), "
                         "skip them, or clean them like music")
    args = ap.parse_args(argv)

    if args.project and args.inputs:
        ap.error("give either input images or --project, not both")
    if not args.output and not args.png_dir and not args.save_project:
        args.output = "sheets.pdf"

    names: list[str] = []
    recipes: list[PageRecipe] = []
    images: list[Image.Image] = []

    if args.project:
        proj = Project.load(args.project)
        settings = proj.settings
        for r in proj.pages:
            if not r.include:
                continue
            try:
                images.append(render(r, settings))
            except Exception as e:
                print(f"  {r.source}: FAILED ({e})", file=sys.stderr)
                continue
            recipes.append(r)
            names.append(Path(r.source).stem)
        all_recipes = proj.pages
    else:
        paths = collect_inputs(args.inputs)
        if not paths:
            ap.error("no input images found")
        settings = _settings_from_args(args)
        all_recipes = []
        for p in paths:
            try:
                r, img, _ = analyze(p, settings, rotate90=args.rotate // 90)
            except Exception as e:  # keep going on a bad file
                hint = ""
                if p.suffix.lower() in (".heic", ".heif") and not HEIF_OK:
                    hint = " -- install HEIC support: pip install pillow-heif"
                print(f"  {p}: FAILED ({e}){hint}", file=sys.stderr)
                continue
            # honour the --no-* switches by undoing the automatic decision
            if args.no_dewarp and r.guides:
                r.guides = []
                img = None
            if args.no_crop:
                r.crop = None
                img = None
            if img is None and r.include:
                img = render(r, settings)
            print(_describe(p.name, r), file=sys.stderr)
            all_recipes.append(r)
            if r.include:
                recipes.append(r)
                images.append(img)
                names.append(p.stem)

    if args.save_project:
        Project(settings=settings, pages=all_recipes).save(args.save_project)
        print(f"saved project {args.save_project}", file=sys.stderr)
    if not images:
        return 0 if args.save_project else 1

    pages = layout(recipes, images, settings)
    if args.png_dir:
        d = Path(args.png_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, (name, page) in enumerate(zip(names, pages), 1):
            page.save(d / f"{i:03d}_{name}.png", dpi=(settings.dpi, settings.dpi))
    if args.output:
        pages[0].save(args.output, "PDF", resolution=settings.dpi, save_all=True,
                      append_images=pages[1:])
        print(f"wrote {args.output} ({len(pages)} pages)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
