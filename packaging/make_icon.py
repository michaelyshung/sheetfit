"""Draw the sheetfit app icon: a white page with five straight staff lines
and a note, on a blue rounded square. Writes icon.png, icon.ico, icon.icns
next to this file.

    python packaging/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
N = 1024
S = 4  # supersample


def draw() -> Image.Image:
    n = N * S
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(n * 0.06)
    d.rounded_rectangle([pad, pad, n - pad, n - pad], radius=int(n * 0.2), fill=(55, 138, 221, 255))
    # page with a soft shadow, slightly tilted feel from a folded corner
    px0, py0, px1, py1 = int(n * 0.24), int(n * 0.17), int(n * 0.76), int(n * 0.83)
    d.rounded_rectangle([px0 + n * 0.015, py0 + n * 0.02, px1 + n * 0.015, py1 + n * 0.02],
                        radius=int(n * 0.03), fill=(20, 60, 110, 90))
    d.rounded_rectangle([px0, py0, px1, py1], radius=int(n * 0.03), fill=(255, 255, 255, 255))
    fold = int(n * 0.11)
    d.polygon([(px1 - fold, py0), (px1, py0), (px1, py0 + fold)], fill=(55, 138, 221, 255))
    d.polygon([(px1 - fold, py0), (px1 - fold, py0 + fold), (px1, py0 + fold)], fill=(205, 222, 240, 255))
    # two staves
    gap = int(n * 0.034)
    lw = max(1, int(n * 0.009))
    for top in (int(n * 0.33), int(n * 0.58)):
        for k in range(5):
            y = top + k * gap
            d.line([px0 + n * 0.06, y, px1 - n * 0.06, y], fill=(30, 30, 35, 255), width=lw)
    # notes
    for (cx, cy) in [(0.40, 0.33 + 0.034 * 3), (0.55, 0.33 + 0.034 * 1.5),
                     (0.45, 0.58 + 0.034 * 2.5), (0.62, 0.58 + 0.034 * 1)]:
        x, y = cx * n, cy * n
        rx, ry = n * 0.03, n * 0.022
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(30, 30, 35, 255))
        d.line([x + rx - lw / 2, y, x + rx - lw / 2, y - n * 0.11], fill=(30, 30, 35, 255), width=lw)
    return img.resize((N, N), Image.LANCZOS)


def main() -> None:
    img = draw()
    img.save(HERE / "icon.png")
    img.save(HERE / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                       (128, 128), (256, 256)])
    img.save(HERE / "icon.icns")
    print("wrote", HERE / "icon.png", HERE / "icon.ico", HERE / "icon.icns")


if __name__ == "__main__":
    main()
