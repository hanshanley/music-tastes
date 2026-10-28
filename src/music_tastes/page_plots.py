"""Export Margin of Error page-background copies of the Substack originals."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

from .paths import FIGURES
from .vizstyle import BG

PAGE_BACKGROUND = "#F5F5F5"


def export(figures: Path = FIGURES) -> list[Path]:
    figures = Path(figures)
    output = figures / "page_background"
    sources = sorted(
        path for path in figures.rglob("*.png") if not path.is_relative_to(output)
    )
    if not sources:
        raise ValueError("No original plot PNGs found")
    background = (*ImageColor.getrgb(BG), 255)
    replacement = (*ImageColor.getrgb(PAGE_BACKGROUND), 255)
    exported = []
    for source in sources:
        with Image.open(source) as image:
            pixels = np.array(image.convert("RGBA"))
            dpi = image.info.get("dpi")
        mask = np.all(pixels == background, axis=2)
        if not mask.any():
            raise ValueError(f"{source}: expected chart background {BG} is missing")
        pixels[mask] = replacement
        destination = output / source.relative_to(figures)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(destination, **({"dpi": dpi} if dpi else {}))
        exported.append(destination)
    for stale in sorted(output.rglob("*.png")):
        if stale not in exported:
            stale.unlink()
    print(f"Exported {len(exported)} {PAGE_BACKGROUND} chart copies to {output}")
    return exported
