"""Convert a lit portrait scene to a colored ASCII grid."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 6, 12
PAPER = np.array((8, 16, 25))
RAMP = " .,:-+=xo08%#@"


def glyph_atlas():
    for fontname in ("DejaVuSansMono-Bold.ttf", "/System/Library/Fonts/Menlo.ttc", "consola.ttf"):
        try:
            font = ImageFont.truetype(fontname, 11)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default()
    atlas = np.zeros((128, CELL_H, CELL_W), dtype=np.float32)
    for glyph in set(RAMP + "/\\_|"):
        tile = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(tile)
        draw.text((0, -1), glyph, font=font, fill=255)
        atlas[ord(glyph)] = np.asarray(tile) / 255
    return atlas


class AsciiCamera:
    def __init__(self, columns, rows=None):
        self.cols = columns
        self.rows = rows if rows is not None else max(2, round(columns * 16 / 9 * CELL_W / CELL_H))
        unit = max(1, round(columns * CELL_W / 9))
        self.size = (9 * unit, 16 * unit)
        self.atlas = glyph_atlas()

    def grid(self, image):
        # The projected shape, lighting and camera come from the input frame.
        # Box sampling integrates coverage across each rectangular glyph cell.
        if "A" in image.getbands() or "transparency" in image.info:
            backing = Image.new("RGBA", image.size, tuple(PAPER.tolist()) + (255,))
            image = Image.alpha_composite(backing, image.convert("RGBA"))
        rgb = np.asarray(image.convert("RGB").resize((self.cols, self.rows), Image.Resampling.BOX),
                         dtype=np.float32)
        light = (rgb @ np.array((.2126, .7152, .0722))) / 255
        levels = np.clip(np.sqrt(light) * len(RAMP) - 2.6, 0, len(RAMP)-1).astype(int)
        codes = np.frombuffer(RAMP.encode(), dtype=np.uint8)[levels]
        dy, dx = np.gradient(light)
        # The contour runs perpendicular to the brightness gradient.
        edge = (np.hypot(dx, dy) > .105) & (light > .15)
        slope = np.where(dx * dy > 0, ord("/"), ord("\\"))
        edge_glyph = np.where(abs(dx) > 2*abs(dy), ord("|"),
                              np.where(abs(dy) > 2*abs(dx), ord("_"), slope))
        codes = np.where(edge, edge_glyph, codes)
        return codes, rgb

    def frame(self, image):
        codes, rgb = self.grid(image)
        mask = self.atlas[codes][..., None]
        ink = np.minimum(255, rgb * 1.15 + 14)
        back = PAPER + .15 * np.maximum(rgb - 55, 0)
        cells = back[:, :, None, None, :] + mask * (ink-back)[:, :, None, None, :]
        pixels = cells.transpose(0, 2, 1, 3, 4).reshape(self.rows * CELL_H, self.cols * CELL_W, 3)
        return Image.fromarray(pixels.astype(np.uint8)).resize(self.size, Image.Resampling.LANCZOS)
