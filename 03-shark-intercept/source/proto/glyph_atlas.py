"""Printable ASCII glyphs as fixed-size coverage masks, indexed by ord(c) - 32."""

import numpy as np
from PIL import Image, ImageDraw


def glyph_atlas(font, cell_width, cell_height):
    out = np.zeros((96, cell_height, cell_width), np.float32)
    for code in range(32, 127):
        tile = Image.new("L", (cell_width, cell_height))
        lift = 5 if code == ord("_") else 3
        ImageDraw.Draw(tile).text((0, -lift), chr(code), font=font, fill=255)
        out[code - 32] = np.asarray(tile, np.float32) / 255
    return out
