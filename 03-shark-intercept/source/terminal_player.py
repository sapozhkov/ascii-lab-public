"""Play a renderer's ASCII character grid in an ANSI terminal."""

import math
import os
import sys
import time

import numpy as np


def fit_grid(layers, columns, lines):
    rows, width = layers[0].shape
    # Leave the last row and column free to avoid wrapping or scrolling.
    scale = min(1, max(1, columns - 1) / width, max(1, lines - 1) / rows)
    if scale == 1:
        return layers

    height, width_on_screen = max(1, int(rows * scale)), max(1, int(width * scale))
    y = ((np.arange(height) + 0.5) * rows / height).astype(int)
    x = ((np.arange(width_on_screen) + 0.5) * width / width_on_screen).astype(int)
    selection = np.ix_(y, x)
    return tuple(layer[selection] for layer in layers)


def rgb_rows(colors):
    return (np.clip(colors, 0, 1) * 255).astype(np.uint8).tolist()


def ansi_frame(chars, ink, paper=None):
    """Encode glyph IDs (ASCII minus 32) and RGB colors on a 0–1 scale."""
    backgrounds = rgb_rows(paper) if paper is not None else None
    parts = []
    previous_ink = previous_paper = None
    for row, (glyphs, rgb) in enumerate(zip(chars.tolist(), rgb_rows(ink))):
        if row:
            parts.append("\x1b[E")
        background = backgrounds[row] if backgrounds is not None else None
        for column, (glyph, color) in enumerate(zip(glyphs, rgb)):
            if background is not None and background[column] != previous_paper:
                previous_paper = background[column]
                red, green, blue = previous_paper
                parts.append(f"\x1b[48;2;{red};{green};{blue}m")
            if glyph and color != previous_ink:
                red, green, blue = color
                parts.append(f"\x1b[38;2;{red};{green};{blue}m")
                previous_ink = color
            parts.append(chr(glyph + 32))
    return "".join(parts)


def play_terminal(frame_at, *, fps, duration):
    """Play glyphs, foreground RGB and optional background RGB from frame_at."""
    if not math.isfinite(duration) or duration <= 0:
        raise SystemExit("Terminal playback duration must be a positive number.")
    output = sys.stdout
    if not output.isatty():
        raise SystemExit("Run terminal playback directly in a terminal.")

    previous_size = None
    try:
        output.write("\x1b[?1049h\x1b[?25l\x1b[48;2;0;0;0m")
        output.flush()
        first_frame = frame_at(0)
        start = time.monotonic()
        index = 0
        while index / fps < duration:
            layers = first_frame if index == 0 else frame_at(index / fps)
            size = os.get_terminal_size(output.fileno())
            if size != previous_size:
                output.write("\x1b[48;2;0;0;0m\x1b[2J")
                previous_size = size
            rows, columns = layers[0].shape
            visible = fit_grid(layers, size.columns or columns + 1, size.lines or rows + 1)
            output.write("\x1b[1;1H" + ansi_frame(*visible))
            output.flush()

            elapsed = time.monotonic() - start
            # Skip overdue frames so rendering delays do not slow the scene.
            index = max(index + 1, int(elapsed * fps))
            delay = min(index / fps, duration) - elapsed
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        output.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        output.flush()
