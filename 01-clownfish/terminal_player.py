"""Play a renderer's ASCII character grid in an ANSI terminal."""

import math
import os
import sys
import time

import numpy as np


def fit_grid(chars, ink, columns, lines):
    rows, width = chars.shape
    # Leave the last row and column free to avoid wrapping or scrolling.
    scale = min(1, max(1, columns - 1) / width, max(1, lines - 1) / rows)
    if scale == 1:
        return chars, ink

    height, width_on_screen = max(1, int(rows * scale)), max(1, int(width * scale))
    y = ((np.arange(height) + 0.5) * rows / height).astype(int)
    x = ((np.arange(width_on_screen) + 0.5) * width / width_on_screen).astype(int)
    selection = np.ix_(y, x)
    return chars[selection], ink[selection]


def ansi_frame(chars, ink):
    """Encode glyph IDs (ASCII minus 32) and RGB colors on a 0–1 scale."""
    colors = (np.clip(ink, 0, 1) * 255).astype(np.uint8)
    parts = []
    previous = None
    for row, (glyphs, rgb) in enumerate(zip(chars.tolist(), colors.tolist())):
        if row:
            parts.append("\x1b[E")
        for glyph, color in zip(glyphs, rgb):
            if glyph and color != previous:
                red, green, blue = color
                parts.append(f"\x1b[38;2;{red};{green};{blue}m")
                previous = color
            parts.append(chr(glyph + 32))
    return "".join(parts)


def play_terminal(frame_at, *, fps, duration):
    """Play one pass of a callable returning a glyph grid and foreground colors."""
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
            chars, ink = first_frame if index == 0 else frame_at(index / fps)
            size = os.get_terminal_size(output.fileno())
            if size != previous_size:
                output.write("\x1b[2J")
                previous_size = size
            chars, ink = fit_grid(chars, ink, size.columns or chars.shape[1] + 1,
                                  size.lines or chars.shape[0] + 1)
            output.write("\x1b[1;1H" + ansi_frame(chars, ink))
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
