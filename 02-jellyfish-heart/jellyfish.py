"""Render the silent ASCII scene from the Jellyfish Heart reel."""

import argparse
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from source.duet import Orbit, SECONDS, render
from source.reef import RockShelf, compose
from source.renderer import FPS
from source.swimmer import WIDTH


# The framing of the reel above the editing panel.
CROP_TOP, HEIGHT = 232, 1382


class Scene:
    def __init__(self):
        self.orbit = Orbit()
        self.shelf = RockShelf(self.orbit)

    def frame(self, time):
        swimmers = np.asarray(render(self.orbit, time))
        pixels = compose(swimmers, self.shelf.frame(time))
        return Image.fromarray(pixels[CROP_TOP:CROP_TOP + HEIGHT])

    def write_video(self, output, frame_count):
        command = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for frame in range(frame_count):
                    encoder.stdin.write(self.frame(frame / FPS).tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise subprocess.CalledProcessError(encoder.returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="format", required=True)
    png = commands.add_parser("png", help="Render one frame")
    png.add_argument("time", type=float, help="Seconds into the reel")
    png.add_argument("output", type=Path)
    mp4 = commands.add_parser("mp4", help="Render a silent video with FFmpeg")
    mp4.add_argument("output", type=Path)
    mp4.add_argument("--duration", type=float, default=SECONDS, help="Duration in seconds")
    args = parser.parse_args()
    if args.format == "png":
        if not math.isfinite(args.time) or not 0 <= args.time < SECONDS:
            png.error(f"time must be at least 0 and less than {SECONDS} seconds")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Scene().frame(args.time).save(args.output)
    else:
        if not math.isfinite(args.duration) or not 1 / FPS <= args.duration <= SECONDS:
            mp4.error(f"duration must be between one frame and {SECONDS} seconds")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Scene().write_video(args.output, int(args.duration * FPS))


if __name__ == "__main__":
    main()
