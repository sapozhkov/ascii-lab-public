"""Play or render the Lotus ASCII animation."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time


RUNTIME = Path(__file__).resolve().parent / 'runtime'
PERIOD = 20


def scene_time(value):
    seconds = float(value)
    if not 0 <= seconds < PERIOD:
        raise argparse.ArgumentTypeError(f'time must be in [0, {PERIOD}) seconds')
    return seconds


def terminal_frame(camera, garden, seconds, first, last, color):
    import numpy as np
    from PIL import Image

    codes, rgb = camera.grid(Image.fromarray(garden.frame(seconds)))
    glyphs = codes[first:last].astype('uint8')
    if not color:
        return '\n'.join(row.tobytes().decode('ascii') for row in glyphs)
    ink = np.clip(rgb[first:last] * 1.15 + 14, 0, 255).astype('uint8')
    rows = []
    for glyph_row, color_row in zip(glyphs, ink):
        cells = []
        for glyph, (red, green, blue) in zip(glyph_row, color_row):
            cells.append(f'\033[38;2;{red};{green};{blue}m{chr(glyph)}' if glyph != 32 else ' ')
        rows.append(''.join(cells) + '\033[0m')
    return '\n'.join(rows)


def play(garden, seconds):
    from ascii_preview import AsciiCamera
    from living import SIZE

    screen = shutil.get_terminal_size((80, 24))
    fraction = garden.panel.scene_height / SIZE[1]
    columns = max(2, min(135, screen.columns - 1, int((screen.lines - 1) * 9 / 8 / fraction)))
    rows = max(2, round(columns * 8 / 9))
    first = round(rows * garden.panel.crop_top / SIZE[1])
    last = round(rows * (garden.panel.crop_top + garden.panel.scene_height) / SIZE[1])
    camera = AsciiCamera(columns, rows)
    color = sys.stdout.isatty()
    if seconds is not None:
        print(terminal_frame(camera, garden, seconds, first, last, color))
        return

    fps = 12
    frames = PERIOD * fps
    frame = 0
    start = time.monotonic()
    prefix = '\033[H' if color else ''
    if color:
        sys.stdout.write('\033[?25l\033[2J')
    try:
        while frame < frames:
            image = terminal_frame(camera, garden, frame / fps, first, last, color)
            sys.stdout.write(prefix + image)
            sys.stdout.flush()
            if not color:
                prefix = '\f\n'

            # Skip past frames that took longer to render than their interval.
            elapsed = time.monotonic() - start
            frame = max(frame + 1, int(elapsed * fps))
            next_tick = min(frame, frames) / fps
            time.sleep(max(0, next_tick - elapsed))
    finally:
        if color:
            sys.stdout.write('\033[0m\033[?25h')
        sys.stdout.write('\n')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest='mode', required=True)
    term = modes.add_parser('terminal', help='play the scene in the terminal')
    term.add_argument('--time', type=scene_time, help='print a single frame in seconds')
    png = modes.add_parser('png', help='export a PNG frame')
    png.add_argument('--time', type=scene_time, required=True, help='frame time in seconds')
    png.add_argument('--output', type=Path, default=Path('lotus.png'))
    mp4 = modes.add_parser('mp4', help='export the full silent MP4')
    mp4.add_argument('--output', type=Path, default=Path('lotus.mp4'))
    args = cli.parse_args()
    if not shutil.which('cc'):
        cli.error('cc is required to build the mesh renderer')
    if args.mode == 'mp4' and not shutil.which('ffmpeg'):
        cli.error('FFmpeg with libx264 is required to export MP4')

    sys.path.insert(0, str(RUNTIME))
    try:
        from leaf_flow import LeafFlowGarden

        garden = LeafFlowGarden()
        if args.mode == 'terminal':
            play(garden, args.time)
        elif args.mode == 'png':
            from PIL import Image

            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(garden.ascii_frame(args.time)).save(args.output, format='PNG')
        else:
            from flow_scene import export

            export(garden, args.output)
    except ModuleNotFoundError as exc:
        cli.exit(1, f'{exc}. Install the packages in requirements.txt.\n')
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f'{exc}\n')
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
