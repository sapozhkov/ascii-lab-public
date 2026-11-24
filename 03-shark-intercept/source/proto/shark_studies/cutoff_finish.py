"""Apply the reel's typography and source panel to checkpoint C.

Run from the project root: python3 -m proto.shark_studies.cutoff_finish
"""
from functools import lru_cache
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import textwrap

from PIL import Image, ImageDraw, ImageFont

from proto.channel import HANDLE
from .cutoff_motion import states
from .school_render import FONT, FRAME_H, SCENE_H, WIDTH

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "shots/shark-cutoff-escape"
OUTPUT = ROOT / "shots/shark-cutoff-finished"
HERE = Path(__file__).resolve().parent
FPS = 24
HEADER_H = 172
PANEL_H = FRAME_H - SCENE_H
STATUS_SIZE = 20
CODE_SIZE = 21
CODE_LEADING = 30

# Contiguous excerpts from checkpoint C. Markers bound the source displayed in
# the editor. The frame at which it opens comes from the recorded model below.
EXCERPTS = {
    "entry": ("cutoff_motion.py", "def entry_pose", "return replace(pose, spine=spine, tangent=tangent)"),
    "school": ("cutoff_motion.py", "gap, outward = body_gap(pos, pose)", "if np.any(start & relay):"),
    "target": ("cutoff_motion.py", "def select_isolated_prey", "return None"),
    "burst": ("dive_pose.py", "tail_load = .62*self.tail_speed**2", "self.center += dt*self.velocity"),
    "catch": ("cutoff_motion.py", "# First contact at the jaw opening", "captured = True"),
    "release": ("cutoff_motion.py", "# The banked first rows slow", "1.4*(z-pos[:, 2])))"),
    "return": ("cutoff_motion.py", "# The incoming rows already move upstream.", "velocity[:, 1] += current-initial_current"),
}


def model_frames(path):
    text = path.read_text(encoding="utf-8")
    _, marker, payload = text.partition("const frames=")
    if not marker:
        raise ValueError(f"Frame data not found in {path}")
    frames, _ = json.JSONDecoder().raw_decode(payload)
    return frames


def panel_timeline(frames):
    first = lambda predicate: next(i for i, f in enumerate(frames) if predicate(f))
    catch = first(lambda frame: frame["caught"])
    return_start = next(i for i in range(catch + 1, len(frames)) if not frames[i]["caught"])
    return [
        (0, "entry"),
        (first(lambda frame: frame["stage"] == "School turns"), "school"),
        (first(lambda frame: frame["target"] is not None), "target"),
        (first(lambda frame: frame["stage"] == "Prey dodges"), "burst"),
        (catch, "catch"),
        (catch + round(2.2 * FPS), "release"),
        (return_start, "return"),
        (len(frames) - round(1.5 * FPS), "entry"),
    ]


@lru_cache(maxsize=7)
def source_panel(section):
    filename, start, end = EXCERPTS[section]
    lines = (HERE / filename).read_text(encoding="utf-8").splitlines()
    lo = next(i for i, line in enumerate(lines) if line.lstrip().startswith(start))
    hi = next(i for i in range(lo, len(lines)) if lines[i].lstrip().startswith(end))
    excerpt = textwrap.dedent("\n".join(lines[lo:hi + 1])).splitlines()

    panel = Image.new("RGBA", (WIDTH, PANEL_H), (8, 10, 14, 255))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(FONT, CODE_SIZE)
    draw.line((0, 0, WIDTH, 0), fill=(40, 48, 60))
    draw.text((88, 14), filename, font=font, fill=(127, 158, 157))

    y = 14 + CODE_LEADING
    left, right = 88, 960
    columns = int((right-left) / font.getlength("M"))
    for number, line in enumerate(excerpt, lo+1):
        if not line:
            y += CODE_LEADING
            continue
        # Continue long lines graphically and show their source number once.
        pieces = textwrap.wrap(line, columns, drop_whitespace=False,
                               break_on_hyphens=False, subsequent_indent="    ")
        tone = (110, 150, 120) if line.lstrip().startswith(("#", '"""')) else (176, 186, 204)
        if number == lo:
            tone = (150, 220, 170)
        draw.text((68, y), str(number), font=font, fill=(92, 105, 121), anchor="ra")
        for piece in pieces:
            if y + CODE_LEADING > panel.height:
                raise ValueError(f"{section} is too tall for the panel")
            draw.text((left, y), piece, font=font, fill=tone)
            y += CODE_LEADING
    return panel.tobytes()


def header(speed, fish, after_catch, font):
    image = Image.new("RGBA", (WIDTH, HEADER_H))
    draw = ImageDraw.Draw(image)
    style = dict(font=font, fill=(159, 177, 191, 245), stroke_width=1,
                 stroke_fill=(7, 21, 29, 210))
    note = f"speed {speed:.2f}   fish {fish}"
    draw.text((96, 130), note, **style)
    draw.text((WIDTH-96, 130), HANDLE, anchor="ra", **style)
    if 0 <= after_catch < 20:
        opacity = round(238 * min(1., (20 - after_catch) / 10))
        offset = 96 + font.getlength(note + "  ")
        draw.text((offset, 130), "−1", font=font, fill=(201, 232, 218, opacity),
                  stroke_width=1, stroke_fill=(7, 21, 29, opacity))
    return image


def main():
    model = SOURCE / "model.html"
    source = SOURCE / "cutoff.mp4"
    frames = model_frames(model)
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(source),
    ])
    stream, = json.loads(probe)["streams"]
    if (stream["width"], stream["height"], stream["avg_frame_rate"], int(stream["nb_frames"])) != (
            WIDTH, FRAME_H, f"{FPS}/1", len(frames)):
        raise ValueError("Checkpoint video and model disagree")
    cuts = panel_timeline(frames)
    catch_frame = next(index for index, section in cuts if section == "catch")
    school_size = len(next(states(evasive_prey=True)).fish)
    for _, section in cuts:
        source_panel(section)
    timeline = iter(cuts)
    _, section = next(timeline)
    change = next(timeline, None)
    panel = source_panel(section)
    status = ImageFont.truetype(FONT, STATUS_SIZE)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    filters = (
        f"[0:v]crop=iw:{SCENE_H}:0:0[scene];[1:v]split[upper][lower];"
        f"[upper]crop=iw:{HEADER_H}:0:0,format=yuva420p[header];"
        f"[lower]crop=iw:{PANEL_H}:0:{HEADER_H},format=yuv420p[panel];"
        "[scene][header]overlay=0:0:shortest=1[top];[top][panel]vstack=shortest=1[v]"
    )
    writer = ["ffmpeg", "-v", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgba",
              "-s", f"{WIDTH}x{HEADER_H+PANEL_H}", "-r", str(FPS), "-i", "-",
              "-filter_complex", filters, "-map", "[v]", "-an",
              "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
              "-movflags", "+faststart"]
    with TemporaryDirectory(dir=OUTPUT) as directory:
        movie = Path(directory) / "cutoff.mp4"
        with subprocess.Popen(writer + [str(movie)], stdin=subprocess.PIPE) as dst:
            for i, frame in enumerate(frames):
                if change is not None and i == change[0]:
                    panel = source_panel(change[1])
                    change = next(timeline, None)
                # During the seam there are two shoals in the simulation.
                # The HUD counts the active shoal, including fish off-camera.
                fish = school_size - int(frame["caught"])
                overlay = header(frame["swim"][0], fish, i - catch_frame, status)
                dst.stdin.write(overlay.tobytes())
                dst.stdin.write(panel)
            dst.stdin.close()
            if dst.wait():
                raise RuntimeError("Could not encode the finished reel")
        movie.replace(OUTPUT / movie.name)
    (OUTPUT / model.name).write_bytes(model.read_bytes())


if __name__ == "__main__":
    main()
