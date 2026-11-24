"""Two ASCII views of one school model. Run python3 -m proto.shark_studies.school_render."""

import argparse
from functools import lru_cache
import inspect
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proto.glyph_atlas import glyph_atlas
from proto.shark_studies import school_motion as model


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "shots" / "shark-school-grid"
ZOOM_OUTPUT = ROOT / "shots" / "shark-school-zoom"
REEF_OUTPUT = ROOT / "shots" / "shark-open-water-contrast"
REEF_ZOOMS = {"solo": .57, "duo": .44}
DIAGONAL_OUTPUT = ROOT / "shots" / "shark-diagonal"
ZOOMS = (.57, .48, .39)
SOURCE = ROOT / "ref" / "shark-school" / "reference.png"
COLS, ROWS, CW, CH = 90, 72, 12, 18
WIDTH, SCENE_H, FRAME_H = COLS * CW, ROWS * CH, 1920
FONT = "/System/Library/Fonts/Menlo.ttc"
ROW, COL = np.indices((ROWS, COLS), dtype=np.float32)
X, Y = (COL + .5) * CW / 2, (ROW + .5) * CH / 2
# Twelve samples measure the coverage and direction of a thin fin in a single cell.
SY, SX = np.mgrid[0:4, 0:3]
SAMPLE_X = X[..., None] + ((SX.ravel() + .5) / 3 - .5) * CW / 2
SAMPLE_Y = Y[..., None] + ((SY.ravel() + .5) / 4 - .5) * CH / 2


def glyphs(text):
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8) - 32


WATER = glyphs(" .,:-~=+xseaoO08%#@")
SKIN = glyphs("  .:;!ilI/+xohO")
SCALES = glyphs(".:~-+xo%8#@")
CORAL = glyphs(" .,:;*+vxsao0%#@")


def select(ramp, value):
    index = np.rint(np.clip(value, 0, 1) * (len(ramp) - 1))
    return ramp[index.astype(np.intp)]


def seabed():
    # The photograph supplies the large turquoise and pale patches on the sand.
    # Remove the photographed shark and shadow before sampling any light or colour.
    photo = np.asarray(Image.open(SOURCE).convert("RGB"))
    contour = np.array([(242,410),(288,410),(310,474),(362,491),(369,532),
                        (314,548),(310,616),(289,676),(272,805),(244,805),
                        (235,686),(243,551),(191,545),(195,503),(237,481)])
    hole = cv2.fillPoly(np.zeros(photo.shape[:2], np.uint8), [contour], 255)
    x, y, w, h = cv2.boundingRect(hole)
    clean = cv2.seamlessClone(np.roll(photo, 167, axis=1), photo, hole,
                              (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
    return cv2.GaussianBlur(clean.astype(np.float32) / 255, (0, 0), 2)


def refracted_sun(x, y, t):
    # Light concentrates on folds where two trains of surface waves cross.
    swell = np.sin(.012 * x + .008 * y - .49 * t)
    cross = np.cos(.009 * x - .014 * y + .35 * t)
    front = .023 * x + .032 * y + 2.3 * swell + 1.1 * cross - .56 * t
    backs = .031 * x - .019 * y + 2.6 * cross + .8 * swell + .31 * t
    a = np.exp(-7 * (1 - np.cos(front)))
    b = np.exp(-9 * (1 - np.cos(backs)))
    return np.maximum(.83 * a, .62 * b) + .23 * a * b


def reef_noise(x, y, scale):
    """Continuous variation in the height of the rough coral surface."""
    x, y = x / scale, y / scale
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx, fy = fx**2 * (3 - 2*fx), fy**2 * (3 - 2*fy)

    def height(dx, dy):
        n = np.sin((ix + dx)*73.257 + (iy + dy)*131.119) * 4375.873
        return n - np.floor(n)

    upper = height(0, 0) * (1-fx) + height(1, 0) * fx
    lower = height(0, 1) * (1-fx) + height(1, 1) * fx
    return upper * (1-fy) + lower * fy


@lru_cache(maxsize=4)
def sand_channel(zoom):
    x, y = 270 + (X-270)/zoom, 324 + (Y-324)/zoom
    bank = reef_noise(x, y, 97)
    knolls = reef_noise(x, y, 48)
    grain = reef_noise(x, y, 19)
    # A shallow meandering channel bifurcates around a bank of coral.
    axis = 135 + .23*(y-324) + 138*np.sin(y/235 + .3)
    main = abs(x-axis) - (128 + 32*np.sin(y/149))
    branch = abs(y-410 + .43*(x-270) + 61*np.sin(x/193)) - 102
    shore = np.minimum(main, branch) + 126*(bank-.5) + 71*(knolls-.5) + 35*(grain-.5)
    sand = np.clip(.5 - shore/49, 0, 1)
    relief = np.clip(.50*bank + .39*knolls + .43*grain - .15, 0, 1)
    return sand, relief


def water_cells(floor, t, camera_y, shadow, zoom=1.0, sunlight=None):
    # Sample a point on the sand at each fixed screen cell. Refraction moves the
    # sampled light, while the screen columns and text baselines remain stationary.
    world_x = 270 + (X - 270) / zoom
    world_y = 324 + (Y - 324) / zoom + .76 * camera_y
    wave_x = np.sin(.018 * world_x + .029 * world_y - .81 * t)
    wave_y = np.sin(.031 * world_x - .016 * world_y + .63 * t)
    sx = (54 + .8 * world_x + 2.3 * wave_x).astype(np.float32)
    sy = (245 + .8 * world_y + 2.1 * wave_y).astype(np.float32)
    sand = cv2.remap(floor, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if zoom < 1:
        # The sandbank gives way to blue water beyond the reference photograph.
        h, w = floor.shape[:2]
        outside = np.maximum.reduce((-sx, sx - w + 1, -sy, sy - h + 1, np.zeros_like(sx)))
        offshore = (1 - np.exp(-outside / 140))[..., None]
        water = np.array([.035, .40, .61]) + .025 * wave_x[..., None]
        sand = sand * (1 - offshore) + water * offshore
    sun = np.clip((sand[..., 1] - .20) / .71, 0, 1)
    ridges = np.sin(.076 * world_x + .12 * world_y + 1.6 * wave_x)
    ripple = np.maximum(0, .55 * wave_x + .45 * wave_y)**3
    crest = np.maximum(0, ridges + .30 * wave_y - .24)
    value = .06 + .67 * sun + .10 * ridges + .23 * ripple + .13 * crest
    if sunlight is not None:
        channel, relief = sand_channel(zoom)
        bright_sand = .24 + .34 * sun + .52 * sunlight + .05 * ridges
        coral = .10 + .63 * relief + .045 * sunlight
        value = coral + channel * (bright_sand - coral)
    chars = select(WATER, value * (1 - .66 * shadow))
    if sunlight is not None:
        coral_cells = channel < .36
        chars[coral_cells] = select(CORAL, value[coral_cells] * (1 - .66 * shadow[coral_cells]))
    # Cool the water in shadow and preserve some sand texture underneath.
    light = (sand ** .78 * 287 + 9) * (.76 + .22 * ripple[..., None])
    light += (ripple + .57 * crest)[..., None] * [47, 46, 32]
    if sunlight is not None:
        lagoon = light * .44 + [48, 102, 93]
        lagoon += sunlight[..., None] * [142, 114, 68]
        coral = np.array([16, 55, 53]) + relief[..., None] * [120, 151, 101]
        coral += (relief**4)[..., None] * [53, 34, 7]
        light = coral + channel[..., None] * (lagoon - coral)
    ink = light * (1 - .59 * shadow[..., None])
    return chars, np.clip(ink, 0, 255).astype(np.uint8)


def coverage_samples(surface):
    inside, reflection = surface
    coverage = inside.mean(axis=-1, dtype=np.float32)
    light = np.where(inside, reflection, 0)
    mean = light.mean(axis=-1) / np.maximum(coverage, .001)
    # Narrow reflections would disappear in the mean when they graze one cell.
    highlight = .70 * mean + .30 * light.max(axis=-1)
    return coverage, highlight


def shadow_on_floor(coverage, offset, blur, zoom=1.0):
    # Project the same posed body away from the light, in units of screen cells.
    dx, dy = offset
    matrix = np.float32([[1, 0, dx * zoom / (CW / 2)], [0, 1, dy * zoom / (CH / 2)]])
    projected = cv2.warpAffine(coverage, matrix, (COLS, ROWS))
    return cv2.GaussianBlur(projected, (0, 0), max(.25, blur * zoom), borderType=cv2.BORDER_CONSTANT)


def write_body(chars, ink, coverage, light, small, pearl=False):
    occupied = coverage >= (.16 if small else .18)
    if small:
        signs = select(SCALES, .16 + .98 * light)
        color = np.array([153, 73, 27]) + light[..., None] * [181, 208, 211]
        if pearl:
            signs = select(glyphs("=+xoa8B%#@$"), .27 + .85 * light)
            color = np.array([231, 150, 67]) + light[..., None] * [45, 168, 245]
    else:
        signs = select(SKIN, 1.85 * light)
        color = np.array([26, 36, 47]) + light[..., None] * [364, 354, 324]
    # Draw a broken contour only where the surface crosses the cell.
    grad_y, grad_x = np.gradient(coverage)
    edge = occupied & (coverage < (.52 if small else .80)) & (light < (.60 if small else .26))
    direction = np.where(abs(grad_x) > 2 * abs(grad_y), ord("|"), ord("_"))
    diagonal = (abs(grad_x) <= 2 * abs(grad_y)) & (abs(grad_y) <= 2 * abs(grad_x))
    direction = np.where(diagonal, np.where(grad_x * grad_y > 0, ord("/"), ord("\\")), direction)
    signs[edge] = direction[edge] - 32
    chars[occupied] = signs[occupied]
    ink[occupied] = np.clip(color[occupied], 0, 255).astype(np.uint8)


def shoal_cells(bodies, world_x, world_y, zoom, sunlight=None):
    # Evaluate each fish only where its nose, fins or bent tail can reach.
    lengths = model.fish_lengths(len(bodies))
    for index, body in enumerate(bodies):
        sx = 270 + (body[0] - 270) * zoom
        sy = 324 + (body[1] - 324) * zoom
        radius = .68 * lengths[index] * zoom
        x0 = max(0, int(np.floor((sx - radius) / (CW / 2))))
        x1 = min(COLS, int(np.ceil((sx + radius) / (CW / 2))))
        y0 = max(0, int(np.floor((sy - radius) / (CH / 2))))
        y1 = min(ROWS, int(np.ceil((sy + radius) / (CH / 2))))
        if x0 >= x1 or y0 >= y1:
            continue
        roi = np.s_[y0:y1, x0:x1]
        surface = model.fish_surface(world_x[roi], world_y[roi], body, index)
        coverage = np.zeros((ROWS, COLS), np.float32)
        light = np.zeros_like(coverage)
        coverage[roi], light[roi] = coverage_samples(surface)
        if sunlight is not None:
            # A turn exposes more of the silver flank to one fixed light source.
            heading, phase, alarm = body[2:]
            bank = np.cos(heading + .18 * np.sin(phase) - .6) ** 4
            light[roi] *= .86 + .23 * sunlight[roi] + (.14 + .22 * alarm) * bank
            np.clip(light, 0, 1, out=light)
        yield coverage, light


def passing_glint(lx, ly, phase, sun):
    """A narrow reflection sweeps the back as its wet surface rolls into the sun."""
    station = np.clip(ly / model.SHARK_LENGTH, 0, 1)
    half_width = np.interp(station, model.BODY_STATIONS, model.HALF_WIDTH)
    roll = phase - 4.15 * station
    flank = (lx - model.tail_sweep(station, phase)) / np.maximum(half_width, 1)
    normal = np.exp(-((flank + .26 - .33 * np.sin(roll)) / .49)**2)
    crown = np.sin(np.pi * station)**.55
    alignment = (.5 + .5 * np.cos(roll - .55))**4
    flash = np.clip((.93 * sun + .62 * alignment - .31) / .77, 0, 1)
    return normal * crown * flash**1.7


def scene_grid(state, t, floor, zoom=1.0, with_backlight=False, diagonal=False):
    world_x = 270 + (SAMPLE_X - 270) / zoom
    world_y = 324 + (SAMPLE_Y - 324) / zoom
    sunlight = None
    glints = []
    if "sharks" in state:
        from .reef_pass import local_xy
        cx = 270 + (X - 270) / zoom
        cy = 324 + (Y - 324) / zoom
        sunlight = refracted_sun(cx, cy, t)
        # The light on the raised back and on the sand below comes from the same wave.
        shark_offset = (33, 25) if diagonal else (21, 17.5)
        on_back = refracted_sun(cx + shark_offset[0], cy + shark_offset[1], t)
        channel, _ = sand_channel(zoom)
        sharks = []
        for animal in state["sharks"]:
            lx, ly = local_xy(world_x, world_y, animal)
            pose = {"nose": (0, 0), "phase": animal["phase"]}
            area, skin = coverage_samples(model.shark_surface(lx, ly, pose))
            # A ridge catching skylight remains visible over dark coral.
            rim = (1 - channel) * np.maximum(skin - .095, 0) * .61
            light = skin * (.94 + .39 * on_back) + .11 * on_back * np.sqrt(skin) + rim
            if diagonal:
                spec = passing_glint(lx, ly, animal["phase"], on_back[..., None])
                glint = .55 * spec.mean(axis=-1) + .45 * spec.max(axis=-1)
                glints.append(glint)
                light = .83 * light + .68 * glint
            sharks.append((area, light))
    else:
        sharks = [coverage_samples(model.shark_surface(world_x, world_y, state))]
        shark_offset = (21, 17.5)
    fish = list(shoal_cells(state["fish"], world_x, world_y, zoom, sunlight))
    shadow = np.zeros((ROWS, COLS), np.float32)
    for area, _ in sharks:
        blur = .48 if diagonal else 1.4
        shadow = np.maximum(shadow, shadow_on_floor(area, shark_offset, blur, zoom) * .95)
    for coverage, _ in fish:
        fish_offset = (7, 7) if diagonal else (4.5, 6)
        direct = shadow_on_floor(coverage, fish_offset, .36 if diagonal else .55, zoom)
        if diagonal:
            shadow = np.maximum(shadow, .96 * direct)
            continue
        # Softer penumbra across the surrounding glyphs keeps the tail readable
        # when a minnow turns over pale sand. Both terms follow its actual pose.
        kernel = np.ones((1 + 2 * round(zoom), 1 + 2 * round(2 * zoom)), np.uint8)
        penumbra = cv2.dilate(direct, kernel)
        penumbra = cv2.GaussianBlur(penumbra, (0, 0), 1.1 * zoom)
        shadow = np.maximum(shadow, np.minimum(1, .63 * direct + .78 * penumbra))
    chars, ink = water_cells(floor, t, state["camera_y"], shadow, zoom, sunlight)
    if with_backlight:
        # Separate the average tone of the water from the fine detail of its
        # letters. A dark animal can cross either bottom and still cut a silhouette.
        ambient = (np.array([21, 57, 65]) + relief_light(zoom, sunlight)) * (1 - .62 * shadow[..., None])
        if diagonal:
            ambient *= .54 * (1 - .70 * shadow[..., None])
        stroke = .74 if diagonal else .49
        ink = (ambient + stroke * (ink - ambient)).clip(0, 255).astype(np.uint8)
    # From floor to camera. The shark swims above the school in the model.
    for coverage, light in fish:
        write_body(chars, ink, coverage, light, small=True, pearl=sunlight is not None)
        if with_backlight:
            weight = np.clip(1.45 * coverage, 0, 1)[..., None]
            lit_scales = np.array([67, 36, 12]) + light[..., None] * [90, 79, 61]
            if diagonal:
                lit_scales *= .55
            ambient += weight * (lit_scales - ambient)
    for index, (area, light) in enumerate(sharks):
        write_body(chars, ink, area, light, small=False)
        if with_backlight:
            weight = np.clip(1.7 * area, 0, 1)[..., None]
            skin = np.array([2, 7, 13]) + light[..., None] * [23, 32, 40]
            ambient += weight * (skin - ambient)
            # The upper left edge and the dorsal ridge pick up skylight. Leave the
            # unlit flank and belly dark, as in the overhead photograph.
            gy, gx = np.gradient(area)
            crest = (area > .18) & ((gx + .30 * gy > .17) | (light > .23))
            strength = np.clip(.18 + .70 * light[crest], 0, 1)
            ink[crest] = np.maximum(ink[crest], (strength[:, None] * [194, 222, 242]).astype(np.uint8))
            if diagonal:
                gleam = glints[index]
                lit = (area >= .18) & (gleam > .16)
                power = np.clip((gleam[lit]-.16) / .58, 0, 1)
                chars[lit] = select(glyphs("il/+xo08#@"), np.sqrt(power))
                ink[lit] = [130, 181, 209] + power[:, None] * [125, 74, 46]
                ambient[lit] += power[:, None]**2.6 * [107, 143, 161]
    if with_backlight:
        return chars, ink, ambient.clip(0, 255).astype(np.uint8)
    return chars, ink


def relief_light(zoom, sunlight):
    sand, relief = sand_channel(zoom)
    return (relief[..., None] * [17, 24, 20] + sand[..., None] * [14, 23, 28]
            + (sunlight * sand)[..., None] * [11, 12, 9])


def paint(chars, ink, atlas, background=None, bloom=1.):
    # One fixed glyph per cell. Its coverage mixes foreground and background ink.
    masks = atlas[chars]
    if background is None:
        pixels = masks[..., None] * ink[:, :, None, None, :]
        return pixels.astype(np.uint8).transpose(0, 2, 1, 3, 4).reshape(SCENE_H, WIDTH, 3)
    shape = (WIDTH, SCENE_H)
    mask = masks.transpose(0, 2, 1, 3).reshape(SCENE_H, WIDTH)
    paper = cv2.resize(background, shape, interpolation=cv2.INTER_LINEAR)
    foreground = cv2.resize(ink, shape, interpolation=cv2.INTER_NEAREST)
    image = cv2.blendLinear(foreground, paper, mask, 1 - mask)
    # Like the clownfish render, spread the light of the actual glyphs
    # at two radii. Only bright strokes illuminate their neighbours.
    energy = masks.mean(axis=(2, 3))[..., None] * np.maximum(ink.astype(np.float32) - 125, 0)
    glow = .65 * cv2.GaussianBlur(energy, (0, 0), .62)
    glow += .27 * cv2.GaussianBlur(energy, (0, 0), 2.1)
    halo = cv2.resize(glow * bloom, shape, interpolation=cv2.INTER_LINEAR)
    return cv2.add(image, halo, dtype=cv2.CV_8U)


def code_panel(functions):
    pane = Image.new("RGB", (WIDTH, FRAME_H - SCENE_H), (8, 10, 14))
    draw = ImageDraw.Draw(pane)
    draw.line((0, 0, WIDTH, 0), fill=(40, 48, 60))
    y = 13
    font = ImageFont.truetype(FONT, 23)
    for func in functions:
        lines, first = inspect.getsourcelines(func)
        for number, line in enumerate(lines, first):
            line = line.rstrip()
            if y > pane.height - 33:
                return np.asarray(pane)
            draw.text((14, y), f"{number:4d}", font=font, fill=(70, 80, 96))
            # Scale an occasional long line to the available width.
            size = min(23, int(23 * (WIDTH - 108) / max(1, font.getlength(line))))
            face = font if size == 23 else ImageFont.truetype(FONT, size)
            tone = (150, 220, 170) if number == first else (157, 175, 189)
            draw.text((90, y), line, font=face, fill=tone)
            y += 33
        y += 18
    return np.asarray(pane)


def render(variant, floor, atlas, zoom=1.0):
    diagonal = variant == "diagonal"
    reef = variant in REEF_ZOOMS or diagonal
    if reef:
        from . import reef_pass
        frames = reef_pass.states(variant)
        panels = [code_panel((reef_pass.reef_current, model.tail_sweep)),
                  code_panel((reef_pass.local_avoidance,)), code_panel((passing_glint if diagonal else refracted_sun,))]
        out = (DIAGONAL_OUTPUT if diagonal else REEF_OUTPUT) / f"{variant}.mp4"
    else:
        frames = model.states(variant)
        reaction = model.avoid_shark if variant == "wide" else model.escape
        panels = [code_panel((model.tail_sweep, shadow_on_floor)),
                  code_panel((reaction,)), code_panel((model.flock,))]
        out = ZOOM_OUTPUT / f"pass-{zoom:.2f}.mp4" if variant == "wide" else OUTPUT / f"{variant}.mp4"
    out.parent.mkdir(exist_ok=True, parents=True)
    staged = out.with_suffix(".part.mp4")
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{FRAME_H}", "-r", str(model.FPS), "-i", "-", "-an", "-c:v",
               "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(staged)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        section = 0
        for i, state in enumerate(frames):
            alarm = max(f[-1] for f in state["fish"])
            if reef and i >= (8 if diagonal else 16) * model.FPS:
                section = 2
            elif section == 0 and alarm > .2:
                section = 1
            elif section == 1 and alarm < .12:
                section = 2
            if reef:
                chars, ink, paper = scene_grid(state, i / model.FPS, floor, zoom, with_backlight=True, diagonal=diagonal)
                scene = paint(chars, ink, atlas, paper, bloom=.36 if diagonal else 1.)
            else:
                chars, ink = scene_grid(state, i / model.FPS, floor, zoom)
                scene = paint(chars, ink, atlas)
            encoder.stdin.write(scene.tobytes())
            encoder.stdin.write(panels[section].tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed")
    staged.replace(out)


def compare_views(views, output):
    # Show the water panes on one clock. Full vertical exports include the source code.
    width = 432 if len(views) == 2 else 384
    height = round(SCENE_H * width / WIDTH / 2) * 2
    labels = Image.new("RGB", (len(views) * width, 42), (8, 10, 14))
    draw = ImageDraw.Draw(labels)
    font = ImageFont.truetype(FONT, 22)
    inputs = []
    filters = []
    for index, (path, label) in enumerate(views):
        draw.text(((index + .5) * width, 21), label,
                  anchor="mm", font=font, fill=(188, 204, 213))
        inputs.extend(["-i", str(path)])
        filters.append(f"[{index}:v]crop=iw:{SCENE_H}:0:0,scale={width}:{height},setsar=1[v{index}]")
    panes = "".join(f"[v{i}]" for i in range(len(views)))
    filters += [f"{panes}hstack=inputs={len(views)}[panes]",
                f"[{len(views)}:v][panes]vstack=shortest=1[v]"]
    with TemporaryDirectory(dir=output.parent) as temp:
        header = Path(temp) / "header.png"
        staged = Path(temp) / "compare.mp4"
        labels.save(header)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-loop", "1", "-framerate",
                        str(model.FPS), "-i", str(header), "-filter_complex", ";".join(filters),
                        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(staged)], check=True)
        staged.replace(output)


def compare_zooms():
    compare_views([(ZOOM_OUTPUT / f"pass-{z:.2f}.mp4", f"{z:.2f}×".replace(".", ","))
                   for z in ZOOMS], ZOOM_OUTPUT / "compare.mp4")


def reef_comparison():
    from .reef_view import write_model
    write_model(REEF_OUTPUT / "model.html")
    compare_views([(REEF_OUTPUT / "solo.mp4", "One shark · 0.57×"),
                   (REEF_OUTPUT / "duo.mp4", "Two sharks · 0.44×")], REEF_OUTPUT / "compare.mp4")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("variants", nargs="*", default=["follow", "pass"])
    args = cli.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    floor = seabed()
    atlas = glyph_atlas(ImageFont.truetype(FONT, 20, index=1), CW, CH)
    for variant in args.variants:
        if variant == "diagonal":
            from .reef_pass import ZOOM
            from .reef_view import write_model
            render(variant, floor, atlas, ZOOM[variant])
            write_model(DIAGONAL_OUTPUT / "model.html", (variant,))
        elif variant == "reef":
            for mode, zoom in REEF_ZOOMS.items():
                render(mode, floor, atlas, zoom)
            reef_comparison()
        elif variant in REEF_ZOOMS:
            render(variant, floor, atlas, REEF_ZOOMS[variant])
        elif variant == "wide":
            from .school_zoom_view import write_model
            for zoom in ZOOMS:
                render(variant, floor, atlas, zoom)
            compare_zooms()
            write_model(ZOOM_OUTPUT / "model.html", ZOOMS)
        else:
            render(variant, floor, atlas)
    if any(variant in ("pass", "follow") for variant in args.variants):
        model.write_model(OUTPUT / "model.html")
