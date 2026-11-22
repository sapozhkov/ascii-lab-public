"""Project geometry onto a character grid and paint the glyphs."""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H, FPS, LOOP = 1080, 1920, 30, 20.0
CW, CH, COLS, ROWS = 12, 18, 90, 72
TAU = 2 * np.pi
FONT_PATHS = ("/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf", "consola.ttf")
BEAT, FIRST_BEAT = 2.0, 0.25                         # the animal pumps once every two seconds


def beat_time(n):
    return FIRST_BEAT + n * BEAT


FORMING = (0.20, 1.70)                               # the dome appears, top first
WHEN_WHITE = beat_time(1)                            # first pump: everything blazes white
LIMBS_OUT = (2.35, 4.6)
DOME_RADIUS, DOME_HEIGHT, ROLL, LIP, TURN = 0.465, 0.58, 1.62, 0.04, 0.38
PITCH, CAM_D = 0.28, 7.0                             # we watch from a little underneath
ZOOM_IN, ZOOM_OUT = 610.0, 480.0
LIGHT = np.array([-0.42, 0.78, -0.46])               # lamp: overhead, slightly left, in front
RAMP = "$QB%8&0#@oauenjzt[]{}<>rcvx+=1il!;:,"       # heaviest letters first


def glyph_id(ch):
    return ord(ch) - 32


RAMP_G = np.array([glyph_id(c) for c in RAMP])
INNER_G = np.array([glyph_id(c) for c in "|1:;|!l:"])


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return (3 - 2 * x) * x * x


def progress(t, span):
    return np.clip((t - span[0]) / (span[1] - span[0]), 0, 1)


def tint(colour, amount):
    return colour * (1 - amount) + amount


def normalised(v):
    return v / np.sqrt((v * v).sum(-1, keepdims=True))


def rgb(*c):
    return np.array(c, float) / 255


def hash01(a, b):
    return np.modf(np.abs(np.sin(a * 127.1 + b * 311.7) * 43758.5453))[0]


def around(radius, angle, height):
    # a point given the way a potter would: distance from the axis, bearing, height
    radius, angle, height = np.broadcast_arrays(radius, angle, height)
    return np.stack([radius * np.cos(angle), height, radius * np.sin(angle)], -1)


def spin_x(p, a):
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.moveaxis(p, -1, 0)
    return np.stack([x + 0 * a, c * y - s * z, s * y + c * z], -1)


def spin_z(p, a):
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.moveaxis(p, -1, 0)
    return np.stack([c * x - s * y, s * x + c * y, z + 0 * a], -1)


def pulse(t):
    # How hard the body is clenched at time t. Just before a beat it swings open
    # (negative), then it clamps shut (positive) and relaxes back to zero.
    t = np.asarray(t, float)
    phase = np.mod(t - FIRST_BEAT, BEAT)
    shut = 0.65 * ease((phase - 0.2) / 0.3) * (1 - ease((phase - 0.5) / 0.6))
    wide = 0.9 * np.exp(-((np.mod(phase + 1, BEAT) - 1.03) / 0.2) ** 2)
    return np.where(t >= WHEN_WHITE, shut - wide, 0.0)


def lens(t):
    # start close to the dome, then back away so the long trailing parts fit
    away = ease((t - 2.25) / 0.9)
    scale = (ZOOM_IN + (ZOOM_OUT - ZOOM_IN) * away) * (0.7 + 0.3 * ease((t - 0.3) / 1.3))
    return scale, 600 - 60 * away, 640 - 133 * away


def on_screen(p, t):
    scale, mid_x, mid_y = lens(t)
    seen = spin_x(p, PITCH)
    shrink = scale * CAM_D / (CAM_D + seen[..., 2])
    return mid_x + shrink * seen[..., 0], mid_y - shrink * seen[..., 1], seen[..., 2]


def line_glyph(dx, dy):
    # which of | _ / \ lies closest to a direction on the screen, judged in
    # cells, not pixels: a cell is half as tall again as it is wide
    dx, dy = dx / CW, dy / CH
    steep, flat = np.abs(dy) > 2.0 * np.abs(dx), np.abs(dx) > 2.0 * np.abs(dy)
    return np.select([steep, flat, dx * dy < 0], [glyph_id("|"), glyph_id("_"), glyph_id("/")], glyph_id("\\"))


# skins: dome, stripes, inner wall, limbs. Skin 0 is worn before the light comes on.
SKINS = [
    (rgb(38, 78, 235), rgb(238, 242, 255), rgb(70, 80, 215), rgb(128, 108, 235)),
    (rgb(60, 205, 240), rgb(225, 95, 225), rgb(70, 120, 240), rgb(255, 115, 165)),
    (rgb(230, 70, 175), rgb(248, 218, 60), rgb(160, 85, 215), rgb(70, 205, 240)),
    (rgb(70, 205, 95), rgb(242, 224, 70), rgb(60, 170, 150), rgb(250, 152, 50)),
    (rgb(238, 70, 60), rgb(252, 162, 50), rgb(215, 100, 70), rgb(195, 155, 245)),
    (rgb(85, 115, 245), rgb(80, 215, 240), rgb(165, 95, 235), rgb(244, 214, 90)),
    (rgb(235, 62, 50), rgb(252, 224, 72), rgb(240, 150, 110), rgb(244, 238, 238)),
    (rgb(92, 214, 244), rgb(242, 242, 252), rgb(120, 110, 235), rgb(255, 135, 185)),
]
SCALLOPS, STRIPES = 16, 8                            # bumps on the hem, wedges on the dome
EDGE_DELAY = 0.20                                    # a pump needs this long to reach the hem
FLOOR = np.cos(ROLL)


def dome_profile(v):
    # v walks from the crown (0) to the hem (1) along an arc of ROLL radians:
    # how far from the axis we are, and how high above the hem
    arc = v * ROLL
    return DOME_RADIUS * np.sqrt(np.sin(arc)), DOME_HEIGHT * (np.cos(arc) - FLOOR) / (1 - FLOOR)


def dome(v, phi, t, move):
    # A point of the dome skin: v down the arc, phi round the axis. The pump is
    # felt at the crown first, so every ring of the dome reads `pulse` a bit late.
    v, phi = np.asarray(v, float), np.asarray(phi, float)
    late = t - EDGE_DELAY * v
    grip = pulse(late)
    girth, rise = dome_profile(v)
    hem = v ** 6                                     # only the last stretch flares and scallops
    girth = girth * (1 + 0.06 * hem * np.cos(SCALLOPS * phi)) * (1 + LIP * hem - 0.34 * grip * v ** 1.6)
    return move(around(girth, phi + TURN * t, rise * (1 + 0.15 * grip)), late)


# The dome skin is a fixed net of points. Each knot owns a patch of texture and
# an ink per skin, so the pattern turns with the animal instead of swimming on it.
DOME_V, DOME_PHI = np.repeat(np.linspace(0.004, 1, 120), 330), np.tile(np.arange(330) * TAU / 330, 120)
_row, _col = np.floor(DOME_V * 30), np.floor(DOME_PHI / TAU * 136)
DOME_GIN = INNER_G[(hash01(_col, _row) * len(INNER_G)).astype(int)]
FACING_OUT, FACING_UP = np.sin(DOME_V * ROLL), np.cos(DOME_V * ROLL)
HEM_PHI = np.arange(520) * TAU / 520
LIMBS, LIMB_REACH, LIMB_HALF, LIMB_DELAY = 4, 1.02, 0.145, 0.45
LIMB_ROWS, LIMB_COLS = 110, 15
LIMB_ID = np.repeat(np.arange(LIMBS), LIMB_ROWS * LIMB_COLS)
LIMB_DOWN = np.tile(np.repeat(np.linspace(0, 1, LIMB_ROWS), LIMB_COLS), LIMBS)
LIMB_SIDE = np.tile(np.linspace(-1, 1, LIMB_COLS), LIMBS * LIMB_ROWS)
_margin = ease((np.abs(LIMB_SIDE) - 0.55) / 0.4)[:, None]
LIMB_INK = np.array([skin[3] * (1 - _margin) + tint(skin[3], 0.6) * _margin for skin in SKINS])
_ld, _ls = np.floor(LIMB_DOWN * 42), np.floor((LIMB_SIDE + 1) * 3.5)
LIMB_TONE = np.clip(0.25 + 0.75 * np.modf(0.31 * _ld + 0.37 * _ls + LIMB_ID * 0.4)[0] ** 0.8
                    - 0.3 * hash01(_ld + LIMB_ID * 9, _ls), 0, 1)
LIMB_G = RAMP_G[((1 - LIMB_TONE) * (len(RAMP) - 1)).astype(int)]


def limb(which, down, side, t, out, move):
    # One of the four frilly mouth-ribbons. `down` is 0 at the root and 1 at the
    # tip, `side` is -1..1 across the ribbon, `out` is how much has unrolled yet.
    late = t - EDGE_DELAY - LIMB_DELAY * down * out
    grip = pulse(late)
    wander = 0.30 * np.sin(0.9 * t + which - 2.2 * down)
    bearing = which * TAU / LIMBS + 0.4 + TURN * late + 1.4 * down * out + wander
    bow = (1 - 0.45 * grip) * (0.04 + 0.11 * np.sin(np.pi * down ** 0.8))
    spine = around(bow, bearing, 0.10 - (1 + 0.10 * grip) * LIMB_REACH * out * down)
    # the flat of the ribbon corkscrews round its spine and narrows to a point
    curl = bearing + 1.2 + 2.6 * down
    taper = np.clip(1 - down, 0, 1) ** 0.55 * ease(0.25 + 6 * down)
    half = LIMB_HALF * (0.35 + 0.65 * out) * taper
    ruffle = 0.05 * np.abs(side) ** 1.5 * np.sin(13 * down + 2.6 * side + 2 * which - 3.2 * t)
    flutter = 0.025 * np.abs(side) * np.sin(9 * down + which - 2 * t)
    sheet = around(half * side, curl, flutter) + around(ruffle, curl + TAU / 4, 0.0)
    return move(spine + sheet, late)


def limbs_out(t):
    return float(ease(progress(t, LIMBS_OUT)))


STRANDS, STRAND_KNOTS, STRAND_DELAY, FALL_TIME = 14, 260, 0.65, 1.1
_dice = np.random.default_rng(5)
STRAND_SPAN = (1.0 + 0.5 * _dice.random(STRANDS))[:, None]
_dice.random(STRANDS)  # Keep the random sequence used for the strand roots.
STRAND_ROOT = (TAU / STRANDS * (np.arange(STRANDS) + 0.5 * _dice.random(STRANDS)))[:, None]
FRONT_SPEED = 2.4                                    # body-lengths per second
JOIN_LIMB, JOIN_STRAND = 0.45, 0.90                  # where limbs and strands hang on the route
SHELL, INSIDE, HEM, LIMB, STRAND, LOOSE = 1, 2, 3, 4, 5, 6


def limb_part(t, dye, move):
    out = limbs_out(t)
    if out <= 0.01:
        return []
    spot = limb(LIMB_ID, LIMB_DOWN, LIMB_SIDE, t, out, move)
    sx, sy, _ = on_screen(spot, t)
    rim = np.abs(LIMB_SIDE) > 0.9                                # pale edges drawn as strokes
    stroke = line_glyph(sx - np.roll(sx, LIMB_COLS), sy - np.roll(sy, LIMB_COLS))
    ink, _ = dye(JOIN_LIMB + 0.5 * LIMB_DOWN * out, LIMB_INK, np.where(rim, 0.8, 0.62 + 0.38 * LIMB_TONE))
    keep = LIMB_DOWN > 0.012
    return [(spot[keep], np.where(rim, stroke, LIMB_G)[keep], ink[keep], LIMB)]


_w = np.random.default_rng(21)
N_DOTS, N_PLUS = 120, 46
DOT = _w.random((N_DOTS, 3))
_font = {}


def font(size):
    if size not in _font:
        override = os.environ.get("ASCII_LAB_FONT")
        if override:
            _font[size] = ImageFont.truetype(override, size)
        else:
            for name in FONT_PATHS:
                try:
                    _font[size] = ImageFont.truetype(name, size)
                    break
                except OSError:
                    continue
            else:
                raise OSError("Set ASCII_LAB_FONT to the path of a monospace TTF or TTC font.")
    return _font[size]


def make_atlas():
    out = np.zeros((96, CH, CW), np.float32)
    f = font(20)
    for i in range(95):
        im = Image.new("L", (CW, CH), 0)
        lift = 5 if chr(32 + i) == "_" else 3        # Menlo's _ would fall out of the cell
        ImageDraw.Draw(im).text((0, -lift), chr(32 + i), font=f, fill=255)
        out[i] = np.asarray(im, np.float32) / 255
    return out


ATLAS = None


def rasterize(t, chars, ink, p, g, col, part, offset=(0, 0)):
    rows, cols = chars.shape
    sx, sy, sz = on_screen(p, t)
    sz = sz - 0.03 * (part == HEM) + 0.004 * (part == INSIDE)
    c, r = np.floor((sx + offset[0]) / CW).astype(int), np.floor((sy + offset[1]) / CH).astype(int)
    ok = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
    order = np.argsort(-sz[ok], kind="stable")                # far first, so the nearest wins
    c, r, g, col, part = c[ok][order], r[ok][order], g[ok][order], col[ok][order], part[ok][order]
    owner = np.zeros(chars.shape, int)
    chars[r, c], ink[r, c], owner[r, c] = g, col, part

    # outline pass: where the shell meets the water, draw its edge instead of its fill
    body = np.pad((owner == SHELL) | (owner == INSIDE) | (owner == HEM), 1)
    here = body[1:-1, 1:-1]
    up, down, left, right = ~body[:-2, 1:-1], ~body[2:, 1:-1], ~body[1:-1, :-2], ~body[1:-1, 2:]
    shell = here & (owner == SHELL)
    edge = np.full(chars.shape, -1)
    edge[shell & up] = glyph_id("-")
    edge[shell & left] = glyph_id("|")
    edge[shell & right] = glyph_id("|")
    edge[shell & left & up] = glyph_id("/")
    edge[shell & right & up] = glyph_id("\\")
    edge[shell & left & down & ~up] = glyph_id("\\")
    edge[shell & right & down & ~up] = glyph_id("/")
    if t < FORMING[1]:
        edge[shell & down & ~left & ~right & ~up] = glyph_id("_")
    m = edge >= 0
    chars[m] = edge[m]
    ink[m] = tint(ink[m], 0.35) * 0.95


def glyph_atlas():
    global ATLAS
    if ATLAS is None:
        ATLAS = make_atlas()
    return ATLAS


def paint(chars, ink):
    height, width = chars.shape[0] * CH, chars.shape[1] * CW
    mask = glyph_atlas()[chars].transpose(0, 2, 1, 3).reshape(height, width)
    col = np.repeat(np.repeat(ink.astype(np.float32), CH, 0), CW, 1)
    img = mask[..., None] * col
    small = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((width // 4, height // 4), Image.BOX)
    near = np.asarray(small.filter(ImageFilter.GaussianBlur(2.2)).resize((width, height), Image.BILINEAR), np.float32)
    far = np.asarray(small.filter(ImageFilter.GaussianBlur(9)).resize((width, height), Image.BILINEAR), np.float32)
    run = np.cumsum(np.pad(img, ((STREAK, STREAK), (0, 0), (0, 0))), 0)   # thin vertical light streaks
    rays = (run[2 * STREAK:] - run[:-2 * STREAK]) / (2 * STREAK)
    return np.clip(img * 255 + GLOW_NEAR * near + GLOW_FAR * far + GLOW_RAYS * 255 * rays, 0, 255).astype(np.uint8)


GLOW_NEAR, GLOW_FAR, GLOW_RAYS, STREAK = 0.7, 0.55, 0.16, 110
