"""Build the rock shelf, sea pen, anemone, and light shafts."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import duet, renderer as j
from .swimmer import COLS, ROWS, WIDTH, HEIGHT


@dataclass(frozen=True)
class Colony:
    x: float
    z: float
    height: float
    lean: float
    bearing: float
    seed: int


def bed_height(x, z):
    return -2.02 + 0.12 * np.sin(2.25 * x + 0.69 * z) + 0.035 * z


class Pen:
    """A flexible axial stalk carrying two staggered rows of tapered pinnules."""

    def __init__(self, site, path, root_height=None):
        self.site = site
        self.t = path.t
        self.u = np.linspace(0, 1, 64)
        self.root = np.array([site.x, bed_height(site.x, site.z) if root_height is None else root_height, site.z])
        self.angles, self.stimulus = self.water_response(path)
        rng = np.random.default_rng(site.seed)
        # Samples in local material coordinates (stalk station, span, normal,
        # height above the insertion point, glyph, pigment, light, polyp tip).
        material = []
        # A bare peduncle leads into a lit axial ridge. Its concave shadow
        # gives the main pen a continuous diagonal even at phone size.
        for station in np.linspace(0, 1, 188):
            radius = 0.004 + 0.015 * np.sin(np.pi * station) ** 0.6
            for flank in (-1, 0, 1):
                mark = '|' if flank else 'l'
                material.append(
                    (
                        station,
                        radius * flank,
                        -0.4 * radius * (1 - flank**2),
                        0,
                        mark,
                        0,
                        (0.75 if flank else 1.1) * (0.76 + 0.24 * station),
                        0,
                    )
                )
        # Flat, staggered pinnules leave water between consecutive blades.
        # The main pen has six ranks, a young colony only three.
        pairs = int(np.clip(np.round(2.8 + 1.55 * site.height), 3, 6))
        for sign in (-1, 1):
            for row in range(pairs):
                age = (row + 0.5) / pairs
                station = 0.17 + 0.72 * age + sign * 0.095 / pairs
                length = (0.043 + 0.218 * (1 - age) ** 0.55) * rng.uniform(0.88, 1.10)
                skew = rng.uniform(-0.21, 0.21)
                blade = np.sin(np.pi * age) ** 0.60
                for u in np.linspace(0, 1, 34):
                    vein = np.sin(np.pi * u) ** 0.9
                    width = (0.008 + 0.019 * blade) * vein
                    rise = (0.11 + 0.092 * (1 - age)) * (u * 0.69 + u * u * 0.31)
                    curl = (0.019 * u**2 + skew * length * u) * sign
                    span = sign * length * u
                    for rim in (-1, 0, 1):
                        mark = ('/' if sign > 0 else '\\') if rim >= 0 else ','
                        material.append(
                            (
                                station,
                                span,
                                curl + 0.38 * width * (1 - rim**2),
                                rise + rim * width,
                                mark,
                                1,
                                0.47 + 0.17 * u + 0.36 * (rim == 1),
                                0,
                            )
                        )
                # One or two tiny polyps near the distal rim form an accent.
                for u in (0.74, 0.95):
                    rise = (0.11 + 0.092 * (1 - age)) * (u * 0.69 + u * u * 0.31)
                    curl = (0.019 * u**2 + skew * length * u) * sign
                    material.append(
                        (station, sign * length * u, curl, rise + 0.014, '*', 2, 0.74 + 0.11 * u, 1)
                    )
        values = np.array([row[:4] for row in material])
        self.station, self.span, self.normal, self.lift = values.T
        self.glyph = np.array([j.glyph_id(row[4]) for row in material])
        pigment = np.array([row[5] for row in material])
        # Salt-green stalks, a little amber on the exposed polyps. Depth
        # attenuates the colony as a whole so distant stalks read together.
        palette = np.array([[97, 153, 132], [68, 133, 115], [139, 164, 120]])
        self.ink = palette[pigment] * np.array([row[6] for row in material])[:, None]
        self.polyp = np.array([row[7] for row in material])
        self.depth_gain = 0.92 * np.exp(-0.29 * max(0, site.z - 0.48))
        self.twist = site.bearing + 0.38 * np.sin(3 * self.station + site.seed) + 0.09 * self.station

    def water_response(self, path):
        t = path.t
        site = self.site
        h0 = path.sample(path.height, 0)
        # Water expelled below the passing bell reaches plants after travelling
        # down and out. Sample the same pulse and orbit as the actual duet.
        gap = np.maximum(0.12, path.height - h0 - (self.root[1] + 0.62 * site.height))
        late = t - gap / 0.94
        ang = np.interp(late, t, path.angle) - path.start_angle
        radius = np.interp(late, t, path.radius)
        displacement = np.zeros((len(t), 2))
        stress = np.zeros(len(t))
        for i in range(2):
            position = np.stack(
                (radius * np.cos(ang + i * np.pi), radius * np.sin(ang + i * np.pi)), axis=-1
            )
            offset = np.array([site.x, site.z]) - position
            distance = np.linalg.norm(offset, axis=1)
            pulse = j.pulse(duet.clock(late, i))
            impulse = np.maximum(0, np.gradient(pulse, duet.DT))
            weight = impulse * np.exp(-(distance**2) / 1.75 - gap / 2.2)
            displacement += offset / (distance[:, None] + 0.25) * weight[:, None]
            stress += weight
        phase = 0.52 * site.x + 0.83 * site.z
        current = np.stack(
            (
                0.11 + 0.28 * np.sin(0.75 * t - phase) + 0.07 * np.sin(1.53 * t + 0.71 * phase),
                0.08 * np.sin(0.69 * t + 0.51 * phase),
            ),
            axis=-1,
        )
        target = current + 0.94 * displacement
        bend = np.zeros_like(target)
        velocity = np.zeros(2)
        activity = np.zeros(len(t))
        for k in range(1, len(t)):
            velocity += (8 * (target[k] - bend[k - 1]) - 3.1 * velocity) * duet.DT
            bend[k] = bend[k - 1] + velocity * duet.DT
            activity[k] = activity[k - 1] + (stress[k] - activity[k - 1]) * (1 - np.exp(-duet.DT / 0.36))
        return bend, activity

    def sample(self, t):
        s = self.u
        h = self.site.height
        delayed = t - 0.68 * s**1.7
        bend = np.stack([np.interp(delayed, self.t, self.angles[:, k]) for k in range(2)], axis=-1)
        tangent = np.column_stack(
            (
                self.site.lean * (0.55 + 0.86 * s) + bend[:, 0] * s**1.4,
                np.ones_like(s),
                0.09 * np.sin(self.site.bearing) + bend[:, 1] * s**1.4,
            )
        )
        tangent = j.normalised(tangent)
        spine = self.root + np.vstack(
            (np.zeros(3), np.cumsum(0.5 * (tangent[1:] + tangent[:-1]) * h / (len(s) - 1), axis=0))
        )
        pos = np.stack([np.interp(self.station, s, spine[:, k]) for k in range(3)], axis=-1)
        axis = np.stack([np.interp(self.station, s, tangent[:, k]) for k in range(3)], axis=-1)
        plane = self.twist + 0.31 * np.interp(self.station, s, bend[:, 0])
        width = np.stack((np.cos(plane), np.zeros_like(plane), np.sin(plane)), axis=-1)
        width = j.normalised(width - axis * np.sum(width * axis, axis=1)[:, None])
        normal = np.cross(width, axis)
        pos += h * (self.span[:, None] * width + self.normal[:, None] * normal + self.lift[:, None] * axis)
        # A brush of moving water can excite polyps in succession along a pen.
        # The response travels over its own stalk and naturally vanishes as
        # the jellyfish rise away from the bottom.
        drive = np.interp(t - 0.90 * (1 - self.station), self.t, self.stimulus)
        glow = 1.55 * self.polyp * j.ease((drive - 0.075) / 0.44)
        color = self.depth_gain * (self.ink + glow[:, None] * np.array([76, 72, 36]))
        return pos, self.glyph, color


def seabed():
    """Patchy ripple crests in the horizontal plane; no line at the horizon."""
    rng = np.random.default_rng(184)
    points = []
    glyphs = []
    colors = []
    for z in np.linspace(-0.6, 5.1, 29):
        for x in np.arange(-4.1, 4.1, 0.027):
            rz = z + 0.10 * np.sin(2.7 * x + 1.4 * z) + 0.045 * np.sin(5.1 * x - z)
            patch = 0.54 + 0.30 * np.sin(x * 3.6 + z) + 0.16 * np.sin(z * 5.5 - x * 0.7)
            if rng.random() > 0.62 * patch:
                continue
            points.append((x, bed_height(x, rz) + rng.uniform(-0.010, 0.010), rz))
            glyphs.append(
                j.glyph_id(rng.choice([',', '.', "'", '~', '_'], p=[0.34, 0.27, 0.16, 0.11, 0.12]))
            )
            colors.append(np.array([35, 72, 79]) * (0.63 + 0.47 * patch) * rng.uniform(0.61, 1))
    return np.array(points), np.array(glyphs), np.array(colors)


def project_material(path, t, parts, chars, ink, exposure=1.0):
    points, glyph, color = [np.concatenate([part[k] for part in parts]) for k in range(3)]
    points[:, 1] -= path.sample(path.height, t) - path.sample(path.height, 0)
    points = duet.yaw(points, path.start_angle - path.camera(t)) / (1 + 0.58 * duet.spread(t))
    sx, sy, depth = j.on_screen(points, duet.CLOCK)
    col = np.floor((sx + duet.OFFSET[0]) / j.CW).astype(int)
    row = np.floor((sy + duet.OFFSET[1]) / j.CH).astype(int)
    visible = (col >= 0) & (col < COLS) & (row >= 0) & (row < ROWS)
    ids = np.flatnonzero(visible)
    order = ids[np.argsort(-depth[ids], kind='stable')]
    chars[row[order], col[order]] = glyph[order]
    ink[row[order], col[order]] = np.clip(exposure * color[order], 0, 255).astype(np.uint8)


def paint_material(chars, ink):
    cells = (j.glyph_atlas()[chars, ..., None] * ink[:, :, None, None, :]).astype(np.uint8)
    pixels = cells.transpose(0, 2, 1, 3, 4).reshape(ROWS * j.CH, COLS * j.CW, 3)[:HEIGHT]
    image = Image.fromarray(pixels)
    quarter = image.resize((WIDTH // 4, HEIGHT // 4), Image.Resampling.BOX)
    glow = quarter.filter(ImageFilter.GaussianBlur(1.15)).resize(
        (WIDTH, HEIGHT), Image.Resampling.BILINEAR
    )
    return np.asarray(ImageChops.add(image, glow))


def compose(source, background):
    # One native cell of clearance around lit foreground glyphs, including
    # their dim phosphor halo. Preserve both empty space inside the heart and
    # the dark underside of the bells if a distant pen passes behind them.
    low = Image.fromarray(source).resize((COLS, ROWS), Image.Resampling.BOX)
    occupied = np.asarray(low).max(axis=2) > 17
    for line in occupied:
        hits = np.flatnonzero(line)
        if len(hits) > 4:
            runs = np.split(hits, np.flatnonzero(np.diff(hits) > 6) + 1)
            for run in runs:
                if len(run) > 3:
                    line[run[0] : run[-1] + 1] = True
    water = Image.fromarray(np.where(occupied, 0, 255).astype(np.uint8))
    water = water.filter(ImageFilter.MinFilter(3)).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    reef = Image.composite(Image.fromarray(background), Image.new('RGB', (WIDTH, HEIGHT)), water)
    return np.maximum(source, np.asarray(reef))


# Each crest focuses sunlight into a sheet. Its neighbours share the swell
# but have different curvatures, so their beams can widen and fade in turn.
# (surface x, lean, half-width at the surface, relative transmission)
CRESTS = (
    (-0.13, 0.47, 0.033, 0.62),
    (0.025, 0.40, 0.049, 1.00),
    (0.145, 0.32, 0.022, 0.68),
    (0.240, 0.31, 0.046, 0.52),
    (0.730, 0.24, 0.036, 0.41),
    (0.900, 0.33, 0.050, 0.58),
)

ROW, COL = np.indices((ROWS, COLS))
X = ((COL + 0.5) * j.CW) / WIDTH
# Coordinates of the visible water after the editor crop.
DEPTH = ((ROW + 0.5) * j.CH - 328) / (1710 - 328)
GRAIN = j.hash01(ROW + 7, COL + 39)
# A loose diagonal hatch follows the incident light. Broken runs leave
# open water between the marks, even in the brightest part of a sheet.
HATCH = ((2 * ROW - COL) % 5 == 0) | ((GRAIN > 0.89) & (ROW % 2 == 0))
MARK = np.where(GRAIN < 0.14, j.glyph_id("'"), j.glyph_id('\\'))
WATER_COLOR = np.array([92, 152, 181])


def draw_shafts(t, chars, ink):
    """Paint the far field into empty cells. The distant sun stays out of the orbit."""
    phase = j.TAU * (t % 20) / 20
    light = np.zeros(chars.shape)
    for origin, lean, width, transmission in CRESTS:
        swell = phase - 6.7 * origin
        ripple = 2 * phase + 13 * origin
        slope = lean + 0.072 * np.cos(swell) + 0.032 * np.cos(ripple)
        crest = origin + 0.022 * np.sin(swell) + 0.010 * np.sin(ripple)
        radius = width * (1 + 0.20 * np.sin(ripple)) + 0.024 * np.maximum(DEPTH, 0)
        across = (X - crest - slope * DEPTH) / radius
        focus = 0.83 + 0.17 * np.cos(ripple + 0.6)
        light += transmission * focus * np.exp(-(across**2))

    # Scattering loses contrast with depth. Keep just a trace of the beams
    # in the central hollow that the two bells open between them.
    light *= np.exp(-1.58 * np.maximum(DEPTH, 0))
    light *= 1 - 0.60 * np.exp(-(((X - 0.5) / 0.205) ** 2) - ((DEPTH - 0.5) / 0.34) ** 2)
    intensity = np.minimum(light, 1) * (0.65 + 0.35 * GRAIN)
    mask = HATCH & (chars == 0) & (intensity > 0.018)
    chars[mask] = MARK[mask]
    ink[mask] = intensity[mask, None] * WATER_COLOR


def boulder(x, z, radii, seed):
    """An uneven half-ellipsoid bedded in the sand, shaded in world space."""
    azimuth, polar = np.meshgrid(
        np.linspace(0, 2 * np.pi, 110, endpoint=False), np.linspace(0.005, 1.69, 48)
    )
    a, p = azimuth.ravel(), polar.ravel()
    normal = np.column_stack((np.sin(p) * np.cos(a), np.cos(p), np.sin(p) * np.sin(a)))
    facets = 1 + 0.095 * np.sin(3 * a + seed) * np.sin(2 * p) + 0.047 * np.sin(7 * a + 2 * p + seed)
    stone = normal * np.array(radii) * facets[:, None]
    stone[:, 1] += 0.10 * stone[:, 0] + bed_height(x, z) + 0.13
    stone[:, 0] += x
    stone[:, 2] += z
    light = np.maximum(0, (j.normalised(normal / radii) * duet.SURFACE_LIGHT).sum(axis=1))
    grain = np.random.default_rng(seed).uniform(0.81, 1.13, len(a))
    level = np.clip((0.16 + 0.84 * light) * grain, 0, 1)
    # Broken sediment seams give each lit face a different texture.
    seam = np.sin(17 * stone[:, 1] + 2.7 * stone[:, 0] + 0.7 * np.sin(3 * stone[:, 2] + seed))
    level *= 0.72 + 0.28 * j.ease((seam + 0.68) / 0.65)
    ramp = np.array([j.glyph_id(c) for c in "..',:;=+"])
    marks = ramp[np.minimum(len(ramp) - 1, (level * len(ramp)).astype(int))]
    color = (np.array([32, 38, 56]) + level[:, None] * [64, 70, 81]) * grain[:, None]
    return stone, marks, color


def flower_space(x, lift, z):
    # The oral disc tilts towards the open water. Both whorls share its plane.
    return np.column_stack((x, 0.70 * lift + 0.714 * z, -0.714 * lift + 0.70 * z))


class Anemone:
    """Two staggered whorls of fleshy, tapered arms on a low oral disc."""

    def __init__(self):
        rng = np.random.default_rng(2519)
        arms = 18
        ring = np.arange(arms) < 11
        self.a = np.repeat(np.r_[np.arange(11) * 2 * np.pi / 11, (np.arange(7) + 0.4) * 2 * np.pi / 7], 50)
        self.u = np.tile(np.linspace(0, 1, 50), arms)
        self.length = np.repeat(np.where(ring, 0.42, 0.27) * rng.uniform(0.80, 1.20, arms), 50)
        self.radius = np.repeat(np.where(ring, 0.10, 0.042), 50)
        self.root = np.array([1.06, -0.96, 0.44])
        # Low radial disc catches the same amber highlight as the arms.
        angle, r = np.meshgrid(np.linspace(0, 2 * np.pi, 91), np.linspace(0.02, 0.21, 16))
        self.disc = (
            flower_space(
                r.ravel() * np.cos(angle.ravel()),
                (0.19 + 0.055 * (1 - r / 0.21)).ravel(),
                r.ravel() * np.sin(angle.ravel()),
            )
            + self.root
        )
        self.disc_color = (0.5 + 0.35 * (1 + np.cos(12 * angle.ravel())) / 2)[:, None] * [154, 90, 63]
        self.disc_marks = np.full(r.size, j.glyph_id(':'))

    def sample(self, t):
        u, a = self.u, self.a
        phase = 2 * np.pi * t / duet.SECONDS
        # A surge bends the tips after the bases; their roots stay on the rock.
        bend = 0.08 * np.sin(3 * phase - 0.8 * u + 0.4 * np.sin(a)) * u**1.7
        span = self.radius + self.length * (0.76 * u + 0.22 * np.sin(np.pi * u))
        a = a + 0.32 * u * u * np.sin(3 * a + 1.1)
        x = span * np.cos(a) + bend
        z = span * np.sin(a)
        y = self.length * (0.78 * u + 0.55 * np.sin(0.91 * np.pi * u)) + 0.025 * np.sin(2 * a + phase) * u
        points = flower_space(x, y, z) + self.root
        width = (0.016 + 0.036 * np.sin(np.pi * u) ** 0.65) * (1 - 0.40 * u)
        rim = flower_space(-np.sin(a) * width, 0 * u, np.cos(a) * width)
        tip = j.ease((u - 0.80) / 0.17)
        lit = (0.73 + 0.21 * np.cos(a + 1.1) + 0.06 * np.sin(phase + a))[:, None]
        color = (np.array([129, 77, 74]) + u[:, None] * [26, 36, 22] + tip[:, None] * [53, 44, 30]) * lit
        marks = np.where(
            tip > 0.6, j.glyph_id('o'), np.where(np.cos(a) > 0.18, j.glyph_id('/'), j.glyph_id('\\'))
        )
        return (
            np.concatenate((self.disc, points - rim, points + rim, points)),
            np.concatenate((self.disc_marks, np.tile(marks, 3))),
            np.concatenate((self.disc_color, color * 0.74, color * 0.86, color)),
        )


class RockShelf:
    def __init__(self, path):
        self.path = path
        self.pen = Pen(Colony(-0.82, 0.52, 1.25, -0.71, 0.31, 51), path, root_height=-1.49)
        self.anemone = Anemone()
        self.rocks = [
            boulder(x, z, size, seed)
            for x, z, size, seed in [
                (-1.27, 0.63, (0.80, 0.63, 0.56), 4),
                (-0.78, 0.86, (0.36, 0.41, 0.29), 7),
                (1.15, 0.50, (0.68, 0.91, 0.62), 3),
                (1.61, 0.73, (0.48, 0.50, 0.41), 11),
                (0.62, 1.22, (0.33, 0.28, 0.24), 8),
                (-1.95, 1.56, (0.66, 0.40, 0.37), 13),
            ]
        ]
        self.static = [seabed(), *self.rocks]

    def terrain(self, t):
        chars = np.zeros((ROWS, COLS), np.uint8)
        ink = np.zeros((ROWS, COLS, 3), np.uint8)
        parts = [*self.static, self.pen.sample(t), self.anemone.sample(t)]
        project_material(self.path, t, parts, chars, ink)
        return chars, ink

    def frame(self, t):
        chars = np.zeros((ROWS, COLS), np.uint8)
        ink = np.zeros((ROWS, COLS, 3), np.uint8)
        draw_shafts(t, chars, ink)
        soil, pigment = self.terrain(t)
        # Once the lower ledge has left the shot, dissolve the outer rocks
        # back in for the loop. The swimmers and their motion never dissolve.
        if t > 16:
            arriving, other = self.terrain(t - duet.SECONDS)
            mix = j.ease((t - 16) / 3.9)
            pigment = pigment * (1 - mix)
            other = other * mix
            soil = np.where(other.max(axis=2) > pigment.max(axis=2), arriving, soil)
            pigment = np.maximum(other, pigment)
        solid = soil != 0
        ink[solid] = pigment[solid]
        chars[solid] = soil[solid]
        return paint_material(chars, ink)
