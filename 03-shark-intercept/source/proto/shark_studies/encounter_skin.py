"""An articulated grey reef shark sampled onto the fixed character grid.

Coordinates follow the landmarks in reef_shark: longitudinal distance from
the animal's centre, distance to starboard, and height above its midline.
Only the anatomical surface is supersampled. A cell emits one glyph.
"""
from functools import lru_cache

import cv2
import numpy as np

from . import reef_shark as anatomy


ROWS, COLS = 72, 90
SAMPLES = 6
RAMP = np.frombuffer(b" .:;i!l|/txcoahkO8B%#@", np.uint8) - 32


@lru_cache(maxsize=1)
def mesh():
    faces, parts = [], []

    def add(strip, part):
        # Adjacent closed rings share their contour vertices.
        rolled = np.roll(strip, -1, axis=1)
        quads = np.stack((strip[:-1], rolled[:-1], rolled[1:], strip[1:]), axis=2)
        faces.extend(quads.reshape(-1, 4, 3))
        parts.extend([part] * ((len(strip) - 1) * strip.shape[1]))

    # Extra rings at the shoulder and head carry the lateral line into the snout.
    axial = np.unique(np.r_[anatomy.STATIONS, np.linspace(-.28, .5, 37)])
    w = np.interp(axial, anatomy.STATIONS, anatomy.HALF_WIDTH)
    h = np.interp(axial, anatomy.STATIONS, anatomy.BACK)
    b = np.interp(axial, anatomy.STATIONS, anatomy.BELLY)
    ang = np.arange(28) * (2 * np.pi / 28)
    z = np.sin(ang)
    trunk = np.stack(np.broadcast_arrays(axial[:, None], w[:, None] * np.cos(ang),
                       np.where(z > 0, h[:, None], b[:, None]) * z), axis=-1)
    add(trunk, 0)

    def fin(points, part, span_axis, thickness):
        # Reconstruct an aerofoil between the leading and trailing margin at
        # each cross section. Foil thickness makes an edge-on fin visible.
        outline = np.array(points)
        span = outline[:, span_axis]
        rings = []
        for level in np.linspace(span.min()+1e-6, span.max()-1e-6, 13):
            ends = []
            for p, q in zip(outline, np.roll(outline, -1, axis=0)):
                if (p[span_axis] <= level < q[span_axis]) or (q[span_axis] <= level < p[span_axis]):
                    ends.append(p + (q - p) * ((level - p[span_axis]) / (q[span_axis] - p[span_axis])))
            if len(ends) < 2:
                continue
            leading, trailing = max(ends, key=lambda p: p[0]), min(ends, key=lambda p: p[0])
            chord = np.linalg.norm(leading - trailing)
            ring = []
            for u, camber in [(0.,0.),(.27,1.),(.78,.35),(1.,0.),(.78,-.35),(.27,-1.)]:
                point = leading*(1-u) + trailing*u
                point[3-span_axis] += camber * chord * thickness
                ring.append(point)
            rings.append(ring)
        add(np.array(rings), part)

    for pts, tag in [(anatomy.DORSAL, 1), (anatomy.SECOND, 1), (anatomy.ANAL, 1),
                     (anatomy.UPPER, 2), (anatomy.LOWER, 2)]:
        fin(pts, tag, 2, .115)
    for side in [-1, 1]:
        for pts in (anatomy.PECTORAL, anatomy.PELVIC):
            fin(np.array(pts) * [1, side, 1], 3, 1, .105)
    faces = np.array(faces)
    vertices, indices = np.unique(faces.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, indices.reshape(-1, 4), np.array(parts), faces.mean(axis=1)


def articulated(vertices, phase):
    x, y, z = vertices.T
    behind = np.clip((.31 - x) / .812, 0, 1)
    wave = phase - 4.5 * behind
    swing = .096 * behind**2.35
    offset = swing * np.sin(wave)
    slope = (-.226 * behind**1.35 * np.sin(wave) + 4.5 * swing * np.cos(wave)) / .812
    tangent = 1 / np.sqrt(1 + slope*slope)
    return np.column_stack((x - y*slope*tangent, offset+y*tangent, z))


def surface_color(rest, normals, parts, rotation, center, length, t):
    x, y, z = rest.T
    half = np.maximum(.004, np.interp(x, anatomy.STATIONS, anatomy.BACK))
    side = z / half
    # Grey reef sharks have a distinct pale ventral field and blue-slate back.
    pale = np.clip((.40 - side) / .80, 0, 1)
    pale = pale*pale*(3 - 2*pale)
    base = np.array([26., 48., 65.]) + pale[:, None] * [176., 161., 137.]
    band = np.exp(-((side - .03 - .065*np.cos((x-.10)*12))/.14)**2)
    base += band[:, None] * [23, 29, 30]
    fin = parts != 0
    base[fin] = [55, 82, 96]
    # The rounded leading margin turns toward the flow and toward skylight.
    leading = np.maximum(0, np.einsum('ij,j->i', normals, rotation[0]))
    base[fin] += (leading[fin]**1.4)[:, None] * [56, 70, 70]
    # The grey reef shark carries charcoal caudal margins and dark pectoral tips.
    caps = ((parts == 2) & (x < -.345)) | ((parts == 3) & (abs(y) > .169))
    base[caps] *= .58
    # Anatomical up follows the model's orientation. Incident light is stable
    # in camera space as the animal banks under it.
    key = np.array([-.45, -.54, .71])
    half_vector = key + [0, 0, 1]
    half_vector /= np.linalg.norm(half_vector)
    diffuse = np.maximum(0, np.einsum("ij,j->i", normals, key))
    facing = np.maximum(0, normals[:, 2])
    fresnel = (1-facing)**2.2
    camera_points = np.einsum("ij,jk->ik", rest, rotation)
    pos = camera_points[:, :2] * length + center
    wave = .24*pos[:, 0] - .13*pos[:, 1] + 1.6*np.sin(.065*pos[:, 1] + .46*t) - .96*t
    caustic = np.exp(-4.9*(1 - np.cos(wave)))
    reflection = np.maximum(0, np.einsum("ij,j->i", normals, half_vector))
    sky = .60 + .44*diffuse + .13*fresnel
    blue = np.array([97, 155, 177])
    white = np.array([239, 247, 241])
    result = base*sky[:, None] + (reflection**5 * (.28 + .28*caustic))[:, None]*blue
    result += (reflection**22 * (.12 + .98*caustic))[:, None]*white
    result += ((.19 + .28*caustic)*fresnel*diffuse)[:, None] * blue
    # The rounded shoulder reflects a broad strip of sky along the lateral
    # line. The fin and the dark caudal margin keep their own material.
    lateral = np.exp(-((side-.16)/.45)**2) * (parts==0)
    result += (lateral*(.45+.55*diffuse))[:,None]*[49,58,59]
    return np.clip(result, 0, 255).astype(np.uint8)


def shark_cells(center, forward, right, up, length, phase, t):
    """Project onto 90×72 cells of aspect 1:1.5. Camera z points to viewer.

    center and length are in column widths; axes are 3-D unit vectors.
    The result is coverage, printable glyph index, glyph RGB and paper RGB.
    """
    rotation = np.array([forward, right, up], dtype=float)
    center = np.asarray(center, dtype=float)
    vertices, indices, parts, rest = mesh()
    posed = np.einsum("ij,jk->ik", articulated(vertices, phase), rotation)
    contours = posed[indices]
    normals = np.cross(contours[:, 2]-contours[:, 0], contours[:, 3]-contours[:, 1])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    normals *= np.where(normals[:, 2] < 0, -1., 1.)[:, None]
    rgb = surface_color(rest, normals, parts, rotation, center, length, t)
    projected = (posed[:, :2]*length + center) * [SAMPLES, SAMPLES/1.5]
    points = np.rint(projected).astype(np.int32)[indices]
    colors = np.column_stack((rgb, np.full(len(rgb), 255))).tolist()
    canvas = np.zeros((ROWS*SAMPLES, COLS*SAMPLES, 4), np.uint8)
    for i in np.argsort(contours[:, :, 2].mean(axis=1)):
        cv2.fillConvexPoly(canvas, points[i], colors[i], lineType=cv2.LINE_8)
    # Mark the near side on the actual cheek. The eye is dark in a narrow
    # orbital rim. Slits follow the gill arch behind it, tapering ventrally.
    eyes = []
    for sign in [-1, 1]:
        def project(x, height, lift=0.):
            x = np.asarray(x)
            height = np.asarray(height)
            width = np.interp(x, anatomy.STATIONS, anatomy.HALF_WIDTH)
            surface_h = np.interp(x, anatomy.STATIONS, anatomy.BACK)
            transverse = sign*(width*np.sqrt(np.maximum(0, 1-(height/surface_h)**2)) + lift)
            pts = np.einsum("ij,jk->ik", articulated(np.column_stack((x, transverse, height)), phase), rotation)
            return np.rint((pts[:, :2]*length + center)*[SAMPLES, SAMPLES/1.5]).astype(np.int32)

        if sign*rotation[1, 2] + .29*rotation[2, 2] + .22*rotation[0, 2] > .17:
            eye = project([.429], [.016], .001)[0]
            eyes.append(eye)
            radius = max(1, round(.009*length*SAMPLES))
            cv2.ellipse(canvas, eye, (radius, max(1, round(.76*radius))), 0, 0, 360, (157,176,166,255), -1)
            cv2.circle(canvas, eye, max(1, round(.62*radius)), (11,24,31,255), -1)
        if sign*rotation[1, 2] > .18:
            for gill in (.265, .283, .301, .319, .335):
                path = project(np.array([gill, gill-.007, gill-.005]), np.array([.019, -.001, -.022]))
                cv2.polylines(canvas, [path], False, (27,46,56,255), max(1, round(length*.003*SAMPLES)))
            mouth = project([.365,.392,.419,.447,.465],[-.026,-.027,-.024,-.019,-.013])
            cv2.polylines(canvas, [mouth], False, (28,43,47,255), max(1,round(length*.003*SAMPLES)))

    reduced = cv2.resize(canvas, (COLS,ROWS), interpolation=cv2.INTER_AREA)
    coverage = reduced[..., 3].astype(np.float32)/255
    light = reduced[..., :3] / np.maximum(coverage[..., None], .001)
    lum = np.einsum("ijk,k->ij", light, [.23, .65, .12])
    strength = np.clip((lum-17)/206, 0, 1)
    chars = RAMP[np.rint((.12 + .88*strength**.82)*(len(RAMP)-1)).astype(int)]
    dy, dx = np.gradient(coverage)
    edge = (coverage > .12) & (coverage < .67)
    direction = np.where(abs(dx) > 1.7*abs(dy), ord('|'), ord('_'))
    slant = (abs(dx) <= 1.7*abs(dy)) & (abs(dy) <= 1.7*abs(dx))
    direction = np.where(slant, np.where(dx*dy > 0, ord('/'), ord('\\')), direction)
    chars[edge] = direction[edge]-32
    # Midtones are printed with ink on tinted dark paper. Reflections have a
    # locally brighter substrate; the glyph is still visible at a phone width.
    ink = np.clip(1.22*light + [8, 13, 15], 0, 255).astype(np.uint8)
    paper = np.clip(light*(.18 + .37*strength[..., None]**1.9), 0, 255).astype(np.uint8)
    # At reel scale a small orbit and the eye share a cell. Give that cell an
    # open glyph: its dark centre carries the pupil even after downsampling.
    for px, py in eyes:
        ex, ey = px//SAMPLES, py//SAMPLES
        if length >= 38 and 0 <= ex < COLS and 0 <= ey < ROWS and coverage[ey, ex] > .65:
            chars[ey, ex] = ord('o')-32
            ink[ey, ex] = [204, 204, 172]
            paper[ey, ex] = [13, 23, 29]
    return coverage, chars, ink, paper
