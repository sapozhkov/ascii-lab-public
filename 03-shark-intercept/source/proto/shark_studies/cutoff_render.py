"""Render the 20-second ASCII reel and its synchronized model."""
import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proto.channel import HANDLE
from .cutoff_motion import FPS, approach, states
from .dive_pose import caudal_reaction, smoothstep, unit, water_acceleration, world_points


HERE = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Menlo.ttc"
COLS, ROWS, CW, CH = 90, 72, 12, 18
WIDTH, SCENE_H, FRAME_H = COLS * CW, ROWS * CH, 1920
LOOP_SECONDS = 20
SAMPLES = 6
RAMP = np.frombuffer(b" .:;i!l|/txcoahkO8B%#@", np.uint8) - 32

Y, X = np.mgrid[0:ROWS, 0:COLS].astype(float)
X += .5
Y = (Y + .5) * 1.5


# Glyph atlas and print


def glyphs(text):
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8) - 32


def select(ramp, value):
    index = np.rint(np.clip(value, 0, 1) * (len(ramp) - 1))
    return ramp[index.astype(np.intp)]


def glyph_atlas(font, cell_width, cell_height):
    out = np.zeros((96, cell_height, cell_width), np.float32)
    for code in range(32, 127):
        tile = Image.new("L", (cell_width, cell_height))
        lift = 5 if code == ord("_") else 3
        ImageDraw.Draw(tile).text((0, -lift), chr(code), font=font, fill=255)
        out[code - 32] = np.asarray(tile, np.float32) / 255
    return out


@lru_cache(maxsize=1)
def atlas():
    return glyph_atlas(ImageFont.truetype(FONT,20,index=1),12,18)


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


# Reef lighting and silver fish


@dataclass
class Camera:
    focus: np.ndarray
    azimuth: float
    elevation: float
    scale: float
    aim: tuple = (45, 53)

    def __post_init__(self):
        c, s = np.cos(self.azimuth), np.sin(self.azimuth)
        h, v = np.cos(self.elevation), np.sin(self.elevation)
        self.axes = np.array([[c, s, 0], [s*v, -c*v, -h], [s*h, -c*h, v]])

    def project(self, points):
        relative = np.einsum("...j,ij->...i", points-self.focus, self.axes)
        return relative[..., :2] * self.scale + self.aim

    def plane(self, height):
        """World intersections of parallel viewing rays with a horizontal plane."""
        u = (X-self.aim[0])/self.scale
        v = (Y-self.aim[1])/self.scale
        depth = (height - self.focus[2] - v*self.axes[1, 2])/self.axes[2, 2]
        return tuple(self.focus[k] + u*self.axes[0,k] + v*self.axes[1,k] +
                     depth*self.axes[2,k] for k in (0,1))


def noise(x, y):
    i, j = np.floor(x), np.floor(y)
    u, v = smoothstep(0, 1, x-i), smoothstep(0, 1, y-j)

    def h(a,b):
        q = np.sin(71.31*(i+a)+127.73*(j+b))*43758.5
        return q-np.floor(q)

    return (1-v)*((1-u)*h(0,0)+u*h(1,0)) + v*((1-u)*h(0,1)+u*h(1,1))


def caustics(x, y, t, period=None):
    # Warp intersecting wave families. Only converging fronts concentrate light.
    rates = np.array([.65, .33, .57, .48, .70, .55])
    if period is not None:
        fundamental = 2*np.pi / period
        rates = fundamental * np.rint(rates / fundamental)
    a, b, c, d, e, f = rates * t
    u = 8.6*x+5*y + 1.7*np.sin(3.1*y-a) + 2.8*np.sin(2*x+1.2*y+b)
    v = 3.7*x-11.9*y + 2.1*np.sin(3*x-1.3*y+c) - 1.4*np.sin(2*y-d)
    fold = np.cos(u-e)+.78*np.cos(v+f)
    return np.exp(-8*(fold-.23)**2)


def environment(cam, t, shadow, *, period):
    """Light from a moving surface attenuates toward the distant reef."""
    x,y = cam.plane(0.)
    macro = noise(1.15*x+2,1.15*y)
    stone = .50*macro+.30*noise(3.6*x,3.6*y)+.20*noise(11*x,11*y)
    edge = .38*np.sin(1.1*x+.5) - .24*x + (stone-.5)*.85
    bank = smoothstep(-.12,.15, y-edge-.57)
    slope = .5+.5*np.sin(48*y+12*x+2.2*np.sin(5*x+3*y))
    sand = .48+.18*macro+.067*slope
    drift_x, drift_y = .06*t, -.03*t
    if period is not None:
        omega = 2*np.pi / period
        drift_x = .06*np.sin(omega*t)/omega
        drift_y = (-.03*np.sin(omega*t)+.05*(1-np.cos(omega*t)))/omega
    shafts = .24+.76*noise(1.7*x+drift_x,1.7*y+drift_y)
    light = caustics(x,y,t,period)*shafts*.57
    # Blue-green sand, deep petrol water and a broken ochre reef lip.
    lit = sand + .52*light
    ink = np.array([18,53,63]) + lit[...,None]*[130,175,162]
    reef = np.array([16,32,39])+stone[...,None]**2*[147,166,117]
    ink += bank[...,None]*(reef-ink)
    paper = np.array([10,35,42])+((sand+.25*light)*(1-bank))[...,None]*[37,63,58]
    value = (.20+.33*sand+.39*light)*(1-bank)+bank*(.05+.64*stone**1.4)
    shadow = np.clip(shadow,0,1)
    ink *= (1-.77*shadow[...,None])
    paper *= (1-.73*shadow[...,None])
    chars = select(glyphs("  ..,,::;==++xsoO8@"),value*(1-.66*shadow))
    # A broad vignette represents loss of side light, not a frame overlay.
    edge = np.clip(1.05-.23*((X-44)/53)**2-.10*((Y-48)/67)**2,.68,1)
    return chars, ink*edge[...,None], paper*edge[...,None]


def small_fish(chars, ink, paper, xy, tangent, lengths, light, depth, phase, sunlit_sand=False):
    """One fusiform body and forked tail, sampled in screen cells."""
    for i in np.argsort(depth):
        cx,cy = xy[i]
        length = lengths[i]
        if cx < -length or cx > 90+length or cy < -length or cy > 108+length:
            continue
        x0,x1 = max(0,int(cx-length)), min(90,int(cx+length)+1)
        y0,y1 = max(0,int((cy-length)/1.5)), min(72,int((cy+length)/1.5)+1)
        region = np.s_[y0:y1,x0:x1]
        dx = X[region]-cx
        dy = Y[region]-cy
        a,b = tangent[i]
        side = (-b*dx+a*dy)
        along = (a*dx+b*dy)/length
        radius = np.maximum(.38, length*.128)
        width = radius*np.sqrt(np.maximum(0,1-((along-.06)/.46)**2))
        tail = np.maximum(0,-along-.24)*length*.72
        wobble = np.sin(phase[i])*.30*tail
        dist = np.maximum(np.maximum(-.57-along,along-.52)*length,
                          np.abs(side-wobble)-np.maximum(width,tail))
        coverage = np.clip(.48-dist,0,1)
        core = coverage>.21
        if sunlit_sand:
            # Under a bright silver belly the thin lower margin stays dark.
            # Its penumbra separates the return leg from the pale sand below.
            border = np.exp(-2.5*(dist-.63)**2)*np.clip(side/radius+.20,0,1)
            paper[region] *= 1-.35*border[...,None]
            ink[region] *= 1-.28*border[...,None]
        if not core.any():
            continue
        sheen = np.clip(light[i]+.13*(1-np.abs(side)/radius)-.12*np.maximum(0,-along),0,1)
        back = np.clip((side-wobble)/radius,-1,1)
        stripe = .74+.26*np.exp(-((back+.10)/.50)**2)
        hue = (np.array([61,110,129])+sheen[...,None]*[190,143,119])*stripe[...,None]
        if sunlit_sand:
            hue = np.minimum(255,hue+[37,24,3])
        feather = smoothstep(.13,.88,coverage)[...,None]
        paper[region] += feather*(np.array([28,43,52])+sheen[...,None]*[102,107,103]-paper[region])
        if sunlit_sand:
            paper[region] += feather*sheen[...,None]*[32,25,12]
        ink[region][core] = hue[core]
        chars[region][core] = select(glyphs("-~=+xoO8%#@"),.22+.76*sheen[core])
        if length>4.8:
            tailtip = core & (along<-.32)
            chars[region][tailtip] = ord('/' if a*b<0 else chr(92))-32


# Shark geometry, reflectance and glyph selection


STATIONS = (-.292,-.260,-.205,-.125,-.035,.060,.155,.245,.330,.395,.448,.480,.498,.505)
HALF_WIDTH = (.000,.014,.023,.036,.050,.067,.077,.078,.068,.057,.044,.032,.016,.000)
BACK = (.005,.023,.028,.040,.054,.067,.075,.073,.061,.048,.034,.023,.012,.000)
BELLY = (.000,.018,.022,.031,.046,.060,.068,.069,.059,.046,.030,.016,.008,.000)
# Each fin is attached in the same longitudinal, lateral, vertical coordinates.
DORSAL = ((.186,0,.063),(.154,0,.091),(.105,0,.148),(.082,0,.165),(.076,0,.154),
          (.077,0,.106),(.065,0,.082),(.035,0,.061),(.019,0,.056))
SECOND = ((-.150,0,.028),(-.180,0,.063),(-.189,0,.065),(-.189,0,.041),(-.209,0,.026))
UPPER = ((-.248,0,.016),(-.299,0,.036),(-.396,0,.093),(-.496,0,.126),
         (-.502,0,.122),(-.478,0,.099),(-.454,0,.082),(-.447,0,.084),
         (-.404,0,.050),(-.335,0,.007),(-.295,0,-.012))
LOWER = ((-.252,0,-.010),(-.304,0,-.024),(-.347,0,-.101),
         (-.340,0,-.114),(-.323,0,-.100),(-.296,0,-.065),(-.276,0,-.029))
PECTORAL = ((.281,.047,-.008),(.267,.086,-.020),(.218,.145,-.050),
            (.157,.205,-.089),(.139,.219,-.102),(.140,.200,-.096),
            (.169,.125,-.063),(.170,.080,-.037),(.150,.056,-.014))
PELVIC = ((-.060,.032,-.021),(-.087,.063,-.049),(-.124,.087,-.063),
          (-.117,.055,-.038),(-.139,.027,-.027))
ANAL = ((-.153,0,-.024),(-.187,0,-.064),(-.194,0,-.057),(-.188,0,-.037),(-.207,0,-.027))


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
    axial = np.unique(np.r_[STATIONS, np.linspace(-.28, .5, 37)])
    w = np.interp(axial, STATIONS, HALF_WIDTH)
    h = np.interp(axial, STATIONS, BACK)
    b = np.interp(axial, STATIONS, BELLY)
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

    for pts, tag in [(DORSAL, 1), (SECOND, 1), (ANAL, 1),
                     (UPPER, 2), (LOWER, 2)]:
        fin(pts, tag, 2, .115)
    for side in [-1, 1]:
        for pts in (PECTORAL, PELVIC):
            fin(np.array(pts) * [1, side, 1], 3, 1, .105)
    faces = np.array(faces)
    vertices, indices = np.unique(faces.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, indices.reshape(-1, 4), np.array(parts), faces.mean(axis=1)


def surface_color(rest, normals, parts, rotation, center, length, t):
    x, y, z = rest.T
    half = np.maximum(.004, np.interp(x, STATIONS, BACK))
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


def cheek(x,z,side):
    x,z=np.broadcast_arrays(x,z)
    width=np.interp(x,STATIONS,HALF_WIDTH)
    half=np.where(z>=0,np.interp(x,STATIONS,BACK),
                  np.interp(x,STATIONS,BELLY))
    y=side*width*np.sqrt(np.clip(1-(z/np.maximum(.004,half))**2,0,1))
    return np.column_stack((x,y,z))


def jaw_contours(pose, cam, side):
    """Project the upper maxillary edge, mobile mandible and dark buccal space."""
    gape=pose.gape
    reach=np.array([0.,.22,.52,.81,.96,1.])
    x=.342+.144*reach
    upper=cheek(x,[-.017,-.016,-.011,-.006,-.002,-.001],side)
    lower=cheek(x-.013*reach*gape,[-.017,-.021,-.026,-.028,-.024,-.014],side)
    # The mandible ends below the projecting rostrum. Its inside faces
    # turn down while the narrow maxillary edge still catches skylight.
    roof=world_points(upper,pose)+(.007*gape*reach)[:,None]*pose.up
    jaw=world_points(lower,pose)-(.014*gape*reach)[:,None]*pose.up
    lower_edge=jaw-(.012*reach)[:,None]*pose.up
    def project(points):
        return np.rint(cam.project(points)*[SAMPLES,SAMPLES/1.5]).astype(np.int32)
    return project(roof),project(jaw),project(lower_edge)


def cell_ink(canvas):
    reduced=cv2.resize(canvas,(COLS,ROWS),interpolation=cv2.INTER_AREA)
    coverage=reduced[...,3].astype(np.float32)/255
    light=reduced[...,:3]/np.maximum(coverage[...,None],.001)
    lum=np.einsum('ijk,k->ij',light,[.23,.65,.12])
    strength=np.clip((lum-17)/206,0,1)
    chars=RAMP[np.rint((.12+.88*strength**.82)*(len(RAMP)-1)).astype(int)]
    dy,dx=np.gradient(coverage)
    edge=(coverage>.12)&(coverage<.67)
    edge_sign=np.where(abs(dx)>1.7*abs(dy),ord('|'),ord('_'))
    slant=(abs(dx)<=1.7*abs(dy))&(abs(dy)<=1.7*abs(dx))
    edge_sign=np.where(slant,np.where(dx*dy>0,ord('/'),ord(chr(92))),edge_sign)
    chars[edge]=edge_sign[edge]-32
    ink=np.clip(1.24*light+[9,15,19],0,255).astype(np.uint8)
    # Most of the light stays on the printed surface. The narrow brightest
    # reflections retain a little scattered light between the strokes.
    paper=np.clip(light*(.09+.37*strength[...,None]**2),0,255).astype(np.uint8)
    return coverage,chars,ink,paper


def shark_cells(pose,cam,t):
    """Rasterize a supersampled moving mesh, then select one symbol per cell."""
    vertices,indices,parts,rest=mesh()
    world=world_points(vertices,pose)
    camera_space=np.einsum('ij,kj->ik',world-pose.center,cam.axes)
    faces=camera_space[indices]
    normals=np.cross(faces[:,2]-faces[:,0],faces[:,3]-faces[:,1])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
    normals*=np.where(normals[:,2]<0,-1.,1.)[:,None]
    rotation=np.array([pose.forward,pose.right,pose.up]) @ cam.axes.T
    length=cam.scale
    rgb=surface_color(rest,normals,parts,rotation,cam.project(pose.center),length,t)
    rgba=np.column_stack((rgb,np.full(len(rgb),255))).tolist()
    resolution=np.array([SAMPLES,SAMPLES/1.5])
    points=np.rint(cam.project(world)*resolution).astype(np.int32)[indices]
    canvas=np.zeros((ROWS*SAMPLES,COLS*SAMPLES,4),np.uint8)
    for i in np.argsort(faces[:,:,2].mean(axis=1)):
        cv2.fillConvexPoly(canvas,points[i],rgba[i])

    def project(points):
        xy=cam.project(world_points(points,pose))
        return np.rint(xy*resolution).astype(np.int32)

    for side in (-1,1):
        facing=side*rotation[1,2]
        if facing>.18:
            for gill in (.265,.283,.301,.319,.335):
                slit=cheek([gill,gill-.007,gill-.005],[.019,-.001,-.022],side)
                cv2.polylines(canvas,[project(slit)],False,(29,44,50,255),max(1,round(length*.003*SAMPLES)))
            roof,jaw,lower_edge=jaw_contours(pose,cam,side)
            if pose.gape>.065:
                # The cold dark cavity has a defined lower lip. Its hue stays
                # neutral so the amber prey remains the distinct focal point.
                opening=np.vstack((roof,jaw[::-1]))
                cv2.fillPoly(canvas,[opening],(4,9,13,255))
                cv2.fillPoly(canvas,[np.vstack((jaw,lower_edge[::-1]))],(141,156,152,255))
                cv2.polylines(canvas,[jaw[1:]],False,(192,199,180,255),max(1,round(length*.0045*SAMPLES)))
                cv2.polylines(canvas,[roof[1:]],False,(154,170,167,255),max(1,round(length*.0035*SAMPLES)))
            else:
                cv2.polylines(canvas,[jaw],False,(40,49,52,255),max(1,round(length*.003*SAMPLES)))

    coverage,chars,ink,paper=cell_ink(canvas)
    return coverage,chars,ink,paper


def grid(f, cam, t, swimmer_time):
    pose = f.shark
    coverage, skin_chars, skin, skin_paper = shark_cells(pose, cam, swimmer_time)
    displacement = cam.project(pose.center + pose.center[2] * np.array([.19, -.14, -1])) - cam.project(pose.center)
    transform = np.float32([[1, 0, displacement[0]], [0, 1, displacement[1] / 1.5]])
    shadow = cv2.warpAffine(coverage, transform, (90, 72))
    shadow = .54 * cv2.GaussianBlur(shadow, (0, 0), .6)
    chars, ink, paper = environment(cam, t, shadow, period=LOOP_SECONDS)
    # Keep enough light on the seabed to read the dark body in negative.
    paper = paper * [.65, .89, .97] + [1, 5, 12]
    ink = .50 * (ink * [.53, .70, .79] + [3, 8, 14]) + .50 * paper
    where = coverage > .28
    weight = smoothstep(.08, .91, coverage)[..., None]
    xy = cam.project(f.fish)
    heading = unit(f.velocity) @ cam.axes.T
    length = cam.scale * f.sizes * np.linalg.norm(heading[:, :2], axis=1)
    depth = (f.fish - pose.center) @ cam.axes[2]
    silver = np.clip(.58 + .21 * np.abs(heading[:, 0]) + .10 * np.sin(f.phase * .34), 0, 1)
    alive = np.ones(len(length), dtype=bool)
    if f.caught and f.target is not None:
        alive[f.target] = False
    for front in (False, True):
        ids = alive & ((depth >= 0) == front)
        under = ink.copy()
        small_fish(chars, ink, paper, xy[ids], unit(heading[ids, :2]),
                   length[ids], silver[ids], depth[ids], f.phase[ids], sunlit_sand=True)
        # Silver flanks cover several bright strokes even when downsampled.
        # Apply the same material to every visible fish, at either depth.
        scales = (ink[..., 0] > under[..., 0] + 10) & (ink[..., 0] > 120)
        paper[scales] += .42 * (ink[scales] - paper[scales])
        if not front:
            ink += weight * (skin - ink)
            paper += weight * (skin_paper - paper)
            chars[where] = skin_chars[where]
    return chars, np.uint8(np.clip(ink, 0, 255)), np.uint8(np.clip(paper, 0, 255))


# Camera framing and loop


def backbone(pose):
    return pose.center + pose.spine @ np.array([pose.forward, pose.right, pose.up])


def fixed_camera(frames):
    # Fit the split, the entire pursuit, and the mouth at contact.
    # The ends of the incoming rows are allowed to enter through the frame.
    target = next(f.target for f in frames if f.target is not None)
    active = [f for f in frames if np.any(f.alarm) and not f.caught]
    points = np.vstack([
        np.stack([f.fish[target] for f in active]),
        np.vstack([backbone(f.shark)[[0, 12]] for f in active[::4]]),
        np.vstack([np.quantile(f.fish, [.08, .92], axis=0) for f in active[::8]]),
    ])
    cam = Camera(points.mean(axis=0), 0., np.deg2rad(83), 1., (0., 0.))
    # Keep the school approaching at an oblique angle across the frame.
    choices = []
    for azimuth in np.linspace(.33, 1.22, 31) - np.pi:
        trial = Camera(cam.focus, azimuth, cam.elevation, 1., (0., 0.))
        xy = trial.project(points)
        scale = min(58., 81. / np.ptp(xy[:, 0]), 96. / np.ptp(xy[:, 1]))
        mid = (np.min(xy, axis=0) + np.max(xy, axis=0)) / 2
        choices.append((scale, azimuth, mid))
    scale, angle, mid = max(choices, key=lambda entry: entry[0])
    return Camera(cam.focus, angle, cam.elevation, scale, np.array([45, 54]) - scale * mid)


def loop_frames(frames, approach, cam):
    """Two consecutive shoals cross the same static water at 20 s intervals."""
    period = LOOP_SECONDS * FPS
    # Switch the tracked swimmer only when its entire body and shadow have left.
    # Both shoals remain in the model throughout their separate passages.
    for i in range(period):
        if i < period // 2:
            neighbour = frames[i + period]
        else:
            neighbour = approach[i - period]
        primary, secondary = frames[i], neighbour
        if i >= period // 2:
            spine = cam.project(backbone(primary.shark))
            if spine[:, 1].max() < -16:
                primary, secondary = neighbour, primary
        yield primary, secondary


def join_shoals(primary, secondary):
    # Track indices remain stable within the active passage. Its caught fish
    # stays at the jaw in the model and remains hidden by the existing shader.
    alive = np.ones(len(secondary.fish), dtype=bool)
    if secondary.caught:
        alive[secondary.target] = False
    return replace(primary, **{
        key: np.concatenate((getattr(primary, key), getattr(secondary, key)[alive]))
        for key in ("fish", "velocity", "phase", "sizes", "alarm")
    })


def write_model(frames, camera, output):
    data = []
    for f in frames:
        def coords(points):
            return camera.project(points).round(2).tolist()
        stage = f.stage
        data.append(dict(fish=coords(f.fish), spine=coords(backbone(f.shark)),
                         target=f.target, stage=stage, caught=bool(f.caught)))
        pose = f.shark
        data[-1]["swim"] = [round(float(value), 3) for value in
                            (pose.speed, pose.thrust, 2.45*pose.speed**2,
                             pose.tail_speed, np.rad2deg(pose.bank))]
    template = Path(__file__).with_name("cutoff_view.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=int)
    page = template.replace("__FRAMES__", payload).replace("__FPS__", str(FPS))
    output.write_text(page, encoding="utf-8")
    return data


# Reel typography and export


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


def render_scene(frames, camera, swimmer_times, output):
    # Keep the first encode so the published scene and type reproduce exactly.
    panel = code_panel((caudal_reaction, water_acceleration))
    # A gentle shoulder keeps the silver flanks legible after phone downsampling.
    levels = np.arange(256, dtype=float) / 255
    silver_print = np.uint8(np.rint(255 * np.minimum(1., levels + .18 *
                                smoothstep(.18, .63, levels) * (1. - levels))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as writer:
        for i, (frame, swimmer_time) in enumerate(zip(frames, swimmer_times)):
            chars, ink, paper = grid(frame, camera, i / FPS, swimmer_time)
            scene = paint(chars, ink, atlas(), paper, bloom=.26)
            writer.stdin.write(cv2.LUT(scene, silver_print).tobytes())
            writer.stdin.write(panel.tobytes())
        writer.stdin.close()
        if writer.wait():
            raise RuntimeError("ffmpeg could not encode the shot")


def finish_movie(source, output, frames, school_size):
    cuts = panel_timeline(frames)
    catch_frame = next(index for index, section in cuts if section == "catch")
    for _, section in cuts:
        source_panel(section)
    timeline = iter(cuts)
    _, section = next(timeline)
    change = next(timeline, None)
    panel = source_panel(section)
    status = ImageFont.truetype(FONT, STATUS_SIZE)

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
    with subprocess.Popen(writer + [str(output)], stdin=subprocess.PIPE) as dst:
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


def release_loop():
    frames = list(states(seconds=LOOP_SECONDS * 3 // 2, wider_escape=True, evasive_prey=True))
    camera = fixed_camera(list(states()))
    incoming = list(approach(frames[0], LOOP_SECONDS // 2, steady_entry=True))
    pairs = list(loop_frames(frames, incoming, camera))
    swimmer_times = [i / FPS if primary is frames[i] else i / FPS - LOOP_SECONDS
                     for i, (primary, _) in enumerate(pairs)]
    joined = [join_shoals(primary, secondary) for primary, secondary in pairs]
    return joined, camera, swimmer_times, len(frames[0].fish)


def play_scene(frames, camera, swimmer_times, duration):
    from terminal_player import play_terminal

    def frame_at(t):
        index = round(t * FPS)
        chars, ink, paper = grid(frames[index], camera, t, swimmer_times[index])
        return chars, ink / 255, paper / 255

    play_terminal(frame_at, fps=FPS, duration=duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="directory for cutoff.mp4 and model.html (default: output)")
    commands = parser.add_subparsers(dest="format")
    terminal = commands.add_parser("terminal", help="Play the scene in an ANSI terminal")
    terminal.add_argument("--duration", type=float, default=LOOP_SECONDS, help="Duration in seconds")
    args = parser.parse_args()

    if args.format == "terminal":
        if not 0 < args.duration <= LOOP_SECONDS:
            terminal.error(f"duration must be greater than 0 and at most {LOOP_SECONDS} seconds")
        frames, camera, swimmer_times, _ = release_loop()
        play_scene(frames, camera, swimmer_times, args.duration)
        return

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames, camera, swimmer_times, school_size = release_loop()
    with TemporaryDirectory(dir=out) as workdir:
        work = Path(workdir)
        model = write_model(frames, camera, work / "model.html")
        render_scene(frames, camera, swimmer_times, work / "scene.mp4")
        finish_movie(work / "scene.mp4", work / "cutoff.mp4", model, school_size)
        (work / "cutoff.mp4").replace(out / "cutoff.mp4")
        (work / "model.html").replace(out / "model.html")


if __name__ == "__main__":
    main()
