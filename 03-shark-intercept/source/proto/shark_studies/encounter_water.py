"""A camera, underwater light and individual silver fish on the 90 x 72 grid."""

from dataclasses import dataclass

import cv2
import numpy as np

from .school_render import glyphs, select


Y, X = np.mgrid[0:72, 0:90].astype(float)
X += .5
Y = (Y + .5) * 1.5


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


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


def smoothstep(lo, hi, x):
    a = np.clip((x-lo)/(hi-lo), 0, 1)
    return a*a*(3-2*a)


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


def environment(cam, t, underwater_floor, shadow, *, period=None):
    """Light from a moving surface attenuates toward the distant reef."""
    if underwater_floor:
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
    else:
        # Cool skylight filtered through open ocean. Large, quiet dark areas
        # separate the silver school from the high-frequency surface texture.
        u = X/90
        v = Y/108
        # A submerged observer sees a luminous, moving ceiling at a low
        # angle. Long waves silhouette the short capillary ripples on it.
        sea_x = u + .13*cam.focus[0]
        sea_y = (1.35/(.44+v)) + .05*cam.focus[1]
        swell = np.sin(8.5*sea_x+2.9*sea_y-.50*t)
        crinkle = np.sin(29*sea_y+7*sea_x+3.1*swell+.7*t)
        ripple = np.sin(62*sea_y-8.2*sea_x+3.4*crinkle-.83*t)
        proximity = np.exp(-4.7*v)
        grain = noise(9*sea_x+1.5*swell,26*sea_y+.15*t)
        ridges = (.51+.29*crinkle+.20*ripple)*(.36+.64*grain)
        # Apparent shafts diverge from one sun, attenuating in the water.
        beam_x = (u+.16)/(v+.38)
        intensity = (.50+.50*np.cos(9*beam_x+.8*np.sin(4*beam_x-.22*t)))**4
        glare = np.exp(-((u-.19+.05*swell)/.34)**2) * np.exp(-1.8*v)
        broad = .13+.65*np.exp(-2.45*v)+.26*glare
        sun = .26*intensity*np.exp(-1.95*v)*(.69+.31*grain)
        spec = ridges*proximity
        ink = np.array([9,27,44])+broad[...,None]*[42,90,126]+sun[...,None]*[129,156,151]+spec[...,None]*[175,170,134]
        paper = np.array([4,16,28])+broad[...,None]*[20,64,87]+sun[...,None]*[65,93,97]+spec[...,None]*[69,89,78]
        value = .06+.19*broad+.68*spec+.42*sun
        chars = select(glyphs("   ..,,;:~=+osO8%@"),value)
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
