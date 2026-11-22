"""Geometry and light for the bell and oral ribbons."""

import numpy as np

from . import renderer as j


COLS, ROWS = 96, 114
WIDTH, HEIGHT = COLS*j.CW, 2048
SPAN = j.STRAND_SPAN / j.STRAND_SPAN.mean() * (3.6 * j.DOME_RADIUS)
# A shallower bell with a luminous, wavy tide line circling its skirt.
# The rim radius stays the same, so strands still start on the bell.
BELL_HEIGHT_SCALE = .70
v, phi = j.DOME_V, j.DOME_PHI
tide = .77 + .07*np.sin(5*phi + .4) + .025*np.sin(2*phi)
band = np.exp(-((v-tide)/.075)**2)
grain = j.hash01(np.floor(v*35), np.floor(phi/j.TAU*112))
pearl = np.exp(-((v-.45-.09*np.sin(3*phi))/.12)**2) * j.ease((grain-.68)/.24)
SHELL_COLOR = (j.rgb(44, 174, 233) + (.22*(1-v) + .14*grain)[:, None]*j.rgb(96, 138, 125))
SHELL_COLOR = j.tint(SHELL_COLOR, (.88*band+.52*pearl)[:, None])
SHELL_WEIGHT = .30 + .22*(1-v) + .17*grain + .52*band + .22*pearl


def shallow_pose(move):
    def place(body, when):
        return move(body * [1, BELL_HEIGHT_SCALE, 1], when)
    return place


def shell_parts(t, track, move, light_direction=None):
    place = shallow_pose(move)
    spot = j.dome(v, phi, t, place)
    # Normals transform by the inverse transpose of the vertical compression.
    normal = j.normalised(j.around(j.FACING_OUT, phi+j.TURN*t, j.FACING_UP/BELL_HEIGHT_SCALE))
    normal = track.rotate(normal, t)
    front = j.spin_x(normal, j.PITCH)[:, 2] < .03
    if light_direction is None:
        light = np.clip(.69 + .52*(normal*j.LIGHT).sum(axis=1), .35, 1)
    else:
        incidence = (normal*light_direction).sum(axis=1)
        # Broad penumbra on the shoulder, with a small bright facing patch.
        light = .31 + 1.38*j.ease((incidence+.14)/.98) + .28*np.maximum(incidence,0)**8
    beat = (t-j.FIRST_BEAT) % j.BEAT
    heat = np.exp(-((beat*j.FRONT_SPEED-.75*v)/.18)**2)
    ink = j.tint(SHELL_COLOR, (.40*heat)[:, None]) * light[:, None]
    weight = np.clip(SHELL_WEIGHT*light+.20*heat, 0, 1)
    glyph = j.RAMP_G[((1-weight)*(len(j.RAMP_G)-1)).astype(int)]
    inner = (.55+.23*light[~front, None])*j.rgb(86, 144, 203)
    hem = j.dome(np.ones_like(j.HEM_PHI), j.HEM_PHI, t, place)
    x, y, z = j.on_screen(hem, t)
    near = z < np.median(z)
    stroke = j.line_glyph(np.roll(x,-1)-x, np.roll(y,-1)-y)
    hem_ink = np.broadcast_to(j.rgb(198,255,243),(near.sum(),3))
    if light_direction is not None:
        hem_normal = track.rotate(j.around(np.ones_like(j.HEM_PHI),j.HEM_PHI+j.TURN*t,0.),t)
        hem_ink = hem_ink*(.32+.83*j.ease(((hem_normal[near]*light_direction).sum(axis=1)+.4)/1.15))[:,None]
        inner = (.17+.41*light[~front,None])*j.rgb(86,144,203)
    return [(spot[front],glyph[front],ink[front],j.SHELL),
            (spot[~front],j.DOME_GIN[~front],inner,j.INSIDE),
            (hem[near],stroke[near],hem_ink,j.HEM)]


def blue_wave(t):
    def dye(route, inks, glow):
        front = ((t - j.FIRST_BEAT) % j.BEAT) * j.FRONT_SPEED
        heat = np.exp(-((front-route)/.18)**2)
        base = .65*inks[0] + .35*inks[1]
        if inks is j.LIMB_INK:
            margin = j.ease((np.abs(j.LIMB_SIDE)-.55)/.4)
            base = j.rgb(86,158,207) + margin[:, None]*j.rgb(110,79,41)
        return j.tint(base, (.55*heat)[:, None]) * (glow+.22*heat)[:, None], heat
    return dye
