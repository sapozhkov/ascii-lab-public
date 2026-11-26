"""Lotus in camera space. Coordinates refer to a 960 × 540 view at 6.70 s."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Petal:
    root: tuple
    tip: tuple
    left: tuple
    right: tuple
    travel: tuple
    tone: int
    swell: float = .05
    lag: float = 0.


# Each pair of handles shapes one margin of a petal from its root to its tip.
# Order is the visible overlap in the clean frame. Hidden margins are inferred.
BACK = [
    # A small lobe emerges between the two left-hand margins in the last frames.
    Petal((478,346), (294,284), ((393,257),(317,256)), ((425,373),(334,325)), (-156,21), 1),
    Petal((485,343), (496,115), ((404,305),(398,135)), ((572,256),(577,129)), (18,9), 1),
    Petal((485,346), (600,54), ((467,194),(482,86)), ((627,249),(642,99)), (61,10), 0),
    Petal((493,346), (737,146), ((581,236),(598,99)), ((629,314),(707,243)), (129,89), 0),
    Petal((490,353), (784,364), ((625,279),(728,292)), ((644,472),(739,383)), (30,49), 3),
    Petal((486,360), (287,386), ((404,383),(339,369)), ((390,510),(325,422)), (-22,96), 3),
    Petal((479,348), (350,92), ((374,308),(275,187)), ((523,244),(469,138)), (-6,-2), 1),
    Petal((477,357), (173,312), ((331,242),(239,244)), ((315,422),(225,345)), (-39,81), 0),
    Petal((474,355), (253,191), ((378,242),(353,158)), ((385,368),(249,253)), (-108,51), 1),
    Petal((492,348), (647,496), ((513,451),(574,491)), ((630,343),(713,438)), (50,80), 2),
    Petal((507,356), (351,414), ((434,342),(293,323)), ((472,437),(388,458)), (-22,112), 0),
]

# Displacements of the two distal Bézier handles in the open pose (image pixels).
# Estimated from the visible outer contour at 8.00 s. The Bernstein weight is
# zero at both ends, so these terms never move an annotated tip.
OPEN_HANDLES = {
    (294,284): ((-40,-40),(-64,54)), (496,115): ((26,68),(-76,2)),
    (600,54): ((34,0),(124,22)), (737,146): ((54,0),(24,40)),
    (784,364): ((-8,0),(32,54)), (287,386): ((58,-90),(44,80)),
    (350,92): ((-48,-42),(58,0)), (173,312): ((-48,32),(8,54)),
    (253,191): ((-14,-26),(-40,0)), (647,496): ((-94,12),(-30,-48)),
    (351,414): ((-22,22),(102,40)),
}

FRONT = [
    Petal((497,350), (536,166), ((563,306),(659,151)), ((667,292),(606,89)), (97,86), 0, .22,.14),
    Petal((480,359), (472,210), ((417,358),(400,270)), ((537,326),(533,263)), (-63,94), 0, .45,.02),
    Petal((506,358), (596,221), ((438,254),(574,256)), ((592,379),(624,269)), (51,123), 1, .38,.10),
]

# Absolute seconds and digitised silhouette landmarks. Frame 402 anchors the fit.
TIMES = np.array([6.70, 7.30, 7.60, 7.80, 8.00])
TRACKS = {
    (737,146): [(737,146),(774,167),(832,190),(851,220),(866,235)],
    (253,191): [(253,191),(216,204),(166,223),(151,236),(145,242)],
    (173,312): [(173,312),(162,330),(139,355),(134,377),(134,393)],
    (350,92):  [(350,92),(344,94),(341,82),(348,88),(344,90)],
    (600,54):  [(600,54),(613,55),(634,54),(650,60),(661,64)],
}


def response(t, midpoint, rate):
    """Solution of dq/dt = rate*q*(1-q), relative to the two measured poses."""
    def q(time):
        return 1 / (1 + np.exp(-rate * (time - midpoint)))
    return (q(t) - q(6.7)) / (q(8.) - q(6.7))


def fit_modes():
    # Jointly fit outward travel and downward travel. Weight by actual pixel error.
    modes = []
    for axis in (0, 1):
        best_error = np.inf
        for mid in np.arange(7.30, 7.76, .01):
            for rate in np.arange(3., 9., .2):
                q = response(TIMES, mid, rate)
                err = 0.
                for points in TRACKS.values():
                    obs = np.array(points)[:, axis]
                    err += np.sum((obs[0] + (obs[-1]-obs[0])*q - obs)**2)
                if err < best_error:
                    best_error, best = err, (mid, rate)
        modes.append(best)
    return modes


MODES = fit_modes()


def bezier(start, h1, h2, end, u):
    w = 1 - u
    return w**3*start + 3*w*w*u*h1 + 3*w*u*u*h2 + u**3*end


class Surface:
    def __init__(self, petal, index):
        self.petal = petal
        u = np.linspace(0, 1, 211, dtype='f')[:, None]
        v = np.linspace(-1, 1, 135, dtype='f')[None, :]
        self.u, self.v = u, v
        a, b = np.array(petal.root), np.array(petal.tip)
        lo = bezier(a, *map(np.array, petal.left), b, u)
        hi = bezier(a, *map(np.array, petal.right), b, u)
        mid, half = (hi+lo)/2, (hi-lo)/2
        # Longitudinal ribs follow the curved sheet. Cup offsets vanish at the margins.
        cup = np.sin(np.pi*u)*(1-v*v)
        self.tangent = np.gradient(mid, u[:,0], axis=0)
        self.x = (mid[:,0:1] + half[:,0:1]*v + 7*cup).astype('f')
        self.y = (mid[:,1:2] + half[:,1:2]*v - 10*cup).astype('f')
        self.side_x = (half[:,0:1]*v).astype('f')
        self.side_y = (half[:,1:2]*v).astype('f')
        left,right = OPEN_HANDLES.get(petal.tip, ((0,0),(0,0)))
        bend = 3*u*u*(1-u)
        self.unroll = [bend*((1-v)*l+(1+v)*r)/2 for l,r in zip(left,right)]
        self.lever = u**1.55
        self.color = self.pigment(index)

    def pigment(self, index, palette=None, contrast=1.0):
        u, v = self.u, self.v
        # Pigment and veins live on material coordinates, so they follow the fold.
        phase = index * 1.74
        flow = v + .095*np.sin(3.8*u + phase)*np.sin(np.pi*u)*(1-v*v)
        groove = np.cos(np.pi*(17*flow + .8*np.sin(4*flow + 2*u + phase)))
        silk = np.cos(np.pi*(59*flow + 1.8*u + .9*np.sin(7*flow)))
        fold = .5+.5*np.cos(4.5*v + 1.2*u + phase)
        # A side-lit ridge and a broad hollow give each sheet its own relief.
        ridge = np.exp(-((flow-.33*np.sin(phase))/.34)**2)
        glow = np.exp(-((u-.26)/.27)**2)
        light = (.29 + .34*fold + .19*ridge + .17*glow + .08*(1-v*v)
                 + .052*groove + .024*silk + .20*np.exp(-((np.abs(v)-.95)/.07)**2))
        light -= .15*np.exp(-((u-.94)/.14)**2)
        light = np.clip(.54 + contrast*(light-.54), 0, 1)[...,None]
        tones = [
            ((139,24,69),(255,221,214)), ((118,16,49),(255,187,191)),
            ((143,43,61),(253,174,158)), ((91,31,38),(212,111,95)),
        ] if palette is None else palette
        shadow, ivory = tones[self.petal.tone]
        rgb = np.array(shadow) + (np.array(ivory)-shadow)*light**1.15
        gold = (np.exp(-(u/.26)**2) * (.40 if self.petal.tone != 3 else .15))[...,None]
        return np.uint8(np.clip(rgb*(1-gold)+np.array([255,182,78])*gold, 0, 255))

    def points(self, t):
        p = self.petal
        spread, fall = [response(t, mid+p.lag, rate) for mid,rate in MODES]
        widen = p.swell*spread
        dx,dy = p.travel[0]*spread, p.travel[1]*fall
        tx,ty = self.tangent[:,0:1],self.tangent[:,1:2]
        derivative = 1.55*self.u**.55
        # Let the cross-section turn with the bending midrib in projection.
        turn = .78*np.arctan2(derivative*(tx*dy-ty*dx),
                               tx*tx+ty*ty+derivative*(tx*dx+ty*dy))
        c,s = np.cos(turn)*(1+widen),np.sin(turn)*(1+widen)
        return (self.x + self.lever*dx + (c-1)*self.side_x - s*self.side_y + spread*self.unroll[0],
                self.y + self.lever*dy + s*self.side_x + (c-1)*self.side_y + spread*self.unroll[1])


SURFACES = [Surface(p, i-1) for i,p in enumerate(BACK+FRONT)]


def camera(t):
    # Smooth approach in the portrait crop, complete by the clean reference pose.
    u = np.clip((t-6.0)/.70, 0., 1.)
    return .72+.28*u*u*(3-2*u)
