"""Coupled wings with a short camera exposure. Distances use the bee body frame."""
import cv2
import numpy as np


class Wingbeat:
    exposure = 1 / 560
    span = np.linspace(0., 1., 19)
    chord = np.sin(np.pi * span)**.70 * (.72 + .46 * span)

    def __init__(self, bee):
        dt = 1 / bee.rate
        self.times = np.arange(len(bee.body)) * dt
        velocity = np.gradient(bee.body, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        turn = np.gradient(np.unwrap(bee.attitude[:, 0]), dt)
        effort = np.clip(np.linalg.norm(acceleration, axis=1) / 360., 0., 1.)
        bank = np.tanh(turn / 2.8)
        # Airborne lift transfers to the legs as they take the weight. The
        # flight muscles stop, then the membrane folds back over the abdomen.
        airborne = 1. - bee.attitude[:, 2]
        spread = np.empty_like(airborne)
        spread[0] = airborne[0]
        for i in range(1, len(spread)):
            spread[i] = spread[i-1] + (airborne[i]-spread[i-1]) * (1-np.exp(-dt/.08))
        self.spread = spread
        self.effort = effort
        self.bank = bank
        self.frequency = (183. + 3.8 * effort) * airborne
        self.phase = 2 * np.pi * np.r_[0., np.cumsum((self.frequency[:-1]+self.frequency[1:])*.5*dt)]

    def draw(self, g, rgb, world, t):
        spread, effort, bank, phase, frequency = (
            float(np.interp(t, self.times, field)) for field in
            (self.spread, self.effort, self.bank, self.phase, self.frequency)
        )
        # Average actual poses over the shutter interval. Coincident poses at
        # a stroke reversal concentrate the light on one edge of the membrane.
        half_shutter = np.pi * frequency * self.exposure
        samples = np.linspace(phase-half_shutter, phase+half_shutter, 5) if spread>.06 else [phase]
        for side in (-1, 1):
            for hind in (True, False):
                outlines = []
                for sample in samples:
                    contour = self.contour(side, hind, spread, effort, bank, sample)
                    outlines.append(g.project(world(contour)))
                self.expose(rgb, g.z, np.stack(outlines), hind)

    def contour(self, side, hind, spread, effort, bank, phase):
        # A coupled hindwing stays slightly aft of the forewing. Delayed tip
        # rotation gives a curved leading edge when the thorax reverses stroke.
        s = self.span
        stroke = np.sin(phase-.22*s)
        amplitude = 48. + 8.*effort - side*7.*bank
        az = np.deg2rad(-80. + spread*(78.+amplitude*stroke) - 11.*hind)
        el = np.deg2rad(8. + spread*(16.-29.*stroke + 6.*side*bank))
        direction = np.column_stack([np.sin(az)*np.cos(el), side*np.cos(az)*np.cos(el), np.sin(el)])
        root = np.array([4.-6.*hind, side*10., 12.])
        spine = root + direction * ((48.-15.*hind)*s[:, None])
        spine[:, 2] += (2. + spread*2.) * np.sin(np.pi*s)
        chord_pitch = np.deg2rad(8.+37.*spread*np.cos(phase-.25*s))
        lateral = np.column_stack([np.cos(az)*np.cos(chord_pitch),
                                   -side*np.sin(az)*np.cos(chord_pitch), np.sin(chord_pitch)])
        width = self.chord[:, None] * (13.-4.*hind) * lateral
        return np.vstack([spine+.50*width, (spine-.93*width)[::-1]])

    def expose(self, rgb, depth, outlines, hind):
        xy = outlines[..., :2]
        low = np.maximum(np.floor(xy.min(axis=(0,1))-2).astype(int), 0)
        high = np.minimum(np.ceil(xy.max(axis=(0,1))+2).astype(int), rgb.shape[1::-1])
        x0,y0 = low
        x1,y1 = high
        if x0>=x1 or y0>=y1:
            return
        background = rgb[y0:y1,x0:x1]
        coverage = np.zeros(background.shape[:2], np.float32)
        yy,xx = np.indices(coverage.shape)
        nearest = depth[y0:y1,x0:x1]
        opacity = 144 if hind else 171
        for outline in outlines:
            uv = outline[:,:2]-low
            mask = np.zeros_like(coverage,dtype=np.uint8)
            poly = np.rint(uv*16).astype(np.int32)
            cv2.fillPoly(mask,[poly],opacity,cv2.LINE_AA,shift=4)
            leading = poly[:len(self.span)]
            cv2.polylines(mask,[leading],False,234,2,cv2.LINE_AA,shift=4)
            # Depth test the membrane against body, legs and any foreground petal.
            plane = np.linalg.lstsq(np.c_[uv,np.ones(len(uv))], outline[:,2], rcond=None)[0]
            coverage += mask * (xx*plane[0]+yy*plane[1]+plane[2]+1.1 >= nearest)
        w = (coverage/(255.*len(outlines)))[...,None]
        sky = np.array([177.,195.,194.]) if hind else np.array([212.,223.,215.])
        background[:] = np.uint8(background*(1.-w)+sky*w)
