"""Parallel sunlight through a wind-carried opening in overhead foliage."""
import cv2
import numpy as np

from day import PERIOD, RATE, base


class PassingLight:
    sun = base.unit(np.array([.03, .55, .98], np.float32))

    def __init__(self, garden):
        # Project the very same key used by the petal shadow maps.
        key = garden.key_direction(self.sun)
        ray = garden.project(key) - garden.project(np.zeros(3))
        self.slope = ray[0] / ray[1]
        y, x = np.mgrid[0:base.H:3, 0:base.W:3].astype(np.float32)
        self.across = x - self.slope*(y-320.)
        self.air = (.46+.54*np.exp(-((np.arange(base.H)[:,None]-280.)/235.)**2)).astype('f')

        # The canopy opening travels on the prevailing flow high above the
        # sheltered plant. Local gusts from the plant simulation perturb its speed.
        self.times = np.arange(PERIOD*RATE+1) / RATE
        gust = garden.airflow(self.times[:-1], np.array([-190., 100., 660.]))[:,0]
        speed = 51. + .10*gust
        self.distance = -300. + np.r_[0., np.cumsum(speed)/RATE]

    def aperture(self, across, depth, t):
        gap = np.interp(t % PERIOD, self.times, self.distance)
        offset = across - gap + .065*depth
        # Two irregular leaf edges soften the light distribution across the gap.
        # Its profile translates as a whole; all rays share the same direction.
        core = np.exp(-((offset+5.)/35.)**2)
        fringe = np.exp(-((offset-29.)/49.)**2)
        return .76*core+.33*fringe

    def canopy(self, garden, rgb, t):
        occupied = garden.z > -1e5
        # A nearby petal intercepts the shaft before the distant pond does.
        depth = np.where(occupied[::3,::3], garden.z[::3,::3], 0.)
        aperture = cv2.resize(self.aperture(self.across, depth, t), (base.W,base.H),
                              interpolation=cv2.INTER_LINEAR)
        aperture = np.minimum(aperture, 1.)
        color = rgb.astype(np.float32)
        petal = occupied & (color[...,0]>1.12*color[...,1]) & (color[...,2]>color[...,1])
        # Skylight keeps the flower legible before and after the pencil of sun.
        ambient = np.where(petal, 1.025, .86)
        lit = color*ambient[...,None]
        headroom = np.maximum(0., 1.-lit/255.)
        direct = color*np.array([.77,.69,.55], np.float32)
        haze = self.air[...,None]*np.array([92.,77.,53.], np.float32)
        lit += aperture[...,None]*headroom*np.where(occupied[...,None], direct, haze)
        return np.uint8(np.clip(lit,0,255)), aperture
