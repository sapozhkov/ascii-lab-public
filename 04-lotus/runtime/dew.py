"""Small dew lens riding the same draining parcel as the approved lotus."""
from dataclasses import replace

import cv2
import numpy as np

from capillary import CapillaryWater
from center_water import _project
from day import RATE, base
from lotus_weather import ripples


class DewWater(CapillaryWater):
    radius_px = 7.1
    sun_glint_gain = 280.
    leaf_sky_fill = 0.
    ripple_gain = 1.
    far_ripple_reflection = 0.

    def __init__(self, garden, adhesion=1., *, show_trail=False):
        super().__init__(garden, adhesion, show_trail=show_trail)
        positions = np.array([state.position for state in self.records])
        velocity = np.gradient(positions, 1 / RATE, axis=0)
        acceleration = np.gradient(velocity, 1 / RATE, axis=0)
        # The pinned foot follows the tissue. Inertia makes the crown lag behind
        # both when the channel turns and when the receiving leaf arrests it.
        lag = np.zeros(3)
        lag_speed = np.zeros(3)
        self.crown_lag = np.zeros_like(positions)
        for i, mode in enumerate(self.modes):
            normal = mode[4:7]
            lateral = acceleration[i] - normal * np.dot(acceleration[i], normal)
            force = -mode[3] * lateral / (860 + np.linalg.norm(lateral))
            lag_speed += (240 * (force - lag) - 17 * lag_speed) / RATE
            lag += lag_speed / RATE
            self.crown_lag[i] = lag

    def shape(self, t):
        mode = super().shape(t)
        # A smaller drop wets a shorter footprint. Surface tension rounds the
        # resting crown; the inertial trough at impact can still compress it.
        mode.strain += .13 * mode.contact
        mode.slip *= .83
        return mode

    def surface(self, garden, t):
        surface = super().surface(garden, t)
        mode = self.shape(t)
        frame = np.clip(t * RATE, 0, len(self.records) - 1)
        k = int(frame)
        lag = self.crown_lag[k] + (frame - k) * (self.crown_lag[min(k + 1, len(self.records) - 1)] - self.crown_lag[k])
        lateral = lag - mode.normal * np.dot(lag, mode.normal)
        height = (surface - surface[0, 0]) @ mode.normal
        # Every horizontal section translates without changing its area or volume.
        return surface + .52 * height[..., None] * lateral

    def draw_pond(self, rgb, garden, t, size_ratio):
        age = t - self.events['splash']
        if not 0. <= age < 5.4:
            return
        wet = rgb.copy()
        ripples(wet, *self.screen(garden, t), age, self.far_ripple_reflection)
        # The smaller impact excites lower-amplitude waves. Fine crests
        # dissipate faster while their propagation stays continuous.
        amplitude = self.ripple_gain * size_ratio * np.exp(-.18*age)
        cv2.addWeighted(wet, amplitude, rgb, 1 - amplitude, 0, dst=rgb)

    def draw(self, rgb, garden, t):
        state = self.sample(t)
        size_ratio = self.radius_px / CapillaryWater.radius_px
        self._meniscus(rgb, garden, t, replace(state, reservoir=state.reservoir * size_ratio**3))
        if state.stage == 'pond':
            self.draw_pond(rgb, garden, t, size_ratio)
            return
        surface = self.surface(garden, t)
        points = _project(garden, surface)
        low = np.maximum([0, 0], np.floor(points[..., :2].min(axis=(0, 1)) - 2).astype(int))
        high = np.minimum(rgb.shape[1::-1], np.ceil(points[..., :2].max(axis=(0, 1)) + 2).astype(int))
        x0, y0 = low
        x1, y1 = high
        if x0 >= x1 or y0 >= y1:
            return

        ring = surface[:, :-1]
        tangent = np.roll(ring, -1, axis=1) - np.roll(ring, 1, axis=1)
        tangent = np.concatenate((tangent, tangent[:, :1]), axis=1)
        normals = base.unit(np.cross(tangent, np.gradient(surface, axis=0)))
        shape = self.shape(t)
        normals[0] = -shape.normal
        normals[-1] = normals[-2]
        view = _project(garden, normals) - _project(garden, np.zeros(3))
        nx, ny = view[..., 0] / garden.scale, view[..., 1] / garden.scale
        nz = np.clip(view[..., 2], 0., 1.)
        # Refract the actual lit petal, leaf or background. Use Snell's law for
        # the incoming ray and the local thickness of the transparent lens.
        eta = 1 / 1.333
        refracted_cos = np.sqrt(1 - eta**2 * (1 - nz**2))
        bend = refracted_cos - eta * nz
        radius = self.radius_px * state.volume**(1/3)
        distance = radius * (1.3 + .85 * nz) * bend / (eta + bend * nz)
        sx = (points[..., 0] - distance * nx).astype('f')
        sy = (points[..., 1] - distance * ny).astype('f')
        tissue = cv2.remap(rgb, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(float)

        x, y = np.rint(self.screen(garden, t)).astype(int)
        aperture = garden.lightfield[np.clip(y, 0, rgb.shape[0]-1), np.clip(x, 0, rgb.shape[1]-1)]
        # A patch of cool sky is reflected in only one side of the lens. The
        # sun keeps its direction as the moving canopy exposes or obscures it.
        rx, ry = 2 * nz * nx, 2 * nz * ny
        sky = np.clip(.38 - .66 * ry + .18 * rx, 0., 1.)
        environment = np.array([24., 31., 27.]) + sky[..., None] * [143., 165., 181.]
        fresnel = .0204 + .9796 * (1 - nz)**5
        color = tissue * (.85 + .15*nz[..., None]) * (1 - fresnel[..., None]) + environment * fresnel[..., None]
        reflected_sky = np.exp(-((rx + .34)/.64)**2 - ((ry + .72)/.31)**2)
        sunlight = np.exp(-((nx + .25)/.28)**2 - ((ny + .46)/.19)**2)
        color += reflected_sky[..., None] * [100., 114., 122.]
        if self.leaf_sky_fill and state.stage in ('leaf', 'air2'):
            # A broad patch of sky reflects along the wetted rim. Its visible
            # length scales with the square root of the shrinking contact area.
            sky_fill = np.exp(-((rx - .16)/.68)**2 - ((ry + .06)/.50)**2)
            strength = self.leaf_sky_fill * np.sqrt(shape.contact)
            color += (strength * sky_fill)[..., None] * [.86, 1., 1.08]
        color += sunlight[..., None] * (124 + self.sun_glint_gain * aperture) * [1., .96, .84]

        # Supersample the depth-tested mesh before feeding it to the ASCII grid.
        points[..., :2] = 3 * (points[..., :2] - low)
        vertex = np.ascontiguousarray(np.concatenate((points, np.clip(color, 0, 255)), axis=-1), dtype='f')
        size = (3*(x1-x0), 3*(y1-y0))
        depths = cv2.resize(garden.z[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST) - .75
        patch = rgb[y0:y1, x0:x1]
        raster = cv2.resize(patch, size, interpolation=cv2.INTER_NEAREST)
        base.GRID(vertex, *surface.shape[:2], *size, np.ascontiguousarray(depths, dtype='f'), raster)
        patch[:] = cv2.resize(raster, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
