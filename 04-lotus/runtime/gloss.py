"""Specular response to the moving opening in the canopy."""
import numpy as np

from bank import BankGarden
from day import base

from pliant import PliantWater




class GlossWater(PliantWater):
    sun_glint_gain = 390.


class GlossGarden(BankGarden):
    water_class = GlossWater
    drawing_foliage = False

    def __init__(self):
        super().__init__()
        self.eye = np.array([np.sin(self.az)*np.cos(self.elev),
                             np.cos(self.az)*np.cos(self.elev), np.sin(self.elev)])
        self.key = self.key_direction(self.light_window.sun)
        self.half_vector = base.unit(self.key + self.eye)

    def reflected(self, points, normal, pigment, t, visibility, strength):
        x, y, depth = np.moveaxis(self.project(points), -1, 0)
        across = x - self.light_window.slope*(y-320.)
        beam = self.light_window.aperture(across, depth, t)
        facing = np.where((normal @ self.eye)[...,None] < 0., -normal, normal)
        incidence = np.maximum(facing @ self.key, 0.)
        alignment = np.clip(facing @ self.half_vector, 0., 1.)
        # A narrow wax highlight sits on a broader satin lobe. Roll the
        # exposure off near white so the ASCII texture retains its gradation.
        specular = .68*alignment**24 + .32*alignment**7
        response = strength * beam * visibility * np.sqrt(incidence) * specular
        reflected_sun = np.array([255., 244., 226.])
        return pigment + np.maximum(reflected_sun-pigment, 0.) * response[...,None]

    def petal_reflection(self, points, normal, pigment, t, visibility):
        return self.reflected(points, normal, pigment, t, visibility, 1.10)

    def bee_reflection(self, points, normal, pigment, t):
        return self.reflected(points, normal, pigment, t, 1., .38)

    def stem(self, t):
        self.drawing_foliage = True
        try:
            super().stem(t)
        finally:
            self.drawing_foliage = False

    def mesh(self, points, color):
        if self.drawing_foliage:
            color = self.reflected(points, base.normals(points), color, self.current, 1., .86)
        super().mesh(points, color)
