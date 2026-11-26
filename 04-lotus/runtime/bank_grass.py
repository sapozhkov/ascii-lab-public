"""A few fixed-root grass blades at the two sides of the lotus scene."""
import numpy as np

from day import base
from wind_plant import ElasticRod


# Root, free length, resting lean, half-width, fold heading, stiffness, drag,
# damping, and pigments for the two edges and the fold of each blade.
BLADES = (
    ((-475., 35., -714.), 414., (-20., 18.), 12.0, 19., .71, .18, 3.0, (87., 113., 55.)),
    ((-434., 78., -728.), 350., (50., 12.), 10.7, 36., .63, .18, 3.6, (72., 103., 57.)),
    ((-359., 116., -735.), 292., (44., -16.), 9.0, 26., .57, .10, 3.9, (76., 106., 63.)),
    ((350., 135., -734.), 353., (-50., 27.), 10.0, -26., .76, .16, 3.3, (74., 100., 53.)),
    ((412., 129., -765.), 409., (35., -26.), 11.8, -11., .65, .16, 3.5, (90., 113., 64.)),
)


class BankGrass:
    def __init__(self, garden):
        station = np.linspace(0., 1., 27)
        self.blades = []
        for root, length, lean, half_width, heading, stiffness, drag, damping, pigment in BLADES:
            # Resting curvature follows growth. The same wind field as the
            # lotus bends each blade about its own fixed root.
            horizontal = station[:, None]**1.45 * lean
            rest = np.asarray(root) + np.column_stack((horizontal, length * station))
            rod = ElasticRod(rest, garden.airflow, stiffness=stiffness, tip_mass=.055,
                             tip_drag=drag, damping=damping)
            angle = np.deg2rad(heading)
            lateral = np.array([np.cos(angle), np.sin(angle), 0.])
            width = half_width * np.sin(np.pi * (.16 + .84*station))**.8
            width[-1] = 0.
            color = np.asarray(pigment) * np.array([.66, .94, .74])[:, None]
            self.blades.append((rod, lateral, width, color))

    def axes(self, t):
        return [rod.axis(t) for rod, _, _, _ in self.blades]

    def draw(self, garden, t):
        for rod, lateral, width, color in self.blades:
            axis = rod.axis(t)
            tangent = base.unit(np.gradient(axis, axis=0))
            across = base.unit(lateral - (tangent @ lateral)[:, None] * tangent)
            fold = np.cross(tangent, across) * (width*.13)[:, None]
            edge = across * width[:, None]
            ribbon = np.stack((axis - edge, axis + fold, axis + edge), axis=1)
            garden.mesh(ribbon, color)
