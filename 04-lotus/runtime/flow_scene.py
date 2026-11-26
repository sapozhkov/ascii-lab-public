"""The 04-C revision: a bee forages while runoff passes from the cup to the pond."""


from pathlib import Path
import subprocess
import tempfile

import numpy as np

from day import DayGarden, PERIOD, RATE, base
from center_water import CenterWater
from foraging import Forager
from visitor_scene import VisitGarden





class FlowGarden(VisitGarden):
    perch_compliance = 39.
    show_trail = True
    water_class = CenterWater
    forager_class = Forager

    def __init__(self, plant_only=False):
        super().__init__('catch', 2)
        # Retain the dew scene's whorl and wind. The old preset's timed water
        # load is replaced by the new model of the supporting surfaces.
        self.theme = 'flow'
        self.bearing.fill(0)
        # Place the receiving leaf under the wet lip, sloped back towards the
        # stalk. The collision solver sees the same mesh as the renderer.
        leaf = self.leaves[1]
        leaf.center = np.array([178., 282., -282.])
        leaf.height += .97 * leaf.x + .53 * leaf.y
        leaf.contact_xy = (10., 15.)
        self.rib_deflection = np.zeros((RATE*PERIOD, len(base.U)), dtype=np.float32)
        self.water = None
        if not plant_only:
            self.populate()

    def populate(self):
        self.bee = self.forager_class(self)
        self._load_perch()
        self.cache.clear()
        self.water = self.water_class(self, show_trail=self.show_trail)
        self.cache.clear()

    def _load_perch(self):
        # Damped bending of the dry petal under the insect's actual contact.
        offset, velocity = 0., 0.
        rib = base.U[:, 0]
        bend = np.zeros_like(rib)
        bend_velocity = np.zeros_like(rib)
        for i in range(len(self.bearing)):
            t = i / RATE
            contact = self.bee.contact(t)
            station = self.bee.uv[round(t*self.bee.rate), 0]
            # Cantilever under a point load: the outer third bends most on
            # touchdown. The moment drops as the insect walks towards the base.
            inside = np.minimum(rib, station)
            outside = np.maximum(rib, station)
            equilibrium = self.perch_compliance * contact * inside**2 * (3*outside-inside) / 2
            bend_velocity += (36*(equilibrium-bend) - 6*bend_velocity) / RATE
            bend += bend_velocity / RATE
            self.rib_deflection[i] = bend
            velocity += (36 * (contact - offset) - 6 * velocity) / RATE
            offset += velocity / RATE
            self.bearing[i, 0] = offset

    def tissue(self, i, t):
        if i in self.cache and self.current == t:
            return self.cache[i]
        points, pigment, opening = DayGarden.tissue(self, i, t)
        if i == 10:
            deflection = self.rib_deflection[round(t*RATE) % len(self.rib_deflection)]
            points[..., 2] -= deflection[:, None] * (.82 + .18 * abs(base.V))
        return points, pigment, opening

    def leaf_load(self, t):
        return self.water.leaf_load(t) if self.water is not None else 0.

    def stem(self, t):
        DayGarden.stem(self, t)
        self.leaves[0].draw(self, t, 0.)
        self.leaves[1].draw(self, t, self.leaf_load(t))

    def frame(self, t):
        rgb = self.illuminated_plant(t)
        self.water.draw(rgb, self, t)
        return self.bee.render(self, rgb, t)

    def ascii_frame(self, t):
        return self.compose_frame(base.ink(self.frame(t)), t)

    def compose_frame(self, scene, t):
        return scene


def export(garden, destination, start=0., end=PERIOD, fps=30):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as td:
        silent = Path(td) / 'silent.mp4'
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '810x1440', '-framerate', str(fps), '-i', '-', '-an', '-c:v',
               'libx264', '-threads', '4', '-preset', 'veryfast', '-crf', '16',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(silent)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
            for frame in range(round((end-start)*fps)):
                rgb = garden.ascii_frame(start+frame/fps)
                enc.stdin.write(rgb.tobytes())
        if enc.returncode:
            raise RuntimeError('ffmpeg could not encode the lotus')
        silent.replace(destination)
