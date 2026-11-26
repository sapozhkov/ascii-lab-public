"""Dew rolls down the midrib and loads the flexible end of the petal."""


import numpy as np

from day import PERIOD, RATE, base
from departure import DepartureGarden

from petal_wind import frame_axes, rotation
from rib_dew import RibDew


WET_PETAL = 9


class MidribWater(RibDew):
    tip_compliance = .20

    def __init__(self, garden, adhesion=1., *, show_trail=False):
        self.tip_angles = garden.wet_tip_angles
        self.angle_speed = 0.
        super().__init__(garden, adhesion, show_trail=show_trail)

    @staticmethod
    def outer_uv(f):
        return .071 + .927*f, .5 + .28*(1.-base.smooth(f/.20))

    def _record_step(self, state):
        tick = len(self.records)
        super()._record_step(state)
        if tick+1 == len(self.tip_angles):
            return
        # A point load near the free end has more leverage on the flexible
        # outer section. On release the load disappears; angle and speed carry on.
        target = 0.
        if state.stage == 'petal':
            u, _ = self.outer_uv((state.channel_index-33)/51)
            leverage = max(0., (u-.52)/.48)
            target = self.tip_compliance * state.volume * leverage**2
        angle = self.tip_angles[tick]
        omega, damping = 12.4, .45
        self.angle_speed += (omega**2*(target-angle) - 2*damping*omega*self.angle_speed)/RATE
        self.tip_angles[tick+1] = angle + self.angle_speed/RATE


class MidribGarden(DepartureGarden):
    water_class = MidribWater
    wet_petal = WET_PETAL
    wet_bend_start = .48

    def __init__(self):
        self.wet_tip_angles = np.zeros(PERIOD*RATE+1)
        self.wet_tip_mode = base.smooth((base.U[:,0]-self.wet_bend_start)/(1.-self.wet_bend_start))
        super().__init__()

    def tissue(self, i, t):
        if i in self.cache and t == self.current:
            return self.cache[i]
        mesh, pigment, opening = super().tissue(i, t)
        angle = self.wet_tip_angles[round(t*RATE) % (PERIOD*RATE)]
        if i != WET_PETAL or abs(angle) < 1.e-8:
            return mesh, pigment, opening
        spine, tangent, across = frame_axes(mesh)
        # Positive load bends towards gravity. The base stays fixed and each
        # rib segment and cross-section keeps its original length.
        down = -np.sign(np.cross(across, tangent)[len(spine)*3//4,2])
        bend = rotation(across, down * angle * self.wet_tip_mode)
        segments = np.einsum('sij,sj->si', bend[1:], np.diff(spine, axis=0))
        rib = np.concatenate([spine[:1], spine[:1]+np.cumsum(segments, axis=0)])
        bent = rib[:,None]+np.einsum('sij,svj->svi', bend, mesh-spine[:,None])
        result = bent.astype(mesh.dtype), pigment, opening
        if t == self.current:
            self.cache[i] = result
        return result
