"""Drain the cup along its near flank, directly into the wet outer rib."""
from dataclasses import replace

import numpy as np

from day import base
from dew import DewWater


class RibDew(DewWater):
    # A narrow pool against the margin feeds the adjacent outer rib. The
    # first part follows the inner vein. The two margins meet near its foot.
    pool_uv = (.435, .065)
    pool_span = (.16, .09)

    @staticmethod
    def inner_uv(f):
        return .435*(1-f) + .071*f, .030*(1-f) + .74*f**3.7

    def _support(self, garden, tick, state, previous_position):
        local_support = super()._support
        normal, tangent, remaining = local_support(garden, tick, state, previous_position)
        if state.stage not in ('cup', 'petal'):
            return normal, tangent, remaining
        # As the lens bridges both sheets its wetted base samples both tangent
        # planes. A single mesh vertex would turn the entire front at the seam.
        groove = self.channels[tick]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(groove, axis=0), axis=1))]
        center = np.interp(state.channel_index, np.arange(len(arc)), arc)
        reach = 2.5*self.radius_px*state.volume**(1/3)/garden.scale
        if abs(center-arc[33]) >= reach:
            return normal, tangent, remaining
        stations = np.interp([center-reach,center+reach], arc, np.arange(len(arc)))
        tail, nose = [self._at(groove, k) for k in stations]
        contact_normals = [local_support(garden, tick, replace(state, channel_index=k),
                                         previous_position)[0] for k in stations]
        weight = base.smooth(.5+.5*(center-arc[33])/reach)
        tangent = base.unit(nose-tail)
        normal = (1-weight)*contact_normals[0]+weight*contact_normals[1]
        normal = base.unit(normal-tangent*np.dot(normal,tangent))
        return normal, tangent, remaining
