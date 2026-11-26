"""Solve the bead's contact with the broad leaf over a finite footprint."""
import numpy as np

from broad_leaves import BroadLeafGarden
from center_water import _leaf_face
from edge_ripples import EdgeRipplesWater

# A regular hexagon has orthogonal axes with squared norm 3*r².
def fit_plane(offsets, heights, r):
    level = heights.mean()
    delta = heights - heights[0]
    moment = offsets.T @ delta
    slope = moment / (3. * r**2)
    return level, slope


class LeafFlowWater(EdgeRipplesWater):
    leaf_sky_fill = 130.
    footprint = np.array([(0., 0.), (1., 0.), (.5, .8660254), (-.5, .8660254),
                          (-1., 0.), (-.5, -.8660254), (.5, -.8660254)])

    def _leaf_contact(self, garden, mesh, xy, volume):
        center = _leaf_face(mesh, xy)
        if center is None:
            return None
        radius = .70 * self.radius_px * volume**(1/3) / garden.scale
        offsets = radius * self.footprint
        heights = np.empty(len(offsets))
        heights[0] = center[0]
        for i, offset in enumerate(offsets[1:], 1):
            face = _leaf_face(mesh, xy + offset)
            # Extend the local surface across a partial footprint at the rim.
            heights[i] = (center[0] + center[1] @ offset) if face is None else face[0]
        # The same contact plane drives both rolling and the meniscus.
        return fit_plane(offsets, heights, radius)


class LeafFlowGarden(BroadLeafGarden):
    water_class = LeafFlowWater
