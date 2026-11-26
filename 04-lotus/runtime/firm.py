"""A sheltered lotus. Wind bends the lower stalk. The crown rides it as one piece."""

import numpy as np

from day import PERIOD, RATE, base, wind
from flow_scene import FlowGarden




def matched_rest_shapes(bud, opened, index):
    """Give each rib the same material stations and attachment in both poses."""
    middle = bud.shape[1] // 2
    # Include the inner petals' characteristic swept tips in their rest shape.
    expanded = base.bend(bud, opened, np.ones_like(base.U), index, 'cup', 0)
    rib = bud[:, middle]
    open_rib = expanded[:, middle]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(rib, axis=0), axis=1))]
    lengths = np.linalg.norm(np.diff(open_rib, axis=0), axis=1)
    stations = np.r_[0., np.cumsum(lengths)]
    row = np.interp(stations / stations[-1], arc / arc[-1], np.arange(len(rib)))
    lower = np.minimum(row.astype(int), len(rib) - 2)
    fraction = (row - lower)[:, None, None]
    wrapped = (1 - fraction) * bud[lower] + fraction * bud[lower + 1]
    # Fold a rib of fixed length into the closed pose. Preserve the rolled
    # cross-sections, so opening the flower still exposes its broad surfaces.
    folded_rib = wrapped[:, middle]
    tangents = base.unit(np.diff(folded_rib, axis=0))
    spine = open_rib[0] + np.vstack([np.zeros(3), np.cumsum(tangents * lengths[:, None], axis=0)])
    return (wrapped + (spine - folded_rib)[:, None]).astype('f'), expanded


class FirmGarden(FlowGarden):
    show_trail = False
    perch_compliance = 8.

    @staticmethod
    def stem_weight(z):
        # Clamped below the swaying stalk. The whole crown, including the
        # lowest hanging petal, is above the collar and shares one translation.
        h = np.clip((z + 815.) / (815. - 160.), 0., 1.)
        return h*h*(3. - 2.*h)

    attachment_weight = stem_weight

    def __init__(self, plant_only=False):
        super().__init__(plant_only=True)
        self.sheets = [matched_rest_shapes(b, o, i) for i, (b, o) in enumerate(self.sheets)]
        self.scene = 'firm'  # Swept tips are already included in the open rest shapes.
        self.cycle.flex.fill(0.)
        self.cycle.stem[:] = self.stalk_response()
        if not plant_only:
            self.populate()

    def stalk_response(self):
        """Steady periodic response of the stiff supporting stalk to wind load."""
        count = RATE * PERIOD
        times = np.arange(count) / RATE
        area = .58 + .42 * self.cycle.open.mean(axis=(1, 2))
        equilibrium = np.column_stack([7.5 * wind(times, 'dew'), 3. * wind(times - .3, 'dew')])
        equilibrium *= area[:, None]
        position = np.zeros(2)
        velocity = np.zeros(2)
        history = np.empty_like(equilibrium)
        for tick in range(2 * count):
            i = tick % count
            velocity += (9. * (equilibrium[i] - position) - 6. * velocity) / RATE
            position += velocity / RATE
            history[i] = position
        return history
