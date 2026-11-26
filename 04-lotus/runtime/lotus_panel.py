"""The classic 67.5/32.5 scene and source layout, composed for the lotus."""
from bisect import bisect_right
import inspect
from pathlib import Path

import numpy as np

from day import PERIOD, base
from lotus_weather import ripples
from source_panel import source_panel









def source_section(function, opening=None):
    path = Path(inspect.getsourcefile(function))
    lines = inspect.getsourcelines(function)[0]
    marker = next(line.strip() for line in lines
                  if line.lstrip().startswith(opening or 'def '))
    return path, marker


class LotusPanel:
    width, height = base.SIZE
    scene_height = 2 * round(height * (72 * 18 / 1920) / 2)
    # Keep the native glyph grid. The top margin in the C master gives room
    # to move both the crown and the pond contact up without changing scale.
    crop_top = 324

    def __init__(self, garden):
        bee, water = garden.bee.events, garden.water.events
        opening = source_section(base.bend)
        deformation = garden.water._shape_dynamics
        cuts = (
            (0., opening),
            (bee['approach'], source_section(garden.light_window.canopy)),
            (bee['descend'], source_section(garden.bee.wings.contour)),
            (bee['probe1'], source_section(deformation, 'footprint =')),
            (water['hit'], source_section(deformation, 'rate -= min(')),
            (water['splash'], source_section(ripples)),
            (bee['depart'], opening),
        )
        panes = {section: np.asarray(source_panel(section, self.width, self.height - self.scene_height))
                 for _, section in cuts}
        self.starts = [time for time, _ in cuts]
        self.panes = [panes[section] for _, section in cuts]

    def compose(self, ascii_frame, t):
        scene = ascii_frame[self.crop_top:self.crop_top + self.scene_height]
        index = bisect_right(self.starts, t % PERIOD) - 1
        return np.concatenate((scene, self.panes[index]), axis=0)
