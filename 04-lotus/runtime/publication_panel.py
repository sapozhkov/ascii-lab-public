"""Static framing for both the reel and its 3:4 profile crop."""
import numpy as np

from day import base
from lotus_panel import LotusPanel, source_panel, source_section
from lotus_weather import ripples


class PublicationPanel(LotusPanel):
    scene_height = 950
    crop_top = 144
    # Intersection of the F1 and earlier reel safe areas, at 810 x 1440.
    code_left = 68
    code_right = 705

    @classmethod
    def frame_camera(cls, garden):
        # Compose once in the orthographic camera, before contact and shading
        # are sampled. The glyphs are then rasterized at their native size.
        # The optical axis passes through the stem just below the flower's cup.
        garden.scale *= .815
        garden.xmid = 193.
        garden.ymid = (cls.crop_top + 497.) / 2

    def __init__(self, garden):
        bee, water = garden.bee.events, garden.water.events
        opening = source_section(base.bend)
        deformation = garden.water._shape_dynamics
        cuts = (
            (0., opening),
            (bee['approach'], source_section(garden.light_window.canopy)),
            (bee['descend'], source_section(garden.bee.wings.contour, 'stroke =')),
            (bee['probe1'], source_section(deformation, 'footprint =')),
            (water['hit'], source_section(deformation, 'rate -= min(')),
            (water['splash'], source_section(ripples)),
            (bee['depart'], opening),
        )
        self.starts = [time for time, _ in cuts]
        panes = {section: self._safe_pane(section) for _, section in cuts}
        self.panes = [panes[section] for _, section in cuts]

    def _safe_pane(self, section):
        height = self.height - self.scene_height
        # Keep the 19 px project type size. Only long, unaccented context
        # lines run past the editor viewport. Each selected line fits it.
        excerpt = np.asarray(source_panel(section, self.width, height))[:, :self.code_right-self.code_left]
        paper = np.array([8, 10, 14])
        panel = np.empty((height, self.width, 3), dtype=np.uint8)
        panel[:] = paper
        # The active line is above the caption reserve (y=990). The lower
        # lines provide editor context and recede underneath the platform UI.
        y = np.arange(height)
        strength = np.interp(y, [0, 40, 160, height-1], [1., 1., .64, .34])
        panel[:, self.code_left:self.code_right] = paper + (excerpt-paper)*strength[:, None, None]
        panel[0] = excerpt[0,0]
        return panel
