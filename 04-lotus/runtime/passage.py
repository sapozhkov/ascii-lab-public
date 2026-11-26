"""A single pass of sunlight across the lotus, framed for publication."""


from day import base
from dew import DewWater

from passing_light import PassingLight
from publication_panel import PublicationPanel
from quiet import QuietGarden




class PassageGarden(QuietGarden):
    water_class = DewWater

    def populate(self):
        PublicationPanel.frame_camera(self)
        super().populate()

    def __init__(self):
        super().__init__()
        self.light_window = PassingLight(self)
        self.panel = PublicationPanel(self)

    def sun_direction(self, t):
        return PassingLight.sun

    @property
    def diagram_sunlight(self):
        return self.lightfield

    def compose_frame(self, scene, t):
        return self.panel.compose(scene, t)
