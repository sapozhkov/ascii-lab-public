"""Place the drop's pond impact below the framed landscape."""
from day import base
from gloss import GlossGarden, GlossWater





class DeepPondWater(GlossWater):
    waterline = -980.


class DeepPondGarden(GlossGarden):
    water_class = DeepPondWater
    mark_scene_edge = True
