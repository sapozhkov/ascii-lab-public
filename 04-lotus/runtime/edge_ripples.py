"""A low pond contact whose spreading waves enter the portrait."""
from day import base
from deep_pond import DeepPondGarden, DeepPondWater





class EdgeRipplesWater(DeepPondWater):
    waterline = -644.
    ripple_gain = 1.50
    far_ripple_reflection = .55

    def draw_pond(self, rgb, garden, t, size_ratio):
        # The front crosses the camera crop, while stems and grass in front
        # of the pond occlude it as it spreads.
        foreground = garden.z > -1.e5
        tissue = rgb[foreground]
        super().draw_pond(rgb, garden, t, size_ratio)
        rgb[foreground] = tissue


class EdgeRipplesGarden(DeepPondGarden):
    water_class = EdgeRipplesWater
