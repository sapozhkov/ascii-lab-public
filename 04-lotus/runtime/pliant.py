"""Lotus with a more compliant wet petal."""
from midrib import MidribGarden, MidribWater




class PliantWater(MidribWater):
    tip_compliance = .64


class PliantGarden(MidribGarden):
    water_class = PliantWater
    wet_bend_start = .18
