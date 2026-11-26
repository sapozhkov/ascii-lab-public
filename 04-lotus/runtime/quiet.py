"""Quiet lotus. Unhurried turgor change, a longer feeding pause and moving shade."""


from day import base

from petal_light import PetalLight
from petal_wind import PetalGarden




class QuietGarden(PetalLight, PetalGarden):
    bloom_pace = 3.15
    # A fuller second patch keeps the bee settled while the drop meets the pond.
    nectar_per_site = (1.03, 2.17)
