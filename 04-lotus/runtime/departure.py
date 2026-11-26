"""Small dew on the lotus rib and a bumblebee turning under limited thrust."""


from banked_forager import BankedForager
from day import base

from passage import PassageGarden
from rib_dew import RibDew




class DepartureGarden(PassageGarden):
    water_class = RibDew
    forager_class = BankedForager
    show_departure_detail = True
