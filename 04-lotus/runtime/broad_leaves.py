"""Increase the two leaf blades around their petiole attachments."""
from breeze import WindLeaf
from day import base
from edge_ripples import EdgeRipplesGarden





class BroadLeaf(WindLeaf):
    scales = (1.24, 1.18)

    def attach(self, air):
        # Enlarge each blade before computing its wind pose and water contacts.
        scale = self.scales[self.index]
        self.x *= scale
        self.y *= scale
        self.height *= scale
        super().attach(air)


class BroadLeafGarden(EdgeRipplesGarden):
    leaf_class = BroadLeaf
