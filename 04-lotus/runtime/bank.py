"""Checkpoint D with windblown bank grass and the classic reel layout."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bank_grass import BankGrass
from day import RATE, base
from lotus_panel import source_panel

from pliant import PliantGarden
from publication_panel import PublicationPanel

from channel import HANDLE




class BankPanel(PublicationPanel):
    scene_height = base.SIZE[1] * 27 // 40
    ink = (147, 166, 181)

    def __init__(self, garden):
        super().__init__(garden)
        self.bloom = garden.cycle.open.mean(axis=(1, 2))
        # The pair of modal angles describes the stem's bend away from rest.
        self.sway = np.rad2deg(np.linalg.norm(garden.rod.tip_angles, axis=1))
        self.font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 20)

    def _safe_pane(self, section):
        return np.asarray(source_panel(section, self.width, self.height-self.scene_height))

    def compose(self, ascii_frame, t):
        frame = super().compose(ascii_frame, t)
        band = Image.fromarray(frame[94:126])
        text = ImageDraw.Draw(band)
        tick = round(t*RATE) % len(self.bloom)
        readout = f'bloom {self.bloom[tick]:4.0%}   sway {self.sway[tick]:4.1f}°'
        text.text((72, 16), readout, font=self.font, fill=self.ink, anchor='lm')
        text.text((self.width-72, 16), HANDLE, font=self.font, fill=self.ink, anchor='rm')
        frame[94:126] = np.asarray(band)
        return frame


class BankGarden(PliantGarden):
    def __init__(self):
        super().__init__()
        self.grass = BankGrass(self)
        self.panel = BankPanel(self)

    def stem(self, t):
        super().stem(t)
        self.grass.draw(self, t)
