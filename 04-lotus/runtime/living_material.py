"""Load 14 petal pigments on the renderer's UV grid."""
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def materials():
    with np.load(Path(__file__).with_name('petal_uv.npz'), allow_pickle=False) as palette:
        return palette['pixels']
