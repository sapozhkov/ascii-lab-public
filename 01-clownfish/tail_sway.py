"""Example of the sideways body wave and turn bend in the clownfish model.

s is the position along the body, beat is the tail phase in radians, and
drive and turn control the amplitudes of the swimming and turning motions.
"""

from math import sin


def sway(s, beat, drive, turn):
    amp = .014 + .105 * s**2
    wave = amp*sin(beat-3.4*s)
    lag = s - .25
    bend = lag * abs(lag)
    return drive*wave+turn*bend


if __name__ == "__main__":
    print(f"{sway(.85, 1.8, .7, .12):.4f}")
