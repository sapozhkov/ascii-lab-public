"""Example of the vertical heart-profile coordinate used by the duet model."""

from math import cos


def heart_height(angle):
    y = 13 * cos(angle)
    y -= 5 * cos(2 * angle)
    y -= 2 * cos(3 * angle)
    y -= cos(4 * angle)
    return y / 16


if __name__ == "__main__":
    print(f"{heart_height(1.2):.4f}")
