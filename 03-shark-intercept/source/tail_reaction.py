"""Standalone example of the model's axial tail reaction per unit mass.

Sweep and advance are speeds in the model's units. Slope is the local tail
slope. The coefficient matches caudal_reaction in dive_pose.py.
"""


def tail_reaction(sweep, slope,
                  advance):
    inflow = advance * slope
    push = sweep**2
    loss = inflow**2
    return .75 * (push - loss)


if __name__ == "__main__":
    print(f"{tail_reaction(0.8, 0.5, 1.2):.3f}")
