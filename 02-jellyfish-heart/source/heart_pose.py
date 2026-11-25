"""Fit one point to the heart outline in the pair's side view."""

# y is the point's height before projection; zoom is its perspective scale.
# rim holds the lower and upper projected edges at this horizontal position.
# v is the point's relative height in the swimming silhouette; w blends the pose.
def heart_y(y, zoom, rim, v, w):
    low, high = rim
    screen = y * zoom
    goal = low + v * (high - low)
    shift = w * (goal - screen)
    return y + shift / zoom


if __name__ == "__main__":
    print(f"{heart_y(-0.6, 0.9, (-1.3, 0.5), 0.25, 0.8):.4f}")
