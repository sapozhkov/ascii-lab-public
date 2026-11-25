# 02 · Jellyfish heart

[Watch on Instagram](https://www.instagram.com/reel/DgBKcSYN73Y/) · [Scene page](https://sapozhkov.github.io/ascii-lab-public/02-jellyfish-heart/) · [Explore the model](https://sapozhkov.github.io/ascii-lab-public/02-jellyfish-heart/model.html)

Two ASCII jellyfish spiral upward together. Seen from the side, their bells and tentacles form a heart.

![Two ASCII jellyfish forming a heart](preview.webp)

The interactive model pairs the silent ASCII scene with geometry sampled from the Python simulation at each of its 600 frames. Use the timeline to compare the swimming geometry with the final pose and see how the camera moves around the pair. The model loads the silent crop from `scene.mp4` and shows `model-poster.webp` before playback.

| File | What it does |
| --- | --- |
| [jellyfish.py](jellyfish.py) | Plays the scene in the terminal and exports PNG or silent MP4. |
| [terminal_player.py](terminal_player.py) | Displays the character grid using ANSI true color. |
| [source/duet.py](source/duet.py) | Integrates the orbit, lift, and tentacles and poses the pair in the camera. |
| [source/heart_pose.py](source/heart_pose.py) | Standalone example of how a point moves into the heart pose. |
| [source/heart_height.py](source/heart_height.py) | Standalone example of the vertical coordinate on the heart outline. |
| [source/swimmer.py](source/swimmer.py) | Builds the bell and lights its surface and ribbons. |
| [source/renderer.py](source/renderer.py) | Projects the geometry to a grid and paints the ASCII glyphs. |
| [source/reef.py](source/reef.py) | Builds the rock shelf, sea pen, anemone, and light shafts. |

The code card opens the main renderer. The preview shows how [`heart_y`](source/heart_pose.py) moves a point toward the heart outline in the side view. It uses the point’s relative height in the swimming silhouette to place it between the edges of the heart. Run `python3 source/heart_pose.py` to try the example.

## Run the renderer

Use Python 3.9 or later. Run the commands from the `02-jellyfish-heart/` directory. For MP4 output, install FFmpeg with H.264 encoding support and make it available on your PATH. Keep the `source/` directory alongside `jellyfish.py`.

```sh
python3 -m pip install numpy==2.0.2 Pillow==11.3.0
python3 jellyfish.py terminal
python3 jellyfish.py png 17.5 heart.png
python3 jellyfish.py mp4 heart.mp4
```

Terminal playback runs once and fits the character grid to the window. Use an ANSI true-color terminal with at least 97 columns and 78 rows to see every cell. Press `Ctrl+C` to stop.

The exports cover the visible ASCII scene and reef. The Instagram reel adds a code panel, a status strip, and music. Choose `--duration 2` for a short MP4 test.

For PNG and MP4, the renderer tries Menlo, DejaVu Sans Mono, and Consolas. Set `ASCII_LAB_FONT` to use a different monospace TTF or TTC font.

[Scientific references](https://sapozhkov.github.io/ascii-lab-public/02-jellyfish-heart/model.html#scientific-references)
