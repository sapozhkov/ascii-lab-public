# 02 · Jellyfish heart

[Watch on Instagram](https://www.instagram.com/reel/DgBKcSYN73Y/) · [Scene page](https://sapozhkov.github.io/ascii-lab-public/02-jellyfish-heart/) · [Explore the model](https://sapozhkov.github.io/ascii-lab-public/02-jellyfish-heart/model.html)

Two ASCII jellyfish spiral upward together. Seen from the side, their bells and tentacles form a heart.

![Two ASCII jellyfish forming a heart](preview.webp)

The interactive model pairs the silent ASCII scene with geometry sampled from the Python simulation at each of its 600 frames. Use the timeline to compare the swimming geometry with the final pose and see how the camera moves around the pair. The model loads the silent crop from `scene.mp4` and shows `model-poster.webp` before playback.

| File | What it does |
| --- | --- |
| [jellyfish.py](jellyfish.py) | Renders the scene to PNG or silent MP4. |
| [source/duet.py](source/duet.py) | Integrates the orbit, lift, and tentacles and poses the pair in the camera. |
| [source/swimmer.py](source/swimmer.py) | Builds the bell and lights its surface and ribbons. |
| [source/renderer.py](source/renderer.py) | Projects the geometry to a grid and paints the ASCII glyphs. |
| [source/reef.py](source/reef.py) | Builds the rock shelf, sea pen, anemone, and light shafts. |

The code card opens the main renderer. The preview shows [the spread function](source/duet.py#L21-L22), which controls how far apart the pair swims.

## Run the renderer

Use Python 3.9 or later. The commands below use the NumPy and Pillow versions tested with Python 3.9. For MP4 output, install FFmpeg with H.264 encoding support and make it available on your PATH. Keep the `source/` directory alongside `jellyfish.py`.

```sh
python3 -m pip install numpy==2.0.2 Pillow==11.3.0
python3 jellyfish.py png 17.5 heart.png
python3 jellyfish.py mp4 heart.mp4
```

The export covers the visible ASCII scene and reef. The Instagram reel adds a code panel, a status strip, and music. Choose `--duration 2` for a short MP4 test.

The renderer tries Menlo, DejaVu Sans Mono, and Consolas. Set `ASCII_LAB_FONT` to use a different monospace TTF or TTC font.
