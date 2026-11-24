# Shark intercept

A shark approaches a school of 80 fish. One breaks away. A segmented spine, tail thrust and water drag drive the pursuit. The counter drops on contact at 7.58 seconds. The next school enters before the end of the 20-second loop.

- [Swimming and thrust](source/proto/shark_studies/dive_pose.py)
- [School, pursuit and contact](source/proto/shark_studies/cutoff_motion.py)
- [ASCII rendering, readouts and export](source/proto/shark_studies/cutoff_render.py)
- [Synchronized model](model.html)

## Run the renderer

Use Python 3.9 or later, FFmpeg with libx264 and the Menlo font included with macOS. From this directory, run:

```sh
cd source
python3 -m pip install -r requirements.txt
python3 -m proto.shark_studies.cutoff_render
```

Open `source/output/cutoff.mp4` for the reel or `source/output/model.html` for the matching model and frame scrubber. The reel is 1080 × 1920 at 24 frames per second.

The swimming model uses a simplified tail reaction, body drag and articulated spine. Its hydrodynamic coefficients were tuned for this scene. See the references in `source/proto/shark_studies/dive_pose.py`.
