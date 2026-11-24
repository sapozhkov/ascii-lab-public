"""One fixed overhead shot of the school and the articulated swimmer.

Run from the repository root: python3 -m proto.shark_studies.cutoff_render
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from . import cutoff_motion
from .dive_shark import shark_cells
from .encounter_render import atlas
from .encounter_water import Camera, environment, small_fish, smoothstep, unit
from .school_render import code_panel, paint

OUT = Path(__file__).resolve().parents[2] / "shots/shark-cutoff"
FPS = cutoff_motion.FPS
LOOP_SECONDS = 20


def backbone(pose):
    return pose.center + pose.spine @ np.array([pose.forward, pose.right, pose.up])


def fixed_camera(frames):
    # Fit the split, the entire pursuit, and the mouth at contact.
    # The ends of the incoming rows are allowed to enter through the frame.
    target = next(f.target for f in frames if f.target is not None)
    active = [f for f in frames if np.any(f.alarm) and not f.caught]
    points = np.vstack([
        np.stack([f.fish[target] for f in active]),
        np.vstack([backbone(f.shark)[[0, 12]] for f in active[::4]]),
        np.vstack([np.quantile(f.fish, [.08, .92], axis=0) for f in active[::8]]),
    ])
    cam = Camera(points.mean(axis=0), 0., np.deg2rad(83), 1., (0., 0.))
    # Keep the school approaching at an oblique angle across the frame.
    choices = []
    for azimuth in np.linspace(.33, 1.22, 31) - np.pi:
        trial = Camera(cam.focus, azimuth, cam.elevation, 1., (0., 0.))
        xy = trial.project(points)
        scale = min(58., 81. / np.ptp(xy[:, 0]), 96. / np.ptp(xy[:, 1]))
        mid = (np.min(xy, axis=0) + np.max(xy, axis=0)) / 2
        choices.append((scale, azimuth, mid))
    scale, angle, mid = max(choices, key=lambda entry: entry[0])
    return Camera(cam.focus, angle, cam.elevation, scale, np.array([45, 54]) - scale * mid)


def loop_frames(frames, approach, cam):
    """Two consecutive shoals cross the same static water at 20 s intervals."""
    period = LOOP_SECONDS * FPS
    # Switch the tracked swimmer only when its entire body and shadow have left.
    # Both shoals remain in the model throughout their separate passages.
    for i in range(period):
        if i < period // 2:
            neighbour = frames[i + period]
        else:
            neighbour = approach[i - period]
        primary, secondary = frames[i], neighbour
        if i >= period // 2:
            spine = cam.project(backbone(primary.shark))
            if spine[:, 1].max() < -16:
                primary, secondary = neighbour, primary
        yield primary, secondary


def join_shoals(primary, secondary):
    # Track indices remain stable within the active passage. Its caught fish
    # stays at the jaw in the model and remains hidden by the existing shader.
    alive = np.ones(len(secondary.fish), dtype=bool)
    if secondary.caught:
        alive[secondary.target] = False
    return replace(primary, **{
        key: np.concatenate((getattr(primary, key), getattr(secondary, key)[alive]))
        for key in ("fish", "velocity", "phase", "sizes", "alarm")
    })


def grid(f, cam, t, plume=None, bubbles=None, *, loop=False, swimmer_time=None):
    pose = f.shark
    coverage, skin_chars, skin, skin_paper = shark_cells(pose, cam, t if swimmer_time is None else swimmer_time, draw_eyes=not loop)
    displacement = cam.project(pose.center + pose.center[2] * np.array([.19, -.14, -1])) - cam.project(pose.center)
    transform = np.float32([[1, 0, displacement[0]], [0, 1, displacement[1] / 1.5]])
    shadow = cv2.warpAffine(coverage, transform, (90, 72))
    shadow = .54 * cv2.GaussianBlur(shadow, (0, 0), .6)
    diving = hasattr(cam, "eye_height")
    if diving:
        from .cutoff_water import background, floor_shadow
        shadow = floor_shadow(cam, pose, shadow)
        chars, ink, paper = background(cam, t, shadow)
    else:
        chars, ink, paper = environment(cam, t, True, shadow, period=LOOP_SECONDS if loop else None)
    # Keep enough light on the seabed to read the dark body in negative.
    paper = paper * [.65, .89, .97] + [1, 5, 12]
    ink = .50 * (ink * [.53, .70, .79] + [3, 8, 14]) + .50 * paper
    if plume is not None:
        from .cutoff_water import suspended_dye
        suspended_dye(cam, plume, chars, ink, paper)
    where = coverage > .28
    weight = smoothstep(.08, .91, coverage)[..., None]
    xy = cam.project(f.fish)
    heading = unit(f.velocity) @ cam.axes.T
    length = cam.scale * f.sizes * np.linalg.norm(heading[:, :2], axis=1)
    if diving:
        # End-on fish still present the width of their body to the camera.
        projected = np.hypot(np.linalg.norm(heading[:, :2], axis=1), .23 * heading[:, 2])
        length = cam.scale * f.sizes * projected
    depth = (f.fish - pose.center) @ cam.axes[2]
    silver = np.clip(.58 + .21 * np.abs(heading[:, 0]) + .10 * np.sin(f.phase * .34), 0, 1)
    alive = np.ones(len(length), dtype=bool)
    if f.caught and f.target is not None:
        alive[f.target] = False
    for front in (False, True):
        ids = alive & ((depth >= 0) == front)
        under = ink.copy()
        small_fish(chars, ink, paper, xy[ids], unit(heading[ids, :2]),
                   length[ids], silver[ids], depth[ids], f.phase[ids], sunlit_sand=True)
        # Silver flanks cover several bright strokes even when downsampled.
        # Apply the same material to every visible fish, at either depth.
        scales = (ink[..., 0] > under[..., 0] + 10) & (ink[..., 0] > 120)
        paper[scales] += .42 * (ink[scales] - paper[scales])
        if not front:
            ink += weight * (skin - ink)
            paper += weight * (skin_paper - paper)
            chars[where] = skin_chars[where]
    if diving:
        from .cutoff_water import bubble_glass, port
        if bubbles is not None:
            bubble_glass(cam, bubbles, chars, ink, paper)
        port(chars, ink, paper, cam, t)
    return chars, np.uint8(np.clip(ink, 0, 255)), np.uint8(np.clip(paper, 0, 255))


def model_page(frames, cameras, out, dive, loop=False, show_forces=False):
    data = []
    for f, cam in zip(frames, cameras):
        def coords(points):
            return cam.project(points).round(2).tolist()
        stage = f.stage
        if dive:
            stage += f" · Camera tilt {np.rad2deg(cam.elevation):.0f}°"
        data.append(dict(fish=coords(f.fish), spine=coords(backbone(f.shark)),
                         target=f.target, stage=stage, caught=bool(f.caught)))
        if show_forces:
            pose = f.shark
            data[-1]["swim"] = [round(float(value), 3) for value in
                                (pose.speed, pose.thrust, 2.45*pose.speed**2,
                                 pose.tail_speed, np.rad2deg(pose.bank))]
    template = Path(__file__).with_name("cutoff_view.html").read_text()
    if loop:
        template = template.replace('src="cutoff.mp4"', 'src="cutoff.mp4" loop')
    if dive:
        template = template.replace("Overhead view", "Camera view")
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=int)
    (out / "model.html").write_text(template.replace("__FRAMES__", payload).replace("__FPS__", str(FPS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--dive", action="store_true", help="descend beside the chase")
    views.add_argument("--escape", action="store_true", help="simulate the prey's last evasive turn")
    args = parser.parse_args()
    dive, escape = args.dive, args.escape
    simulation_seconds = cutoff_motion.SECONDS if dive else LOOP_SECONDS * 3 // 2
    frames = list(cutoff_motion.states(seconds=simulation_seconds, wider_escape=not dive,
                                       evasive_prey=escape))
    contact = next(i for i, frame in enumerate(frames) if frame.caught)
    framing = list(cutoff_motion.states()) if escape else frames
    overhead = fixed_camera(framing)
    swimmer_times = None
    if dive:
        frames = frames[:contact + 1 + round(1.65 * FPS)]
        from .cutoff_camera import dive_cameras
        from .cutoff_water import contact_wake, port_bubbles
        cameras = list(dive_cameras(frames, overhead, FPS))
        plumes = list(contact_wake(frames, FPS))
        bubbles = list(port_bubbles(cameras, FPS))
        out = OUT.with_name("shark-cutoff-dive")
    else:
        approach = list(cutoff_motion.approach(frames[0], LOOP_SECONDS // 2, steady_entry=escape))
        pairs = list(loop_frames(frames, approach, overhead))
        swimmer_times = [(i/FPS if f is frames[i] else i/FPS - LOOP_SECONDS)
                         for i, (f, _) in enumerate(pairs)]
        frames = [join_shoals(f, other) for f, other in pairs]
        bubbles = [None] * len(frames)
        plumes = [None] * len(frames)
        cameras = [overhead] * len(frames)
        out = OUT.with_name("shark-cutoff-escape" if escape else "shark-cutoff-loop")
    out.mkdir(parents=True, exist_ok=True)
    model_page(frames, cameras, out, dive, loop=not dive, show_forces=escape)
    from .dive_pose import caudal_reaction, muscle_wave, water_acceleration
    functions = (caudal_reaction, water_acceleration) if escape else (cutoff_motion.select_isolated_prey, muscle_wave)
    panel = code_panel(functions)
    # A gentle shoulder keeps the silver flanks legible after phone downsampling.
    levels = np.arange(256, dtype=float) / 255
    silver_print = np.uint8(np.rint(255 * np.minimum(1., levels + .18 *
                                smoothstep(.18, .63, levels) * (1. - levels))))
    output = out / "cutoff.tmp.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as writer:
        for i, frame in enumerate(frames):
            chars, ink, paper = grid(frame, cameras[i], i / FPS, plumes[i], bubbles[i],
                                     loop=not dive, swimmer_time=swimmer_times[i] if swimmer_times else None)
            scene = paint(chars, ink, atlas(), paper, bloom=.26)
            writer.stdin.write(cv2.LUT(scene, silver_print).tobytes())
            writer.stdin.write(panel.tobytes())
        writer.stdin.close()
        if writer.wait():
            raise RuntimeError("ffmpeg could not encode the shot")
    output.replace(out / "cutoff.mp4")


if __name__ == "__main__":
    main()
