"""Motion shared by the ASCII renderer and the two-camera model viewer."""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


FPS = 24
DURATION = 20
SHARK_LENGTH = 450.0
SHARK_CLEARANCE = SHARK_LENGTH / 2
NOSE_X = 270.0
# Stations are measured from nose to tail. All widths are in the model's world units.
BODY_STATIONS = [0, .012, .035, .075, .13, .21, .34, .46, .57, .69, .78, .9, 1]
HALF_WIDTH = [1, 9, 16, 23, 27, 29, 26, 22, 16, 8, 4.5, 3.3, 0.5]
FIN_REACH = [16, 30, 50, 71, 83, 90]
FIN_LEAD = [.176, .183, .198, .222, .242, .265]
FIN_TRAIL = [.304, .271, .252, .253, .26, .265]
FISH_LENGTH = np.array([58., 54., 61., 56., 59.])
FISH_STATIONS = [0, .045, .12, .25, .38, .50, .62, .74, .79, .82, .90, .96, 1.]
FISH_WIDTH = [0, .065, .125, .166, .17, .15, .12, .068, .04, .035, .15, .21, .25]
FISH_FORK = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .030, .105, .192]


@lru_cache(maxsize=4)
def fish_lengths(count):
    return np.resize(FISH_LENGTH, count)


def tail_sweep(u, phase):
    """A muscular wave reaches the tail after the trunk."""
    return 29 * u**2.15 * np.sin(phase - 4.15 * u)


def shark_surface(x, y, state):
    """Sample the bent trunk, pectoral fins and dorsal ridge seen from above."""
    nose_x, nose_y = state["nose"]
    phase = state["phase"]
    raw = (y - nose_y) / SHARK_LENGTH
    u = np.clip(raw, 0, 1)
    local_x = x - nose_x - tail_sweep(u, phase)
    width = np.interp(u, BODY_STATIONS, HALF_WIDTH)
    flank = local_x / np.maximum(width, 1.)
    body = (raw >= 0) & (raw <= 1) & (np.abs(flank) <= 1)
    reach = np.abs(local_x)
    # The left tip is swept slightly farther back in the overhead view.
    sweep = np.maximum(0, -local_x) * (.025 / 90)
    lead = np.interp(reach, FIN_REACH, FIN_LEAD) + sweep
    trail = np.interp(reach, FIN_REACH, FIN_TRAIL) + sweep
    fins = (reach <= FIN_REACH[-1]) & (reach >= FIN_REACH[0]) & (raw >= lead) & (raw <= trail)

    roll = .18 * np.sin(phase - 4.15 * u)
    lit_side = np.exp(-((flank + .56 - roll) / .22)**2)
    trunk_light = .035 + .17 * lit_side + .09 * np.exp(-((u-.075)/.09)**2)
    # The tall dorsal fin catches a thin streak on its left face.
    ridge = np.exp(-((local_x + 2 - 3 * roll) / 3.3)**2)
    dorsal = .47 * ridge * np.exp(-((u-.34)/.12)**4)
    tail_light = .24 * lit_side * np.exp(-((u-.57)/.12)**2)
    fin_light = .045 + .13 * np.exp(-((raw-lead-.009)/.014)**2)
    reflectance = np.where(body, trunk_light + dorsal + tail_light, fin_light)
    inside = body | fins
    return inside, np.where(inside, reflectance, 0.)


def fish_sweep(u, phase):
    return (.008 + .075 * u**2) * np.sin(phase - 4.0 * u)


def fish_surface(x, y, fish, index):
    """A round-backed minnow tapers to a narrow peduncle and a forked tail."""
    cx, cy, heading, phase, alarm = fish
    length = FISH_LENGTH[index % len(FISH_LENGTH)]
    cs, sn = math.cos(heading), math.sin(heading)
    dx, dy = (x - cx) / length, (y - cy) / length
    raw = .5 - (dx * cs + dy * sn)
    u = np.clip(raw, 0, 1)
    cross = -dx * sn + dy * cs - fish_sweep(u, phase)
    width = np.interp(u, FISH_STATIONS, FISH_WIDTH)
    gap = np.interp(u, FISH_STATIONS, FISH_FORK)
    inside = (raw >= 0) & (raw <= 1) & (np.abs(cross) >= gap) & (np.abs(cross) <= width)
    flank = cross / np.maximum(.015, width)
    # Silver along the back; the paired dark eyes stay attached to the head.
    silver = np.exp(-((flank + .25 - .12*np.sin(phase)) / .53)**2)
    eyes = np.exp(-((u-.135)/.035)**2 - ((np.abs(flank)-.68)/.22)**2)
    shoulder = np.exp(-((u-.35)/.18)**2)
    gill = np.exp(-((u-.225)/.038)**2)
    light = .25 + (.26 + .48 * shoulder) * silver
    light += .12 * np.exp(-((u-.1)/.09)**2) + .07 * alarm - .21 * gill - .25 * eyes
    light = np.where(u > .81, .54 + .21*silver, light)
    return inside, np.where(inside, np.clip(light, 0, 1), 0.)


def shark_sections():
    # Each closed ribbon is expressed as (longitudinal station, cross section).
    u = np.unique(np.r_[BODY_STATIONS, np.linspace(0, 1, 39)])
    w = np.interp(u, BODY_STATIONS, HALF_WIDTH)
    result = [np.column_stack([np.r_[u, u[::-1]], np.r_[-w, w[::-1]]])]
    reach = np.array(FIN_REACH)
    for side in (-1, 1):
        shift = np.maximum(0, -side * reach) * (.025 / 90)
        leading = np.array(FIN_LEAD) + shift
        trailing = np.array(FIN_TRAIL) + shift
        result.append(np.column_stack([np.r_[leading, trailing[::-1]], side * np.r_[reach, reach[::-1]]]))
    return result


def fish_section():
    u = np.array(FISH_STATIONS)
    w = np.array(FISH_WIDTH)
    # The return via the inside of each lobe cuts the V out of the tail.
    notch = u >= .82
    gap = np.array(FISH_FORK)[notch]
    inner = u[notch]
    return np.column_stack([np.r_[u, inner[::-1], inner[1:], u[::-1]],
                            np.r_[-w, -gap[::-1], gap[1:], w[::-1]]])


SHARK_SECTIONS = shark_sections()
FISH_SECTION = fish_section()
SHARK_U, SHARK_CROSS = np.concatenate(SHARK_SECTIONS).T


def minnow_offsets(headings, phases):
    u, cross = FISH_SECTION.T
    length = fish_lengths(len(headings))[:, None]
    ahead = (.5 - u) * length
    lateral = (cross + fish_sweep(u, phases[:, None])) * length
    cs = np.cos(headings)[:, None]
    sn = np.sin(headings)[:, None]
    return np.stack([ahead * cs - lateral * sn, ahead * sn + lateral * cs], axis=-1)


def keep_clearance(positions, headings, phases):
    # A hard personal-space boundary during the tightest part of a turn.
    # Corrections are equal and opposite and preserve the centre of the shoal.
    offsets = minnow_offsets(headings, phases)
    for _ in range(3):
        delta = positions[None, :, :] - positions[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        direction = delta / np.maximum(distance[:, :, None], 1e-6)
        reach = np.einsum('ikd,ijd->ijk', offsets, direction).max(axis=2)
        penetration = np.maximum(0., reach + reach.T + 5 - distance)
        np.fill_diagonal(penetration, 0)
        if np.max(penetration) < .01:
            break
        positions -= .5 * (direction * penetration[:, :, None]).sum(axis=1)


def escape(nose_y, phase, positions, headings, fish_phase):
    """Sense clearance between the caudal sweep, fin tips and each fish."""
    sx = NOSE_X + SHARK_CROSS + tail_sweep(SHARK_U, phase)
    sy = nose_y + SHARK_LENGTH * SHARK_U
    offsets = minnow_offsets(headings, fish_phase)
    fx = positions[:, 0, None] + offsets[:, :, 0]
    fy = positions[:, 1, None] + offsets[:, :, 1]
    delta_x = fx[:, :, None] - sx
    delta_y = fy[:, :, None] - sy
    clearance = np.sqrt((delta_x**2 + delta_y**2).min(axis=(1, 2)))
    urgency = np.clip((107 - clearance) / 91, 0, 1)
    station = np.clip((positions[:, 1] - nose_y) / SHARK_LENGTH, 0, 1)
    side = np.sign(positions[:, 0] - NOSE_X - tail_sweep(station, phase))
    forward = np.clip((positions[:, 1] - nose_y - station * SHARK_LENGTH) / 85, -.8, .8)
    return urgency**2, np.column_stack([side, forward])


def body_clearance(nose_y, phase, positions):
    # Closest point on a segment of the posed shark perimeter. Bound each minnow
    # by its longest possible diagonal (including both lobes of its beating tail).
    edges = []
    for section in SHARK_SECTIONS:
        u, cross = section.T
        points = np.column_stack((NOSE_X + cross + tail_sweep(u, phase),
                                  nose_y + u * SHARK_LENGTH))
        edges.append(np.stack((points, np.roll(points, -1, axis=0)), axis=1))
    start, end = np.concatenate(edges).transpose(1, 0, 2)
    relative = positions[:, None, :] - start
    edge = end - start
    projection = (relative * edge).sum(axis=2) / np.maximum(1e-12, (edge**2).sum(axis=1))
    normal = relative - np.clip(projection, 0, 1)[:, :, None] * edge
    squared = (normal**2).sum(axis=2)
    nearest = squared.argmin(axis=1)
    distance = np.sqrt(squared[np.arange(len(positions)), nearest])
    normal = normal[np.arange(len(positions)), nearest] / np.maximum(1e-9, distance[:, None])
    return distance - .61 * fish_lengths(len(positions)), normal


def avoid_shark(nose_y, phase, positions, speed):
    # Anticipate the advancing snout so the shoal has time to part.
    predicted_y = nose_y - .72 * speed
    clearance, _ = body_clearance(predicted_y, phase, positions)
    urgency = np.clip((SHARK_CLEARANCE + 110 - clearance) / 110, 0, 1)
    sideways = np.where(positions[:, 0] < NOSE_X, -1., 1.)
    forward = -.40 * np.clip((predicted_y - positions[:, 1]) / 110, 0, 1)
    return urgency, np.column_stack((sideways, forward))


def flock(positions, velocities):
    delta = positions[None, :, :] - positions[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    near = (distance > 0) & (distance < 125)
    count = np.maximum(1, near.sum(axis=1))[:, None]
    attraction = (near[:, :, None] * delta).sum(axis=1) / count
    alignment = near @ velocities / count - velocities
    # The front-to-back clearance of two minnows is longer than their beam.
    direction = delta / np.maximum(1., distance[:, :, None])
    axes = velocities / np.maximum(1., np.linalg.norm(velocities, axis=1, keepdims=True))
    along = np.abs((direction * axes[:, None, :]).sum(axis=2))
    extent = fish_lengths(len(positions))[:, None] * np.hypot(.50 * along, .17 * np.sqrt(np.maximum(0., 1 - along**2)))
    clearance = distance - extent - extent.T
    spacing = np.maximum(0, 18 - clearance) / np.maximum(1, distance)
    separation = -(delta * spacing[:, :, None]).sum(axis=1)
    return 0.15 * attraction + 0.16 * alignment + 3.0 * separation


@lru_cache(maxsize=3)
def _trajectory(variant):
    if variant not in ("follow", "pass", "wide"):
        raise ValueError(f"Unknown camera: {variant}")
    tracking = variant == "follow"
    if tracking:
        positions = np.array([(68, 5), (127, 40), (181, -4), (277, 22), (332, -14)], dtype=float)
    elif variant == "pass":
        positions = np.array([(64, 285), (117, 333), (152, 264), (236, 314), (280, 249)], dtype=float)
    else:
        # Five staggered ranks keep neighbours visible in the initial wide shot.
        positions = np.array([(66 + 69*k + 26*(r % 2) + 9*math.sin(r+k),
                               215 + 48*r + 12*math.sin(2*k+.4*r))
                              for r, count in enumerate((3, 5, 5, 5, 3))
                              for k in range(count)], dtype=float)
    headings = np.array([-.10, .08, -.18, .13, -.04])
    fish_speed = np.array([13.0, 12.5, 13.9, 12.8, 13.6])
    tail = np.array([0.1, 2.9, 5.2, 1.0, 4.1])
    wide = variant == "wide"
    if wide:
        rng = np.random.default_rng(33)
        headings = rng.uniform(-.24, .20, len(positions))
        fish_speed = rng.uniform(10, 16, len(positions))
        tail = rng.uniform(0, 2 * math.pi, len(positions))
    alarm = np.zeros(len(positions))
    startled = np.zeros(len(positions))
    cruise = 35.0 if tracking else 64.0
    nose_y = 99.0 if tracking else 722.0
    if wide:
        cruise, nose_y = 112., 1178.
    speed = cruise
    phase = 0.0
    dt = 1 / (FPS * 2)
    frames = []
    for step in range(FPS * DURATION * 2):
        camera_y = nose_y - 99.0 if tracking else 0.0
        if step % 2 == 0:
            fish = tuple(zip(positions[:, 0], positions[:, 1] - camera_y, headings, tail, alarm))
            frames.append(dict(nose=(NOSE_X, nose_y - camera_y), phase=phase, camera_y=camera_y, fish=fish))
        # Swimming thrust beats twice per tail cycle, against quadratic drag.
        thrust = 0.65 * cruise * (1.0 + 0.19 * math.cos(2 * phase))
        speed += (thrust - 0.65 * speed**2 / cruise) * dt
        nose_y -= speed * dt
        phase += 2 * math.pi * (0.43 + speed / 180) * dt
        if wide:
            urgency, outward = avoid_shark(nose_y, phase, positions, speed)
        else:
            urgency, outward = escape(nose_y, phase, positions, headings, tail)
        response_time = np.where(urgency > alarm, 0.13, 1.10)
        alarm += (urgency - alarm) * dt / response_time
        heading_vectors = np.column_stack([np.cos(headings), np.sin(headings)])
        velocities = heading_vectors * fish_speed[:, None]
        social = flock(positions, velocities)
        desired = np.array([14., -1.]) + social * (1 - .45 * alarm[:, None])
        startled = np.maximum(startled, np.minimum(1.0, 4.0 * alarm))
        # Across the open gap they can still see the rest of the shoal.
        regroup = positions.mean(axis=0) - positions
        desired += 0.22 * regroup * (startled * (1 - alarm))[:, None]
        desired += outward * ((158 if wide else 105) * alarm[:, None])
        target_heading = np.arctan2(desired[:, 1], desired[:, 0])
        delta = np.arctan2(np.sin(target_heading - headings), np.cos(target_heading - headings))
        turn_rate = 1.55 + 3.8 * alarm
        headings += np.clip(delta * 3.2, -turn_rate, turn_rate) * dt
        target_speed = np.clip(np.linalg.norm(desired, axis=1), 9, 140 if wide else 85)
        fish_speed += (target_speed - fish_speed) * (1.8 + 2 * alarm) * dt
        positions += np.column_stack([np.cos(headings), np.sin(headings)]) * fish_speed[:, None] * dt
        tail += 2 * math.pi * (0.9 + fish_speed / 60) * dt
        keep_clearance(positions, headings, tail)
        if wide:
            # Resolve the rare approach inside the margin after neighbour forces.
            clearance, normal = body_clearance(nose_y, phase, positions)
            positions += normal * np.maximum(0, SHARK_CLEARANCE + 2 - clearance)[:, None]
    return tuple(frames)


def states(variant):
    return iter(_trajectory(variant))


def write_model(path):
    rounds = {}
    for variant in ("follow", "pass"):
        rounds[variant] = [
            {"nose": [round(v, 2) for v in s["nose"]], "phase": round(s["phase"], 3),
             "camera_y": round(s["camera_y"], 2), "fish": [[round(v, 3) for v in fish] for fish in s["fish"]]}
            for s in states(variant)
        ]
    template = Path(__file__).with_name("school_model.html").read_text()
    # Each point stores the two coefficients of the sinusoidal spine wave.
    # The viewer can reconstruct any recorded phase without a second geometry.
    def wave_vertices(section, wave):
        u, cross = section.T
        return np.column_stack([u, cross, wave(u, 0), wave(u, math.pi / 2)]).round(6).tolist()

    data = dict(fps=FPS, length=SHARK_LENGTH, tracks=rounds,
                shark=[wave_vertices(path, tail_sweep) for path in SHARK_SECTIONS],
                minnow=wave_vertices(FISH_SECTION, fish_sweep), fishLength=FISH_LENGTH.tolist())
    Path(path).write_text(template.replace("/* TRACKS */", json.dumps(data, separators=(",", ":"))))
