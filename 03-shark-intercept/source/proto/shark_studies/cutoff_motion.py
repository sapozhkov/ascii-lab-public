"""One pass through a turning shoal. Lengths are measured in shark bodies."""
from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np

from .dive_pose import STATION, SwimPose, Swimmer, centreline, mouth, muscle_wave
from .encounter_motion import DT, FPS, body_gap
from .encounter_water import smoothstep, unit


SECONDS = 12
# Each display frame samples 0.84 of a simulation frame for gentle slow motion.
STEP = .84 * DT


@dataclass
class CutoffFrame:
    shark: SwimPose
    fish: np.ndarray
    velocity: np.ndarray
    phase: np.ndarray
    sizes: np.ndarray
    alarm: np.ndarray
    target: int | None
    stage: str
    caught: bool


def select_isolated_prey(fish, pose, nose):
    distance = np.linalg.norm(fish[:, None] - fish[None, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    isolation = distance.min(axis=1)
    bearing = fish - nose
    visible = (bearing @ pose.forward > -.12) & (np.linalg.norm(bearing, axis=1) < .95)
    exposed = visible & (isolation > .33)
    if np.any(exposed):
        return int(np.argmax(np.where(exposed, isolation, -np.inf)))
    return None


def entry_pose(pose, weight):
    """Carry the shoulder bend through the forebody during the straight entry."""
    links = -np.diff(pose.spine, axis=0)
    yaw = np.arctan2(links[:, 1], links[:, 0])
    pitch = np.arctan2(links[:, 2], np.hypot(links[:, 0], links[:, 1]))
    shoulder = np.interp(.29, STATION, yaw)
    forebody = 1. - smoothstep(.07, .47, STATION)
    yaw += weight * forebody * (.30 * shoulder - yaw)
    spine, tangent = centreline(yaw, pitch)
    return replace(pose, spine=spine, tangent=tangent)


def approach(first, seconds, *, steady_entry=False):
    """Sample the approaching school up to the frame immediately before first."""
    pose = first.shark
    cruise = np.linalg.norm(first.velocity[:, :2], axis=1)
    beat = 4.4 + 18*cruise
    # The incoming rows already move upstream. Reconstruct their path in
    # the same cross-current used in the encounter, relative to frame zero.
    rate = 5.8*first.velocity[:, 0] + .4
    angle = 5.8*first.fish[:, 0]
    for index in range(-round(seconds*FPS), 0):
        time = index*STEP
        pos = first.fish + time*first.velocity
        current = .034*np.sin(angle + rate*time)
        initial_current = .034*np.sin(angle)
        pos[:, 1] += .034*(np.cos(angle)-np.cos(angle+rate*time))/rate - time*initial_current
        velocity = first.velocity.copy()
        velocity[:, 1] += current-initial_current
        # Initial cruising effort in Swimmer; backbone and heading agree
        # with its initial pose when the first physical step begins.
        phase = pose.phase + time*2*np.pi*(.44+.73*.22)
        spine, tangent = centreline(muscle_wave(phase, .22, 0.), np.zeros(len(pose.tangent)-1))
        shark = replace(pose, center=pose.center + time*pose.speed*pose.forward,
                        phase=phase, spine=spine, tangent=tangent)
        if steady_entry:
            shark = entry_pose(shark, 1.)
        yield replace(first, shark=shark, fish=pos, velocity=velocity,
                      phase=first.phase+time*beat)


def states(seconds=SECONDS, *, wider_escape=True, evasive_prey=False):
    rng = np.random.default_rng(3794)
    ranks, lanes = 20, 4
    count = ranks * lanes
    rank = np.repeat(np.arange(ranks), lanes)
    across = np.tile(np.arange(lanes) - 1.5, ranks)
    x = .15 + .069*rank + .054*np.sin(.52*rank) + rng.normal(0., .023, count)
    width = .052 + .034*np.sin(.36*rank)**2
    centerline = .070*np.sin(.42*rank) + .035*np.sin(.83*rank)
    y = width*across + centerline + rng.normal(0., .011, count)
    z = .70 + rng.normal(0., .012, count) + .012*np.sin(.42*rank)
    pos = np.column_stack((x, y, z))
    sizes = rng.uniform(.049, .071, count)
    phase = rng.uniform(0, 2*np.pi, count)
    cruise = rng.uniform(.176, .184, count)
    speed = cruise.copy()
    heading = np.full(count, np.pi)
    omega = np.zeros(count)
    alerted = np.zeros(count, dtype=bool)
    alive = np.ones(count, dtype=bool)
    turn_side = np.ones(count)
    crease = np.zeros(count)
    turn_rate = 1.33 + .03*across + rng.normal(0, .035, count)
    ahead = np.maximum(np.arange(count)-lanes, 0)
    swimmer = Swimmer([-1.55, -.168, .735], [1., .095, 0.], .13)
    chase = None
    captured = False
    escape_age = None
    for frame in range(round(seconds*FPS)):
        pose = swimmer.pose
        lips = mouth(pose)
        # A front-rank fish can see the predator. The following ranks see
        # their neighbours bank; the signal travels back along the ranks.
        gap, outward = body_gap(pos, pose)
        front = (rank == 0) & (gap < .43)
        relay = (rank > 0) & alerted[ahead] & (x <= crease[ahead] + .018)
        start = ~alerted & (front | relay)
        if np.any(start & front):
            # A head-on encounter divides the leading edge at the predator's
            # projected nose. Each fish takes its own side of the bow.
            lead = start & front
            crossing_y = lips[1] + (x[lead]-lips[0])*pose.forward[1]/pose.forward[0]
            turn_side[lead] = np.where(y[lead] >= crossing_y, 1., -1.)
        if np.any(start & relay):
            # Peripheral fish copy the neighbouring rank's majority turn.
            # One fish turning outwards cannot pull a whole lane after it.
            votes = np.sign(turn_side.reshape(ranks, lanes).sum(axis=1))
            turn_side[start & relay] = votes[rank[start & relay]-1]
        crease[start] = x[start]
        alerted |= start
        remaining = np.maximum(np.pi - (np.pi-heading)*turn_side, 0.)
        desired_omega = -turn_side*np.minimum(turn_rate, 3.1*remaining)*alerted
        omega += STEP*4.4*(desired_omega-omega)
        heading += STEP*omega
        finished = alerted & (remaining < .20)
        # The banked first rows slow so the turn can travel back through
        # the school. Once the chase is over they accelerate out together.
        exit_drive = .19 if captured and wider_escape else 0.
        speed_goal = cruise + .068*alerted - (.154-exit_drive)*finished
        speed += STEP*1.8*(speed_goal-speed)
        vel = np.column_stack((speed*np.cos(heading), speed*np.sin(heading),
                               1.4*(z-pos[:, 2])))
        # A cross-current varies with position. Fish on one wave of the
        # ribbon drift together while their neighbours feel the next wave.
        vel[:, 1] += .034*np.sin(5.8*pos[:, 0] + .4*frame*STEP)
        # Individual fish veer away from close neighbours in dense rows.
        delta = pos[:, None] - pos[None, :]
        distance = np.linalg.norm(delta, axis=2)
        distance[:, ~alive] = np.inf
        repulsion = np.maximum(.053-distance, 0.)/np.maximum(distance, .0001)
        vel += 1.5*np.einsum('ij,ijk->ik', repulsion, delta)
        # A fish in the small outside branch tries to rejoin across the
        # gap. A projected shark capsule blocks that route until it passes.
        majority = 1.
        outsiders = np.flatnonzero(alerted & (turn_side != majority) & alive)
        threat = False
        occluded = False
        for i in outsiders:
            neighbours = np.flatnonzero(alive & (turn_side == majority))
            nearest = neighbours[np.argmin(np.linalg.norm(pos[neighbours]-pos[i], axis=1))]
            home = pos[nearest]-pos[i]
            path = pos[i]+np.linspace(0, 1, 9)[:, None]*home
            clearance, _ = body_gap(path, pose)
            threat = clearance.min() < .18 and np.linalg.norm(home) > .13
            occluded |= clearance.min() < .01
            if remaining[i] < .85:
                # The animal can accelerate along its open flank; turning
                # back is only possible when its route clears the shark.
                forward = np.array([np.cos(heading[i]), np.sin(heading[i]), 0.])
                away = unit(pos[i] - lips)
                aim = unit(1.15*forward + .32*away) if threat else unit(home)
                angle = np.arctan2(aim[1], aim[0])
                error = np.arctan2(np.sin(angle-heading[i]), np.cos(angle-heading[i]))
                heading[i] += STEP*np.clip(2.5*error, -1.5, 1.5)
                vel[i, :2] = (.31 if threat else .20)*np.array(
                    [np.cos(heading[i]), np.sin(heading[i])])
        # Fish along the main ribbon yield to the hull and pectoral fins.
        # A faster predator covers more water during a fish's reaction time.
        # Respond earlier to its charge without altering the outsider's route.
        stand_off = .08
        if wider_escape:
            stand_off += min(.14, 1.8*max(0., pose.speed-.19))
        spring = np.maximum(stand_off - gap, 0.) * (turn_side == majority)
        vel += 4.5 * spring[:, None] * outward
        if evasive_prey and chase is not None and not captured:
            offset = pos[chase] - lips
            if escape_age is None and np.linalg.norm(offset) < .175 and offset @ pose.forward > .10:
                escape_age = 0.
                prey_velocity = vel[chase].copy()
                prey_yaw = np.arctan2(prey_velocity[1], prey_velocity[0])
                prey_turn = 0.
                free_water = pos[chase] - pos[turn_side == majority].mean(axis=0)
                escape_axis = unit(unit(prey_velocity[:2]) + .85*unit(free_water[:2]))
            if escape_age is not None:
                # A short turn precedes the kick. Momentum carries the fish
                # out while its muscles return to ordinary swimming.
                kick_time = np.clip((escape_age-.10)/.37, 0., 1.)
                kick = np.sin(np.pi*kick_time)**2
                release = np.clip((escape_age-.42)/.42, 0., 1.)
                direction = unit((1.-release)*escape_axis + release*unit(vel[chase, :2]))
                error = np.arctan2(direction[1], direction[0]) - prey_yaw
                error = np.arctan2(np.sin(error), np.cos(error))
                prey_turn += STEP*16.*(np.clip(9.*error, -6., 6.)-prey_turn)
                prey_yaw += STEP*prey_turn
                axis = np.array([np.cos(prey_yaw), np.sin(prey_yaw)])
                fast_thrust = 2.4*np.linalg.norm(vel[chase, :2])**2 + .85*kick
                speed_xy = np.linalg.norm(prey_velocity[:2])
                slip = prey_velocity[:2] - (prey_velocity[:2] @ axis)*axis
                prey_velocity[:2] += STEP*(fast_thrust*axis - 2.4*speed_xy*prey_velocity[:2] - 7.*slip)
                prey_velocity[2] = vel[chase, 2]
                vel[chase] = prey_velocity
                escape_age += STEP
        pos += STEP*vel
        x, y = pos[:, 0], pos[:, 1]
        # Only commit once a visible fish has clear water around it.
        if chase is None and occluded:
            chase = select_isolated_prey(pos, pose, lips)
        burst = 0.
        if chase is None:
            aim = np.array([1., .095, 0.])
            effort, gape = (.29 if np.any(alerted) else .12), 0.
        elif captured:
            aim = pose.forward.copy()
            aim[2] = .5*(.73-pose.center[2])
            effort, gape = .09, 0.
        else:
            displacement = pos[chase] - lips
            distance = np.linalg.norm(displacement)
            lead_time = min(.75, distance / max(.28, pose.speed))
            if escape_age is not None:
                # The prey is changing course. Trust a shorter prediction.
                lead_time = min(lead_time, .32)
            intercept = pos[chase]+lead_time*vel[chase]
            # The mouth is ventral and moves with bank and head recoil.
            aim = intercept-pose.center + (pose.center+.451*pose.forward-lips)
            effort = .66
            if escape_age is not None:
                alignment = aim @ pose.forward / np.linalg.norm(aim)
                effort += .34*np.clip((alignment-.93)/.065, 0., 1.)
                stroke = np.clip((escape_age-.06)/.36, 0., 1.)
                burst = 1.7*np.sin(np.pi*stroke)**2
            gape = float(np.clip((.23-distance)/.15, 0., 1.))
        pose = swimmer.step(aim, effort, STEP, gape=gape, use_tail_flow=escape_age is not None,
                            burst=burst, coordinated_bank=evasive_prey)
        if chase is not None and not captured:
            # First contact at the jaw opening, including half the prey's
            # length along its own longitudinal axis.
            lips = mouth(pose)
            axis = unit(vel[chase])
            along = np.clip((lips-pos[chase])@axis, -sizes[chase]/2, sizes[chase]/2)
            if np.linalg.norm(lips-pos[chase]-along*axis) < .040 and pose.gape > .30:
                alive[chase] = False
                captured = True
        phase += STEP*(4.4 + 18*np.linalg.norm(vel, axis=1) + .55*np.abs(omega))
        if captured:
            stage = "After capture"
        elif chase is not None:
            stage = "Prey dodges" if escape_age is not None and escape_age < .55 else "Pursuit"
        elif threat:
            stage = "Fish breaks away"
        elif np.any(alerted):
            stage = "School turns"
        else:
            stage = "Approach"
        if captured:
            pos[chase] = mouth(pose)
        if evasive_prey and chase is None:
            # Blend back to the ordinary steering pose before the shoal reacts.
            entry_weight = smoothstep(.55, .86, gap[rank == 0].min())
            if entry_weight > 0.:
                pose = entry_pose(pose, entry_weight)
        yield CutoffFrame(pose, pos.copy(), vel, phase.copy(), sizes,
                          alerted.astype(float), chase,
                          stage, captured)
