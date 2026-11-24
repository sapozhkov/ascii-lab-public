"""Predator pursuit and delayed, local escape responses. Distances in shark lengths."""

from dataclasses import dataclass
import numpy as np

from .encounter_water import unit


FPS = 24
DT = 1/FPS


@dataclass
class Pose:
    center: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    phase: float
    speed: float
    bank: float


@dataclass
class Frame:
    shark: Pose
    fish: np.ndarray
    velocity: np.ndarray
    alarm: np.ndarray
    phase: np.ndarray
    sizes: np.ndarray


def swim(center, heading, speed, phase, bank, aim, effort, turn_limit, dt):
    """Thrust works against drag. A faster animal needs more space to turn."""
    steering = np.clip(aim @ heading, -1, 1)
    lateral = aim - steering * heading
    curvature = min(turn_limit*dt/max(.45, speed/.23), np.arccos(steering))
    turn = unit(lateral)
    forward = unit(heading*np.cos(curvature) + turn*np.sin(curvature))
    speed += dt*(effort - 1.7*speed**2)
    center = center + dt*speed*forward
    side = unit(np.cross(forward, [0,0,1]))
    dorsal = np.cross(side, forward)
    lean = np.clip(2.1*(turn @ side)*np.sin(curvature)/dt, -.72,.72)
    bank += dt*(lean-bank)*1.9
    right = side*np.cos(bank)-dorsal*np.sin(bank)
    up = dorsal*np.cos(bank)+side*np.sin(bank)
    phase += dt*(2.0+12.5*speed)
    return Pose(center, forward, right, up, phase, speed, bank)


def body_gap(points, shark):
    station = np.clip((points-shark.center) @ shark.forward, -.44,.45)
    nearest = shark.center + station[:,None]*shark.forward
    offset = points-nearest
    # Bounding capsule encloses the pectoral fins and the oscillating tail.
    radius = np.interp(station, [-.44,-.31,-.16,.01,.18,.32,.45],
                       [.15,.17,.15,.25,.23,.15,.09])
    return np.linalg.norm(offset,axis=1)-radius, unit(offset)


def alarm_wave(alarm, gap, neighbours, separation, sensitivity, dt):
    """Threat seen directly or relayed by nearby fish that have already turned."""
    nearby = separation < .39
    relay = np.max(np.where(nearby,alarm[neighbours],0), axis=1)
    visual = np.clip((sensitivity-gap)/.24,0,1)
    excitation = np.maximum(visual, np.clip(1.80*relay-.58,0,1))
    return np.clip(alarm+dt*(1.85*excitation*(1-alarm)-.17*(1-excitation)*alarm),0,1)


def update_school(pos, vel, desired, shark, neighbours, separation, dt):
    """Keep lanes elastic and cap acceleration. Near the predator, turn outward."""
    gap, outward = body_gap(pos,shark)
    near = np.clip((.76-gap)/.24,0,1)
    repulsion = outward*(near*.48)[:,None]
    delta = pos[:,None,:]-pos[neighbours]
    crowd = np.maximum(.095-separation,0)
    repulsion += 1.5*(delta * (crowd / np.maximum(.005,separation))[...,None]).sum(axis=1)
    acc = (desired+repulsion-vel)*3.0
    vel = vel + dt*acc / np.maximum(1,np.linalg.norm(acc,axis=1,keepdims=True)/.85)
    pos = pos+dt*vel
    gap, outward = body_gap(pos,shark)
    pos += np.maximum(.535-gap,0)[:,None]*outward
    return pos,vel


def pursuit_states():
    rng = np.random.default_rng(710)
    count = 84
    rest = rng.normal(0,1,(count,3)).clip(-2.3,2.3)
    sizes = rng.uniform(.055,.091,count)
    phase = rng.uniform(0,2*np.pi,count)
    rest *= [.30,.10,.095]
    origin = np.array([.83,-.35,1.03])
    cruise = np.array([-.053,.011,.006])
    start = np.array([-1.60,.06,.45])
    initial = [.94,-.04,.32]
    pos = origin+rest
    vel = np.tile(cruise,(count,1))
    alarm = np.zeros(count)
    startled = np.zeros(count)
    shark = Pose(start, unit(np.array(initial)),np.zeros(3),np.zeros(3),0,.21,0)
    for k in range(20*FPS):
        t = k*DT
        # Pursue schooling fish while visible ahead; after a miss, coast out
        # and recover a level heading with finite angular acceleration.
        centroid = np.mean(pos,axis=0)
        ahead = (centroid-shark.center) @ shark.forward
        intercept = centroid+np.mean(vel,axis=0)*.65
        if startled.mean()<.35 and ahead>.50:
            aim = unit(intercept-shark.center)
            effort = .17
            limit = .19
        else:
            # Searching turn on the far side: momentum preserves the climb
            # while the shoulder banks back toward the lost school.
            aim = unit(np.array([.94,-.22,.30]) if ahead>.0 and shark.forward[0]>.85 else np.array([-.90,-.56,-.20 if shark.center[2]>.65 else .11]))
            effort = .061
            limit = .29
        shark = swim(shark.center,shark.forward,shark.speed,shark.phase,shark.bank,
                     aim,effort,limit,DT)
        distances = np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=2)
        np.fill_diagonal(distances,np.inf)
        neighbours = np.argpartition(distances,5,axis=1)[:,:5]
        separation = np.take_along_axis(distances,neighbours,axis=1)
        gap,_ = body_gap(pos,shark)
        alarm = alarm_wave(alarm,gap,neighbours,separation,.72,DT)
        startled = np.maximum(startled,alarm)
        # A coordinated dive below the nose, followed by level swimming
        # once each individual reaches its refuge above the bottom.
        bottom = .21+rest[:,2]*.60
        escape = np.column_stack((np.full(count,-.16), -.36*(pos[:,1]-origin[1]-.5*rest[:,1]),
                                  np.clip(1.25*(bottom-pos[:,2]),-.43,.26)))
        desired = cruise + startled[:,None]*(escape-cruise)
        desired[:,1]+=.021*np.sin(.81*t+phase*.3)
        desired[:,2]+=.017*np.cos(.53*t+phase*.2)
        pos,vel = update_school(pos,vel,desired,shark,neighbours,separation,DT)
        pos[:,2] = np.maximum(pos[:,2],.06)
        gap, outward = body_gap(pos,shark)
        pos += np.maximum(.51+.57*sizes-gap,0)[:,None]*outward
        phase = phase+DT*(5+15*np.linalg.norm(vel,axis=1))
        yield Frame(shark,pos.copy(),vel.copy(),alarm.copy(),phase.copy(),sizes)
