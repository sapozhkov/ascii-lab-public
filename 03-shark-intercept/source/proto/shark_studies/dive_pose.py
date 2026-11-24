"""Reduced swimmer dynamics in shark lengths and seconds.

Torques, hydrodynamic gains and muscle strength are staging coefficients.
The posterior travelling wave, pectoral steering and ram approach follow
Ferry & Lauder (1996), Hoffmann et al. (2019), and Motta et al. (1997).
"""
from dataclasses import dataclass
import math

import numpy as np

from .encounter_water import unit


AXIAL = np.linspace(.505, -.505, 27)
SEGMENT = AXIAL[0] - AXIAL[1]
STATION = (np.arange(len(AXIAL)-1) + .5) / (len(AXIAL)-1)


@dataclass
class SwimPose:
    center: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    phase: float
    speed: float
    bank: float
    spine: np.ndarray             # fixed-length backbone, nose to caudal tip, body space
    tangent: np.ndarray
    fin_angle: np.ndarray         # port and starboard elevation change
    gape: float
    thrust: float
    tail_speed: float


def centreline(yaw, pitch):
    """Integrate the link directions with a fixed arc-length step."""
    segments = np.column_stack((np.cos(yaw)*np.cos(pitch),
                                np.sin(yaw)*np.cos(pitch), np.sin(pitch)))
    spine = np.vstack((np.zeros(3), -SEGMENT*np.cumsum(segments,axis=0)))
    spine -= spine[len(spine)//2]
    tangent = unit(np.vstack((segments[0], segments[1:]+segments[:-1], segments[-1])))
    return spine, tangent


def muscle_wave(phase, drive, turn):
    # The head counter-rotates slightly; the flexible peduncle carries a
    # much larger wave. Turn curvature appears first at the shoulders.
    recoil = (.07+.04*drive)*np.sin(phase+.5)*np.exp(-4*STATION)
    wave = (.079*STATION + .59*STATION**2)*(.75+.73*drive)
    bend = -turn*(STATION-.43)*1.28
    return recoil + wave*np.sin(phase-4.55*STATION) + bend


def caudal_reaction(sweep, slope, advance):
    """Axial reaction per unit mass from a reduced momentum balance at the tail."""
    return .75 * (sweep**2 - (advance * slope)**2)


def water_acceleration(velocity, forward, thrust, speed):
    """Combine axial tail thrust with hull drag and resistance to sideslip."""
    lateral = velocity - forward * (velocity @ forward)
    return thrust * forward - 2.45 * speed * velocity - 2.9 * lateral


class Swimmer:
    def __init__(self, center, forward, speed=.20):
        self.center = np.array(center, dtype=float)
        direction = unit(np.array(forward, dtype=float))
        self.azimuth = math.atan2(direction[1],direction[0])
        self.pitch = math.asin(np.clip(direction[2],-.999,.999))
        self.velocity = speed*direction
        self.speed = speed
        self.phase = .52
        self.drive = .22
        self.yaw_rate = self.pitch_rate = self.roll = self.roll_rate = 0.
        self.fins = np.zeros(2)
        self.gape = 0.
        self.thrust = .08
        self.tail_speed = 0.
        self.flow_weight = 0.
        target = muscle_wave(self.phase,self.drive,0.)
        self.joints = np.diff(np.r_[0.,target])
        self.joint_rates = np.zeros_like(self.joints)
        self.spine, self.tangent = centreline(target,np.zeros_like(target))
        self.pose = self.snapshot()

    def snapshot(self):
        cs,sn = math.cos(self.azimuth),math.sin(self.azimuth)
        cp,sp = math.cos(self.pitch),math.sin(self.pitch)
        forward = np.array([cs*cp,sn*cp,sp])
        side = np.array([sn,-cs,0.])
        dorsal = np.cross(side, forward)
        cr,sr = math.cos(self.roll),math.sin(self.roll)
        return SwimPose(self.center.copy(),forward,side*cr-dorsal*sr,dorsal*cr+side*sr,
                        self.phase,self.speed,self.roll,self.spine.copy(),self.tangent.copy(),
                        self.fins.copy(),self.gape,self.thrust,self.tail_speed)

    def step(self, aim, effort, dt, gape=0., *, use_tail_flow=False, burst=0., coordinated_bank=False):
        aim = unit(np.array(aim,dtype=float))
        pieces = max(1,math.ceil(dt*96))
        for _ in range(pieces):
            self.advance(aim,float(np.clip(effort,0,1)),dt/pieces,float(np.clip(gape,0,1)),
                         use_tail_flow, burst, coordinated_bank)
        self.pose = self.snapshot()
        return self.pose

    def advance(self, aim, effort, dt, gape, use_tail_flow=False, burst=0., coordinated_bank=False):
        response = 2.5 + 5.*burst
        self.drive += dt*response*(effort+burst-self.drive)
        self.phase += dt*2*np.pi*(.44+.73*min(self.drive, 1.15))
        yaw_error = self.azimuth-math.atan2(aim[1],aim[0])
        yaw_error = math.atan2(math.sin(yaw_error),math.cos(yaw_error))
        pitch_error = math.asin(np.clip(aim[2],-.98,.98))-self.pitch
        rudder = np.clip(2.15*yaw_error,-1.,1.)
        elevator = np.clip(2.50*pitch_error,-1.,1.)
        # Inner pectoral lowers in a yaw turn; both alter incidence in pitch.
        targets = .31*elevator - .39*np.maximum(np.array([-rudder,rudder]),0.)
        self.fins += dt*5.2*(targets-self.fins)
        fin_asymmetry = self.fins[0]-self.fins[1]
        authority = .52+1.8*min(self.speed,.65)**2+.20*self.drive
        self.yaw_rate += dt*(authority*(1.85*rudder+1.55*fin_asymmetry)-2.15*self.yaw_rate)
        self.pitch_rate += dt*(authority*(1.70*elevator+1.25*self.fins.mean())-2.3*self.pitch_rate)
        self.azimuth -= self.yaw_rate*dt
        self.pitch = np.clip(self.pitch+self.pitch_rate*dt,-1.13,1.13)
        if coordinated_bank:
            # Fin deflection starts the bank as the shark commits to a turn.
            # Positive yaw rate and bank both refer to the starboard turn.
            bank_goal = .38*self.yaw_rate + .68*fin_asymmetry
            self.roll_rate += dt*(22.*(bank_goal-self.roll)-8.8*self.roll_rate)
        else:
            self.roll_rate += dt*(9.0*(.66*self.yaw_rate-self.roll)-4.7*self.roll_rate)
        self.roll += dt*self.roll_rate
        self.gape += (1-math.exp(-dt*(18 if gape>self.gape else 23)))*(gape-self.gape)

        target = muscle_wave(self.phase,self.drive,self.yaw_rate)
        neutral = np.diff(np.r_[0.,target])
        # Each angular spring supplies muscle torque against angular inertia
        # and damping. The softer posterior joints also lag their drive.
        omega = 18.4-3.8*STATION
        torque = omega**2*(neutral-self.joints)-1.22*omega*self.joint_rates
        self.joint_rates += dt*torque
        self.joints += dt*self.joint_rates
        prev = self.spine[-1,1]
        vertical_bend = -.76*self.pitch_rate*(STATION-.46)
        spine,tangent = centreline(np.cumsum(self.joints),vertical_bend)
        self.tail_speed = (spine[-1,1]-prev)/max(dt,1e-9)
        self.spine,self.tangent = spine,tangent
        forward = np.array([math.cos(self.azimuth)*math.cos(self.pitch),
                            math.sin(self.azimuth)*math.cos(self.pitch),math.sin(self.pitch)])
        advance_speed = self.velocity @ forward
        # Keep the inherited cruising load as the boundary condition. In the
        # chase, subtract the flux carried along an inclined fin as it advances.
        self.flow_weight += dt*8.*(float(use_tail_flow)-self.flow_weight)
        tail_tangent = tangent[-1]
        tail_slope = tail_tangent[1] / max(.35, tail_tangent[0])
        tail_load = .62*self.tail_speed**2
        reaction = caudal_reaction(self.tail_speed, tail_slope, advance_speed)
        tail_load += self.flow_weight * (reaction - tail_load)
        # The tail load follows the sweep within a fraction of a beat. The
        # shark's mass still smooths the change in forward velocity.
        load_response = 3. + 6.*self.flow_weight
        self.thrust += dt*load_response*(tail_load-self.thrust)
        self.velocity += dt*water_acceleration(self.velocity, forward, self.thrust, self.speed)
        self.speed = np.linalg.norm(self.velocity)
        self.center += dt*self.velocity


def deform(points, pose):
    """Attach each landmark to its vertebra and local cross-section."""
    p = np.asarray(points,dtype=float)
    x,y,z = p.T
    index = np.clip((AXIAL[0]-x)/SEGMENT,0.,len(AXIAL)-1.000001)
    joint = index.astype(int)
    blend = (index-joint)[:,None]
    spine = pose.spine[joint]+blend*(pose.spine[joint+1]-pose.spine[joint])
    tangent = unit(pose.tangent[joint]+blend*(pose.tangent[joint+1]-pose.tangent[joint]))
    side = unit(np.column_stack((-tangent[:,1],tangent[:,0],np.zeros_like(x))))
    dorsal = np.cross(tangent,side)
    # Cross sections of each pectoral rotate about its embedded root.
    pectoral = (x>.11)&(x<.29)&(np.abs(y)>.065)
    radius = np.maximum(np.abs(y)-.065,0.)
    angle = np.where(y<0,pose.fin_angle[0],pose.fin_angle[1])*pectoral
    y = y - np.sign(y)*radius*(1-np.cos(angle))
    z = z + radius*np.sin(angle)
    # Water loads the unsupported ventral caudal lobe as the terminal lobe
    # sweeps. Twisting its rays introduces a lag and preserves ray length.
    ventral_depth = np.maximum(-z,0.)*np.clip((-x-.245)/.065,0,1)
    twist = np.clip(-.72*pose.tail_speed,-.80,.80)
    y = y + ventral_depth*np.sin(twist)
    z = z + ventral_depth*(1-np.cos(twist))
    # Mandibular depression and a little lateral flare open the ventral
    # mouth while the snout retains its shape.
    jaw = np.clip((x-.335)/.115,0,1)*np.clip((-z-.007)/.019,0,1)
    z = z - .063*pose.gape*jaw
    return spine + y[:,None]*side + z[:,None]*dorsal


def world_points(points,pose):
    rotation = np.array([pose.forward,pose.right,pose.up])
    return pose.center + np.einsum('ij,jk->ik',deform(points,pose),rotation)


def mouth(pose):
    # Mid-aperture between the upper rostrum and the moving mandible.
    lips = world_points([[.451,0,-.006],[.451,0,-.029]],pose)
    return lips.mean(axis=0)
