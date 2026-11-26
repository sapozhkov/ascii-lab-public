"""A pollinator that finds a moving lip, walks its surface and leaves after probing.

Positions in the flight integrator are inertial. VisitGarden.project accepts
undeformed plant coordinates and itself adds stem bending; convert once at the
render boundary. The clock is the 20-second accelerated observation in this reel.
"""
from math import atan2, cos, exp, pi, sin

import numpy as np

from day import PERIOD, RATE, wind
from lotus_weather import sample_patch
from visitor import Pose, Visitor


def limit(vector, maximum):
    return vector * min(1., maximum / max(float(np.linalg.norm(vector)), 1.e-7))


def bend_position(g, point, t, sign):
    shifted = point.copy()
    stem = g.cycle.stem[round(t*RATE) % len(g.cycle.stem)]
    shifted[:2] += sign*stem*g.attachment_weight(point[2])
    return shifted


def angle_error(target, angle):
    return (target-angle+pi) % (2*pi) - pi


class Forager(Visitor):
    rate = 90
    departure_solver = None
    landing_uv = np.array([.88, .32])
    # Two patches along the dry petal toward the anthers. The insect changes
    # sites when its proboscis has sampled a patch, not when a clock strikes.
    patches = np.array([[.68,.26],[.46,.66]])

    def __init__(self,g):
        self.story = 'catch'
        self.variant = g.variant
        self.start,self.end = 0.,float(PERIOD)
        self.thorax = self._coat(34,46,'thorax')
        self.abdomen = self._coat(46,46,'abdomen')
        self.head = self._coat(22,26,'head')
        self.eye = self._coat(16,20,'eye')
        self.events = {}
        self._integrate(g)
        self.wings = None
        if getattr(g, 'wingbeat', False):
            from bee_wings import Wingbeat
            self.wings = Wingbeat(self)

    def _integrate(self,g):
        dt = 1/self.rate
        ticks = PERIOD*self.rate+1
        self.body = np.empty((ticks,3))
        self.reference = np.empty((ticks,3))
        self.uv = np.empty((ticks,2))
        self.attitude = np.empty((ticks,4))
        self.gait = np.zeros((ticks,2))
        self.states = []
        pos = np.array([-1020.,245.,188.])
        vel = np.array([140.,-9.,7.])
        thrust = np.zeros(3)
        yaw,pitch,turn_speed = .05,0.,0.
        believed = None
        seen_v = np.zeros(3)
        state = 'arrival'
        parallax = 0.
        grip = 0.
        foot_uv = self.landing_uv.copy()
        site = 0
        dose = 0.
        nectar = getattr(g, 'nectar_per_site', (1.03, 1.03))
        tongue = 0.
        stride = 0.
        walk_v = 0.
        pitch_trim = 0.
        crawl_heading = 0.
        sampled_step = None
        for i in range(ticks):
            t=i*dt
            step=round(t*RATE)
            if step!=sampled_step:
                sheet = g.tissue(10,step/RATE)[0]
                sampled_step=step
            foot=bend_position(g,sample_patch(sheet,*foot_uv),t,1)
            if believed is None:
                believed=foot.copy()
            past=believed.copy()
            believed+=(foot-believed)*(1-exp(-dt/.12))
            seen_v+=( (believed-past)/dt - seen_v)*(1-exp(-dt/.16))
            delta=believed+np.array([0,0,25.])-pos
            horizontal=np.linalg.norm(delta[:2])
            altitude=-delta[2]
            if hasattr(g, 'airflow'):
                air = g.airflow(t, pos)
            else:
                breeze=wind(t,'dew')
                air=np.array([35*breeze,11*wind(t-.32,'dew'),3*breeze])
            on_ground=state in ('grasp','walk','probe','unload')
            desired_yaw=yaw
            if not on_ground:
                if state=='arrival':
                    target=delta+[0,42,139]
                    desired_v=limit(target*1.25,185)
                    desired_yaw=atan2(desired_v[1],desired_v[0])
                    if horizontal<252:
                        state='inspect'
                        self.events['approach']=t
                        last_bearing=atan2(-delta[1],-delta[0])
                elif state=='inspect':
                    # Maintain an angular image size while slipping sideways
                    # to estimate the depth of the exposed rim by parallax.
                    outward=-delta[:2]/max(horizontal,1)
                    bearing=atan2(outward[1],outward[0])
                    parallax+=abs(angle_error(bearing,last_bearing))
                    last_bearing=bearing
                    tangent=np.array([outward[1],-outward[0]])
                    sideways=152*tangent+(149-horizontal)*1.85*outward
                    desired_v=np.r_[limit(sideways,180),1.9*(108-altitude)]
                    desired_v+=seen_v*.6
                    desired_yaw=atan2(delta[1],delta[0])
                    if parallax>1.08 and horizontal<191:
                        state='brake'
                        self.events['brake']=t
                elif state=='brake':
                    # Arrest sideways motion with a body-length of clearance.
                    desired_v=seen_v+limit((delta+[0,0,66])*2.1,146)
                    desired_yaw=atan2(delta[1],delta[0]) if horizontal>22 else crawl_heading
                    if horizontal<31 and altitude<92:
                        state='descend'
                        self.events['descend']=t
                elif state=='descend':
                    desired_v=seen_v+limit(delta*3.4,91)
                    # Keep the head facing the base while legs extend.
                    toward=sample_patch(sheet,.69,.30)-sample_patch(sheet,*self.landing_uv)
                    desired_yaw=atan2(toward[1],toward[0])
                    if np.linalg.norm(delta)<8.5 and np.linalg.norm(vel-seen_v)<53:
                        state='grasp'
                        self.events['land']=t
                        crawl_heading=desired_yaw
                elif state=='launch':
                    # A short push from the legs clears both the cup and the head.
                    desired_v=np.array([-44.,96.,162.])+air*.18
                    desired_yaw=atan2(foot[1]-pos[1],foot[0]-pos[0])
                    if pos[2]-foot[2]>100:
                        state='orientation'
                        self.events['clear']=t
                        origin_bearing=atan2(pos[1]-foot[1],pos[0]-foot[0])
                elif state=='orientation':
                    outward=-delta[:2]/max(horizontal,1)
                    tangent=np.array([-outward[1],outward[0]])
                    desired_v=np.r_[145*tangent+70*outward,45.]
                    desired_yaw=atan2(delta[1],delta[0])
                    current=atan2(outward[1],outward[0])
                    if abs(angle_error(current,origin_bearing))>.8:
                        state='depart'
                        self.events['depart']=t
                else:
                    desired_v=np.array([-278.,77.,83.])+air*.32
                    desired_yaw=atan2(desired_v[1],desired_v[0])

                # Wing thrust takes time to change. Drag responds immediately
                # to relative air velocity; gusts can move a hovering thorax.
                drive=limit((desired_v-vel)*4.6+.47*desired_v,760.)
                thrust+=(drive-thrust)*(1-exp(-dt/.145))
                acc=thrust-.47*(vel-air)
                vel+=acc*dt
                pos+=vel*dt
                pitch_trim=np.clip(-.0016*(cos(yaw)*acc[0]+sin(yaw)*acc[1])
                                   + .0013*vel[2],-.35,.36)
                tongue*=exp(-dt/.14)
                grip*=exp(-dt/.062)
            else:
                # Steps along the actual surface metric. The two tripods
                # transfer support at different phases of each short stride.
                target_uv=self.patches[site]
                target=bend_position(g,sample_patch(sheet,*target_uv),t,1)
                run=target-foot
                remaining=np.linalg.norm(run)
                if state=='grasp':
                    grip+=(1-grip)*(1-exp(-dt/.085))
                    if grip>.98:
                        state='walk'
                        self.events['walk']=t
                if state=='walk':
                    desired_walk=min(37.,remaining*4)
                    walk_v+=(desired_walk-walk_v)*(1-exp(-dt/.18))
                    stride+=dt*walk_v/15
                    step_gain=.82+.32*sin(2*pi*stride)
                    fraction=min(1.,walk_v*dt*step_gain/max(remaining,.001))
                    foot_uv+=(target_uv-foot_uv)*fraction
                    if remaining>2:
                        crawl_heading=atan2(run[1],run[0])
                    if remaining<2.7:
                        state='probe'
                        dose=0.
                        self.events['probe'+str(site+1)]=t
                if state=='probe':
                    # The head dips and takes a sample; the finite patch
                    # is depleted as long as the mouth is in contact.
                    tongue_target=.5+.5*cos(2*pi*dose*1.28)
                    tongue+=(tongue_target-tongue)*(1-exp(-dt/.09))
                    dose+=dt*(.65+.35*tongue)
                    if dose>nectar[site]:
                        if site+1<len(self.patches):
                            site+=1
                            state='walk'
                        else:
                            state='unload'
                else:
                    tongue*=exp(-dt/.14)
                if state=='unload':
                    grip*=exp(-dt/.12)
                    if grip<.12:
                        self.events['lift']=t
                        state='launch'
                        thrust=np.array([0.,10.,380.])
                        vel+=np.array([-4.,12.,27.])
                rocking=(1.9*sin(2*pi*stride)**2 if state=='walk' else 0.)
                desired_pos=foot+np.array([2.7*tongue*cos(yaw),2.7*tongue*sin(yaw),
                                           23+rocking-4.9*tongue])
                # The compliance of the legs cushions touchdown and closes
                # the last few units of clearance without a position snap.
                vel+=(94*(desired_pos-pos)-15*(vel-seen_v))*dt
                pos+=vel*dt
                desired_yaw=crawl_heading+.14*tongue
                pitch_trim=-.12-.38*tongue+.10*sin(2*pi*stride) if state=='walk' else -.12-.38*tongue

            # A single short turn precedes drift into the next direction.
            error=angle_error(desired_yaw,yaw)
            turn_speed+=(25*error-5.8*turn_speed)*dt
            yaw+=np.clip(turn_speed,-5.5,5.5)*dt
            pitch+=(pitch_trim-pitch)*(1-exp(-dt/.10))
            self.body[i]=pos
            self.reference[i]=foot
            self.uv[i]=foot_uv
            self.attitude[i]=[yaw,pitch,grip,tongue]
            if state=='walk':
                self.gait[i]=stride,min(walk_v/24,1.)
            self.states.append(state)
            if state == 'orientation' and self.departure_solver is not None:
                self.departure_solver(g, i, vel, thrust, turn_speed)
                break

    def leg_pose(self,pose,t,side,pair):
        i=round((t%PERIOD)*self.rate)
        state=self.states[i]
        reach=pose.contact
        swing=lift=0.
        if state in ('descend','grasp'):
            # The leading pair feels for the rim while the wings still carry
            # the body. The middle and hind pairs open when the rim is closer.
            gap=np.linalg.norm(pose.center-pose.foot)
            flare=np.clip((85-14*pair-gap)/38,0,1)
            reach=max(reach,flare)
            swing=(3-pair)*flare
        if state=='walk':
            stride,intensity=self.gait[i]
            phase=2*pi*stride+(pi if (side<0)^(pair==1) else 0.)
            swing=6*intensity*sin(phase)
            lift=7*intensity*max(0.,cos(phase))
        return reach,swing,lift

    def contact(self,t):
        return float(self.attitude[round(float(t)*self.rate)%(len(self.attitude)-1),2])

    def pose(self,g,t):
        t=float(t)%PERIOD
        i=round(t*self.rate)
        yaw,pitch,grip,tongue=self.attitude[i]
        foot=sample_patch(g.tissue(10,t)[0],*self.uv[i])
        reference=bend_position(g,self.reference[i],t,-1)
        center=bend_position(g,self.body[i],t,-1)+(foot-reference)*grip
        return Pose(center,foot,float(yaw),float(pitch),float(grip),float(tongue),0.,grip<.86)
