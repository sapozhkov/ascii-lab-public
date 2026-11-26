"""Lotus E: a rooted plant responding to the same passing gust as its visitor."""

import numpy as np

from day import PERIOD, RATE, base
from firm import FirmGarden

from visitor_leaves import ShieldLeaf
from wind_plant import AirField, ElasticRod, rotations




class WindLeaf(ShieldLeaf):
    def attach(self,air):
        rest = super().stalk_axis(0., self.center)
        i = self.index
        self.rod = ElasticRod(rest, air, stiffness=(1.7,2.25)[i], tip_mass=(.24,.19)[i],
                              tip_drag=(.26,.22)[i], damping=(2.6,2.9)[i])
        # Pressure across an inclined shield makes it tilt on its petiole.
        n = RATE * PERIOD
        centers = self.rod.nodes[:,-1]
        velocity = (np.roll(centers,-1,axis=0)-np.roll(centers,1,axis=0))*RATE/2
        relative = (air(np.arange(n)/RATE,centers)-velocity)/100.
        # Mean resting normal from the measured x/y slope of this blade.
        slope = np.linalg.lstsq(np.stack([self.x.ravel(),self.y.ravel(),np.ones(self.x.size)],axis=1),
                                self.height.ravel(),rcond=None)[0][:2]
        normal = np.r_[-slope,1.]
        normal /= np.linalg.norm(normal)
        side = relative[:,:2]
        drive = np.sum(relative*normal,axis=1)[:,None]*side
        angles = np.zeros(2)
        rate = np.zeros(2)
        history = np.empty((n,2))
        for tick in range(3*n):
            j = tick % n
            rate += ((18.,22.)*drive[j] - 24.*angles - 4.1*rate)/RATE
            angles += rate/RATE
            history[j] = angles
        self.rotation = self.rod.tip_rotation @ rotations(history)
        self.landing_patch = np.exp(-((self.x-self.contact_xy[0])/58)**2-((self.y-self.contact_xy[1])/49)**2)

    def geometry(self,t,load):
        tick = round(t*RATE) % (PERIOD*RATE)
        center = self.rod.nodes[tick,-1]
        height = self.height-load*(22*self.landing_patch+3*self.r)
        sheet = np.stack([self.x,self.y,height],axis=-1)
        return center + sheet @ self.rotation[tick].T,center

    def stalk_axis(self,t,center):
        return self.rod.axis(t)

    def carry(self,point,t0,t1):
        """Transport a contact point between successive rigid blade poses."""
        i,j = (round(t*RATE) % (PERIOD*RATE) for t in (t0,t1))
        local = (point-self.rod.nodes[i,-1])@self.rotation[i]
        return self.rod.nodes[j,-1]+local@self.rotation[j].T


class BreezeGarden(FirmGarden):
    wingbeat = True
    leaf_class = WindLeaf

    def __init__(self,plant_only=False,wind_strength=1.):
        super().__init__(plant_only=True)
        # All subsequent geometry, including contacts, lives in world space.
        self.cycle.stem.fill(0.)
        self.airflow = AirField(wind_strength)
        a = np.linspace(1.,0.,46)
        rest = np.stack([26*np.sin(3.8*a+.08)-2+91*a**1.27,9*a-12,-815*a],axis=-1)
        exposure = self.cycle.open.mean(axis=(1,2))
        self.rod = ElasticRod(rest,self.airflow,stiffness=2.65,tip_mass=.42,
                              tip_drag=.20+.19*exposure,damping=2.5)
        for leaf in self.leaves:
            leaf.attach(self.airflow)
        self._heart_pose = None
        if not plant_only:
            self.populate()

    def crown(self,points,t):
        tick = round(t*RATE) % (PERIOD*RATE)
        return self.rod.nodes[tick,-1] + (points-self.rod.rest[-1])@self.rod.tip_rotation[tick].T

    def tissue(self,i,t):
        if i in self.cache and self.current == t:
            return self.cache[i]
        local,pigment,opening = super().tissue(i,t)
        result = self.crown(local,t).astype('f'),pigment,opening
        if self.current == t:
            self.cache[i] = result
        return result

    def heart(self,open_fraction):
        self._heart_pose = self.current
        try:
            super().heart(open_fraction)
        finally:
            self._heart_pose = None

    def project(self,p):
        if self._heart_pose is not None:
            p = self.crown(p,self._heart_pose)
        return super().project(p)

    def stem(self,t):
        axis = self.rod.axis(t)
        # Continue the planted base below the crop. Its position and tangent are fixed.
        root = self.rod.rest[0]
        direction = base.unit(self.rod.rest[0]-self.rod.rest[1])
        axis = np.vstack([root+640*direction,axis])
        u = np.clip(-self.rod.rest[:,2]/815.,0.,1.)
        radius = np.r_[11.,6.1+3*u+2.4*np.exp(-((u-.025)/.04)**2)][:,None]
        angle = np.linspace(0.,2*np.pi,15)[None,:]
        rings = np.stack([radius*np.cos(angle),radius*np.sin(angle),np.zeros((len(axis),15))],axis=-1)
        tint = .38+.54*(.5+.5*np.cos(angle-2.2))+.07*np.sin(angle*5)
        self.mesh(axis[:,None,:]+rings,np.array([99,121,59])*tint[...,None])
        self.leaves[0].draw(self,t,0.)
        self.leaves[1].draw(self,t,self.leaf_load(t))
