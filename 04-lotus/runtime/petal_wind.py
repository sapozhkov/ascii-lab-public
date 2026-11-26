"""Individually loaded petals above the checkpoint B stalk and leaf simulation."""

import numpy as np

from breeze import BreezeGarden
from day import base, PERIOD, RATE



# Quadrature patches for integrating air pressure across each material surface.
ROWS = np.array([8, 21, 36, 51, 65])
COLS = np.array([5, 16, 28, 40])
MID = base.V.size // 2


def frame_axes(mesh):
    rib = mesh[:, MID]
    tangent = base.unit(np.gradient(rib, axis=0))
    width = mesh[:, -2] - mesh[:, 1]
    width = width - np.sum(width*tangent, axis=-1, keepdims=True)*tangent
    # Tapering can collapse the final transverse sample to one point.
    width[[0,-1]] = width[[1,-2]]
    return rib, tangent, base.unit(width)


def rotation(axis, angle):
    """One orthogonal rotation per rib station."""
    x,y,z = axis.T
    zero = np.zeros_like(x)
    cross = np.stack([zero,-z,y,z,zero,-x,-y,x,zero],axis=-1).reshape(-1,3,3)
    return (np.eye(3) + np.sin(angle)[:,None,None]*cross
            + (1-np.cos(angle))[:,None,None]*(cross@cross))


class PetalGarden(BreezeGarden):
    def __init__(self, plant_only=False, wind_strength=1.):
        super().__init__(plant_only=True, wind_strength=wind_strength)
        u = base.U[:,0]
        self.basal_mode = base.smooth((u-.10)/.67)
        self.tip_mode = base.smooth((u-.48)/.52)
        self.torsion_mode = base.smooth((u-.16)/.84)
        self.angles = self.petal_response()
        self.cache.clear()
        if not plant_only:
            self.populate()

    def petal_response(self):
        """Integrate bending and torsion under the same local gust as the supporting stalk."""
        samples = 12
        times = np.arange(PERIOD*samples)/samples
        count = len(self.sheets)
        moments = np.empty((len(times),count,3))
        length = np.array([np.linalg.norm(np.diff(o[:,MID],axis=0),axis=1).sum()
                           for _,o in self.sheets])
        for k,t in enumerate(times):
            # Sampling the opening shapes before elastic wind deflection is the
            # small-deflection approximation in this reduced strip model.
            meshes = np.array([BreezeGarden.tissue(self,i,t)[0] for i in range(count)])
            p = meshes[:,ROWS[:,None],COLS]
            du = (meshes[:,(ROWS+1)[:,None],COLS]-meshes[:,(ROWS-1)[:,None],COLS])*.5
            dv = (meshes[:,ROWS[:,None],COLS+1]-meshes[:,ROWS[:,None],COLS-1])*.5
            area_vector = np.cross(du,dv)
            area = np.linalg.norm(area_vector,axis=-1)*(70/len(ROWS))*(45/len(COLS))
            normal = base.unit(area_vector)
            air = self.airflow(t,p)
            tick = round(t*RATE) % (PERIOD*RATE)
            # Subtract the velocity of this material point due to crown motion.
            tip = self.rod.nodes[tick,-1]
            local = (p-tip)@self.rod.tip_rotation[tick]
            prev,nxt = (tick-1)%(PERIOD*RATE),(tick+1)%(PERIOD*RATE)
            carrier = (self.rod.nodes[nxt,-1] - self.rod.nodes[prev,-1]
                       + local@(self.rod.tip_rotation[nxt]-self.rod.tip_rotation[prev]).T)*(.5*RATE)
            relative = (air-carrier)/100.
            # A blade upwind of this patch shelters it. Clearance is measured
            # perpendicular to the flow using the neighboring blade's width.
            centers = p.mean(axis=(1,2))
            delta = p[:,:,:,None,:]-centers
            upstream = np.sum(delta*base.unit(relative)[:,:,:,None,:],axis=-1)
            separation = np.maximum(0.,np.sum(delta*delta,axis=-1)-upstream**2)
            shade = np.exp(-separation/(56.**2)) * base.smooth(upstream/40.) * np.exp(-np.maximum(0.,upstream)/330.)
            shade[np.arange(count),:,:,np.arange(count)] = 0.
            shelter = 1.-.70*shade.max(axis=-1)
            incidence = np.sum(relative*normal,axis=-1)
            pressure = normal*(incidence*abs(incidence))[...,None] + .11*relative*np.linalg.norm(relative,axis=-1,keepdims=True)
            force = pressure*(area*shelter)[...,None]
            # Neighboring blades brace the closed bud. Moment arms come from
            # the current geometry, so the same gust loads each blade differently.
            opening = np.clip(self.cycle.open[tick].mean(axis=1),0.,1.)
            force *= (.04+.96*opening[:,None,None,None]**1.5)
            for i,mesh in enumerate(meshes):
                rib,tangent,lateral = frame_axes(mesh)
                torque = np.cross(p[i]-rib[7],force[i])
                distal = np.cross(p[i]-rib[33],force[i])
                twist = np.cross(p[i]-rib[ROWS,None],force[i])
                moments[k,i] = [np.sum(torque*lateral[ROWS,None]),
                                np.sum(distal[2:]*lateral[ROWS[2:],None]),
                                np.sum(twist*tangent[ROWS,None])]
        # Torque divided by stiffness gives an equilibrium angle. Modest
        # geometrical stiffening limits large bends. Long ribs respond more slowly.
        compliance = (length/285.)[:,None] * np.array([2.3e-7,2.9e-7,1.05e-6])
        limit = np.deg2rad([32.,21.,24.])
        target = limit*np.tanh(moments*compliance/limit)
        omega = (285./length[:,None])**1.5*np.array([5.2,8.5,7.])
        zeta = np.array([.52,.42,.50])
        steps = PERIOD*RATE
        hist = np.empty((steps,count,3))
        q = np.zeros((count,3))
        speed = np.zeros_like(q)
        for n in range(3*steps):
            j = n % steps
            f = j*samples/RATE
            a = int(f)
            equilibrium = target[a] + (f-a)*(target[(a+1)%len(times)]-target[a])
            speed += (omega**2*(equilibrium-q)-2*zeta*omega*speed)/RATE
            q += speed/RATE
            hist[j] = q
        return hist

    def tissue(self,i,t):
        if i in self.cache and self.current == t:
            return self.cache[i]
        mesh,pigment,opening = super().tissue(i,t)
        rib,tangent,width = frame_axes(mesh)
        basal,tip,twist = self.angles[round(t*RATE)%(PERIOD*RATE),i]
        bend = rotation(width,basal*self.basal_mode+tip*self.tip_mode)
        torsion = rotation(tangent,twist*self.torsion_mode)
        segments = np.einsum('sij,sj->si',bend[1:],np.diff(rib,axis=0))
        spine = np.concatenate([rib[:1],rib[:1]+np.cumsum(segments,axis=0)],axis=0)
        # Each midrib segment and each transverse section retains its length.
        offset = mesh-rib[:,None]
        out = spine[:,None]+np.einsum('sij,svj->svi',bend@torsion,offset)
        result = out.astype('f'),pigment,opening
        if self.current == t:
            self.cache[i] = result
        return result
