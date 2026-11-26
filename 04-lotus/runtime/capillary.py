"""Reduced free-surface model for one drop on the existing world-space trajectory."""
from dataclasses import dataclass

import cv2
import numpy as np

from center_water import CenterWater, _project
from day import RATE, base
from lotus_weather import ripples


@dataclass
class DropShape:
    strain: float
    slip: float
    neck: float
    contact: float
    normal: np.ndarray
    tangent: np.ndarray


class CapillaryWater(CenterWater):
    # The camera sees a macro view; these time and material scales are in reel units.
    radius_px = 11.8
    ring = np.linspace(0., 2*np.pi, 49)[None, :]
    latitude = np.linspace(0., np.pi, 28)[:, None]
    layer = .5 - .5*np.cos(latitude)

    def __init__(self, g, adhesion=1., *, show_trail=False):
        super().__init__(g, adhesion, show_trail=show_trail)
        self._shape_dynamics(g)

    def _support(self, g, tick, state, previous_position):
        t = tick / RATE
        if state.stage == 'leaf':
            mesh = self._leaf(g, t, self.response[tick])
            face = self._leaf_contact(g, mesh, state.position[:2], state.volume)
            normal = base.unit(np.r_[-face[1], 1.]) if face is not None else np.array([0., 0., 1.])
            carried = g.leaves[1].carry(previous_position, t-1/RATE, t)
            direction = state.position - carried
            if np.linalg.norm(direction) < .01:
                direction = np.array([0., 0., -1.]) + normal[2]*normal
            remaining = np.min(np.linalg.norm(mesh[-1] - state.position, axis=1))
        else:
            channel = self.channels[tick]
            j = min(int(state.channel_index), len(channel)-2)
            direction = channel[j+1] - channel[j]
            remaining = (np.linalg.norm(channel[j+1] - state.position)
                         + np.linalg.norm(np.diff(channel[j+1:], axis=0), axis=1).sum())
            if j < 33:
                f = state.channel_index/33
                petal = 13
                u, v = self.inner_uv(f)
            else:
                f = (state.channel_index-33)/51
                petal = 9
                u, v = self.outer_uv(f)
            mesh = g.tissue(petal, t)[0]
            a = np.clip(round(u*(len(mesh)-1)), 1, len(mesh)-2)
            # _uv() in the drainage model parameterizes both mesh axes in [0, 1].
            b = np.clip(round(v*(mesh.shape[1]-1)), 1, mesh.shape[1]-2)
            normal = base.unit(np.cross(mesh[a+1,b]-mesh[a-1,b], mesh[a,b+1]-mesh[a,b-1]))
            if normal[2] < 0.:
                normal = -normal
        tangent = base.unit(direction-normal*np.dot(direction,normal))
        return normal, tangent, remaining

    def _shape_dynamics(self, g):
        dt = 1/RATE
        strain, rate, slip, neck, contact = -.50, 0., .10, 0., 1.
        previous = self.records[0]
        vertical = np.array([0., 0., 1.])
        normal, tangent = vertical.copy(), np.array([1., 0., 0.])
        self.modes = []
        for tick, state in enumerate(self.records):
            radius = self.radius_px*max(state.volume,.015)**(1/3)/g.scale
            supported = state.stage in ('cup', 'petal', 'leaf')
            if supported:
                support_normal, downhill, remaining = self._support(g,tick,state,previous.position)
                # As part of the footprint overhangs, the rear contact line pins
                # while gravity pulls the bulk forward and below the lip.
                footprint = radius * (1.75 + slip)
                overhang = base.smooth(1.-remaining/footprint) if state.stage != 'cup' else 0.
                pendant = overhang * min(1., self.adhesion)
                rolling = np.tanh(state.speed/90.)
                target = -.70 + 1.43*pendant
                target_slip = (.22 + .37*rolling)*(1.-.75*pendant)
                target_contact = 1.-.96*pendant
                target_normal = base.unit((1.-pendant)*support_normal + pendant*vertical)
                aim = downhill
                if previous.stage == 'air1':
                    # Convert normal impact speed to an inward surface strain rate.
                    rate -= min(20., self.impact_speed/radius)
                    slip += .18
            else:
                # In free fall surface tension restores the sphere; relative air
                # pressure gives a small oblate equilibrium at higher speed.
                target = -.07*np.tanh((state.speed/360.)**2)
                target_slip = target_contact = pendant = 0.
                target_normal, aim = vertical, tangent
            # The capillary frequency scales as r^-3/2. Substrate contact damps
            # the normal mode more strongly than air does.
            omega = 13. / np.sqrt(max(state.volume,.15))
            stiffness = omega**2
            damping = (1.20 if supported else .27)*omega
            rate += (stiffness * (target - strain) - damping * rate) * dt
            strain += rate*dt
            slip += (target_slip-slip)*(1.-np.exp(-dt/.095))
            neck += (pendant-neck)*(1.-np.exp(-dt/(.045 if supported else .11)))
            contact += (target_contact-contact)*(1.-np.exp(-dt/(.042 if supported else .055)))
            normal = base.unit(normal+(target_normal-normal)*(1.-np.exp(-dt/.065)))
            along = tangent+(aim-tangent)*(1.-np.exp(-dt/.095))
            tangent = base.unit(along-normal*np.dot(along,normal))
            self.modes.append(np.r_[strain, slip, neck, contact, normal, tangent])
            previous = state
        self.modes = np.asarray(self.modes)

    def shape(self, t):
        frame = np.clip(t*RATE, 0, len(self.modes)-1)
        i = int(frame)
        mode = self.modes[i] + (frame-i)*(self.modes[min(i+1,len(self.modes)-1)]-self.modes[i])
        normal = base.unit(mode[4:7])
        tangent = base.unit(mode[7:10]-normal*np.dot(mode[7:10],normal))
        return DropShape(*mode[:4], normal, tangent)

    def surface(self, g, t):
        state, shape = self.sample(t), self.shape(t)
        bottom = -1. + .78*shape.contact
        h = bottom + (1.-bottom)*self.layer
        width = np.sqrt(np.maximum(0.,1.-h*h))
        shoulder = base.smooth((self.layer-.48)/.43)
        width *= 1.-.83*shape.neck*shoulder
        # Front and rear contact angles differ as the line slips. The trailing
        # side narrows; the leading shoulder stays round.
        co, si = np.cos(self.ring), np.sin(self.ring)
        skew = np.tanh(2.*shape.slip) * shape.contact
        along = np.exp(shape.slip-.5*shape.strain)*width*(co-.45*skew*((1.-co)*.5)**2)
        across = np.exp(-.42*shape.slip-.5*shape.strain)*width*si*(1.+.40*skew*co)
        along += shape.neck*(.6-1.30*shoulder)
        height = np.exp(shape.strain)*(h-bottom) + .43*shape.neck*shoulder
        # Integrate horizontal section areas before applying the conserved
        # parcel volume. Closing the wetted base with a disk adds no volume.
        area = .5*np.abs(np.sum(along[:,:-1]*across[:,1:]-across[:,:-1]*along[:,1:],axis=1))
        section_mean = (area[:-1]+np.sqrt(area[:-1]*area[1:])+area[1:])/3
        volume = np.sum(section_mean*np.diff(height[:,0]))
        scale = self.radius_px/g.scale*(4*np.pi*state.volume/(3*max(volume,1.e-9)))**(1/3)
        origin = state.position - .85*scale*shape.neck*shape.normal
        side = np.cross(shape.normal,shape.tangent)
        surface = (origin + scale*(along[...,None]*shape.tangent + across[...,None]*side
                                   + height[...,None]*shape.normal))
        base_center = surface[:1,:-1].mean(axis=1,keepdims=True)
        return np.concatenate([np.broadcast_to(base_center,surface[:1].shape),surface])

    def draw(self, rgb, g, t):
        state = self.sample(t)
        self._meniscus(rgb, g, t, state)
        if state.stage == 'pond':
            ripples(rgb, *self.screen(g,t), t-self.events['splash'])
            return
        surface = self.surface(g,t)
        points = _project(g,surface)
        low = np.maximum([0,0], np.floor(points[...,:2].min(axis=(0,1))-2).astype(int))
        high = np.minimum(rgb.shape[1::-1], np.ceil(points[...,:2].max(axis=(0,1))+2).astype(int))
        x0,y0 = low
        x1,y1 = high
        if x0 >= x1 or y0 >= y1:
            return
        # Render refraction through the surface geometry and a reflection of
        # the bright canopy slot. Both shift with the surface normals.
        ring_tangent = np.roll(surface[:,:-1],-1,axis=1)-np.roll(surface[:,:-1],1,axis=1)
        ring_tangent = np.concatenate([ring_tangent,ring_tangent[:,:1]],axis=1)
        normals = base.unit(np.cross(ring_tangent, np.gradient(surface,axis=0)))
        normals[0] = -self.shape(t).normal
        normals[-1] = normals[-2]
        view = _project(g,normals)-_project(g,np.zeros(3))
        nx,ny,nz = view[...,0]/g.scale,view[...,1]/g.scale,np.clip(view[...,2],0.,1.)
        radius = self.radius_px*state.volume**(1/3)
        xx = np.clip(np.rint(points[...,0]-radius*1.55*nx).astype(int),0,rgb.shape[1]-1)
        yy = np.clip(np.rint(points[...,1]-radius*1.55*ny).astype(int),0,rgb.shape[0]-1)
        transmitted = rgb[yy,xx].astype(float)*.58+[14.,26.,35.]
        fresnel = (1.-nz)**1.85
        sky = np.array([150.,191.,208.])+np.maximum(0.,-ny)[...,None]*43
        glint = np.exp(-((nx+.37)/.44)**2-((ny+.55)/.22)**2)
        rim = (.14+.80*fresnel*(.65+.35*np.maximum(0.,-ny)))[...,None]
        color = transmitted*(1-rim)+sky*rim+glint[...,None]*[218.,214.,191.]
        points[...,:2] = 2*(points[...,:2]-low)
        vertex = np.ascontiguousarray(np.concatenate([points,np.clip(color,0.,255.)],axis=-1),dtype='f')
        patch = rgb[y0:y1,x0:x1]
        size = (2*(x1-x0),2*(y1-y0))
        depths = cv2.resize(g.z[y0:y1,x0:x1],size,interpolation=cv2.INTER_NEAREST)-1.5
        raster = cv2.resize(patch,size,interpolation=cv2.INTER_NEAREST)
        base.GRID(vertex,*surface.shape[:2],*size,np.ascontiguousarray(depths,dtype='f'),raster)
        patch[:] = cv2.resize(raster,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
