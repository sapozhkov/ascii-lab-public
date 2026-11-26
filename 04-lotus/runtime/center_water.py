"""A finite parcel draining from the inner lotus cup, through a rib and onto a leaf.

Lengths and masses are scene units. The fixed-step model conserves a normalized
volume. Its trajectories use world coordinates, including stem deflection;
orthographic projection is applied only when reading a sample for the camera.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from day import PERIOD, RATE
from lotus_weather import dew_bead, ripples


@dataclass
class WaterState:
    position: np.ndarray
    stage: str
    volume: float
    reservoir: float
    film: float
    speed: float
    distance: float
    pond: float
    channel_index: float


def _world(g,points,t):
    p=np.asarray(points,dtype=float).copy()
    bend=g.stem_weight(p[...,2])
    sway=g.cycle.stem[round(t*RATE)%len(g.cycle.stem)]
    p[...,:2]+=bend[...,None]*sway
    return p


def _project(g,points):
    """The stored points already include the displacement normally added by project()."""
    previous=g.load
    g.load=np.zeros(2)
    try:
        return g.project(points)
    finally:
        g.load=previous


def _uv(mesh,u,v):
    u=np.clip(u,0,1)*(mesh.shape[0]-1)
    v=np.clip(v,0,1)*(mesh.shape[1]-1)
    i=np.minimum(u.astype(int),mesh.shape[0]-2)
    j=np.minimum(v.astype(int),mesh.shape[1]-2)
    a,b=(u-i)[:,None],(v-j)[:,None]
    return (1-b)*((1-a)*mesh[i,j]+a*mesh[i+1,j])+b*((1-a)*mesh[i,j+1]+a*mesh[i+1,j+1])


def _leaf_face(mesh,xy):
    """Exact planar height and gradient on the intersected rendered triangle."""
    p, q, r, s=mesh[:-1,:-1],mesh[1:,:-1],mesh[:-1,1:],mesh[1:,1:]
    # Broad phase over the XY bounding boxes of the deforming polar cells.
    low=np.minimum(np.minimum(p[...,:2],q[...,:2]),np.minimum(r[...,:2],s[...,:2]))
    high=np.maximum(np.maximum(p[...,:2],q[...,:2]),np.maximum(r[...,:2],s[...,:2]))
    cells=np.argwhere(((low<=xy)&(high>=xy)).all(-1))
    for i,j in cells:
        for a,b,c in ((p[i,j],q[i,j],r[i,j]),(q[i,j],s[i,j],r[i,j])):
            ab,ac=b-a,c-a
            basis=np.stack((ab[:2],ac[:2]),axis=1)
            if abs(np.linalg.det(basis))<1.e-9:continue
            uv=np.linalg.solve(basis,xy-a[:2])
            if min(uv)>=-1.e-7 and sum(uv)<=1.0000001:
                gradient=np.linalg.solve(basis.T,[ab[2],ac[2]])
                return a[2]+uv[0]*ab[2]+uv[1]*ac[2],gradient
    return None


class CenterWater:
    gravity=650.
    waterline=-535.
    pool_uv=(.643,.49)
    pool_span=(.182,.267)
    # Contact-angle hysteresis and rolling resistance are effective material
    # parameters, not measurements of a particular lotus cultivar.
    static_friction=.155
    moving_friction=.018

    def __init__(self,g,adhesion=1.,*,show_trail=True):
        self.events={}
        self.adhesion=adhesion
        self.show_trail=show_trail
        self.records=[]
        self.channels=[]
        self.response=[]
        self._simulate(g)

    @staticmethod
    def inner_uv(f):
        return .63*(1-f)+.071*f, .49+.25*f

    @staticmethod
    def outer_uv(f):
        return .071+.927*f, .78-.27*np.sqrt(f)

    def _channel(self,g,t):
        # The inner forward wall faces the golden crown. Its foot meets the
        # adjacent outer rib. The tangent is rebuilt as the sheets bend.
        f=np.linspace(0,1,34)
        petal13=_uv(g.tissue(13,t)[0],*self.inner_uv(f))
        f=np.linspace(0,1,52)
        petal9=_uv(g.tissue(9,t)[0],*self.outer_uv(f))
        # Unite the two contact points at the physical junction. Small differences
        # in the sampled mesh positions are smaller than the water footprint.
        contact=(petal13[-1]+petal9[0])*.5
        groove=np.concatenate([petal13[:-1],contact[None,:],petal9[1:]])
        return _world(g,groove,t)

    @staticmethod
    def _at(channel,index):
        i=min(int(index),len(channel)-2)
        f=index-i
        return channel[i]*(1-f)+channel[i+1]*f

    def _leaf(self,g,t,load):
        return _world(g,g.leaves[1].geometry(t,load)[0],t)

    def _leaf_contact(self, garden, mesh, xy, volume):
        return _leaf_face(mesh, xy)

    def _record_step(self, state):
        self.records.append(state)

    def _simulate(self,g):
        dt=1/RATE
        duration=PERIOD*RATE+1
        index=distance=speed=0.
        reservoir=.965; volume=.035; film=pond=0.
        stage='cup';pinned=True
        previous=None
        velocity=np.zeros(3)
        air = getattr(g, 'airflow', None)
        carry = getattr(g.leaves[1], 'carry', None)
        support_velocity = np.zeros(3)
        deflection=bending_velocity=0.
        self.impact_speed=0.
        for tick in range(duration):
            surface_load=deflection
            t=tick/RATE
            channel=self._channel(g,t)
            self.channels.append(channel)
            length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(channel,axis=0),axis=1))]
            if stage in ('cup','petal'):
                opening=float(g.cycle.open[round(t*RATE)%len(g.cycle.open),13,41])
                # Drainage from a finite pool into the moving front. A closed
                # whorl has negligible hydraulic conductance.
                influx=reservoir*(1-np.exp(-dt*1.72*max(0,opening)**2.1))
                reservoir-=influx
                volume+=influx
                i=min(int(index),len(channel)-2)
                tangent=channel[min(i+2,len(channel)-1)]-channel[max(0,i-1)]
                tangent/=max(np.linalg.norm(tangent),1.e-6)
                # A finite globule can overtop a shallow ridge. Its hydrostatic
                # head acts over the wetted footprint near the inner junction.
                foot_pressure=.33*volume**(1/3)*np.exp(-((index-38)/10)**2)
                downhill=-tangent[2]+foot_pressure
                # Contact line scales like radius; weight scales like volume.
                threshold=self.adhesion*self.static_friction/(volume**(2/3))
                if downhill>threshold: pinned=False
                if not pinned:
                    acceleration=.78*self.gravity*(downhill-self.moving_friction*self.adhesion/(volume**(2/3)))-2.25*speed
                    speed=max(0,speed+acceleration*dt)
                    if speed<.2 and downhill<threshold:pinned=True
                old_s=np.interp(index,np.arange(len(channel)),length)
                advance=min(speed*dt,length[-1]-old_s)
                distance+=advance
                index=float(np.interp(old_s+advance,length,np.arange(len(channel))))
                deposited=min(volume*.12,advance*.00012)
                film+=deposited;volume-=deposited
                pos=self._at(channel,index)
                if index>=len(channel)-1-1.e-6:
                    # Once the footprint runs out of tissue only a narrow neck
                    # carries the drop. Its resisting force is proportional to
                    # the remaining wetted fraction of the contact line.
                    if volume*(-tangent[2]+.28)>.082*self.adhesion*volume**(1/3):
                        stage='air1';self.events['release']=t
                        velocity=tangent*speed
                        if tick:velocity+=(channel[-1]-self.channels[-2][-1])/dt
                    else:speed=0.
                else:
                    stage='cup' if index<33 else 'petal'
            elif stage in ('air1','air2'):
                old=pos.copy()
                # Horizontal motion is inherited at detachment. The solver
                # does not steer towards the next support or a projected target.
                air_velocity = air(t,old) if air is not None else 0.
                velocity+=dt*(np.array([0.,0.,-self.gravity])-.26*(velocity-air_velocity))
                pos=old+velocity*dt
                if stage=='air1':
                    leaf=self._leaf(g,t,deflection)
                    hit=self._leaf_contact(g,leaf,pos[:2],volume)
                    if hit is not None and old[2]>=hit[0]-1. and pos[2]<=hit[0]:
                        normal=np.r_[-hit[1],1.]
                        normal/=np.linalg.norm(normal)
                        if carry is not None:
                            support_velocity = (pos-carry(pos,t,t-dt))/dt
                            velocity -= support_velocity
                        self.impact_speed=max(0,-np.dot(velocity,normal))
                        velocity=(velocity-np.dot(velocity,normal)*normal)*.17
                        pos[2]=hit[0]
                        self.events['hit']=t;stage='leaf'
                        bending_velocity=min(3.0,self.impact_speed*.007)
                if pos[2]<=self.waterline:
                    pos[2]=self.waterline;stage='pond';self.events['splash']=t
                    pond+=volume;volume=0.;velocity[:]=0
                speed=np.linalg.norm(velocity)
            elif stage=='leaf':
                if carry is not None:
                    carried = carry(pos,t-dt,t)
                    support_velocity = (carried-pos)/dt
                    pos = carried
                leaf=self._leaf(g,t,deflection)
                here=self._leaf_contact(g,leaf,pos[:2],volume)
                if here is None:
                    stage='air2';self.events['leave']=t
                else:
                    gradient=here[1]
                    slope=np.linalg.norm(gradient)
                    threshold=.04*self.adhesion/(volume**(2/3))
                    drive=-.47*self.gravity*gradient/(1+slope*slope)
                    if slope>threshold or np.linalg.norm(velocity[:2])>.5:
                        velocity[:2]+=(drive-2.7*velocity[:2])*dt
                    else:velocity[:2]=0.
                    xy=pos[:2]+velocity[:2]*dt
                    there=self._leaf_contact(g,leaf,xy,volume)
                    velocity[2]=np.dot(velocity[:2],gradient)
                    pos=np.r_[xy,here[0]+velocity[2]*dt if there is None else there[0]]
                    if there is None:
                        stage='air2';self.events['leave']=t
                    speed=np.linalg.norm(velocity)
                if stage=='air2':
                    velocity += support_velocity
            if previous is not None and stage not in ('cup','petal'):
                distance+=np.linalg.norm(pos-previous)
            previous=pos.copy()
            # Spring impulse plus the bead's static load on the receiver.
            target=.18*volume if stage=='leaf' else 0.
            bending_velocity+=(48*(target-deflection)-6.8*bending_velocity)*dt
            deflection+=bending_velocity*dt
            self.response.append(surface_load)
            self._record_step(WaterState(pos.copy(),stage,volume,reservoir,film,speed,distance,pond,index))
        self.channels=np.asarray(self.channels)

    def sample(self,t):
        f=np.clip(t*RATE+1.e-9,0,len(self.records)-1)
        i=int(f);w=f-i
        a,b=self.records[i],self.records[min(i+1,len(self.records)-1)]
        return WaterState(a.position*(1-w)+b.position*w,a.stage,
                          *(getattr(a,k)*(1-w)+getattr(b,k)*w for k in
                            ('volume','reservoir','film','speed','distance','pond','channel_index')))

    def screen(self,g,t):
        return _project(g,self.sample(t).position)[:2]

    def leaf_load(self,t):
        return float(np.interp(t*RATE,np.arange(len(self.response)),self.response))

    def _meniscus(self,rgb,g,t,s):
        # A shallow irregular puddle on the forward, concave wall of the cup.
        # Its rim, reflection and depth are all sampled on the same petal.
        size=s.reservoir**.42
        if size<.10:return
        mesh=g.tissue(13,t)[0]
        a=np.linspace(0,2*np.pi,49)
        h=(1+.08*np.cos(3*a+.4)+.035*np.sin(5*a))*size
        u=self.pool_uv[0]+self.pool_span[0]*h*np.cos(a)
        v=self.pool_uv[1]+self.pool_span[1]*h*np.sin(a)+.044*h*np.cos(a)
        pts=_project(g,_world(g,_uv(mesh,u,v),t))
        x0,y0=np.maximum(0,np.floor(pts[:,:2].min(axis=0)-3).astype(int))
        x1,y1=np.minimum([rgb.shape[1],rgb.shape[0]],np.ceil(pts[:,:2].max(axis=0)+4).astype(int))
        if x1<=x0 or y1<=y0:return
        view=rgb[y0:y1,x0:x1];dry=view.astype(float)
        contour=np.rint(pts[:,:2]-[x0,y0]).astype(np.int32)
        film=np.zeros(view.shape[:2],np.uint8)
        glint=np.zeros_like(film)
        cv2.fillPoly(film,[contour],255,cv2.LINE_AA)
        cv2.polylines(glint,[contour[5:21]],False,245,3,cv2.LINE_AA)
        cv2.polylines(glint,[contour[30:37]],False,110,2,cv2.LINE_AA)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        normal=np.linalg.lstsq(np.c_[pts[:,:2],np.ones(len(pts))],pts[:,2],rcond=None)[0]
        depth=xx*normal[0]+yy*normal[1]+normal[2]
        visibility=np.clip((depth+5-g.z[y0:y1,x0:x1])/9,0,1)
        # Outside petals shutter the reflected sky until the cup opens.
        aperture=np.mean(g.cycle.open[round(t*RATE)%len(g.cycle.open),9])
        sky_fraction=np.clip((aperture-.24)/.58,0,1)
        alpha=(visibility*film/255*.87*sky_fraction)[...,None]
        reflection=(glint/255.)[...,None]
        color=.17*dry+[20.,62.,89.]+reflection*[156.,179.,160.]
        view[:]=np.uint8(np.clip(dry*(1-alpha)+color*alpha,0,255))

    def draw(self,rgb,g,t):
        s=self.sample(t)
        self._meniscus(rgb,g,t,s)
        index=min(len(self.channels)-1,round(t*RATE))
        channel=self.channels[index]
        # The remnant film gradually loses its specular reflection as it thins.
        age=max(0.,t-self.events.get('release',t))
        reflectivity=np.exp(-age*1.3)
        n=int(s.channel_index)+1
        if self.show_trail and n>1 and reflectivity>.015:
            points=_project(g,channel[:n]);overlay=rgb.copy()
            for a,b in zip(points,points[1:]):
                x,y,_=b
                if not (0<=x<rgb.shape[1]-1 and 0<=y<rgb.shape[0]-1):continue
                if b[2]<g.z[round(y),round(x)]-12:continue
                start=tuple(np.rint(a[:2]).astype(int));end=tuple(np.rint(b[:2]).astype(int))
                cv2.line(overlay,start,end,(76,144,171),5,cv2.LINE_AA)
                cv2.line(overlay,start,end,(167,218,227),2,cv2.LINE_AA)
            alpha=.77*reflectivity
            cv2.addWeighted(overlay,alpha,rgb,1-alpha,0,dst=rgb)
        if s.stage=='pond':
            x,y=self.screen(g,t); ripples(rgb,x,y,t-self.events['splash']);return
        # Contact radius follows conserved volume. Compression at impact relaxes
        # as a damped capillary mode; the airborne drop approaches a sphere.
        aspect=.68 if s.stage in ('cup','petal','leaf') else 1.04
        if 'hit' in self.events and t>=self.events['hit'] and s.stage=='leaf':
            age=t-self.events['hit']
            aspect-=.31*np.exp(-age*4.6)*np.cos(age*16.4)
        if s.stage in ('air1','air2'):
            start=self.events['leave' if s.stage=='air2' else 'release']
            aspect=1.04-.36*np.exp(-max(0,t-start)*13)
        radius=13.4*(s.volume**(1/3))
        x,y,z=_project(g,s.position)
        cy=y-radius*aspect*.66
        if 0<=x<rgb.shape[1]-1 and 0<=cy<rgb.shape[0]-1:
            # A thin refractive film is continuously revealed as the foreground
            # folds part. Per-pixel depth feathering avoids whole-object popping.
            reach=int(np.ceil(1.85*radius))+2
            x0=max(0,int(x)-reach);x1=min(rgb.shape[1],int(x)+reach+1)
            y0=max(0,int(cy)-reach);y1=min(rgb.shape[0],int(cy)+reach+1)
            patch=rgb[y0:y1,x0:x1]
            dry=patch.copy()
            dew_bead(patch,x-x0,cy-y0,radius,aspect)
            if s.stage in ('cup','petal','leaf'):
                depth=g.z[y0:y1,x0:x1]
                vis=np.clip((z+radius*2.8-depth)/max(radius*1.6,.001),0,1)[...,None]
                patch[:]=np.uint8(patch*vis+dry*(1-vis))
