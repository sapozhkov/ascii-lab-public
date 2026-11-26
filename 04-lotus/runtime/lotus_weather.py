"""Local water, pollen and canopy light for the twenty-second lotus studies.

All locations are resolved for an absolute scene time. Seeking to a single
frame gives the same image as stepping through the whole reel.
"""
from math import cos, exp, pi, sin

import cv2
import numpy as np

from living import ANG, FIL_LENGTH, FIL_PUSH, FIL_RADIUS, FIL_TILT


RELEASE = 8.45
IMPACT = 9.60
WIDTH, HEIGHT = 405, 720


def ease(value):
    x = np.clip(value, 0., 1.)
    return x * x * (3. - 2. * x)


def sample_patch(mesh, u, v):
    """Bilinear position on the tissue, including at the collapsed tip."""
    s = min(u * (len(mesh) - 1), len(mesh) - 1.000001)
    j = v * (mesh.shape[1] - 1)
    i, k = int(s), int(j)
    a, b = s-i, j-k
    return ((1-a)*((1-b)*mesh[i,k]+b*mesh[i,k+1]) +
            a*((1-b)*mesh[i+1,k]+b*mesh[i+1,k+1]))


def projected_at(garden, point, birth):
    """Capture the wind-bent stem at release, leaving free particles in space."""
    current = garden.load
    loads = garden.sway
    index = min(len(loads)-1, max(0, round(birth*120)))
    garden.load = loads[index]
    try:
        return garden.project(point)
    finally:
        garden.load = current


def dew_bead(frame, x, y, radius, stretch):
    """Refracted tissue under the bead, reflected sky and a single specular point."""
    h, w = frame.shape[:2]
    rx, ry = radius / stretch**.5, radius * stretch
    x0, x1 = max(0, int(x-rx-2)), min(w, int(x+rx+3))
    y0, y1 = max(0, int(y-ry-2)), min(h, int(y+ry+3))
    if x0>=x1 or y0>=y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    nx, ny = (xx-x)/rx, (yy-y)/ry
    rr = nx*nx+ny*ny
    nz = np.sqrt(np.maximum(0, 1-rr))
    fresnel = (1-nz)**1.65
    sample_x = np.clip(np.rint(x-nx*radius*1.25), 0, w-1).astype(int)
    sample_y = np.clip(np.rint(y+radius-ny*radius*1.8), 0, h-1).astype(int)
    tissue = frame[sample_y,sample_x].astype('f')
    sky = np.array([134,177,188], dtype='f') + np.maximum(0, -ny[...,None])*58
    dark = tissue*.54 + np.array([18,30,37])
    rim = np.maximum(0, .38-nx*.5+ny*.8)*fresnel
    color = dark + (sky-dark)*np.clip(.18+rim[...,None]*1.2,0,1)
    specular = np.exp(-((nx+.31)/.37)**2-((ny+.47)/.29)**2)
    reflected = np.exp(-((nx-.23)/.5)**2-((ny-.69)/.14)**2)
    color += specular[...,None]*[210,214,196]+reflected[...,None]*[92,110,97]
    coverage = np.clip((1-np.sqrt(rr))*min(rx,ry)+.5,0,1)[...,None]
    view = frame[y0:y1,x0:x1]
    view[:] = np.uint8(np.clip(view*(1-coverage)+color*coverage,0,255))


def ripples(frame, x, y, age, far_reflection=0.):
    """Dispersing capillary waves, seen as short curved reflections on dark water."""
    if not 0 <= age < 5.4:
        return
    h,w = frame.shape[:2]
    distance = 85 * age**.73
    x0,x1 = max(0,int(x-distance-12)), min(w,int(x+distance+13))
    y0,y1 = max(0,int(y-distance*.24-10)), min(h,int(y+distance*.24+11))
    if x0>=x1 or y0>=y1:
        return
    yy,xx=np.mgrid[y0:y1,x0:x1]
    dx,dy=xx-x,(yy-y)/.23
    radial=np.hypot(dx,dy)
    angle=np.arctan2(dy,dx)
    glints=.22+.78*(.5+.5*np.cos(angle*2+.55))**2
    glints*=.60+.40*np.cos(angle*5-.7)**2
    # The far faces of the crests can reflect the sky towards the camera.
    if far_reflection:
        glints = np.minimum(1., glints + far_reflection*np.maximum(0., -np.sin(angle))**2)
    waves=np.zeros_like(radial)
    for delay, amplitude in ((0,1.),(.26,.76),(.58,.47),(.93,.24)):
        dt=age-delay
        if dt<=0:
            continue
        front=85*dt**.73
        width=3.6+.020*front
        waves+=amplitude*np.exp(-((radial-front)/width)**2)*min(1,dt/.12)
    waves*=glints*exp(-age*.52)*ease((5.4-age)/1.)
    view=frame[y0:y1,x0:x1]
    view[:]=np.uint8(np.clip(view+waves[...,None]*[125,178,184],0,255))


def anther_tips():
    """Open anther endpoints in the coordinate system of Garden.heart."""
    ex,ey=np.cos(ANG),np.sin(ANG)
    base=np.stack([75*ex*FIL_RADIUS-20, 51*ey*FIL_RADIUS+7,
                   83+FIL_RADIUS*(30*ex-7*ey)],-1)
    direction=np.stack([.7*FIL_PUSH*np.cos(ANG+FIL_TILT),
                        .7*FIL_PUSH*np.sin(ANG+FIL_TILT),.93+0*ex],-1)
    return base+FIL_LENGTH[:,None]*direction


TIPS=anther_tips()


class Weather:
    def __init__(self,scene):
        if scene not in ('dew','gust','sun'):
            raise ValueError(f'Unknown environment: {scene}')
        self.scene=scene
        rng=np.random.default_rng({'dew':1207,'gust':1303,'sun':2103}[scene])
        self.particles=[]
        self.sources={}
        self.drop_sources={}
        self.yy,self.xx=np.mgrid[0:HEIGHT:3,0:WIDTH:3].astype(np.float32)
        # Fluctuations are shared within a pocket, but drag, age, size and lift
        # vary per aggregate. Pollen is visible as clumps of illuminated grains.
        emissions={
            'gust':((6.15,2.25,1,43,1.),(11.35,2.75,-1,54,1.14)),
            'dew':((10.70,3.7,-1,19,.76),),
            'sun':((5.35,3.9,1,22,.69),(11.60,2.5,-1,18,.76)),
        }
        for start,span,direction,count,strength in emissions[scene]:
            pockets=rng.uniform(.1,span-.12,6)
            pockets.sort()
            for i in range(count):
                group=i%len(pockets)
                birth=start+np.clip(pockets[group]+rng.normal(0,.10),0,span)
                life=rng.uniform(2.,4.0)
                scale=rng.uniform(1.45,3.3) * (1 if scene=='gust' else .8)
                drag=rng.uniform(.60,.94)
                # Each burst picks an actual sector of anthers on its upwind side.
                source=int(rng.choice(np.flatnonzero(direction*np.cos(ANG) > -.35)))
                flow=(direction*(58+6*group)*strength, -rng.uniform(20,49)*strength)
                kinetic=(direction*rng.uniform(9,42),-rng.uniform(46,91)*strength)
                self.particles.append((birth,life,source,drag,scale,flow,kinetic,
                                       rng.uniform(0,2*pi),rng.uniform(.78,1.15)))

    def canopy(self,g,rgb,t):
        # An opaque branch above the flower rocks in the passing breeze. Distant
        # sunlight projects two overlapping penumbrae onto each surface. Depth
        # offsets the shadows where one petal stands in front of the next.
        phase=t*(2*pi/20)
        depth=np.clip(g.z[::3,::3],-100,300)
        x=self.xx+.16*depth
        y=self.yy-.075*depth
        axial=x*.93+y*.36
        cross=-x*.36+y*.93
        pivot=320-177*cos(phase-.10)+37*sin(phase*2-.6)+20*sin(phase*4+.65)
        slope=.19+.22*sin(phase+.3)
        passage=axial-pivot-slope*(cross-220)
        aperture=np.exp(-((passage+8*np.sin(cross/48+phase))/(46+11*cos(phase)))**2)
        second=np.exp(-((passage-99+13*np.sin(cross/83))/(31+6*sin(phase)))**2)
        cloud=np.clip(aperture+.50*second,0,1)
        full=cv2.resize(cloud,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        occupied=g.z>-1e5
        diffuse=.69+.60*full
        # The upper bud still sees pale skylight when the moving branch covers
        # the lower bank. Restrict this fill to tissue at the end of each cycle.
        if hasattr(g,"cycle"):
            step=round((t%20)*len(g.cycle.open)/20)%len(g.cycle.open)
            closed=1-ease(float(g.cycle.open[step].mean())/.76)
            petal=(rgb[...,0].astype(float)>1.12*rgb[...,1])&(rgb[...,2]>rgb[...,1])&occupied
            # Cool reflection from the pond also reaches the undersides in open shade.
            diffuse=diffuse+(.32*closed+.145*(1-closed)*(1-full))*petal
        # Amber transmitted and reflected light. Nearly black outside the shaft.
        result=rgb.astype(np.float32)*diffuse[...,None]
        pigment=np.max(rgb,axis=2)/255.
        result+=full[...,None]*pigment[...,None]*[33,19,7]
        air=full*(~occupied)*(.43+.57*np.exp(-((np.arange(rgb.shape[0])[:,None]-280)/220)**2))
        result+=air[...,None]*[14,13,12]
        return np.uint8(np.clip(result,0,255)),full

    def dew(self,g,rgb,t):
        if t < 1.05 or t > IMPACT+5.4:
            return
        if t<RELEASE:
            # The grain is pinned to the cuticle. A growing bead depins and
            # accelerates down a rib once its weight beats surface adhesion.
            slide=np.clip((t-4.15)/(RELEASE-4.15),0,1)
            u=.34+.659*slide**1.65
            v=.62-.13*slide+.015*sin(pi*slide)
            p=sample_patch(g.tissue(9,t)[0],u,v)
            x,y,z=g.project(p)
            r=(6.5+6.4*ease((t-1.05)/4))*ease((t-1.05)/.65)
            if r<.15:
                return
            if 0<=int(y)<HEIGHT and 0<=int(x)<WIDTH and g.z[int(y),int(x)]<z+12:
                dew_bead(rgb,x,y-r*.65,r,.72+.45*ease((slide-.86)/.14))
            return
        key=id(g)
        if key not in self.drop_sources:
            p=sample_patch(g.tissue(9,RELEASE)[0],.999,.49)
            self.drop_sources[key]=projected_at(g,p,RELEASE)
        x,y,z=self.drop_sources[key]
        flight=IMPACT-RELEASE
        speed_x=16*cos(g.az)
        depth_y=g.scale*309*cos(g.elev)
        a=min(t-RELEASE,flight)
        px=x+speed_x*a
        py=y-6+depth_y*(a/flight)**2
        if t<IMPACT:
            dew_bead(rgb,px,py,11.3,1.16+.20*sin(pi*a/flight))
        else:
            age=t-IMPACT
            ripples(rgb,px,py,age)
            # Short recoil and two satellites at the impact point.
            if age<.43:
                for dx in (-1,1):
                    rise=34*age-.5*145*age**2
                    dew_bead(rgb,px+dx*27*age, py-rise, 2.6*(1-age/.43),1.05)

    def dust(self,g,rgb,t,illuminated):
        h,w=rgb.shape[:2]
        sheet=np.zeros((h,w),np.uint8)
        for index,(birth,life,source,drag,size,air,velocity,phase,energy) in enumerate(self.particles):
            age=t-birth
            if age<=0 or age>=life:
                continue
            key=(id(g),index)
            if key not in self.sources:
                self.sources[key]=projected_at(g,TIPS[source],birth)
            x,y,z=self.sources[key]
            # Analytic drag towards moving air, plus a broad eddy shear and slow
            # settling. These offsets live in camera space after detachment.
            response=(1-exp(-age/drag))*drag
            x+=air[0]*age+(velocity[0]-air[0])*response + 5.6*(sin(1.5*age+phase)-sin(phase))
            y+=air[1]*age+(velocity[1]-air[1])*response + 3.8*age**2 + 7*(cos(1.8*age+phase)-cos(phase))
            z+=42*(1-exp(-age)) + 25*age
            ix,iy=round(x),round(y)
            if not(0<=ix<w and 0<=iy<h) or g.z[iy,ix]>z+7:
                continue
            fade=ease(age/.19)*ease((life-age)/.84)
            if illuminated is not None:
                fade*=.035+.965*illuminated[iy,ix]
            alpha=int(np.clip(236*energy*fade,0,255))
            if alpha<3:
                continue
            # Subpixel coverage survives the 3 x 4.5 source-pixel ASCII cell.
            cv2.circle(sheet,(round(x*16),round(y*16)),round(size*16),
                       alpha,-1,cv2.LINE_AA,4)
        if not np.any(sheet):
            return
        # Air carries a faint, soft concentration around each lit aggregate.
        field=cv2.GaussianBlur(sheet,(15,15),3.8)
        values=np.clip((sheet.astype(np.float32)+field*.90)/255,0,1)[...,None]
        color={'dew':(207,185,139),'gust':(255,204,103),'sun':(250,213,152)}[self.scene]
        rgb[:]=np.uint8(rgb+(np.asarray(color)-rgb)*values)

    def apply(self,rgb,g,t):
        """Shade the rendered frame and project effects before the ASCII pass."""
        lit=None
        if self.scene=='sun':
            rgb,lit=self.canopy(g,rgb,t)
        if self.scene=='dew':
            self.dew(g,rgb,t)
        self.dust(g,rgb,t,lit)
        return rgb
