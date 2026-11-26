"""Depth-rendered lotus with photo-sampled material and a spring response along each midrib."""

import ctypes


from pathlib import Path
import subprocess

import tempfile

import cv2
import numpy as np
from PIL import Image

from model import SURFACES
from living_material import materials
from living_hem import folded_edge

# The release entry point puts these modules on its import path.

from ascii_preview import AsciiCamera, RAMP

W,H=405,720
FPS=30
CAM=AsciiCamera(135,160)
U=np.linspace(0,1,71,dtype='f')[:,None]
V=np.linspace(-1,1,46,dtype='f')[None,:]
REF=np.deg2rad(31)
DEPTH=np.array([-38,-150,-115,-48,-37,62,-87,-1,-55,225,155,53,119,137])
# Different periods and phases are attached to the tissue, so edges never jitter.
PHASE=np.random.default_rng(574).uniform(-np.pi,np.pi,len(SURFACES))
GOLD=np.random.default_rng(47)
ANG=GOLD.uniform(0,2*np.pi,156)
FIL_LENGTH=GOLD.uniform(23,49,len(ANG))
FIL_PUSH=GOLD.uniform(.65,1.1,len(ANG))
FIL_TILT=GOLD.uniform(-.16,.17,len(ANG))
FIL_RADIUS=.50+.50*np.sqrt(GOLD.random(len(ANG)))


def rasterizer():
    src=Path(__file__).with_name('living_mesh.c')
    # macOS keeps the loaded library mapped after the temporary file is removed.
    with tempfile.TemporaryDirectory() as folder:
        lib = Path(folder) / 'lotus-grid.so'
        subprocess.run(['cc', '-shared', '-fPIC', '-O3', str(src), '-o', str(lib)], check=True)
        shared = ctypes.CDLL(str(lib))


    function=shared.grid
    pointer=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    function.argtypes=[pointer,*[ctypes.c_int]*4,pointer,np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')]
    function.restype=None
    filament=shared.stamens
    filament.argtypes=[pointer,pointer,*[ctypes.c_int]*3,pointer,np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')]
    filament.restype=None
    return function,filament


GRID,STAMENS=rasterizer()


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-7)


def smooth(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def bud_and_open(s,i):
    # Anchor the visible open margins to the clean three-quarter reference.
    # The unseen depth of each tip is inferred; it is not measured from video.
    x,y=s.points(7.14 if i>=11 else 7.27)
    x,y=[cv2.resize(a.astype('f'),(V.size,U.size)) for a in (x,y)]
    cx=x[:,V.size//2:V.size//2+1]
    root_x,root_y=s.petal.root
    phi=np.arctan2(DEPTH[i],s.petal.tip[0]-root_x)
    hand=np.sign(np.mean(x[:,-1]-x[:,0])*(-np.sin(phi))) or 1
    width=np.hypot(x[:,-1:]-x[:,:1],y[:,-1:]-y[:,:1])/2
    crest=np.maximum(np.sin(np.pi*U),0)
    # A small, fixed irregularity in the distal margins breaks the even cut edge.
    ripple=crest**.52*(abs(V)**3)*(2.8*np.sin(15*U+PHASE[i]+3*V)+1.1*np.sin(38*U+2*V))
    zdepth=DEPTH[i]*U**.96+hand*width*V*np.cos(phi)*.68+8*crest*(1-V*V)
    zheight=(zdepth*np.sin(REF)-(y-350))/np.cos(REF)+ripple
    opened=np.stack([x-490,zdepth,zheight],axis=-1)
    # Each blade wraps a sector of the bud. The outer whorl has a shoulder
    # below the apex; front and rear blades end at different heights.
    long=(238+27*np.sin(phi-1)+14*np.cos(phi*2.1)+9*np.sin(i*3.1))
    if i>=11: long-=22
    radius=11*(1-U)+86*crest**.82+U*(8+16*np.sin(phi+.3)**2)
    theta=phi+.36*(1-U)+.41*U**2+hand*V*(.87+.16*np.sin(i))*(.37+.63*crest**.40)
    off=3.5*np.sin(8*U+PHASE[i])*crest*(V**2)
    z=long*U + 6*V*crest*np.cos(phi)+off
    bud=np.stack([radius*np.cos(theta)-16*U**1.3, radius*np.sin(theta),
                  z+ripple*.35],axis=-1)
    # UV handedness in the traced outlines may be reversed. Keep the same
    # physical margin on its side of the rib throughout the release.
    if np.sum((bud[32,-1]-bud[32,0])*(opened[32,-1]-opened[32,0]))<0:
        bud=bud[:,::-1].copy()
    return bud.astype('f'),opened.astype('f')


def springs(i,scene,var,contact=None):
    phi=PHASE[i]
    base=.02+.21*(1+np.sin(phi))+(.27 if i>=11 else 0)
    duration=1.35+.45*(1+np.cos(phi*1.4))
    if scene=='cup':
        base=([.85,1.13,.63][i-11] if i>=11 else -.22+.16*(1+np.sin(phi)))
        if i>=11: duration=1.74
    if scene=='hem':
        active=(10,9,4)[var]
        adjacent=(7,10,3)[var]
        if i==active:
            base=-.24; duration=1.42
        elif i==adjacent:
            base=.16; duration=1.76
        else:
            base+=.36 if var!=2 and i in (3,4,9) else .16
    if scene=='overlap' and i in (12,13):
        base=-.28 if (i==12) != (var==1) else .05
        duration=1.85 if base<0 else 2.14
    k=28+4*np.cos(phi)-10*U
    damp=2*np.sqrt(k)*(.77+.10*np.cos(phi))
    q=np.zeros_like(U);v=q.copy();hist=[]
    for n in range(481):
        hist.append(q.copy())
        time=n/120
        goal=smooth((time-base-.36*U+.055*np.sin(4*U+phi))/duration)
        if contact is not None:
            # The following lip carries load while the covering edge is still
            # wrapped across it. A little elastic travel remains under contact.
            clearance=smooth((float(contact[n,44,0])-.19)/.55)
            goal=np.minimum(goal,.058+.942*clearance)
        force=k*(goal-q)-damp*v
        v+=force/120; q=q+v/120
    return np.asarray(hist)


def bend(bud,opened,q,i,scene,var):
    # Rebuild the spine by integrating the direction and length of each segment.
    # Intermediate poses preserve blade length even when the tip describes an arc.
    mid=V.size//2
    b=bud[:,mid]; o=opened[:,mid]
    bt=np.diff(b,axis=0); ot=np.diff(o,axis=0)
    bb,oo=unit(bt),unit(ot)
    angle=np.arccos(np.clip(np.sum(bb*oo,axis=-1,keepdims=True),-.9999,.9999))
    f=q[1:]
    tang=(np.sin((1-f)*angle)*bb+np.sin(f*angle)*oo)/np.sin(angle)
    length=(1-f)*np.linalg.norm(bt,axis=1,keepdims=True)+f*np.linalg.norm(ot,axis=1,keepdims=True)
    root=(1-q[0])*b[0]+q[0]*o[0]
    midrib=root+np.vstack((np.zeros(3),np.cumsum(tang*length,axis=0)))
    # Lower edges carry more load. Unrolling propagates from the attached
    # tissue towards the free margins, and asymmetry is fixed to this blade.
    edge=q[:,None,:]-(.17+.05*np.sin(PHASE[i]))*abs(V)[...,None]*np.sin(np.pi*np.clip(q[:,None,:],0,1))
    pts=midrib[:,None,:]+(bud-b[:,None,:])*(1-edge)+(opened-o[:,None,:])*edge
    bulge=np.maximum(np.sin(np.pi*U),0)*U**.55*(1+.32*V*np.sin(PHASE[i]))
    if scene=='hem' and i in (7,9,10):
        weight=(20,37,50)[var]
        pts[:,:,2]+=weight*np.sin(np.pi*np.clip(q,0,1))*bulge*V
        pts[:,:,2]-=weight*(q*(1-q)*3)*U**2
    if scene=='overlap' and i in (12,13):
        direct=-1 if i==12 else 1
        slip=smooth(q)*U**1.4
        pts[:,:,0]+=direct*(74 if (i==12)!=(var==1) else 33)*slip
        pts[:,:,1]+=22*slip
        pts[:,:,2]-=29*slip
    if scene=='cup' and i in (11,12,13):
        pts[:,:,0]+=(-10,-29,35)[i-11]*smooth(q)*U
        pts[:,:,2]-=(8,13,4)[i-11]*smooth(q)*U
    return pts.astype('f')


def normals(p):
    return unit(np.cross(np.gradient(p,axis=0),np.gradient(p,axis=1)))


class Garden:
    def __init__(self,scene,var):
        self.scene=scene;self.var=var
        self.sheets=[bud_and_open(s,i) for i,s in enumerate(SURFACES)]
        self.tex=[cv2.resize(a,(V.size,U.size),interpolation=cv2.INTER_AREA).astype('f') for a in materials()]
        self.h=[springs(i,scene,var) for i in range(len(SURFACES))]
        angles={'hem': [(-6,31),(17,43),(-22,23)],
                'cup': [(-3,33),(18,22),(-17,47)],
                'overlap': [(-7,32),(22,25),(-18,45)]}
        if scene=='overlap':
            driver, follower=(13,12) if var==1 else (12,13)
            self.h[follower]=springs(follower,scene,var,contact=self.h[driver])
        # Bending load follows the displacement of the sheets' centres of mass.
        # The foot of the cantilever is fixed below the portrait.
        moment=np.zeros((481,2))
        for (b,o),h in zip(self.sheets,self.h):
            moment+=h.mean(axis=(1,2))[:,None]*(o-b).mean(axis=(0,1))[:2]/len(self.h)
        # The last small motion begins when most of the whorl has released its
        # lateral support. Relax only the free third of one blade under that load.
        support=np.mean([h[:,48,0] for h in self.h],axis=0)
        demand=smooth((support-.91)/.09)
        self.relax=np.zeros_like(support)
        for j in range(1,len(support)):
            self.relax[j]=self.relax[j-1]+(demand[j]-self.relax[j-1])*(1-np.exp(-1/(120*.34)))
        self.last_blade={'cup':12,'overlap':12 if var==1 else 13,'hem':(10,9,4)[var]}[scene]
        self.sway=np.zeros_like(moment)
        vel=np.zeros(2)
        for n in range(1,len(moment)):
            vel+=(19*(.25*moment[n]-self.sway[n-1])-6*vel)/120
            self.sway[n]=self.sway[n-1]+vel/120
        self.load=np.zeros(2)
        self.az,self.elev=np.deg2rad(angles[scene][var])
        self.roll=np.deg2rad((-3,3,-5)[var])
        self.scale=(.565,.535,.555)[var]
        self.xmid=(.51,.49,.535)[var]*W
        self.ymid=(.49,.51,.48)[var]*H
        self.light=unit(np.array([-.46,.35,.82],dtype='f'))

    @staticmethod
    def stem_weight(z):
        bend=np.clip((z+760)/760,0,1)
        return bend*bend*(3-bend)*.5+np.maximum(z,0)/510

    @staticmethod
    def attachment_weight(z):
        # The earlier flight controller and top view used this shorter lever.
        bend=np.clip((z+760)/760,0,1)
        return bend*bend*(3-2*bend)*.5+np.maximum(z,0)/510

    def project(self,p):
        x,d,z=p[...,0],p[...,1],p[...,2]
        bend=self.stem_weight(z)
        x=x+self.load[0]*bend;d=d+self.load[1]*bend
        xx=x*np.cos(self.az)-d*np.sin(self.az)
        dd=x*np.sin(self.az)+d*np.cos(self.az)
        yy=dd*np.sin(self.elev)-z*np.cos(self.elev)
        dep=dd*np.cos(self.elev)+z*np.sin(self.elev)
        return np.stack([self.xmid+self.scale*(np.cos(self.roll)*xx-np.sin(self.roll)*yy),
                         self.ymid+self.scale*(np.sin(self.roll)*xx+np.cos(self.roll)*yy),dep],-1)

    def mesh(self,p,color):
        points=self.project(p)
        vertex=np.ascontiguousarray(np.concatenate([points,np.broadcast_to(np.clip(color,0,255),p.shape)],axis=-1),dtype='f')
        GRID(vertex,*p.shape[:2],W,H,self.z,self.rgb)

    def stem(self,t):
        a=np.linspace(0,1,46)[:,None];theta=np.linspace(0,2*np.pi,15)[None,:]
        radius=6.1+3*a+2.4*np.exp(-((a-.025)/.040)**2)
        x=26*np.sin(3.8*a+.08)-2+91*a**1.27 + radius*np.cos(theta)
        d=9*a-12+radius*np.sin(theta)
        z=-815*a+np.zeros_like(theta)
        c=(.38+.54*(.5+.5*np.cos(theta-2.2))+.07*np.sin(theta*5))
        self.mesh(np.stack([x,d,z],-1),np.array([99,121,59])*c[...,None])

    def heart(self,open_fraction):
        # Real filaments have separate insertion points in a ring. A few
        # clustered ochre anthers form an irregular crescent behind the lip.
        a=ANG
        ex,ey=np.cos(a),np.sin(a)
        spread=.32+.68*smooth(open_fraction*1.6)
        root=np.stack([(75*ex*FIL_RADIUS-20)*spread,(51*ey*FIL_RADIUS+7)*spread,76+7*spread+spread*FIL_RADIUS*(30*ex-7*ey)],-1)
        radial=FIL_PUSH*(.14+.86*open_fraction)
        tips=root+FIL_LENGTH[:,None]*np.stack([(.7*radial)*np.cos(a+FIL_TILT),
                                              (.7*radial)*np.sin(a+FIL_TILT),.93+0*ex],-1)
        p0=self.project(root);p1=self.project(tips)
        p0[:,2]+=68*open_fraction; p1[:,2]+=68*open_fraction
        gold=np.stack([236+16*FIL_RADIUS,144+49*FIL_PUSH,54+44*np.sin(a)**2],-1)
        STAMENS(np.ascontiguousarray(np.column_stack([p0,gold]),dtype='f'),
                np.ascontiguousarray(np.column_stack([p1,gold]),dtype='f'),
                len(a),W,H,self.z,self.rgb)
        theta=np.linspace(0,2*np.pi,40)[None,:];r=np.linspace(0,1,15)[:,None]
        x=(44*r*np.cos(theta)-16)*spread;d=(33*r*np.sin(theta)-11)*spread
        z=74+7*(1-r*r)+.39*x
        self.mesh(np.stack([x,d,z],-1),np.array([224,171,67])*(.8+.2*(1-r*r))[...,None])

    def tissue(self,i,t):
        step=min(round(t*120),480)
        q=self.h[i][step]
        b,o=self.sheets[i]
        p=bend(b,o,q,i,self.scene,self.var)
        tex=self.tex[i]
        if self.scene=='hem':
            p,tex=folded_edge(p,tex,q,i,self.var)
        if i==self.last_blade:
            free=U**2.7*self.relax[step]*(1+.15*V)
            direction=1 if SURFACES[i].petal.tip[0]>490 else -1
            p[:,:,0]+=direction*(22 if self.scene=='cup' else 15)*free
            p[:,:,2]-=(32 if self.scene=='hem' else 21)*free
            p[:,:,1]+=8*free
        return p,tex,q

    def render(self,t):
        self.rgb=np.zeros((H,W,3),np.uint8)
        self.z=np.full((H,W),-1e6,'f')
        self.load=self.sway[min(round(t*120),480)]
        self.stem(t)
        qs=[h[min(round(t*120),480)] for h in self.h]
        opening=float(np.mean([q.mean() for q in qs]))
        for i,((b,o),q,tex) in enumerate(zip(self.sheets,qs,self.tex)):
            if i==11:
                # The receptacle sits in the whorl between the outer and inner
                # leaves. Constrain the centre to that layer at their contact.
                self.heart(opening)
                dark=self.rgb.astype(np.int16)
                anther=(dark[:,:,0]>110)&(dark[:,:,1]>dark[:,:,2]*1.48)&(dark[:,:,0]>dark[:,:,1])
                self.z[anther]=np.minimum(self.z[anther],50)
            p,tex,q=self.tissue(i,t)
            n=normals(p)
            no=normals(o)
            # Reference pigment includes the neutral studio light. Relight only
            # the change in curvature; the rim is translucent, its base denser.
            lum=np.abs(n@self.light); rest=np.abs(no@self.light)
            gain=np.clip((.67+.45*lum)/(.67+.45*rest),.60,1.36)
            cup=.83+.17*q+.10*U
            self.mesh(p,tex*(gain*cup)[...,None])
        # Local shadow where another surface actually crosses the lighting ray
        # in depth. Apply only inside the silhouette; penumbrae have soft edges.
        occupied=self.z>-1e5
        occlusion=np.zeros_like(self.z)
        for delta in (2,5,9):
            front=np.roll(self.z,(delta,delta),axis=(0,1))
            occlusion+=np.clip((front-self.z-18)/52,0,1)*occupied
        occlusion=cv2.GaussianBlur(occlusion,(7,7),1.25)
        return np.uint8(self.rgb*(1-.072*occlusion[:,:,None]))


# Depth-of-field background is stationary. It has much lower contrast than
# the flower, including in the empty cells across the long stem.
yy,xx=np.mgrid[0:160,0:135]
field=np.exp(-((xx-20)/50)**2-((yy-145)/53)**2)
haze=np.exp(-((xx-113)/56)**2-((yy-57)/52)**2)
BACKGROUND=np.array([7,9,14])+field[...,None]*[7,13,6]+haze[...,None]*[11,8,13]
SIZE=(810,1440)
CAM.size=SIZE
# Offset the type baseline by at most two pixels between columns. Letters keep
# their grid positions, but their dark interlines no longer form a screen-wide band.
shift=np.random.default_rng(231).choice([-2,-1,0,0,1,2],135).repeat(6)
row_lookup=np.clip(np.arange(SIZE[1])[:,None]+shift[None,:],0,SIZE[1]-1)
col_lookup=np.arange(SIZE[0])[None,:]
paper=cv2.resize(BACKGROUND.astype('f'),SIZE,interpolation=cv2.INTER_CUBIC)


def ink(rgb):
    body=cv2.resize(rgb,(135,160),interpolation=cv2.INTER_AREA)
    # Use the casting skill's actual ASCII converter. The soft fill is the
    # colour of each type cell, sampled in the same material coordinates.
    glyph=np.asarray(CAM.frame(Image.fromarray(rgb)))[row_lookup,col_lookup]
    tint=cv2.resize(body,SIZE,interpolation=cv2.INTER_NEAREST)
    occupied=np.minimum(1,cv2.resize(np.max(body,axis=2),SIZE,interpolation=cv2.INTER_LINEAR).astype('f')/85)
    # Remove the converter's uniform paper and expose the scene's subdued ambient.
    glyph=np.maximum(glyph.astype('f')-np.array([8,16,25],dtype='f'),0)
    result=paper+glyph*.97+tint*.41+occupied[...,None]*[4,2,5]
    return np.uint8(np.clip(result,0,255))
