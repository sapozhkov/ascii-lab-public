"""Simulate two jellyfish and render the heart from the reel camera."""

import numpy as np
from PIL import Image

from . import renderer as j
from . import swimmer as solo


SECONDS, DT, WARM = 20, 1/60, 24
PULSE_RATE = .60
NODES = 34
# Start the animals fully formed; keep independent body times during the approach.
CLOCK = 30.25
ROOT_ANGLES = j.STRAND_ROOT[:, 0]
LINK = solo.SPAN[:, 0, None] / (NODES-1)
OFFSET = ((solo.WIDTH-j.W)/2, 300)
DISPLAY_INK = np.rint(255*(np.arange(256)/255)**.70).astype(np.uint8)


def spread(t):
    return j.ease((t-.35)/4.2) * (1-j.ease((t-6)/10))


def yaw(p, angle):
    x,y,z = np.moveaxis(p,-1,0)
    c,s = np.cos(angle),np.sin(angle)
    return np.stack((c*x-s*z,y,s*x+c*z),axis=-1)


def orient(body, tilt, angle):
    # Cant outwards for the two lobes, along the orbit for forward thrust,
    # and upstream in the world frame to lean into the cross-current.
    forward = j.spin_x(j.spin_z(body, -tilt), .15)
    return j.spin_z(yaw(forward, angle), .045)


def clock(t, index):
    return CLOCK + PULSE_RATE*t + index*.18*spread(t)


class Orbit:
    """Joint orbit, rise and wake, integrated before sampling the camera."""
    def __init__(self):
        self.t = np.arange(-WARM, SECONDS+DT/2, DT)
        n = len(self.t)
        self.radius, self.angle, self.height, self.tilt = np.zeros((4,n))
        radius, radial, omega, height, climb, angle, wake = .43,0.,1.7,0.,.55,0.,.8
        self.nodes = np.empty((n,2,j.STRANDS,NODES,3),dtype=np.float32)
        chains = None
        for k,t in enumerate(self.t):
            open_ = spread(t)
            phases = clock(t,np.arange(2))
            pump = float(np.mean(np.maximum(0,(j.pulse(phases)-j.pulse(phases-PULSE_RATE*DT))/DT)))
            # The pair steers for a changing separation. Pulses supply the
            # angular impulse and lift; drag dissipates both between beats.
            target = .43 + .53*open_
            radial += ((target-radius)*9 - radial*5)*DT
            radius += radial*DT
            omega += ((.54-.16*open_)*pump - .62*omega)*DT
            climb += (.20*pump - .65*climb - .6*climb*abs(climb))*DT
            height += climb*DT
            angle += omega*DT
            self.radius[k],self.angle[k],self.height[k]=radius,angle,height
            # Outward cant opens a notch between the two lobes.
            self.tilt[k] = .44 - .21*open_
            wake += (pump-wake)*DT/.75
            # The solo bell samples its rim pose EDGE_DELAY seconds behind
            # its crown. Anchor the chains to that same material point.
            rim_k=max(0,k-round(j.EDGE_DELAY/(PULSE_RATE*DT)))
            rim_radius=self.radius[rim_k]
            roots=[]
            for index,phase in enumerate(phases):
                local=j.dome(np.ones(j.STRANDS),ROOT_ANGLES,phase,lambda p,_:p)
                bearing=self.angle[rim_k]+index*np.pi
                center=np.array([rim_radius*np.cos(bearing),self.height[rim_k],rim_radius*np.sin(bearing)])
                roots.append(orient(local,self.tilt[rim_k],bearing)+center)
            roots=np.array(roots)
            if chains is None:
                chains=np.repeat(roots[:,:,None,:],NODES,axis=2)
                chains[:,:,:,1]-=LINK[:,0,None]*np.arange(NODES)
            # Approximate entrainment by the two pulsed wakes. Only nearby
            # swimmers have a shared wake; each unconstrained node advects.
            radial_rate=(.15+1.7*wake)*(1-open_)**2
            chains[:,:,:,[0,2]]*=np.exp(-radial_rate*DT)
            chains[:,:,:,0]+=.023*DT
            chains[:,:,:,1]-=(.09+.34*wake)*DT
            chains[:,:,0,:]=roots
            for knot in range(1,NODES):
                prev=chains[:,:,knot-1,:]
                d=chains[:,:,knot,:]-prev
                chains[:,:,knot,:]=prev+d*(LINK/np.maximum(1e-8,np.linalg.norm(d,axis=-1,keepdims=True)))
            self.nodes[k]=chains
        self.first=int(round(WARM/DT))
        self.start_angle=self.angle[self.first]
        # Match angular velocity gradually. Correct the remaining phase over
        # the same interval so the two crowns are broadside once locked.
        engage = j.ease((self.t-12)/5)
        self.camera_angle = self.start_angle + np.cumsum(np.diff(self.angle,prepend=self.angle[0])*engage)
        error = self.angle[-1]-self.camera_angle[-1]
        self.camera_angle += engage*(error-2*np.pi*round(error/(2*np.pi)))

    def sample(self, values,t):
        return np.interp(t,self.t,values)

    def camera(self,t):
        return self.sample(self.camera_angle,t)


class View:
    def __init__(self, path,t,index):
        self.path,self.t,self.index=path,t,index
        self.body_time=clock(t,index)
        self.camera=path.camera(t)
        self.height=path.sample(path.height,t)
        self.scale=1/(1+.58*spread(t))

    def rotate(self,p,when):
        scene_t=self.t+(when-self.body_time)/PULSE_RATE
        tilt=self.path.sample(self.path.tilt,scene_t)
        angle=self.path.sample(self.path.angle,scene_t)+self.index*np.pi
        return yaw(orient(p,tilt,angle),-self.camera)

    def move(self,p,when):
        scene_t=self.t+(when-self.body_time)/PULSE_RATE
        a=self.path.sample(self.path.angle,scene_t)+self.index*np.pi-self.camera
        r=self.path.sample(self.path.radius,scene_t)
        h=self.path.sample(self.path.height,scene_t)-self.height
        return (self.rotate(p,when)+np.stack((r*np.cos(a),h,r*np.sin(a)),axis=-1))*self.scale

    def strand_part(self):
        k=round((self.t+WARM)/DT)
        p=self.path.nodes[k,self.index].astype(float)
        p[...,1]-=self.height
        p=yaw(p,-self.camera)*self.scale
        # Render between the simulated knots as well, so the long diagonals
        # remain connected on the character grid when the tails fan out.
        weights=np.arange(6)/6
        between=p[:,:-1,None,:]+np.diff(p,axis=1)[:,:,None,:]*weights[None,None,:,None]
        p=np.concatenate((between.reshape(j.STRANDS,-1,3),p[:,-1:,:]),axis=1)
        sx,sy,depth=j.on_screen(p,self.body_time)
        glyph=j.line_glyph(np.gradient(sx,axis=1),np.gradient(sy,axis=1)).ravel()
        glow=(.80-.16*np.linspace(0,1,p.shape[1]))*(.86+.14*j.ease(.5-depth))
        ink=glow[...,None]*j.rgb(134,219,254)
        return p.reshape(-1,3),glyph,ink.reshape(-1,3),j.STRAND


# Side profile of the closed pose. Each positive x has two boundary heights
# on the parametric heart; reflection supplies the opposite half.
def heart_edges(u):
    a=np.arcsin(np.cbrt(np.clip(u,0,1)))
    def height(angle):
        return (13*np.cos(angle)-5*np.cos(2*angle)-2*np.cos(3*angle)-np.cos(4*angle))/16
    return height(np.pi-a),height(a)


# Horizontal half-width on the lower branch of the same outline as closed_pose.
HEART_X=np.linspace(0,1,513)
HEART_BOTTOM=heart_edges(HEART_X)[0]


def overlap(points, kinds, glyph, colors, path, t):
    """Let a few trailing strands cross inside the fitted heart volume."""
    closed=1-float(spread(t))
    if closed<=.001:
        return
    bearing=path.sample(path.angle,t)-path.camera(t)
    side=j.spin_x(yaw(points,-bearing),j.PITCH)
    zoom=j.CAM_D/(j.CAM_D+side[:,2])
    radius=np.max(np.abs(side[:,0]*zoom))
    strands=np.flatnonzero(kinds==j.STRAND).reshape(2,j.STRANDS,-1)
    ribbons=np.flatnonzero(kinds==j.LIMB).reshape(2,j.LIMBS,-1,j.LIMB_COLS)
    chosen_strands=strands[:,1::3]
    chosen_ribbons=ribbons[:,::2]
    ids=np.concatenate((chosen_strands.ravel(),chosen_ribbons.ravel()))
    x,y,z=side[ids].T
    sx,sy=x*zoom[ids],y*zoom[ids]
    level=(sy+.16)/radius
    width=radius*np.interp(level,HEART_BOTTOM,HEART_X)
    u=np.clip(sx/np.maximum(width,1e-6),-1,1)
    v=np.sqrt(1-u*u)*np.tanh(z/.18)
    # Lower sections trail an earlier bearing of the pair. Feather the
    # rotation to zero at the outline, at the tip and under the two rims.
    lag=7.8*j.ease((.36-level)/1.10)
    turn=path.sample(path.angle,t-lag)-path.sample(path.angle,t)
    turn*=closed*j.ease((1-abs(u))/.26)*j.ease((level-HEART_BOTTOM[0])/.15)
    c,s=np.cos(turn),np.sin(turn)
    nx=sx+width*(u*(c-1)-v*s)
    nz=z+.65*width*(u*s+v*(c-1))
    scale=j.CAM_D/(j.CAM_D+nz)
    points[ids]=yaw(j.spin_x(np.stack((nx/scale,sy/scale,nz),axis=-1),-j.PITCH),bearing)
    px,py,_=j.on_screen(points[chosen_strands],CLOCK)
    strokes=j.line_glyph(np.gradient(px,axis=-1),np.gradient(py,axis=-1))
    moving=np.abs(turn[:chosen_strands.size].reshape(chosen_strands.shape))>.01
    glyph[chosen_strands[moving]]=strokes[moving]
    px,py,_=j.on_screen(points[chosen_ribbons],CLOCK)
    edge_strokes=j.line_glyph(np.gradient(px,axis=2),np.gradient(py,axis=2))
    ribbon_turn=turn[chosen_strands.size:].reshape(chosen_ribbons.shape)
    for edge in (0,-1):
        changed=np.abs(ribbon_turn[...,edge])>.01
        glyph[chosen_ribbons[...,edge][changed]]=edge_strokes[...,edge][changed]
    colors[ids]*=1+.20*np.abs(np.sin(turn/2))[:,None]


def closed_pose(points, path, t):
    """Blend the swimming mesh into a shaped rest pose in the pair's frame."""
    amount=1-float(spread(t))
    if amount<=.001:
        return points
    bearing=path.sample(path.angle,t)-path.camera(t)
    # Keep depth so the two animals still occlude each other in profile.
    side=j.spin_x(yaw(points,-bearing),j.PITCH)
    x,y,z=side.T
    perspective=j.CAM_D/(j.CAM_D+z)
    across=np.abs(x*perspective)
    up=y*perspective
    radius=(abs((x*perspective).min())+(x*perspective).max())/2
    knots=np.linspace(0,1,49)
    for half in (x<0,x>=0):
        u=across[half]/across[half].max()
        bins=np.minimum(np.rint(u*(len(knots)-1)).astype(int),len(knots)-1)
        bottom=np.full(len(knots),np.inf)
        top=np.full(len(knots),-np.inf)
        np.minimum.at(bottom,bins,up[half])
        np.maximum.at(top,bins,up[half])
        valid=np.isfinite(top)
        low=np.interp(u,knots[valid],bottom[valid])
        high=np.interp(u,knots[valid],top[valid])
        within=np.clip((up[half]-low)/np.maximum(high-low,1e-6),0,1)
        lower,upper=heart_edges(u)
        target=radius*(lower+(upper-lower)*within)-.16
        side[half,0]+=(np.sign(x[half])*radius*u-x[half]*perspective[half])*amount/perspective[half]
        side[half,1]+=(target-up[half])*amount/perspective[half]
    return yaw(j.spin_x(side,-j.PITCH),bearing)


# Direction to a distant patch of surface light, in the opening view.
SURFACE_LIGHT=j.normalised(np.array([-.52,.92,-.78]))


def shade_tentacles(points,kinds,colors,key,release):
    """Sample both bells' soft, transmitted shadow on the final posed tails."""
    tail=(kinds==j.LIMB)|(kinds==j.STRAND)
    bell=~tail
    # Orthographic shadow map in the plane perpendicular to the incident ray.
    right=j.normalised(np.cross(key,[0.,1.,0.]))
    up=np.cross(right,key)
    plane=np.column_stack(((points*right).sum(axis=1),(points*up).sum(axis=1)))
    uv=np.floor(plane*30).astype(int)
    uv-=uv.min(axis=0)-2
    depth=(points*key).sum(axis=1)
    shadow=np.full(uv.max(axis=0)+3,-np.inf)
    np.maximum.at(shadow,tuple(uv[bell].T),depth[bell])
    x,y=uv[tail].T
    occlusion=np.zeros(len(x))
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            occlusion+=j.ease((shadow[x+dx,y+dy]-depth[tail]-.015)/.13)
    # The bell transmits some light. Water also separates near and far layers.
    _,_,z=j.on_screen(points[tail],CLOCK)
    near=j.ease((.48-z)/.96)
    gain=(.46+.35*near)*(1-.52*occlusion/9)
    colors[tail]*=(gain+(1-gain)*release)[:,None]


def swimmer_parts(path, t, index, key=None):
    view = View(path, t, index)
    phase = view.body_time
    parts = solo.shell_parts(phase, view, view.move, light_direction=key)
    parts += j.limb_part(phase, solo.blue_wave(phase), view.move)
    parts.append(view.strand_part())
    if index:
        parts = [(p, g, ink[:, [2, 0, 1]] * np.array([1, .94, .96]), kind)
                 for p, g, ink, kind in parts]
    return parts


def combine_parts(parts):
    points, glyph, color = (np.concatenate([part[k] for part in parts]) for k in range(3))
    kinds = np.concatenate([np.full(len(p), kind) for p, _, _, kind in parts])
    return points, glyph, color, kinds


def pose_parts(points, glyph, color, kinds, path, t):
    posed = closed_pose(points, path, t)
    overlap(posed, kinds, glyph, color, path, t)
    return posed


def render(path, t):
    key = yaw(SURFACE_LIGHT, path.start_angle - path.camera(t))
    # While the opening heart separates, let the closing shot's side light
    # turn back towards the normal opening light along the shorter arc.
    opening = 0.
    if t < 6:
        opening = 1 - spread(t)
        turn = path.start_angle - path.camera(SECONDS)
        key = yaw(key, opening * np.arctan2(np.sin(turn), np.cos(turn)))
    parts = swimmer_parts(path, t, 0, key) + swimmer_parts(path, t, 1, key)
    points, glyph, color, kinds = combine_parts(parts)
    points = pose_parts(points, glyph, color, kinds, path, t)
    release = max(j.ease((t - 8) / 8) * (1 - spread(t)), opening)
    shade_tentacles(points, kinds, color, key, release)
    chars=np.zeros((solo.ROWS,solo.COLS),int)
    ink=np.zeros((*chars.shape,3))
    # Static plankton in world coordinates shows ascent and camera rotation.
    bearing=j.DOT[:,0]*j.TAU-path.camera(t)
    r=1.1+2*j.DOT[:,2]
    h=(j.DOT[:,1]*5-path.sample(path.height,t)+2.5)%5-2.5
    dust=j.around(r,bearing,h)*View(path,t,0).scale
    x,y,z=j.on_screen(dust,CLOCK)
    x=((x+OFFSET[0])/j.CW).astype(int); y=((y+OFFSET[1])/j.CH).astype(int)
    ok=(x>=0)&(x<solo.COLS)&(y>=0)&(y<solo.ROWS)
    chars[y[ok],x[ok]]=np.where(j.DOT[ok,2]>.8,j.glyph_id('+'),j.glyph_id('.'))
    ink[y[ok],x[ok]]=j.rgb(107,144,162)*(.8-.1*z[ok,None])
    j.rasterize(CLOCK,chars,ink,points,glyph,color,kinds,OFFSET)
    pixels=j.paint(chars,ink)[:solo.HEIGHT]
    # Thicken strokes by a fraction of a pixel to keep the pair legible when
    # the full portrait is reduced to phone size.
    left=pixels[:,:-1]
    extra=np.maximum(left,pixels[:,1:])-left
    left+=extra-extra//3
    return Image.fromarray(DISPLAY_INK[pixels])
