"""Three continuous 20-second studies derived from the approved lotus meshes."""




import cv2
import numpy as np


import living as base

PERIOD = 20
RATE = 60
TAU = 2 * np.pi



VIEWS = {"dew": [(-3,33), (9,27), (-14,39)],
         "gust": [(-7,32), (-19,26), (8,40)],
         "sun": [(-11,38), (5,30), (-20,44)]}
U, V = base.U, base.V


def pulse(t, center, width):
    """Periodic wind packet. Its centre can cross the camera's time origin."""
    distance = (t - center + PERIOD/2) % PERIOD - PERIOD/2
    return np.exp(-.5*(distance/width)**2)


def wind(t, theme):
    phase = TAU*t/PERIOD
    flow = .27*np.sin(3*phase+.6) + .16*np.sin(7*phase-1.5) + .09*np.sin(12*phase)
    packets = {"dew": [(3.8,.85,.45),(8.1,1.0,.58),(12.2,1.05,-1.0),(15.0,1.2,.96)],
               "gust": [(4.1,.9,-.78),(6.25,.76,1.18),(9.3,.8,-.85),(11.9,1.35,-1.55),(15.0,.98,.86)],
               "sun": [(3.4,.9,.38),(7.7,1.05,-.8),(11.0,.85,.85),(14.9,1.42,-1.11)]}
    return flow + sum(strength*pulse(t, center, width) for center,width,strength in packets[theme])


class Cycle:
    """Periodic integration of turgor, stem bending, and the load on each free edge."""
    def __init__(self, theme, bloom_pace=1.):
        count = len(base.SURFACES)
        ids = np.arange(count)
        phase = base.PHASE
        outer = ids < 11
        delay = np.where(outer, -.55+.54*np.sin(phase), 1.32+.85*np.sin(phase))
        rise = 1.45+.59*(1+np.cos(phase))
        if theme == "gust":
            delay[[11,12,13]] = [1.6,-.4,1.25]
            rise[[12,13]] = [3.3,4.3]
        if theme == "sun":
            delay[[11,12,13]] = [.9,2.0,7.6]
            rise[[11,12,13]] = [3.1,4.2,5.6]
        # Outer petals expose the crown rapidly. Inner tissue closes earlier at dusk;
        # the closing sequence differs from opening, even with the lamp held fixed.
        closing = np.where(outer, 16.1+.85*np.sin(phase*1.4), 15.1+.68*np.cos(phase))
        # The turgor ramp runs on its own clock. Wind, flight and falling water
        # keep their physical time. Let the longer evening ramp end at the same
        # phase so the bud settles before the next dawn.
        rise *= bloom_pace
        delay *= 1. + .45*(bloom_pace-1.)
        fall = 2.4+.36*np.cos(phase)
        extra_fall = .46*(bloom_pace-1.)*fall
        closing -= extra_fall
        fall += extra_fall
        stiffness = 27+4*np.cos(phase[:,None])-9*U.T
        drag = 2*np.sqrt(stiffness)*(.78+.09*np.cos(phase[:,None]))
        length = U.T
        state = np.zeros((count, len(U)))
        velocity = np.zeros_like(state)
        angle = np.zeros(count)
        angular_v = np.zeros(count)
        stem = np.zeros(2)
        stem_v = np.zeros(2)
        ncycle = RATE*PERIOD
        self.open = np.empty((ncycle,count,len(U)), np.float32)
        self.flex = np.empty((ncycle,count), np.float32)
        self.stem = np.empty((ncycle,2), np.float32)
        for step in range(2*ncycle):
            frame = step % ncycle
            t = frame/RATE
            # The new opening starts just before zero, while the former cycle settles.
            dawn = t if t < 17 else t-PERIOD
            opening = base.smooth((dawn-delay[:,None]-.36*length)/rise[:,None])
            day = 1-base.smooth((t-closing[:,None]-.28*length)/fall[:,None])
            goal = np.minimum(day, opening) if t < 17 else np.maximum(day, opening)
            if theme in ("gust","sun"):
                clearance = base.smooth((state[12,46]-.22)/.65)
                goal[13] = np.minimum(goal[13], .095+.905*clearance) if t<9 else goal[13]
            force = wind(t, theme)
            exposure = np.mean(state,axis=1)
            # Wind bends the midrib then twists the thin margin, according to incidence.
            target = force*(.25+.75*exposure)*(.56+.53*np.sin(phase+.7))
            angular_v += (18*(target-angle) - (3.9+.5*np.cos(phase))*angular_v)/RATE
            angle += angular_v/RATE
            moment = np.array([49*force, 19*wind(t-.58,theme)])*(.58+.42*exposure.mean())
            moment[0] += 9*np.mean(exposure*np.sin(phase))
            stem_v += (11*(moment-stem)-3.5*stem_v)/RATE
            stem += stem_v/RATE
            velocity += (stiffness*(goal-state)-drag*velocity)/RATE
            state += velocity/RATE
            if step>=ncycle:
                self.open[frame]=state
                self.flex[frame]=angle
                self.stem[frame]=stem


def pond(theme):
    """Defocused banks, distant stems and folded pads behind the lotus."""
    rng=np.random.default_rng({"dew":528,"gust":532,"sun":618}[theme])
    h,w=base.H,base.W
    yy,xx=np.mgrid[0:h,0:w]
    sky=np.exp(-((xx-w*.18)/(w*.78))**2-((yy-h*.36)/(h*.63))**2)
    water=base.smooth((yy/h-.65)/.30)
    bg=np.zeros((h,w,3),dtype=np.float32)
    bg[:]=[4,8,12]
    bg+=sky[...,None]*[10,19,17]+water[...,None]*[3,8,13]
    # Distant shaded reed blades, out of the depth of field.
    reeds=np.zeros_like(bg)
    for side in (-1,1):
        for _ in range(22):
            x=rng.uniform(-150,95) if side<0 else rng.uniform(w-85,w+150)
            y=rng.uniform(250,570)
            tilt=rng.uniform(-55,55)
            tips=np.array([[x-7,y+130],[x+tilt-3,y-50],[x+tilt*1.65,y-160],
                           [x+tilt+4,y-57],[x+6,y+130]],np.int32)
            cv2.fillPoly(reeds,[tips],tuple(rng.uniform(.4,1.1)*np.array([8,16,7])))
    bg+=cv2.GaussianBlur(reeds,(0,0),9)
    for x,y,rx,ry in [(-18,560,136,32),(352,532,79,14),(406,650,175,54),(53,685,112,32)]:
        angle=rng.uniform(-.2,.19)
        a=np.linspace(.2,2*np.pi-.16,100)
        radial=1+.031*np.sin(5*a+1)+.017*np.sin(13*a)
        def contour(radius):
            ex=rx*np.cos(a)*radial*radius; ey=ry*np.sin(a)*radial*radius
            return np.int32(np.column_stack([x+ex*np.cos(angle)-ey*np.sin(angle),y+ex*np.sin(angle)+ey*np.cos(angle)]))
        points=np.vstack(([x,y],contour(1),[x,y])).astype('int32')
        cv2.fillPoly(bg,[points],(20,42,36))
        cv2.polylines(bg,[contour(1)],False,(33,56,45),2,cv2.LINE_AA)
        for p in contour(.93)[3::8]:
            cv2.line(bg,(x,y),tuple(p),(26,50,40),1)
    return cv2.GaussianBlur(bg,(0,0),3.2).astype(np.uint8)


class DayGarden(base.Garden):
    bloom_pace = 1.

    def __init__(self, theme, variant):
        super().__init__("cup" if theme=="dew" else "overlap",0)
        self.theme=theme
        self.cycle=Cycle(theme,self.bloom_pace)
        self.sway=np.repeat(self.cycle.stem,2,axis=0)
        self.az,self.elev=np.deg2rad(VIEWS[theme][variant])
        self.roll=np.deg2rad({"dew":-4,"gust":-2,"sun":3}[theme])
        self.xmid={"dew":.499,"gust":.552,"sun":.52}[theme]*base.W
        self.ymid={"dew":.49,"gust":.512,"sun":.50}[theme]*base.H
        self.scale={"dew":.495,"gust":.470,"sun":.548}[theme]
        self.original_normals=[base.normals(o) for b,o in self.sheets]
        self.ground=pond(theme)
        self.current=0
        self.cache={}
        self.periphery=U**1.85*(.74+.26*abs(V))
        self.pondmap=np.mgrid[:base.H,:base.W][::-1].astype('f')

    def tissue(self,i,t):
        if i in self.cache and t==self.current:
            return self.cache[i]
        step=round(t*RATE)%(RATE*PERIOD)
        q=self.cycle.open[step,i,:,None]
        b,o=self.sheets[i]
        mode="cup" if self.theme=="sun" and i<12 else self.scene
        p=base.bend(b,o,q,i,mode,0)
        angle=self.cycle.flex[step,i]
        blade=base.PHASE[i]
        free=self.periphery
        p[:,:,0]+=36*angle*free
        p[:,:,1]+=(16*np.cos(blade)*angle)*free
        p[:,:,2]+=(30*np.sin(blade+.5)*angle+22*angle*V)*free
        # A drop adds weight until it leaves. Recoil decays under damping.
        if self.theme=="dew" and i==9:
            elapsed=t-8.45
            if elapsed<0:
                stress=base.smooth((t-1.3)/4.8)
            else:
                stress=np.exp(-2.2*elapsed)*np.cos(6.7*elapsed)
            p[:,:,2]-=45*stress*free
        value=(p,self.tex[i],q)
        if t==self.current:
            self.cache[i]=value
        return value

    def stem(self,t):
        # Continue the same stalk beyond the portrait in the higher camera view.
        # At s=0 the continuation matches the position, radius and tangent of the stalk.
        super().stem(t)
        s=np.linspace(0,.88,30)[:,None]
        theta=np.linspace(0,2*np.pi,15)[None,:]
        radius=9.1+2.2*s
        x=26*np.sin(3.88)-2+91+s*(26*3.8*np.cos(3.88)+91*1.27)+radius*np.cos(theta)
        depth=-3+9*s+radius*np.sin(theta)
        z=-815*(1+s)+np.zeros_like(theta)
        shade=.38+.54*(.5+.5*np.cos(theta-2.2))+.07*np.sin(theta*5)
        self.mesh(np.stack([x,depth,z],axis=-1),np.array([99,121,59])*shade[...,None])

    def sun_direction(self, t):
        theta = TAU*t/PERIOD
        return base.unit(np.array([-.53+.34*np.sin(theta-.6),.37,.87],dtype='f'))

    def render(self,t):
        t=float(t)
        self.current=t
        self.cache={}
        step=round(t*RATE)%(RATE*PERIOD)
        self.load=self.cycle.stem[step]
        self.z=np.full((base.H,base.W),-1e6,'f')
        self.rgb=np.zeros((base.H,base.W,3),np.uint8)
        self.stem(t)
        opening=float(np.mean(self.cycle.open[step]))
        theta=TAU*t/PERIOD
        sun=self.sun_direction(t)
        lighting = self.petal_lighting(t, sun) if hasattr(self, 'petal_lighting') else None
        for i,rest in enumerate(self.original_normals):
            if i==11:
                self.heart(opening)
                red,green,blue=np.moveaxis(self.rgb.astype(np.int16),-1,0)
                anther=(red>110)&(green>blue*1.48)&(red>green)
                self.z[anther]=np.minimum(self.z[anther],50)
            p,tex,q=self.tissue(i,t)
            n=base.normals(p)
            gain=np.clip((.59+.57*abs(n@sun))/(.66+.45*abs(rest@self.light)),.59,1.46)
            # The closed waxy outer wall presents a brighter continuous face to the
            # broad sky. Keep the vein pattern and the dark lower attachments.
            closed=1-base.smooth(opening/.76)
            face=(.66+.34*U)*(.48+.52*abs(n@sun))
            pigment=tex*(gain*(.84+.16*q+.10*U))[...,None]
            # Compress strong reflections toward the sky tint to preserve the lilac hue.
            sky_reflection=np.minimum(.76,closed*face*(.49+.52*np.mean(tex,axis=2)/220))
            pigment+=sky_reflection[...,None]*(np.array([249,207,247])-pigment)
            if lighting is not None:
                pigment = lighting(i, p, n, pigment)
            self.mesh(p,pigment)
        occupied=self.z>-1e5
        shade=np.zeros_like(self.z)
        for d in (3,7):
            near=np.roll(self.z,(d,d),axis=(0,1))
            shade+=np.clip((near-self.z-17)/56,0,1)*occupied
        colored=np.uint8(self.rgb*(1-.091*cv2.GaussianBlur(shade,(7,7),1.35)[...,None]))
        # One low-frequency surface oscillation distorts the distant bank's reflection.
        x,y=self.pondmap
        warp=.8*np.sin(y*.095+theta*4)+1.25*np.sin(y*.047-theta*3)
        water=base.smooth((y-490)/140)
        ground=cv2.remap(self.ground,x+warp*water,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        return np.where(occupied[...,None],colored,ground)
