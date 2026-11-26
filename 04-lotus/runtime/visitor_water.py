"""Two-stage water path from the loaded petal onto a second lotus leaf."""
import numpy as np
from lotus_weather import RELEASE, dew_bead, ease, projected_at, ripples, sample_patch

HIT=9.38
LEAVE=11.42
SPLASH=12.35


def leaf_impulse(t):
    """Transient tilt caused by the bead striking the raised blade."""
    a=t-HIT
    return 0 if a<=0 else .48*np.sin(8.2*a)*np.exp(-2.8*a)


class LeafTransfer:
    def __init__(self):
        self.anchors=None

    def point(self,g,t):
        """Return projected position, deformation and support at the current time."""
        def leaf(time,u):
            sheet,_=g.leaves[1].geometry(time,leaf_impulse(time))
            return sample_patch(sheet,u,.091-.035*u)

        if self.anchors is None:
            tip=sample_patch(g.tissue(9,RELEASE)[0],.999,.49)
            launch=projected_at(g,tip,RELEASE).copy()
            contact=projected_at(g,leaf(HIT,.16),HIT).copy()
            second=projected_at(g,leaf(LEAVE,.995),LEAVE).copy()
            water=second+np.array([14,g.scale*190*np.cos(g.elev),0])
            self.anchors=launch,contact,second,water
        launch,contact,second,water=self.anchors
        if t<HIT:
            f=np.clip((t-RELEASE)/(HIT-RELEASE),0,1)
            pos=launch+(contact-launch)*[f,f*f,f]
            return pos,1.19,'air'
        if t<LEAVE:
            elapsed=t-HIT
            f=np.clip(elapsed/(LEAVE-HIT),0,1)
            # Gravity overcomes pinning on the inclined waxy surface.
            u=.16+.835*f**1.46
            pos=g.project(leaf(t,u))
            aspect=.85-.44*np.exp(-elapsed/.14)+.14*np.sin(14*elapsed)*np.exp(-elapsed/.4)
            return pos,aspect,'leaf'
        if t<SPLASH:
            f=np.clip((t-LEAVE)/(SPLASH-LEAVE),0,1)
            return second+(water-second)*[f,f*f,f],1.18,'air'
        return water,1.,'water'

    def draw(self,rgb,g,t,rain):
        if t<RELEASE:
            rain.dew(g,rgb,t)
            return
        pos,aspect,support=self.point(g,t)
        x,y,z=pos
        if support=='water':
            ripples(rgb,x,y-4,t-SPLASH)
        else:
            dew_bead(rgb,x,y-5,10.7,aspect)


class LoadedDrop:
    """Slide along the loaded rib until the contact line reaches the pointed lip."""
    def __init__(self,bearing,rate):
        self.rate=rate
        self.travel=np.zeros(len(bearing))
        progress=speed=0.
        self.release=None
        for i,load in enumerate(bearing[:,0]):
            t=i/rate
            # Increasing water mass and slope are opposed by pinning and viscous drag.
            gravity=.25+.25*ease((t-2.8)/3)+.66*load
            speed=max(0,speed+(1.9*(gravity-.34)-1.1*speed)/rate)
            progress+=speed*1.04/rate
            self.travel[i]=min(1,progress)
            if progress>=1 and self.release is None:self.release=t
        if self.release is None: raise ValueError('The water never reaches the tip')
        self.impact=self.release+1.15
        self.launch=None

    def draw(self,rgb,g,t):
        if t<1.05:return
        if t<self.release:
            slide=self.travel[round(t*self.rate)%len(self.travel)]
            p=sample_patch(g.tissue(9,t)[0],.34+.659*slide,.45-.01*slide)
            x,y,z=g.project(p)
            r=(6.3+6.0*ease((t-1.05)/4))*ease((t-1.05)/.65)
            iy,ix=round(y),round(x)
            if r>.18 and 0<=iy<rgb.shape[0] and 0<=ix<rgb.shape[1] and g.z[iy,ix]<z+16:
                dew_bead(rgb,x,y-r*.65,r,.74+.41*ease((slide-.87)/.13))
        else:
            if self.launch is None:
                p=sample_patch(g.tissue(9,self.release)[0],.999,.44)
                self.launch=projected_at(g,p,self.release)
            x,y,_=self.launch
            f=min(1,(t-self.release)/(self.impact-self.release))
            x+=18*f
            y+=g.scale*309*np.cos(g.elev)*f*f-6
            if t<self.impact:dew_bead(rgb,x,y,11.3,1.19)
            else:ripples(rgb,x,y,t-self.impact)
