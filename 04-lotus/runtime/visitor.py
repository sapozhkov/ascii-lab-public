"""A bumblebee in the lotus' coordinates. The flower and insect share depth."""
from math import cos, exp, pi, sin
from typing import NamedTuple

import cv2
import numpy as np

from lotus_weather import sample_patch


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x*x*(3-2*x)


def support(t, on, off):
    return ease((t-on)/.23)*(1-ease((t-off)/.19))


def visit_load(story, t):
    """Fraction of weight on the landing petal (rim) and on the green pad."""
    if story in ('rim','catch'):
        return support(t, 7.48, 11.66), 0.
    if story == 'leaf':
        return support(t, 11.35, 14.25), support(t, 3.38, 6.92)
    return support(t,10.25,13.95), 0.


class Pose(NamedTuple):
    center: np.ndarray       # centre of the thorax, world coordinates
    foot: np.ndarray         # current target on the plant
    yaw: float
    pitch: float
    contact: float
    looking: float           # fraction of an active probing/feeding stroke
    pollen: float            # pollen visible on the hind tibia
    flying: bool
    bank: float = 0.


class Visitor:
    def __init__(self, story, variant):
        if story not in ('rim','cup','leaf','catch'):
            raise ValueError(story)
        self.story = story
        self.variant = variant
        self.step = 1/120
        self.start, self.end = {'rim': (3.0, 15.1), 'cup': (5.3,17.2),
                                'leaf': (.8,17.8), 'catch': (3.,15.1)}[story]
        self.sides = {'rim': 1., 'cup': -1., 'leaf': -1., 'catch': -1.}
        self._motion = self._flight()
        self.thorax = self._coat(34, 46, 'thorax')
        self.abdomen = self._coat(46, 46, 'abdomen')
        self.head = self._coat(22, 26, 'head')
        self.eye = self._coat(16, 20, 'eye')

    def _flight(self):
        """Integrate tracking thrust, drag and torque between sighted perches.

        A target consists of a perch interpolation coordinate and XYZ offset.
        The physical coordinates of both perches are evaluated in the current
        plant state, so tarsi stay on the flexing tissue while in contact.
        """
        side = self.sides[self.story]
        if self.story in ('rim','catch'):
            stops = [(0, [0, 690, 60, 180]), (3.05,[0, 144,38,150]),
                     (4.62,[0, 62,35,145]), (5.25,[0, 93,63,109]),
                     (6.37,[0, 5,8,31]), (7.04,[0,0,0,0]),
                     (11.65,[0,30,30,97]),(12.61,[0,144,49,209]),
                     (13.34,[0,680,5,311])]
        elif self.story == 'cup':
            stops = [(0,[1,-700,90,280]), (5.5,[1,-178,52,181]),
                     (7.24,[1,-100,73,155]), (7.96,[1,-113,53,105]),
                     (8.96,[1,-48,44,48]), (9.79,[1,0,0,0]),
                     (13.95,[1,23,67,100]),(14.86,[1,135,72,165]),
                     (15.75,[1,680,35,261])]
        else:
            stops = [(0,[0,-720,30,152]),(.94,[0,-105,58,133]),
                     (1.94,[0,-59,32,51]),(2.69,[0,0,0,0]),
                     (6.9,[0,24,11,115]),(7.84,[.50,-14,60,154]),
                     (8.97,[1,-84,55,148]),(9.74,[1,-51,40,67]),
                     (10.73,[1,0,0,0]),(14.25,[1,25,43,130]),
                     (15.25,[1,157,70,165]),(16.05,[1,680,50,220])]
        if self.story == 'catch':
            # Enter from the far bank and approach the left-hand lip. The wet
            # petal and the receiving pad stay clear of the insect silhouette.
            stops=[(t,[w,-x,y,z]) for t,(w,x,y,z) in stops]
        n = int(20/self.step) + 1
        history = np.empty((n,7))
        pos = np.array(stops[0][1],float)
        velocity = np.zeros(4)
        yaw = pi if side>0 else .1
        pitch = 0.
        perch = 0
        for i in range(n):
            t = i*self.step
            if perch+1<len(stops) and t>=stops[perch+1][0]:
                perch += 1
            goal = np.array(stops[perch][1],float)
            on, pad = visit_load(self.story,t)
            contact = max(on,pad)
            if self.story in ('cup','leaf'):
                feeding = support(t,10.25 if self.story=='cup' else 11.35,
                                  13.95 if self.story=='cup' else 14.25)
                contact = max(contact,feeding)
            # Small changes of thrust oppose the local breeze in a hover. The
            # touchdown controller damps them within one body length of the perch.
            airborne = min(1., np.linalg.norm(goal[1:])/60)*(1-contact)
            goal[1:] += airborne*np.array([6*sin(2.9*t)+3*cos(6.7*t),
                                            3*sin(3.1*t+.6),6*cos(3.9*t)])
            acceleration = 18*(goal-pos)-7.0*velocity
            acceleration[1:] = np.clip(acceleration[1:],-1080,1080)
            velocity += self.step*acceleration
            pos += self.step*velocity
            approach = pi-.28 if side>0 else .18
            if self.story=='leaf' and pos[0]<.66:
                approach = -.18
            if abs(velocity[1])>38:
                approach = np.arctan2(velocity[2],velocity[1])
            # Maximum turn rate and pitch depend on thrust rather than frame time.
            turn = (approach-yaw+pi)%(2*pi)-pi
            yaw += np.clip(5.8*turn,-4.8,4.8)*self.step
            target_pitch = np.clip(velocity[3]/(180+abs(velocity[1])),-.29,.29)
            if contact:
                target_pitch -= .20*contact
            pitch += (target_pitch-pitch)*min(1,6*self.step)
            history[i] = [*pos,yaw,pitch,contact]
        return history

    def pose(self,g,t):
        t = float(t)%20
        k = int(round(t/self.step))
        weight, dx,dy,dz,yaw,pitch,contact = self._motion[k]
        if self.story=='rim':
            support_point = sample_patch(g.tissue(9,t)[0], .56, .20)
        elif self.story=='catch':
            support_point = sample_patch(g.tissue(10,t)[0], .67, .49)
        elif self.story=='cup':
            support_point = sample_patch(g.tissue(12,t)[0],.39,.16)
        else:
            support_point = ((1-weight)*np.asarray(g.leaf_point(t)) +
                             weight*sample_patch(g.tissue(12,t)[0],.39,.16))
        feeding = 0.
        if self.story in ('cup','leaf'):
            feeding = support(t,10.33 if self.story=='cup' else 11.47,
                              13.86 if self.story=='cup' else 14.18)
        # Short exploratory steps lead towards the next patch of anthers.
        probe = feeding*(.46+.54*cos(5.7*t+.45*sin(1.8*t)))
        foot = support_point+np.array([3.8*probe,1.6*probe,0.])
        offset = (1-contact)*np.array([dx,dy,dz])
        center = foot+offset+[0,0,27.5-2.9*contact-2.7*probe]
        pollen = ease((t-(10.5 if self.story=='cup' else 11.8))/2.3) if self.story in ('cup','leaf') else 0.
        return Pose(center,foot,yaw,pitch-.08*probe,contact,probe,pollen,contact<.87)

    @staticmethod
    def _coat(rows, columns, part):
        polar = np.linspace(0,pi,rows)[:,None]
        azimuth = np.linspace(0,2*pi,columns)[None,:]
        x = np.cos(polar)*np.ones_like(azimuth)
        y,z = np.sin(polar)*np.cos(azimuth),np.sin(polar)*np.sin(azimuth)
        normals = np.stack([x,y,z],-1)
        if part=='eye':
            color = np.empty_like(normals)
            color[:] = (40,43,44)
            return normals,color,normals
        grain = (.34*np.sin(23*azimuth+37*polar)*np.sin(14*polar-19*azimuth)+
                 .28*np.cos(29*azimuth-46*polar)+.27*np.cos(19*azimuth+52*polar))
        roughness = 1+(0.025 if part=='head' else .065)*grain
        mesh = normals*roughness[...,None]
        def blend(a,b,w):
            return np.asarray(a)+(np.asarray(b)-a)*w[...,None]
        if part=='abdomen':
            # Copper-yellow first tergite, black middle and a pale buff tail.
            border = x+.045*np.sin(3*azimuth)+.04*grain
            band = np.clip((border-.10)*5.5,0,1)
            tail = np.clip((-.42-border)*5.3,0,1)
            color = blend([31,26,32],[218,145,44],band)
            color = blend(color,[220,206,176],tail)
            mesh[:,:,1:] *= (1+.11*x)[...,None]
        elif part=='thorax':
            collar = np.clip((x+.12+.04*grain)*3.2,0,1)
            color=blend([49,36,29],[233,172,55],collar)
        else:
            color = blend([22,23,26],[56,49,41],np.clip(.4+.5*z,0,1))
        return mesh,color*(1+.17*grain[...,None]),normals

    def leg_pose(self,pose,t,side,pair):
        """Leg extension, fore-aft stride and toe lift in body coordinates."""
        return pose.contact, 0., 0.

    def render(self,g,rgb,t):
        """Composite 3D surfaces and translucent wings, sharing the garden z-buffer."""
        if not self.start<=t<=self.end:
            return rgb
        pose=self.pose(g,t)
        center=g.project(pose.center)
        if center[0]<-65 or center[0]>rgb.shape[1]+65 or center[1]<-55 or center[1]>rgb.shape[0]+55:
            return rgb
        yaw,pitch=pose.yaw,pose.pitch
        fwd=np.array([cos(yaw)*cos(pitch),sin(yaw)*cos(pitch),sin(pitch)])
        lateral=np.array([-sin(yaw),cos(yaw),0])
        up=np.cross(fwd,lateral)
        if pose.bank:
            cb,sb=cos(pose.bank),sin(pose.bank)
            lateral,up=cb*lateral-sb*up,sb*lateral+cb*up
        axes=np.stack([fwd,lateral,up])
        def world(local):
            return (1.23*np.asarray(local))@axes+pose.center
        def paint_line(points,color,width):
            points=g.project(world(points))
            for a,b in zip(points[:-1],points[1:]):
                delta=b-a
                count=max(2,int(np.linalg.norm(delta[:2])/2))
                line=np.linspace(a,b,count)
                for mark in line:
                    x,y,z=mark
                    ix,iy=int(round(x)),int(round(y))
                    if 0<=ix<rgb.shape[1] and 0<=iy<rgb.shape[0] and g.z[iy,ix]<z+1:
                        cv2.circle(rgb,(round(x*16),round(y*16)),round(width*8),color,-1,cv2.LINE_AA,4)
        if hasattr(g,'lightfield'):
            x=int(np.clip(round(center[0]),0,rgb.shape[1]-1))
            y=int(np.clip(round(center[1]),0,rgb.shape[0]-1))
            sun=float(g.lightfield[y,x])
        else:
            sun=.45
        reflection = getattr(g, 'bee_reflection', None)
        def solid(coat, origin, radii):
            mesh,pigment,normal=coat
            n=normal@axes
            light=np.maximum(0,n@np.array([-.43,.42,.80]))
            sky=np.maximum(0,n[:,:,2])
            shade=.59+(.42+.25*sun)*light+.23*sky
            color=np.clip(pigment*shade[...,None]+light[...,None]*[13,10,5],0,255)
            points = world(mesh*np.asarray(radii)+origin)
            if reflection is not None:
                color = reflection(points,n,color,t)
            g.mesh(points,color)
        # Foreshortened shadow follows the receiving surface and softens before landing.
        ival=np.clip(1-np.linalg.norm(pose.center-pose.foot)/120,0,1)
        if ival>0:
            fx,fy,fz=g.project(pose.foot)
            if -28<fx<rgb.shape[1]+28 and -16<fy<rgb.shape[0]+16:
                x0,x1=max(0,int(fx-27)),min(rgb.shape[1],int(fx+28))
                y0,y1=max(0,int(fy-15)),min(rgb.shape[0],int(fy+16))
                yy,xx=np.mgrid[y0:y1,x0:x1]
                nearby=abs(g.z[y0:y1,x0:x1]-fz)<55
                shadow=.48*ival*np.exp(-((xx-fx-3)/16)**2-((yy-fy)/5.5)**2)*nearby
                view=rgb[y0:y1,x0:x1]
                view[:]=np.uint8(view*(1-shadow[...,None]))
        prior=g.rgb
        g.rgb=rgb
        try:
            # Pairwise tarsi, tibiae and femora. The hind tibia carries a load of pollen.
            for sign in (-1,1):
                for i,(hip,toe) in enumerate(((9,24),(0,-1),(-10,-31))):
                    reach,swing,lift=self.leg_pose(pose,t,sign,i)
                    knee=np.array([hip-4,sign*(10+reach*3),-7.])
                    ankle=np.array([toe-4+.7*swing,sign*(14+reach*5),-15-2.1*reach+.7*lift])
                    tip=np.array([toe+1.5+swing,sign*(14+reach*7),-17-4*reach+lift])
                    paint_line([[hip,sign*7,-4],knee,ankle,tip],(58,46,43),2.2)
                    if i==2 and pose.pollen>.04:
                        solid(self.eye,ankle+[0,0,1],[3.6,2.6,5])
                        paint_line([knee*.32+ankle*.68,ankle],(215,151,49),3.9*pose.pollen+.2)
            solid(self.abdomen,[-23,0,0],[25.4,15.5,14.2])
            solid(self.thorax,[1,0,2],[17.5,16.7,17.3])
            solid(self.head,[20,0,-2.6],[11.0,12.2,11.8])
            for sign in (-1,1):
                solid(self.eye,[23.4,sign*10.,-.6],[5.4,3.6,6.6])
                paint_line([[25,sign*6,3],[29,sign*6.5,12],[34.2,sign*9,15],
                            [37,sign*11,13]],(100,87,72),1.5)
            # Short guard hairs grow continuously out of the thoracic coat.
            # Draw only the hairs grazing the view ray; the coat texture supplies the rest.
            mesh,pigment,nrm=self.thorax
            strands=mesh[2:-2:3,::3].reshape(-1,3)
            outward=nrm[2:-2:3,::3].reshape(-1,3)
            tint=pigment[2:-2:3,::3].reshape(-1,3)
            facing=(outward@axes)@np.array([sin(g.az)*cos(g.elev),cos(g.az)*cos(g.elev),sin(g.elev)])
            rim=(abs(facing)<.27)&(outward[:,2]>-.25)
            for a,n,c in zip(strands[rim],outward[rim],tint[rim]):
                root=a*[17.5,16.7,17.3]+[1,0,2]
                tip=root+n*(2.3+.8*n[2])-[.8,0,0]
                paint_line([root,tip],tuple(np.clip(c*(.88+.21*sun)+9,0,255)),.95)
            # Long tongue is visible only while its tip is actually sampling the crown.
            if pose.looking>.2:
                paint_line([[28,0,-6],[33,0,-12],[35,0,-17-pose.looking*4]],(72,43,34),1.4)
            if getattr(self, 'wings', None) is not None:
                self.wings.draw(g, rgb, world, t)
            else:
                opening=1-pose.contact
                for sign in (-1,1):
                    for hind in (True,False):
                        self._wing(g,rgb,world,sign,hind,opening,t)
        finally:
            g.rgb=prior
        return rgb

    @staticmethod
    def _wing(g,rgb,world,sign,hind,opening,t):
        """A translucent membrane integrates a fan of wing positions during exposure."""
        root=np.array([-.5 if hind else 5,sign*9.5,10])
        n=25
        s=np.linspace(0,1,n)
        widths=(6.2 if hind else 8.8)*np.sin(np.pi*s)**.65
        length=(30 if hind else 42)
        # Small slow variation is the change of the mean stroke plane on banking.
        # Fast wingbeats are integrated in the image, avoiding sampled flicker.
        for delta,alpha in ((-23,.15),(22,.18),(0,.49)):
            if delta and opening<.08: continue
            a=np.deg2rad(11+opening*(65+delta+4*sin(3.4*t))+(-14 if hind else 0))
            direction=np.array([-cos(a),sign*sin(a),.18+.32*opening])
            width=np.array([sin(a),sign*cos(a),.16])
            mid=root+s[:,None]*direction*length
            mid[:,2]+=np.sin(np.pi*s)*(4.2+2*opening)
            outline=np.concatenate((mid+widths[:,None]*width, (mid-.63*widths[:,None]*width)[::-1]))
            projected=g.project(world(outline))
            low=np.floor(projected[:,:2].min(axis=0)-2).astype(int)
            high=np.ceil(projected[:,:2].max(axis=0)+2).astype(int)
            x0,y0=np.maximum(low,[0,0]); x1,y1=np.minimum(high,[rgb.shape[1],rgb.shape[0]])
            if x0>=x1 or y0>=y1:continue
            shape=(y1-y0,x1-x0)
            uv=projected[:,:2]-[x0,y0]
            mask=np.zeros(shape,np.uint8)
            poly=np.rint(uv*16).astype('int32')
            cv2.fillPoly(mask,[poly],round(255*alpha),cv2.LINE_AA,shift=4)
            if delta==0:
                cv2.polylines(mask,[poly],True,round(235*(.82 if opening<.2 else .65)),1,cv2.LINE_AA,4)
                vein=g.project(world(mid - widths[:,None]*width*.17))[:,:2]-[x0,y0]
                cv2.polylines(mask,[np.int32(np.rint(vein*16))],False,155,1,cv2.LINE_AA,4)
                for start,end in ((.25,.65),(.48,.88)):
                    a,b=round(start*(n-1)),round(end*(n-1))
                    branch=mid[a:b]+widths[a:b,None]*width*np.linspace(-.17,.75,b-a)[:,None]
                    branch=g.project(world(branch))[:,:2]-[x0,y0]
                    cv2.polylines(mask,[np.int32(np.rint(branch*16))],False,144,1,cv2.LINE_AA,4)
            # Local tangent depth interpolates along the membrane; neighbouring
            # petals and the near side of the insect can hide a portion of it.
            plane=np.linalg.lstsq(np.c_[uv,np.ones(len(uv))],projected[:,2],rcond=None)[0]
            yy,xx=np.mgrid[0:shape[0],0:shape[1]]
            visible=xx*plane[0]+yy*plane[1]+plane[2]+1.2>=g.z[y0:y1,x0:x1]
            tint=np.array([183,195,192]) if delta==0 else np.array([145,158,164])
            w=(visible*mask.astype('f')/255)[...,None]
            bg=rgb[y0:y1,x0:x1]
            bg[:]=np.uint8(bg*(1-w)+tint*w)
