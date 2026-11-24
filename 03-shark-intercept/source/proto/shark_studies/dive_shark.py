"""ASCII surface of the dive swimmer. All features use the physical backbone."""

import cv2
import numpy as np

from . import reef_shark as anatomy
from .dive_pose import Swimmer, mouth, world_points
from .encounter_skin import COLS, ROWS, SAMPLES, RAMP, mesh, surface_color


def cheek(x,z,side):
    x,z=np.broadcast_arrays(x,z)
    width=np.interp(x,anatomy.STATIONS,anatomy.HALF_WIDTH)
    half=np.where(z>=0,np.interp(x,anatomy.STATIONS,anatomy.BACK),
                  np.interp(x,anatomy.STATIONS,anatomy.BELLY))
    y=side*width*np.sqrt(np.clip(1-(z/np.maximum(.004,half))**2,0,1))
    return np.column_stack((x,y,z))


def jaw_contours(pose, cam, side):
    """Project the upper maxillary edge, mobile mandible and dark buccal space."""
    gape=pose.gape
    reach=np.array([0.,.22,.52,.81,.96,1.])
    x=.342+.144*reach
    upper=cheek(x,[-.017,-.016,-.011,-.006,-.002,-.001],side)
    lower=cheek(x-.013*reach*gape,[-.017,-.021,-.026,-.028,-.024,-.014],side)
    # The mandible ends below the projecting rostrum. Its inside faces
    # turn down while the narrow maxillary edge still catches skylight.
    roof=world_points(upper,pose)+(.007*gape*reach)[:,None]*pose.up
    jaw=world_points(lower,pose)-(.014*gape*reach)[:,None]*pose.up
    lower_edge=jaw-(.012*reach)[:,None]*pose.up
    def project(points):
        return np.rint(cam.project(points)*[SAMPLES,SAMPLES/1.5]).astype(np.int32)
    return project(roof),project(jaw),project(lower_edge)


def bite_visibility(pose,cam,coverage):
    """Opacity for the prey portion still visible outside the near lip."""
    visible=1-np.clip((coverage-.16)/.49,0,1)
    if pose.gape < .025:
        return visible
    sign=1 if pose.right @ cam.axes[2] >= 0 else -1
    roof,jaw,_=jaw_contours(pose,cam,sign)
    cuts=np.zeros((ROWS*SAMPLES,COLS*SAMPLES),np.uint8)
    # Only the outer lip can show a fish sliding in. Once it crosses the
    # cheek the near side of the animal hides its swallowed portion.
    cv2.fillPoly(cuts,[np.vstack((roof[-3:],jaw[:-4:-1]))],255)
    opening=cv2.resize(cuts,(COLS,ROWS),interpolation=cv2.INTER_AREA)/255
    return np.maximum(visible,opening)


def cell_ink(canvas):
    reduced=cv2.resize(canvas,(COLS,ROWS),interpolation=cv2.INTER_AREA)
    coverage=reduced[...,3].astype(np.float32)/255
    light=reduced[...,:3]/np.maximum(coverage[...,None],.001)
    lum=np.einsum('ijk,k->ij',light,[.23,.65,.12])
    strength=np.clip((lum-17)/206,0,1)
    chars=RAMP[np.rint((.12+.88*strength**.82)*(len(RAMP)-1)).astype(int)]
    dy,dx=np.gradient(coverage)
    edge=(coverage>.12)&(coverage<.67)
    edge_sign=np.where(abs(dx)>1.7*abs(dy),ord('|'),ord('_'))
    slant=(abs(dx)<=1.7*abs(dy))&(abs(dy)<=1.7*abs(dx))
    edge_sign=np.where(slant,np.where(dx*dy>0,ord('/'),ord(chr(92))),edge_sign)
    chars[edge]=edge_sign[edge]-32
    ink=np.clip(1.24*light+[9,15,19],0,255).astype(np.uint8)
    # Most of the light stays on the printed surface. The narrow brightest
    # reflections retain a little scattered light between the strokes.
    paper=np.clip(light*(.09+.37*strength[...,None]**2),0,255).astype(np.uint8)
    return coverage,chars,ink,paper


def shark_cells(pose,cam,t, *, draw_eyes=True):
    """Rasterize a supersampled moving mesh, then select one symbol per cell."""
    vertices,indices,parts,rest=mesh()
    world=world_points(vertices,pose)
    camera_space=np.einsum('ij,kj->ik',world-pose.center,cam.axes)
    faces=camera_space[indices]
    normals=np.cross(faces[:,2]-faces[:,0],faces[:,3]-faces[:,1])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
    normals*=np.where(normals[:,2]<0,-1.,1.)[:,None]
    rotation=np.array([pose.forward,pose.right,pose.up]) @ cam.axes.T
    length=cam.scale_at(pose.center) if hasattr(cam,'scale_at') else cam.scale
    rgb=surface_color(rest,normals,parts,rotation,cam.project(pose.center),length,t)
    rgba=np.column_stack((rgb,np.full(len(rgb),255))).tolist()
    resolution=np.array([SAMPLES,SAMPLES/1.5])
    points=np.rint(cam.project(world)*resolution).astype(np.int32)[indices]
    canvas=np.zeros((ROWS*SAMPLES,COLS*SAMPLES,4),np.uint8)
    for i in np.argsort(faces[:,:,2].mean(axis=1)):
        cv2.fillConvexPoly(canvas,points[i],rgba[i])

    def project(points):
        xy=cam.project(world_points(points,pose))
        return np.rint(xy*resolution).astype(np.int32)

    eyes=[]
    for side in (-1,1):
        facing=side*rotation[1,2]
        if draw_eyes and facing+.29*rotation[2,2]+.22*rotation[0,2]>.17:
            eye=project(cheek([.429],[.016],side))[0]
            eyes.append(eye)
            r=max(1,round(.009*length*SAMPLES))
            cv2.ellipse(canvas,eye,(r,max(1,round(.75*r))),0,0,360,(155,174,160,255),-1)
            cv2.circle(canvas,eye,max(1,round(.62*r)),(12,22,27,255),-1)
        if facing>.18:
            for gill in (.265,.283,.301,.319,.335):
                slit=cheek([gill,gill-.007,gill-.005],[.019,-.001,-.022],side)
                cv2.polylines(canvas,[project(slit)],False,(29,44,50,255),max(1,round(length*.003*SAMPLES)))
            roof,jaw,lower_edge=jaw_contours(pose,cam,side)
            if pose.gape>.065:
                # The cold dark cavity has a defined lower lip. Its hue stays
                # neutral so the amber prey remains the distinct focal point.
                opening=np.vstack((roof,jaw[::-1]))
                cv2.fillPoly(canvas,[opening],(4,9,13,255))
                cv2.fillPoly(canvas,[np.vstack((jaw,lower_edge[::-1]))],(141,156,152,255))
                cv2.polylines(canvas,[jaw[1:]],False,(192,199,180,255),max(1,round(length*.0045*SAMPLES)))
                cv2.polylines(canvas,[roof[1:]],False,(154,170,167,255),max(1,round(length*.0035*SAMPLES)))
            else:
                cv2.polylines(canvas,[jaw],False,(40,49,52,255),max(1,round(length*.003*SAMPLES)))

    coverage,chars,ink,paper=cell_ink(canvas)
    for px,py in eyes:
        x,y=px//SAMPLES,py//SAMPLES
        if length>=38 and 0<=x<COLS and 0<=y<ROWS and coverage[y,x]>.65:
            chars[y,x]=ord('o')-32
            ink[y,x]=[192,194,166]
            paper[y,x]=[13,23,29]
    return coverage,chars,ink,paper
