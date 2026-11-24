"""Render the three 20-second shark encounters and their casting camera studies.

Run from the project root: python3 -m proto.shark_studies.encounter_render 01 a
"""
import argparse
import inspect
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proto.glyph_atlas import glyph_atlas
from . import encounter_motion as motion
from .encounter_skin import shark_cells
from .encounter_water import (Camera, X, Y, environment, noise, select, glyphs,
                              small_fish, smoothstep, unit)
from .school_render import FONT, paint, WIDTH, SCENE_H, FRAME_H


cv2.setNumThreads(2)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/scene-selection/2026-11-24-shark-encounters"
FPS = motion.FPS


def camera(scene, frame, t, variant):
    if scene == '01':
        from .encounter_fold import fold_camera
        return fold_camera(frame,t,variant)
    if scene == '10':
        from .encounter_window import window_camera
        return window_camera(frame,t,variant)
    shark = frame.shark
    centre = shark.center.copy()
    fish = np.median(frame.fish,axis=0)
    # Let the reef tip serve as a fixed depth reference through the miss.
    interest = .47*smoothstep(1.4,6,t)
    focus = centre + interest*np.clip(fish-centre,-.92,.92)
    # Swing around the outside shoulder during the banking recovery.
    yaw = np.arctan2(shark.forward[1],shark.forward[0])
    az = [-.08,-.56,.29][variant] + .79*yaw*smoothstep(7,14,t)
    elevation = [.18,.43,-.28][variant]
    scale = [79,76,78][variant]-23*smoothstep(2,6,t)+2*smoothstep(14,19,t)
    aim = [43+3*smoothstep(7,15,t),50]
    return Camera(focus,az,elevation,scale,aim)


def reef_tip(cam,t,reef_geometry=None):
    """A fixed line of far corals and a nearer eroded outcrop for visual scale."""
    # A near exposed lip catches amber skylight. Behind it the same limestone
    # breaks into softer blue ledges. Both are fixed in world coordinates.
    # Sample along horizontal camera rays. At grazing angles this keeps
    # the visible end of the outcrop bounded as the camera rounds it.
    rgb = np.zeros((72,90,3))
    substrate = np.zeros_like(rgb)
    symbols = np.zeros((72,90),np.uint8)
    occupied = np.zeros((72,90),bool)
    sideways = cam.axes[0,:2]
    far = np.array([-sideways[1],sideways[0]])
    u = (X[0]-cam.aim[0])/cam.scale
    depths = (2.15,1.68,1.23,.88,.60,.34)
    if reef_geometry is not None:
        reef, height_at = reef_geometry
        peak_depth = (np.array([reef["x"], reef["y"]]) - cam.focus[:2]) @ far
        depths = sorted((*depths, peak_depth), reverse=True)
    for depth in depths:
        ground = cam.focus[:2]+u[:,None]*sideways+depth*far
        wx,wy = ground.T
        grain = noise(wx*7,wy*7)
        if reef_geometry is None:
            height = .024+.035*np.sin(8*wx+3*wy)+.058*grain
            ledge = np.exp(-((wy-.65-.23*np.sin(2.1*wx))/.42)**2)
            height += ledge*(.18+.10*np.sin(6*wx)+.08*grain)
            height += .44*np.exp(-((wx-.15)/.14)**2-((wy-.53)/.23)**2)
        else:
            height = height_at(wx, wy)
        tops = cam.project(np.column_stack((ground,height)))
        h = Y - tops[:,1][None,:]
        wx = wx[None,:]
        opacity = np.exp(-.53*depth)
        grain = noise(17*wx+.18*h,.23*h+depth)
        relief = .61*noise(5*wx+.03*h,.073*h+depth)+.39*grain
        step = .5+.5*np.sin(.79*h+4*relief+8*wx)
        lip = np.exp(-np.maximum(h,0)/5)*(1+.19*np.sin(t*.35+11*wx))
        calcium = smoothstep(.49,.76,relief)*(.22+.78*lip)
        sun = np.maximum(0,lip*(.21+.58*grain)+.37*calcium+.08*step)
        edge = h>=0
        ink = np.array([14,29,37])+relief[...,None]*[24,49,44]+sun[...,None]*[185,147,89]
        paper = np.array([6,17,25])+relief[...,None]*[11,22,23]+sun[...,None]*[41,37,25]
        value = (.16+.34*relief+.56*sun)*opacity
        # Distant rock lies under a longer path through blue water.
        rgb[edge]=(opacity*ink+(1-opacity)*np.array([28,69,77]))[edge]
        substrate[edge]=(opacity*paper+(1-opacity)*np.array([11,35,44]))[edge]
        symbols[edge]=select(glyphs(" ..,:;-+vaxso08@"),value[edge])
        occupied |= edge
    return occupied,symbols,rgb,substrate


def scene_grid(scene,frame,cam,t, *, fish_scale_origin=None, reef_geometry=None):
    p = frame.shark
    center = cam.project(p.center)
    axes = np.array([p.forward,p.right,p.up]) @ cam.axes.T
    area,shark_chars,shark_ink,shark_paper = shark_cells(center,*axes,cam.scale,p.phase,t)
    on_sand = scene=='01' or (scene=='10' and cam.elevation>.1)
    shadow = np.zeros_like(area)
    if on_sand:
        on_floor = p.center + p.center[2]*np.array([.32,-.26,-1])
        shift = cam.project(on_floor)-center
        matrix = np.float32([[1,0,shift[0]],[0,1,shift[1]/1.5]])
        shadow = .52*cv2.GaussianBlur(cv2.warpAffine(area,matrix,(90,72)),(0,0),1.04)
    chars,ink,paper = environment(cam,t,on_sand,shadow)
    if scene=='10' and cam.elevation<0:
        # A low side view looks through a column of sky-lit water. The
        # ceiling is bright behind the animal and its underside loses fill.
        sheet = np.exp(-((Y-41-.26*(X-45))/39)**2)
        shaft = .5+.5*np.sin(.064*X-.029*Y+.33*np.sin(t*.30))
        scatter = sheet*(.43+.57*shaft)
        paper += scatter[...,None]*[26,62,55]
        ink += scatter[...,None]*[28,62,52]
        gloss = smoothstep(187,252,shark_ink.max(axis=2))
        shark_ink = shark_ink*(.66+.32*gloss[...,None])
        shark_paper = shark_paper*(.49+.36*gloss[...,None])
    if scene=='07':
        bg,rc,ri,rp = reef_tip(cam,t,reef_geometry)
        chars[bg],ink[bg],paper[bg] = rc[bg],ri[bg],rp[bg]
        from .encounter_reef import sea_fans
        sea_fans(cam,t,chars,ink,paper)
    pos = frame.fish
    xy = cam.project(pos)
    dirs = unit(frame.velocity) @ cam.axes.T
    side = unit(dirs[:,:2])
    lengths = np.maximum(1.7,cam.scale*frame.sizes*np.maximum(.42,np.linalg.norm(dirs[:,:2],axis=1)))
    depth = np.einsum("ij,j->i",pos-p.center,cam.axes[2])
    # In a revision of a shot, retain the original depth cue for fish size.
    # Occlusion below still follows the current position of the shark.
    scale_depth = depth if fish_scale_origin is None else (pos-fish_scale_origin) @ cam.axes[2]
    lengths *= np.clip(1+.23*scale_depth,.77,1.25)
    # Reflected silver tracks bank and headings. A warm band at the gill
    # becomes visible only while broadside to the light.
    silver = np.clip(.25+.26*np.cos(frame.phase*.34)+.25*abs(dirs[:,0])+
                     .27*frame.alarm*np.cos(1.7*pos[:,0]+t)**4,0,1)
    if on_sand:
        floor_pos = pos+pos[:,2,None]*[.32,-.26,-1]
        projected = cam.project(floor_pos)
        m = np.zeros_like(shadow)
        for (x,y),length in zip(projected,lengths):
            if -3<x<93 and -3<y<111:
                cv2.ellipse(m,(round(x),round(y/1.5)),(max(1,round(.36*length)),1),0,0,360,.60,-1)
        fishshade = cv2.GaussianBlur(m,(0,0),.66)
        ink *= 1-.59*fishshade[...,None]
        paper *= 1-.57*fishshade[...,None]
    for foreground in (False, True):
        ids = depth>=0 if foreground else depth<0
        small_fish(chars,ink,paper,xy[ids],side[ids],lengths[ids],silver[ids],depth[ids],frame.phase[ids],sunlit_sand=scene=="01")
        if not foreground:
            weight = smoothstep(.07,.93,area)[...,None]
            ink += weight*(shark_ink-ink)
            paper += weight*(shark_paper-paper)
            inside = area>.27
            chars[inside]=shark_chars[inside]
    return chars,ink.clip(0,255).astype(np.uint8),paper.clip(0,255).astype(np.uint8),area,xy


@lru_cache(maxsize=16)
def panel(scene,section):
    if scene=='01':
        from .encounter_fold import fold_states
        funcs = (motion.swim,fold_states,fold_states)
        blocks = (0,34,57)
    elif scene=='10':
        from .encounter_window import school_volume,window_states
        funcs = (school_volume,window_states,window_states)
        blocks = (0,38,65)
    else:
        funcs = (motion.swim,motion.pursuit_states,motion.update_school)
        blocks = (0,28,0)
    return source_panel(funcs[section], blocks[section])


@lru_cache(maxsize=16)
def source_panel(f, offset=0):
    width,height = WIDTH,FRAME_H-SCENE_H
    sheet = Image.new('RGB',(width,height),(8,14,20))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT,21)
    small = ImageFont.truetype(FONT,18)
    d.line((0,0,width,0),fill=(45,65,72),width=2)
    d.text((42,24),f'{Path(inspect.getfile(f)).name}   /   {f.__name__}',font=small,fill=(123,155,159))
    import io,tokenize,token
    lines,n = inspect.getsourcelines(f)
    colored = {}
    for tk in tokenize.generate_tokens(io.StringIO(''.join(lines)).readline):
        tone = (144,162,170)
        if tk.type==tokenize.COMMENT: tone=(86,114,113)
        if tk.type==token.STRING: tone=(135,163,131)
        if tk.type==token.NUMBER: tone=(201,164,122)
        if tk.string in ('def','return','if','for','in','else'): tone=(137,181,196)
        colored.setdefault(tk.start[0],[]).append((tk.start[1],tk.string,tone))
    start = min(offset, max(0,len(lines)-14))
    for i,line in enumerate(lines[start:start+15]):
        y=68+i*33
        d.text((18,y),str(n+start+i).rjust(4),font=small,fill=(69,88,94))
        for begin,s,tone in colored.get(start+i+1,[]):
            if s.strip():
                d.text((84+begin*font.getlength('M'),y),s,font=font,fill=tone)
    return np.asarray(sheet)


def frames_for(scene):
    if scene=='01':
        from .encounter_fold import fold_states
        return list(fold_states())
    if scene=='10':
        from .encounter_window import window_states
        return list(window_states())
    return list(motion.pursuit_states())


@lru_cache(maxsize=1)
def atlas():
    return glyph_atlas(ImageFont.truetype(FONT,20,index=1),12,18)


@lru_cache(maxsize=2)
def exposure(gain):
    levels=np.linspace(0,1,256)
    mids=4*levels*(1-levels)
    return np.rint(255*(levels+gain*mids*(np.sqrt(levels)-levels))).astype(np.uint8)


def render_frame(scene, frame, variant, t):
    cam = camera(scene,frame,t,variant)
    chars,ink,paper,area,xy=scene_grid(scene,frame,cam,t)
    picture=paint(chars,ink,atlas(),paper,bloom=.46)
    if scene in ("01","07"):
        picture=cv2.LUT(picture,exposure(.62 if scene=="01" else .29))
    return np.vstack((picture,panel(scene,0 if t<5.8 else 1 if t<13.4 else 2))),area,xy,cam


# Each study samples the same deterministic run as its full-length film.
PEAK = {'01':(11.9,12.6,12.3),'07':(6.7,6.1,7.25),'10':(10.6,10.9,10.4)}
START = {'01':8.3,'07':3.75,'10':3.5}
STUDY_SECONDS = {'01':6,'07':6,'10':15}


def camera_studies(scene,frames):
    folder=OUT/'previews'
    folder.mkdir(exist_ok=True,parents=True)
    for variant,letter in enumerate('abc'):
        name=f"{scene}-{letter}"
        t=PEAK[scene][variant]
        image,*_=render_frame(scene,frames[round(t*FPS)],variant,t)
        Image.fromarray(image).save(folder/(name+'.png'))
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s','540x960','-r','12','-i','-','-an','-c:v','libx264','-threads','2',
             '-preset','veryfast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',
             str(folder/(name+'.mp4'))]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
            start=round(START[scene]*FPS)
            for i in range(start,start+STUDY_SECONDS[scene]*FPS,2):
                a,*_=render_frame(scene,frames[i],variant,i/FPS)
                encoder.stdin.write(memoryview(cv2.resize(a,(540,960),interpolation=cv2.INTER_AREA)))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Study encoder failed')


def run(scene, variant, video, peak):
    frames = frames_for(scene)
    if variant is None:
        return camera_studies(scene,frames)
    preview_dir = OUT/'previews'
    preview_dir.mkdir(exist_ok=True,parents=True)
    name = f"{scene}-{'abc'[variant]}"
    image,*_ = render_frame(scene,frames[round(peak*FPS)],variant,peak)
    Image.fromarray(image).save(preview_dir/(name+'.png'))
    if not video:
        return
    media = OUT/'video'
    media.mkdir(exist_ok=True)
    # Model geometry is sampled alongside the film for the synchronised overlay.
    path=media/(name+'.mp4')
    stage=path.with_suffix('.part.mp4')
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s','1080x1920','-r',str(FPS),'-i','-','-an','-c:v','libx264',
             '-threads','3','-preset','veryfast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(stage)]
    tracks=[]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for i,frame in enumerate(frames):
            image,area,xy,cam=render_frame(scene,frame,variant,i/FPS)
            encoder.stdin.write(memoryview(image))
            if i%3==0:
                outlines,_=cv2.findContours((area>.30).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                outline=max(outlines,key=cv2.contourArea).reshape(-1,2).tolist() if outlines else []
                tracks.append([outline,np.round(xy*[1,1/1.5],1).tolist(),round(frame.shark.speed,2)])
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoder failed')
    stage.replace(path)
    (media/(name+'.js')).write_text('const tracks='+json.dumps(tracks,separators=(',',':'))+';')
    template=Path(__file__).with_name('encounter_view.html').read_text()
    (media/(name+'.html')).write_text(template.replace('FILM',name))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scene',choices=['01','07','10'])
    ap.add_argument('variant',choices=['a','b','c','studies'])
    ap.add_argument('--peak',type=float,default=None)
    ap.add_argument('--video',action='store_true')
    args=ap.parse_args()
    variant = None if args.variant=='studies' else 'abc'.index(args.variant)
    peak = args.peak if args.peak is not None else PEAK[args.scene][variant or 0]
    run(args.scene,variant,args.video,peak)
