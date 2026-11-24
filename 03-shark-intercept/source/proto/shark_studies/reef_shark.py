"""Side and dorsal projections of one grey reef shark, in total-length units.

Side profile: FAO species sheet CARCH Carch 19 and Ilan Ben Tov's Fakarava
juvenile in profile. Fin placement and caudal margin: Florida Museum,
Carcharhinus amblyrhynchos. Widths and swimming roll are artistic estimates.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

# From tail stock to snout. Body thickness and side profile share the same landmarks.
STATIONS = (-.292,-.260,-.205,-.125,-.035,.060,.155,.245,.330,.395,.448,.480,.498,.505)
HALF_WIDTH = (.000,.014,.023,.036,.050,.067,.077,.078,.068,.057,.044,.032,.016,.000)
BACK = (.005,.023,.028,.040,.054,.067,.075,.073,.061,.048,.034,.023,.012,.000)
BELLY = (.000,.018,.022,.031,.046,.060,.068,.069,.059,.046,.030,.016,.008,.000)
# Each fin is attached in the same longitudinal, lateral, vertical coordinates.
DORSAL = ((.186,0,.063),(.154,0,.091),(.105,0,.148),(.082,0,.165),(.076,0,.154),
          (.077,0,.106),(.065,0,.082),(.035,0,.061),(.019,0,.056))
SECOND = ((-.150,0,.028),(-.180,0,.063),(-.189,0,.065),(-.189,0,.041),(-.209,0,.026))
UPPER = ((-.248,0,.016),(-.299,0,.036),(-.396,0,.093),(-.496,0,.126),
         (-.502,0,.122),(-.478,0,.099),(-.454,0,.082),(-.447,0,.084),
         (-.404,0,.050),(-.335,0,.007),(-.295,0,-.012))
LOWER = ((-.252,0,-.010),(-.304,0,-.024),(-.347,0,-.101),
         (-.340,0,-.114),(-.323,0,-.100),(-.296,0,-.065),(-.276,0,-.029))
PECTORAL = ((.281,.047,-.008),(.267,.086,-.020),(.218,.145,-.050),
            (.157,.205,-.089),(.139,.219,-.102),(.140,.200,-.096),
            (.169,.125,-.063),(.170,.080,-.037),(.150,.056,-.014))
PELVIC = ((-.060,.032,-.021),(-.087,.063,-.049),(-.124,.087,-.063),
          (-.117,.055,-.038),(-.139,.027,-.027))
ANAL = ((-.153,0,-.024),(-.187,0,-.064),(-.194,0,-.057),(-.188,0,-.037),(-.207,0,-.027))


def render(g, x, y, length, angle, beat, top, color, bend):
    from grid import COL, ROW, COLS, ROWS, RAMP

    cs, sn = math.cos(angle), math.sin(angle)
    dx, dy = (COL-x)/length, 2*(ROW-y)/length
    u, across = dx*cs+dy*sn, -dx*sn+dy*cs
    # A side-facing camera keeps the dorsal side up in both travel directions.
    if not top and cs < 0:
        across = -across
    roll = (.09*math.sin(beat-.9)+1.2*bend) if top else .23+.09*math.sin(beat-.4)
    lateral, up = (math.cos(roll), -math.sin(roll)) if top else (math.sin(roll), -math.cos(roll))

    def spine(station):
        tail = np.clip((.28-station)/.78,0,1)
        return bend*tail**2 + (.006+.106*tail**2.3)*np.sin(beat-4.1*tail)

    span = across-lateral*spine(u)
    sea = np.array(color)*.28 + np.array([120,143,155])*.72
    if not top:
        # The breach camera sees reflected surface light on the flank and fins.
        sea *= 1.20

    def paint(mask, level, rgb):
        g.chars[mask] = RAMP[np.clip((np.broadcast_to(level,u.shape)[mask]*9).astype(int),0,9)]
        g.ink[mask] = np.clip(np.broadcast_to(rgb, (*u.shape,3))[mask],0,255)

    def fin(points, side=1, shade=.90, caudal=False):
        # Supersample thin tips. A centre-only grid test breaks the swept tail
        # into disjoint dashes whenever it passes between consecutive rows.
        previous = points[-1]
        samples = []
        for point in points:
            steps = max(1, math.ceil(abs(point[0]-previous[0])/.04))
            for step in range(1,steps+1):
                samples.append(tuple(a+(b-a)*step/steps for a,b in zip(previous,point)))
            previous = point
        a,b,c = np.array(samples).T
        cross = lateral*(spine(a)+side*b)+up*c
        if not top and cs<0:
            cross = -cross
        px = 4*(x+.5+length*(a*cs-cross*sn))
        py = 4*(y+.5+.5*length*(a*sn+cross*cs))
        path = list(zip(px,py))
        tile = Image.new("L",(COLS*4,ROWS*4))
        pen = ImageDraw.Draw(tile)
        pen.polygon(path,fill=255)
        if top and caudal:
            pen.line(path+[path[0]],fill=255,width=3)
        coverage = np.asarray(tile.resize((COLS,ROWS),Image.Resampling.BOX))/255
        mask = coverage>.15
        tone = .84 + .12*np.sin((u+.13)*18) + .08*span
        ink = sea[None,None,:] * (tone*shade)[...,None]
        if caudal:
            ink *= np.where((u<-.37)&(coverage<.78),.76,1)[...,None]
        ink *= (.79+.32*coverage)[...,None]
        paint(mask, .30+.58*coverage, ink)

    fin(PECTORAL,-1,.90)
    fin(PELVIC,-1,.72)
    fin(UPPER,caudal=True); fin(LOWER,caudal=True)
    fin(SECOND); fin(ANAL)
    if not top:
        fin(DORSAL,shade=1.05)
    if top:
        fin(PECTORAL); fin(PELVIC)
    width=np.interp(u,STATIONS,HALF_WIDTH)
    back=np.interp(u,STATIONS,BACK)
    belly=np.interp(u,STATIONS,BELLY)
    mid = up*(back-belly)/2
    radius = np.hypot(lateral*width,up*(back+belly)/2)
    rel = (span-mid)/np.maximum(.001,radius)
    mask = (abs(rel)<1) & (u>STATIONS[0]) & (u<STATIONS[-1])
    dome = np.sqrt(np.maximum(0,1-rel**2))
    if top:
        lum = .65+.49*dome-.14*rel
        fleck = .17+.45*np.exp(-((rel+.32)/.39)**2)
    else:
        lum = .83+.38*dome+.10*rel
        fleck = .16+.75*np.clip((rel+.05)*2,0,1)
    rgb = (sea*(1-fleck[...,None])+np.array([222,231,226])*fleck[...,None])*lum[...,None]
    paint(mask, .22+.60*dome+.12*np.sin(169*u+21*rel)*np.cos(63*u-3*rel),rgb)
    if top:
        fin(DORSAL,shade=1.03)
    else:
        fin(PECTORAL,1,.92); fin(PELVIC,1,.76)

    def mark(a,b,c,glyph,ink):
        cross = lateral*(spine(a)+b)+up*c
        if not top and cs<0:
            cross = -cross
        g.put(x+length*(a*cs-cross*sn),y+.5*length*(a*sn+cross*cs),glyph,ink)

    if not top:
        for a in (.265,.286,.307):
            mark(a,.060,.002,"/",(70,94,103))
        mark(.439,.035,.013,"o",(253,235,186))
        mark(.409,.014,-.020,"_",(61,82,93))
