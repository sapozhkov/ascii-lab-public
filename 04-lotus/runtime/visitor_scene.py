"""Cast a bumblebee's visit in one dew-lit, planted lotus scene."""



import tempfile


import numpy as np


from day import DayGarden, PERIOD, RATE, U, base
from lotus_weather import Weather
from visitor_leaves import ShieldLeaf
from visitor_water import LeafTransfer, LoadedDrop, leaf_impulse


ANGLES={'rim':[(8,22),(-10,35),(19,45)],'cup':[(-9,32),(-20,41),(6,36)],
        'leaf':[(-2,23),(-21,44),(13,28)]}
ANGLES['catch']=[(-2,21),(19,31),(-15,37)]
TIMES={'catch':(7.25,13.5,9.65),'rim':(5.8,10.6,8.17),'cup':(9.0,15.2,12.3),'leaf':(4.5,11.6,6.30)}


class VisitGarden(DayGarden):
    leaf_class = ShieldLeaf
    def __init__(self,story,variant):
        super().__init__('dew',1)
        behavior=story
        self.story=story
        self.variant=variant
        self.az,self.elev=np.deg2rad(ANGLES[story][variant])
        self.xmid=.501*base.W
        self.ymid=.48*base.H
        self.scale={'catch':.485,'rim':.492,'cup':.493,'leaf':.488}[story]
        self.leaves=[self.leaf_class(0),self.leaf_class(1)]
        from visitor import Visitor,visit_load
        self.bee=Visitor(behavior,variant)
        self.transfer=LeafTransfer()
        if story=='catch':
            # A separate raised blade intercepts the flight below the wet petal.
            leaf=self.leaves[1]
            leaf.center=np.array([138.,189.,-298.])
            leaf.height-=.60*leaf.x+.03*leaf.y
            leaf.contact_xy=(55.,25.)
            leaf.reach=np.array([62.,48.,-1190.])
        if story=='rim':
            # This younger leaf is on a shorter petiole. Its rim stays out of the
            # falling bead's projected route, which leads directly to open water.
            leaf=self.leaves[1]
            leaf.center+=np.array([-63.,-15.,-16.])
            leaf.x*=.82
            leaf.y*=.82
            leaf.height*=.82
        self.light_window=Weather('sun')
        self.rain=Weather('dew')
        # The visitor's weight acts on its support. The free bladder of tissue
        # stores elastic energy and rebounds after lift-off.
        n=RATE*PERIOD
        self.bearing=np.empty((n,2),dtype='f')
        offset=np.zeros(2);velocity=np.zeros(2)
        for i in range(2*n):
            t=(i%n)/RATE
            weight=np.asarray(visit_load(behavior,t))
            velocity+=(np.array([36,24])*(weight-offset)-np.array([6,4.9])*velocity)/RATE
            offset+=velocity/RATE
            if i>=n:self.bearing[i%n]=offset
        self.loaded_drop=LoadedDrop(self.bearing,RATE) if story=='rim' else None

    def leaf_point(self,t):
        return self.leaves[0].landing(t,self.bearing[round(t*RATE)%len(self.bearing),1])

    def stem(self,t):
        super().stem(t)
        fraction=self.bearing[round(t*RATE)%len(self.bearing),1]
        for i,leaf in enumerate(self.leaves): leaf.draw(self,t,fraction if i==0 else leaf_impulse(t) if self.story=='catch' else 0)

    def tissue(self,i,t):
        if i in self.cache and self.current==t:
            return self.cache[i]
        p,tex,q=super().tissue(i,t)
        target={'rim':9,'catch':10,'leaf':12,'cup':12}[self.story]
        if i==target:
            weight=self.bearing[round(t*RATE)%len(self.bearing),0]
            p[:,:,2]-=(31 if self.story in ('rim','catch') else 18)*weight*self.periphery
        return p,tex,q

    def illuminated_plant(self,t):
        rgb=self.render(t)
        # Both the wet petal and the same foliage receive the changing light.
        rgb,self.lightfield=self.light_window.canopy(self,rgb,t)
        # Broad reflected light from the two raised leaves reaches shaded petal faces.
        petal=(rgb[...,0].astype('f')>1.16*rgb[...,1])&(rgb[...,2]>rgb[...,1])&(self.z>-1e5)
        bounce=(1-self.lightfield)*petal
        rgb=np.uint8(np.clip(rgb.astype('f')*(1+bounce[...,None]*[.065,.135,.078]),0,255))
        return rgb

    def frame(self,t):
        rgb=self.illuminated_plant(t)
        if self.story=='catch': self.transfer.draw(rgb,self,t,self.rain)
        elif self.loaded_drop: self.loaded_drop.draw(rgb,self,t)
        else: self.rain.dew(self,rgb,t)
        return self.bee.render(self,rgb,t)
