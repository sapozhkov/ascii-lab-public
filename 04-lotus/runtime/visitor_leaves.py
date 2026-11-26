"""Peltate leaf blades with independent petioles, radial veins and wind loading."""
import numpy as np

from day import RATE, PERIOD, wind
from living import normals


class ShieldLeaf:
    def __init__(self, index):
        self.index=index
        self.contact_xy=(-45.,24.)
        r=self.r=np.linspace(.006,1,57,dtype='f')[:,None]
        a=self.a=np.linspace(0,2*np.pi,181,dtype='f')[None,:]
        phi=.83*index
        self.radius=(147.,114.)[index]
        self.center=np.array([(-179.,-58.,-337.),(165.,-118.,-541.)][index])
        self.reach=np.array([(59.,-165.,-1080.),(88.,-183.,-1210.)][index])
        self.edge=1+.048*np.sin(5*a+.5+phi)+.026*np.sin(11*a+phi)+.018*np.cos(3*a-phi)
        # Primary veins start at the petiole insertion and fan out. The lateral
        # branches converge on the same ribs as the skin, so markings bend with it.
        sector=(a+.052*np.sin(3*a+phi)*r+.021*np.sin(7*a)*r*r)*13/(2*np.pi)
        phase=(sector+.5)%1-.5
        self.rib=np.exp(-(phase/(.047+.073/(1+5*r)))**2)
        side=(r*13+np.abs(phase)*1.46+.17*np.sin(3*a+2*r))%1
        branches=np.exp(-(np.minimum(side,1-side)/.054)**2)*np.sin(np.pi*np.minimum(.5,abs(phase))*2)
        bloom=.5+.5*np.cos(2*a+.7+phi)
        fleck=np.sin(97*r+1.8*np.cos(31*a))*np.sin(57*r+4*np.sin(29*a))
        margin=np.exp(-((1-r)/.017)**2)
        self.height=13*r**1.5+9*r**3*(.45+np.sin(4*a+.81+phi)) + 4*self.rib*r + 3.9*r**4*np.sin(8*a+phi)
        self.x=self.radius*r*self.edge*np.cos(a)
        self.y=self.radius*r*self.edge*np.sin(a)*(.89 if index==0 else .82)
        self.height+=(-.12 if index==0 else .30)*self.x+(-.40 if index==0 else -.30)*self.y
        self.color=(np.array([47.,91.,74.])+bloom[...,None]*[17,23,17]+self.rib[...,None]*[41,40,25]
                    +branches[...,None]*[8,11,6]+margin[...,None]*[20,25,15]+fleck[...,None]*[2,2.7,1.7])
        hub=np.exp(-(r/.082)**2)
        self.color+=hub[...,None]*[33,39,16]
        if index==1: self.color*=np.array([.92,.93,.79])
        n=RATE*PERIOD
        self.breeze=np.zeros(n)
        tilt=velocity=0.
        for k in range(n*2):
            t=(k%n)/RATE
            target=wind(t-.28*index,'dew')*.73
            velocity+=(13*(target-tilt)-4.1*velocity)/RATE
            tilt+=velocity/RATE
            if k>=n:self.breeze[k%n]=tilt

    def geometry(self,t,load):
        beta=self.breeze[round(t*RATE)%len(self.breeze)]
        center=self.center+np.array([beta*(17-2*self.index),beta*5,-6*load])
        deformation=(.10*beta*self.x+.055*beta*self.y)*self.r
        landing=np.exp(-((self.x-self.contact_xy[0])/58)**2-((self.y-self.contact_xy[1])/49)**2)
        z=self.height+deformation-load*(22*landing+3*self.r)+1.8*beta*self.r**2*np.sin(6*self.a)
        return center+np.stack([self.x,self.y,z],axis=-1),center

    def stalk_axis(self,t,center):
        s=np.linspace(0,1,38)[:,None]
        axis=self.reach*(1-s)+center*s
        axis[:,0]+=np.sin(s[:,0]*np.pi)*(30 if self.index==0 else -27)
        axis[:,1]-=14*np.sin(np.pi*s[:,0])
        return axis

    def draw(self,g,t,load):
        sheet,center=self.geometry(t,load)
        axis=self.stalk_axis(t,center)
        s=np.linspace(0,1,len(axis))[:,None]
        theta=np.linspace(0,2*np.pi,15)[None,:]
        radius=(4.8 if self.index==0 else 3.7)*(1.14-.28*s)
        stalk=axis[:,None,:]+np.stack([radius*np.cos(theta),radius*np.sin(theta),np.zeros_like(radius*np.cos(theta))],axis=-1)
        tint=.56+.39*(.5+.5*np.cos(theta-2))+.04*np.sin(theta*6)
        g.mesh(stalk,np.array([79,112,72])*tint[...,None])
        normal=normals(sheet)
        sunlight=np.maximum(0,normal@np.array([-.46,.20,.86]))
        g.mesh(sheet,self.color*(.78+.32*sunlight[...,None]))

    def landing(self,t,load):
        pts,_=self.geometry(t,load)
        return pts[22,76].copy()
