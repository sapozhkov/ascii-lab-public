"""An advected air field and reduced elastic rods, in the lotus scene's units."""
import numpy as np

from day import PERIOD, RATE


class AirField:
    """Periodic gust fronts travel across the patch. Near the ground the air slows."""
    def __init__(self, strength=1.):
        self.strength = strength

    def __call__(self, time, position):
        x, y, z = np.moveaxis(np.asarray(position), -1, 0)
        t = time - (x + .27*y)/540.
        phase = 2*np.pi*t/PERIOD
        def front(center, width):
            distance = (t - center + PERIOD/2) % PERIOD - PERIOD/2
            return np.exp(-.5*(distance/width)**2)
        forward = 18 + 14*np.sin(2*phase+.3) + 8*np.sin(5*phase+.5)
        forward += 108*front(3.8, .92) - 83*front(8.5, .73) + 125*front(12.45, 1.02) - 90*front(16.2, .68)
        cross = 14*np.cos(3*phase+.7) + .24*forward + 22*front(10.4, .85)
        up = 6*np.sin(4*phase-.6) + .065*forward
        height = np.clip((z+850)/900, 0, 1)
        exposure = self.strength * (.40+.60*height)
        return np.stack(np.broadcast_arrays(forward, cross, up), axis=-1)*exposure[..., None]


def rotations(angles):
    """Orthogonal rotations about two horizontal axes; the vertical bends toward x,y."""
    x,y = np.moveaxis(angles,-1,0)
    cx,sx,cy,sy = np.cos(x),np.sin(x),np.cos(y),np.sin(y)
    zero = np.zeros_like(x)
    return np.stack([cx,-sx*sy,sx*cy,zero,cy,sy,-sx,-cx*sy,cx*cy],axis=-1).reshape(x.shape+(3,3))


class ElasticRod:
    """Two bending modes per axis. Air, bending stiffness, damping and inertia set the pose."""
    def __init__(self, rest, air, stiffness, tip_mass, tip_drag, damping):
        self.rest = rest
        links = np.diff(rest, axis=0)
        lengths = np.linalg.norm(links,axis=1)
        length = lengths.sum()
        edges = np.r_[0.,np.cumsum(lengths)]/length
        mid = (edges[1:] + edges[:-1])/2
        widths = lengths / length
        basis = np.column_stack([mid*(2-mid), mid*(1-mid)*(1-2*mid)])
        gradient = np.column_stack([2*(1-mid), 1-6*mid+6*mid**2])
        # The Jacobian maps angular modal velocities to lateral speeds at each station.
        jacobian = np.cumsum(basis*widths[:,None],axis=0)
        masses = .28*widths
        masses[-1] += tip_mass
        mass = jacobian.T @ (masses[:,None]*jacobian)
        spring = stiffness * (gradient.T @ (widths[:,None]*gradient))
        drag = damping*mass + .013*spring
        dt = 1/RATE
        inverse = np.linalg.inv(mass + dt*drag + dt**2*spring)
        n = RATE*PERIOD
        times = np.arange(n)/RATE
        winds = air(times[:,None],rest[None,1:,:])[...,:2]
        area = np.broadcast_to(.048*widths,(n,len(widths))).copy()
        area[:,-1] += tip_drag
        q = np.zeros((2,2))
        rate = np.zeros_like(q)
        history = np.empty((n,2,2))
        for tick in range(3*n):
            i = tick % n
            relative = (winds[i] - length*(jacobian@rate))/100.
            pressure = relative * np.linalg.norm(relative,axis=1,keepdims=True)
            force = jacobian.T @ (area[i,:,None]*pressure)
            rate = inverse @ (mass@rate + dt*(force-spring@q))
            q += rate*dt
            history[i] = q
        rotation = rotations(np.einsum('sm,tma->tsa',basis,history))
        new_links = np.einsum('tsij,sj->tsi',rotation,links)
        roots = np.broadcast_to(rest[0],(n,1,3))
        self.nodes = np.concatenate([roots,roots+np.cumsum(new_links,axis=1)],axis=1)
        self.tip_angles = history[:,0]
        self.tip_rotation = rotations(self.tip_angles)

    def axis(self,t):
        return self.nodes[round(t*RATE) % len(self.nodes)]
