"""A hanging petal margin with a softer, warmer reverse side."""
import numpy as np


def _ease(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


def folded_edge(p, tex, q, i, var):
    """Bend transverse strips about the local midrib, keeping strip lengths.

    The lip stays raised while the midrib opens. Once the supporting curvature
    relaxes, the free margin rolls back towards the viewer. The root is clamped.
    Coordinates are (right, towards camera, up).
    """
    target, order, turn = [(10, 1, -1.), (9, -1, 1.), (4, 1, 1.)][var]
    if i != target:
        return p, tex
    source = p[:, ::order]
    base_colour = tex[:, ::order]
    nu, nv = source.shape[:2]
    u = np.linspace(0., 1., nu, dtype='f')[:, None]
    w = np.linspace(0., 1., nv, dtype='f')[None, :]
    midrib = source[:,nv//2]
    tangent = np.gradient(midrib,axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-7)
    tangent = tangent[:,None]
    progress = np.clip(q,0.,1.)
    free = _ease((u-.075)/.53)
    # Keep a broad face toward the light while the far third of the width rolls.
    # The crease is oblique and fades at both pointed ends.
    crease = .51-.10*np.sin(np.pi*u)+.09*u
    curve = _ease((w-crease)/.47)
    load = _ease((progress-.16)/.57)
    width_turn = turn * (3.15 if var!=1 else 3.45) * curve * free * load
    edge_angle = .5*(width_turn[:,1:]+width_turn[:,:-1])[...,None]
    strips = np.diff(source,axis=1)
    cos, sin = np.cos(edge_angle), np.sin(edge_angle)
    rotated = (strips*cos + np.cross(tangent,strips)*sin
               + tangent*np.sum(tangent*strips,axis=-1,keepdims=True)*(1-cos))
    result = np.concatenate((source[:,:1],source[:,:1]+np.cumsum(rotated,axis=1)),axis=1)
    # A delayed bend through the distal rib also brings the common tip down.
    lift = 35*np.sin(np.pi*progress)*(1-progress)
    sag = (32 if var==1 else 24)*_ease((progress-.56)/.43)
    result[:,:,2] += (lift-sag)*u**1.75
    # Pigment follows the changing surface orientation. Broad transitional
    # half-tones show that the two faces meet along the same sheet.
    reverse = _ease((abs(width_turn)-.48)/1.33)[...,None]
    reverse_colour = base_colour*np.array([.76,.55,.68],dtype='f') + [13,4,8]
    crease_light = np.maximum(0.,1.-abs(abs(width_turn)-.57)/.47)*free*load
    color = (1-reverse)*base_colour+reverse*reverse_colour+crease_light[...,None]*[8,6,10]
    return result[:,::order].astype('f'),np.clip(color[:,::order],0,255).astype('f')
