"""Light cast from a broad opening in the canopy onto the moving corolla."""
import cv2
import numpy as np

from day import base


class LightDepth:
    """An orthographic depth buffer as seen from one point of the area light."""
    size = 384
    scale = .49

    def __init__(self, direction, meshes):
        right = base.unit(np.cross([0., 0., 1.], direction))
        up = np.cross(direction, right)
        self.axes = np.asarray([right*self.scale, up*self.scale, direction], np.float32).T
        self.depth = np.full((self.size, self.size), -1.e6, np.float32)
        self.owner = np.zeros((self.size, self.size, 3), np.uint8)
        for i, mesh in enumerate(meshes):
            projected = self.project(mesh)
            vertex = np.empty((*mesh.shape[:2], 6), np.float32)
            vertex[..., :3] = projected
            vertex[..., 3:] = i + 1
            base.GRID(vertex, *mesh.shape[:2], self.size, self.size, self.depth, self.owner)

    def project(self, p):
        xy_depth = p @ self.axes
        xy_depth[..., :2] += self.size*.5
        return xy_depth

    def occlusion(self, i, mesh):
        projected = self.project(mesh)
        x = np.clip(projected[..., 0].astype(int), 0, self.size-1)
        y = np.clip(projected[..., 1].astype(int), 0, self.size-1)
        gap = self.depth[y, x] - projected[..., 2]
        # Ignore this sheet itself; its normals provide the shading of its folds.
        return (self.owner[y, x, 0] != i+1) * base.smooth((gap-1.5)/7.)


class PetalLight:
    @staticmethod
    def key_direction(sun):
        return base.unit(sun + [-.40, -.73, -.16])

    def petal_lighting(self, t, sun):
        meshes = []
        for i in range(len(self.sheets)):
            mesh = self.tissue(i, t)[0]
            # Keep every outer edge in the caster silhouette.
            rows = np.r_[np.arange(0, len(mesh)-1, 3), len(mesh)-1]
            cols = np.r_[np.arange(0, mesh.shape[1]-1, 3), mesh.shape[1]-1]
            meshes.append(mesh[rows[:, None], cols])
        key = self.key_direction(sun)
        horizontal = base.unit(np.cross([0., 0., 1.], key))
        vertical = np.cross(key, horizontal)
        samples = ((-.11, -.055), (.11, -.055), (0., .12))
        maps = [LightDepth(base.unit(key+h*horizontal+v*vertical), meshes) for h,v in samples]
        eye = np.array([np.sin(self.az)*np.cos(self.elev),
                        np.cos(self.az)*np.cos(self.elev), np.sin(self.elev)])
        half = base.unit(key + eye)
        reflection = getattr(self, 'petal_reflection', None)

        def shade(i, mesh, normals, pigment):
            # Shade the face visible in the camera. Some light still passes
            # through the back of the thin petal.
            n = base.unit(cv2.GaussianBlur(normals, (0,0), 1.35))
            n = np.where(np.sum(n*eye, axis=-1, keepdims=True) < 0., -n, n)
            incidence = np.sum(n*key, axis=-1)
            diffuse = np.maximum(incidence, 0.)
            through = np.maximum(-incidence, 0.)
            sky = .5+.5*n[..., 2]
            wax = np.maximum(np.sum(n*half, axis=-1), 0.)**9
            fresnel = (1.-np.clip(np.sum(n*eye, axis=-1),0.,1.))**2
            blocked = sum(light.occlusion(i, mesh) for light in maps)/len(maps)
            blocked = cv2.GaussianBlur(blocked, (0,0), .85)
            lit = 1.-blocked
            # Violet skylight stays in the shadow. Where the upper petal
            # hides the key, both direct light and the satin glint disappear.
            fill = .45 + .14*sky
            energy = fill + lit*(.385 + .32*diffuse + .14*through + .11*wax + .13*fresnel)
            tint = np.array([.99, .96, 1.11])
            color = (pigment*energy[..., None]*(1.+blocked[..., None]*(tint-1.))
                     + (lit*wax)[...,None]*[18., 12., 17.])
            if reflection is not None:
                color = reflection(mesh, n, color, t, lit)
            return color

        return shade
