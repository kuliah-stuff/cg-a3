"""Scenes, camera, parallel render loop, and image output for Assignment 3.

Do not modify this file. Only student.py is submitted.
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit, prange

import student
from framework import (DIELECTRIC, EMISSIVE, LAMBERTIAN, METAL, SDF, SPHERE, TRIANGLE,
                       normalize, rng_next)

Scene = namedtuple('Scene', [
    'kind', 'ref', 'mat_id', 'spheres', 'tris', 'sdfs', 'bounds', 'materials', 'background',
    'node_min', 'node_max', 'node_left', 'node_first', 'node_count', 'order'])

SCENES = ('cornell', 'spheres', 'materials', 'sdf')


# Building scenes

class SceneBuilder:
    """Collects primitives and materials in Python lists, then packs them into a Scene."""

    def __init__(self, background=(0.0, 0.0, 0.0)):
        self.materials = []
        self.spheres, self.tris, self.sdfs = [], [], []
        self.items = []  # (kind, ref, material)
        self.background = np.array(background, float)

    def material(self, kind, albedo=(0, 0, 0), roughness=0.0, ior=1.0, emission=(0, 0, 0)):
        self.materials.append([kind, *albedo, roughness, ior, *emission])
        return len(self.materials) - 1

    def sphere(self, center, radius, mat):
        self.spheres.append([*center, radius])
        self.items.append((SPHERE, len(self.spheres) - 1, mat))

    def triangle(self, a, b, c, mat):
        self.tris.append([a, b, c])
        self.items.append((TRIANGLE, len(self.tris) - 1, mat))

    def quad(self, a, b, c, d, mat):
        """Two triangles; corners counter-clockwise seen from the side the normal points to."""
        self.triangle(a, b, c, mat)
        self.triangle(a, c, d, mat)

    def mesh(self, vertices, faces, mat):
        for f in faces:
            self.triangle(vertices[f[0]], vertices[f[1]], vertices[f[2]], mat)

    def sdf(self, params, mat):
        self.sdfs.append(list(params))
        self.items.append((SDF, len(self.sdfs) - 1, mat))

    def build(self):
        kind = np.array([k for k, _, _ in self.items], np.int64)
        ref = np.array([r for _, r, _ in self.items], np.int64)
        mat_id = np.array([m for _, _, m in self.items], np.int64)
        spheres = np.array(self.spheres, float).reshape(-1, 4)
        tris = np.array(self.tris, float).reshape(-1, 3, 3)
        sdfs = np.array(self.sdfs, float).reshape(-1, 8)
        bounds = primitive_bounds(kind, ref, spheres, tris, sdfs)
        return Scene(kind, ref, mat_id, spheres, tris, sdfs, bounds,
                     np.array(self.materials, float).reshape(-1, 9), self.background,
                     *empty_bvh())


def primitive_bounds(kind, ref, spheres, tris, sdfs):
    """Axis-aligned box (N, 2, 3) of every primitive."""
    bounds = np.empty((len(kind), 2, 3))
    for i, (k, j) in enumerate(zip(kind, ref)):
        if k == SPHERE:
            bounds[i, 0] = spheres[j, :3] - spheres[j, 3]
            bounds[i, 1] = spheres[j, :3] + spheres[j, 3]
        elif k == TRIANGLE:
            bounds[i, 0] = tris[j].min(axis=0)
            bounds[i, 1] = tris[j].max(axis=0)
        else:
            extent = sdfs[j, 3:6] + sdfs[j, 6]
            bounds[i, 0] = sdfs[j, :3] - extent
            bounds[i, 1] = sdfs[j, :3] + extent
    return bounds


def empty_bvh():
    return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64),
            np.zeros(0, np.int64), np.zeros(0, np.int64))


def with_bvh(scene, leaf_size=2):
    """Return the scene with a BVH built by student.build_bvh."""
    node_min, node_max, node_left, node_first, node_count, order = student.build_bvh(
        scene.bounds, leaf_size)
    return scene._replace(
        node_min=np.ascontiguousarray(node_min, dtype=float),
        node_max=np.ascontiguousarray(node_max, dtype=float),
        node_left=np.ascontiguousarray(node_left, dtype=np.int64),
        node_first=np.ascontiguousarray(node_first, dtype=np.int64),
        node_count=np.ascontiguousarray(node_count, dtype=np.int64),
        order=np.ascontiguousarray(order, dtype=np.int64))


def without_sdf(scene):
    """Return the scene without its SDF primitives (used until Task 5 is implemented)."""
    keep = scene.kind != SDF
    return scene._replace(kind=scene.kind[keep], ref=scene.ref[keep], mat_id=scene.mat_id[keep],
                          bounds=scene.bounds[keep])


def icosphere(level):
    """Unit icosphere: 20 * 4**level triangles, counter-clockwise seen from outside."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
             (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts), np.array(faces)


def cornell(mesh_level=4):
    """Cornell box in [0, 1]^3: glass ball, metal ball, triangle-mesh ball, SDF object."""
    b = SceneBuilder()
    white = b.material(LAMBERTIAN, (0.73, 0.73, 0.73))
    red = b.material(LAMBERTIAN, (0.65, 0.05, 0.05))
    green = b.material(LAMBERTIAN, (0.12, 0.45, 0.15))
    light = b.material(EMISSIVE, emission=(15.0, 15.0, 15.0))
    glass = b.material(DIELECTRIC, ior=1.5)
    metal = b.material(METAL, (0.8, 0.85, 0.88), roughness=0.05)
    blue = b.material(LAMBERTIAN, (0.2, 0.3, 0.8))
    b.quad((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0), white)        # floor
    b.quad((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1), white)        # ceiling
    b.quad((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), white)        # back wall
    b.quad((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0), red)          # wall at x = 1, seen on the left
    b.quad((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), green)        # wall at x = 0, seen on the right
    b.quad((0.37, 0.999, 0.37), (0.63, 0.999, 0.37), (0.63, 0.999, 0.63), (0.37, 0.999, 0.63),
           light)
    b.sphere((0.77, 0.12, 0.30), 0.12, glass)
    b.sdf((0.50, 0.09, 0.22, 0.07, 0.07, 0.07, 0.015, 0.095), white)
    verts, faces = icosphere(mesh_level)
    b.mesh(verts * 0.14 + np.array([0.23, 0.14, 0.32]), faces, blue)
    b.sphere((0.55, 0.20, 0.72), 0.20, metal)
    cam = camera((0.5, 0.55, -1.3), (0.5, 0.42, 0.5), 40.0)
    return b.build(), cam


def spheres(count=60, seed=7):
    """Many small spheres of all materials on a large ground sphere, lit by the sky."""
    b = SceneBuilder(background=(0.7, 0.8, 1.0))
    ground = b.material(LAMBERTIAN, (0.5, 0.5, 0.5))
    b.sphere((0.0, -1000.0, 0.0), 1000.0, ground)
    rng = np.random.default_rng(seed)
    placed = []
    while len(placed) < count:
        x, z = rng.uniform(-6, 6, 2)
        r = rng.uniform(0.15, 0.35)
        if any((x - px) ** 2 + (z - pz) ** 2 < (r + pr + 0.05) ** 2 for px, pz, pr in placed):
            continue
        placed.append((x, z, r))
        choice = rng.uniform()
        if choice < 0.6:
            mat = b.material(LAMBERTIAN, tuple(rng.uniform(0.1, 0.9, 3)))
        elif choice < 0.85:
            mat = b.material(METAL, tuple(rng.uniform(0.5, 1.0, 3)), roughness=rng.uniform(0, 0.4))
        else:
            mat = b.material(DIELECTRIC, ior=1.5)
        b.sphere((x, r, z), r, mat)
    cam = camera((0.0, 2.2, -9.0), (0.0, 0.3, 0.0), 30.0)
    return b.build(), cam


def materials():
    """Metal spheres of increasing roughness and glass spheres of increasing refractive index."""
    b = SceneBuilder(background=(0.7, 0.8, 1.0))
    floor = b.material(LAMBERTIAN, (0.8, 0.8, 0.8))
    checker = b.material(LAMBERTIAN, (0.2, 0.2, 0.2))
    for i in range(-8, 8):
        for k in range(-14, 30):
            mat = floor if (i + k) % 2 == 0 else checker
            b.quad((i, 0, k), (i, 0, k + 1), (i + 1, 0, k + 1), (i + 1, 0, k), mat)
    for i, rough in enumerate((0.0, 0.1, 0.3, 0.6)):
        b.sphere((3.5 - i, 0.45, 0.0), 0.45, b.material(METAL, (0.9, 0.8, 0.6), roughness=rough))
    for i, ior in enumerate((1.0, 1.33, 1.5, 2.4)):
        b.sphere((-0.5 - i, 0.45, 0.0), 0.45, b.material(DIELECTRIC, ior=ior))
    cam = camera((0.0, 1.2, -9.0), (0.0, 0.45, 0.0), 38.0)
    return b.build(), cam


def sdf():
    """The SDF object alone on a ground plane, lit by the sky."""
    b = SceneBuilder(background=(0.7, 0.8, 1.0))
    ground = b.material(LAMBERTIAN, (0.5, 0.5, 0.5))
    orange = b.material(LAMBERTIAN, (0.9, 0.5, 0.2))
    b.quad((-5, 0, -5), (-5, 0, 5), (5, 0, 5), (5, 0, -5), ground)
    b.sdf((0.0, 0.6, 0.0, 0.45, 0.45, 0.45, 0.08, 0.62), orange)
    cam = camera((1.6, 1.8, -2.4), (0.0, 0.55, 0.0), 40.0)
    return b.build(), cam


def load_scene(name, mesh_level=4):
    if name == 'cornell':
        return cornell(mesh_level)
    if name == 'spheres':
        return spheres()
    if name == 'materials':
        return materials()
    if name == 'sdf':
        return sdf()
    raise ValueError(f'unknown scene {name!r}; choose from {", ".join(SCENES)}')


# Camera and rendering

def camera(look_from, look_at, vfov, up=(0.0, 1.0, 0.0)):
    """Pinhole camera as a (4, 3) array: eye, lower-left corner, horizontal, vertical.

    The image plane is at distance 1; the aspect ratio is set when rendering.
    """
    eye = np.array(look_from, float)
    w = eye - np.array(look_at, float)
    w /= np.linalg.norm(w)
    u = np.cross(np.array(up, float), w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    half_h = math.tan(math.radians(vfov) / 2)
    return np.array([eye, u, v, w]), half_h


@njit(parallel=True, cache=True, nogil=True)
def render(scene, cam, half_h, width, height, spp, max_depth, seed, use_roulette):
    """Linear radiance image (height, width, 3); one random state per pixel."""
    eye, u, v, w = cam[0], cam[1], cam[2], cam[3]
    half_w = half_h * width / height
    image = np.zeros((height, width, 3))
    for y in prange(height):
        state = np.empty(1, np.uint64)
        for x in range(width):
            # seeds are spread with an odd constant that is not a small multiple of the
            # generator step, so different seeds give unrelated random sequences
            state[0] = np.uint64(seed) * np.uint64(0xD1B54A32D192ED03) + np.uint64(y * width + x)
            rng_next(state)
            total = np.zeros(3)
            for _ in range(spp):
                sx = (2.0 * (x + rng_next(state)) / width - 1.0) * half_w
                sy = (1.0 - 2.0 * (y + rng_next(state)) / height) * half_h
                rd = normalize(sx * u + sy * v - w)
                total += student.trace_path(eye.copy(), rd, scene, state, max_depth, use_roulette)
            image[y, x] = total / spp
    return image


def tone_map(radiance):
    """ACES filmic curve and sRGB gamma; returns uint8 (height, width, 3)."""
    x = np.clip(radiance, 0.0, None)
    x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    x = np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_png(radiance, path):
    from PIL import Image
    Image.fromarray(tone_map(radiance)).save(path)
