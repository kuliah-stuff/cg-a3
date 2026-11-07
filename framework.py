"""Constants, vector functions, random numbers, and the SDF shape for Assignment 3.

Do not modify this file. Only student.py is submitted.
Every function here is compiled with Numba and may be called from student.py.
"""

import math

import numpy as np
from numba import njit

INF = 1e30
EPS = 1e-4

# Primitive kinds (scene.kind)
SPHERE = 0
TRIANGLE = 1
SDF = 2

# Material types (column M_TYPE of scene.materials)
LAMBERTIAN = 0
METAL = 1
DIELECTRIC = 2
EMISSIVE = 3

# Columns of a row of scene.materials
M_TYPE = 0
M_ALBEDO = 1      # three columns: 1, 2, 3
M_ROUGHNESS = 4
M_IOR = 5
M_EMISSION = 6    # three columns: 6, 7, 8


# Vectors

@njit(cache=True)
def dot(a, b):
    """Dot product of two (3,) arrays."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def cross(a, b):
    """Cross product of two (3,) arrays."""
    return np.array((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))


@njit(cache=True)
def length(a):
    return math.sqrt(dot(a, a))


@njit(cache=True)
def normalize(a):
    """a divided by its length."""
    return a / math.sqrt(dot(a, a))


@njit(cache=True)
def reflect(d, n):
    """Mirror direction of d around the unit normal n."""
    return d - 2.0 * dot(d, n) * n


# Random numbers

@njit(cache=True)
def rng_next(state):
    """Uniform float in [0, 1). state is a uint64 array of length 1 and is advanced."""
    s = state[0] + np.uint64(0x9E3779B97F4A7C15)
    state[0] = s
    z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def rng_unit_vector(state):
    """Uniformly distributed direction on the unit sphere."""
    z = 1.0 - 2.0 * rng_next(state)
    phi = 2.0 * math.pi * rng_next(state)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array((r * math.cos(phi), r * math.sin(phi), z))


@njit(cache=True)
def rng_in_unit_sphere(state):
    """Uniformly distributed point inside the unit ball."""
    return rng_unit_vector(state) * rng_next(state) ** (1.0 / 3.0)


# Signed distance function of the SDF primitive

@njit(cache=True)
def sdf_eval(p, params):
    """Signed distance from p to a rounded box minus a sphere.

    params (8,): box centre (3), box half extents (3), edge radius, hole radius.
    The hole sphere has the same centre as the box.
    """
    c0, c1, c2 = params[0], params[1], params[2]
    q0 = abs(p[0] - c0) - params[3]
    q1 = abs(p[1] - c1) - params[4]
    q2 = abs(p[2] - c2) - params[5]
    outside = math.sqrt(max(q0, 0.0) ** 2 + max(q1, 0.0) ** 2 + max(q2, 0.0) ** 2)
    box = outside + min(max(q0, max(q1, q2)), 0.0) - params[6]
    sphere = math.sqrt((p[0] - c0) ** 2 + (p[1] - c1) ** 2 + (p[2] - c2) ** 2) - params[7]
    return max(box, -sphere)
