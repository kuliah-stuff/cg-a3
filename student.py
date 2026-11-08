"""Implement the tasks described in the Assignment 3 PDF.

Only this file is submitted. Keep every @njit decorator and every signature.
Functions marked "Provided" are complete: do not change them.

Every function is compiled by Numba. Inside them use loops, scalar arithmetic,
small NumPy arrays, and the functions imported from framework.py. Run
    python run.py --scene cornell --no-jit
to execute the same code as plain Python when Numba reports an error.
"""

import math

import numpy as np
from numba import njit

from framework import (DIELECTRIC, EMISSIVE, EPS, INF, LAMBERTIAN, M_ALBEDO, M_EMISSION,
                       M_IOR, M_ROUGHNESS, M_TYPE, METAL, SDF, SPHERE, TRIANGLE, cross, dot,
                       normalize, reflect, rng_in_unit_sphere, rng_next, rng_unit_vector,
                       sdf_eval) 


# Task 1: ray-primitive intersection

@njit(cache=True)
def hit_sphere(center, radius, ro, rd, t_min, t_max):
    """Task 1. Smallest t in [t_min, t_max] where the ray hits the sphere, or INF.

    center (3,), radius float, ro and rd (3,) with rd of unit length.
    Check: python check.py 1
    """
    a = np.dot(rd, rd)
    oc = ro - center
    h = np.dot(oc, rd)
    c = np.dot(oc, oc) - radius**2
    D = h**2 - a*c
    if(D < 0): return INF
    else:
        D = np.sqrt(D)
        t = (-h + np.array([-D, D]))/a
        # print(t)
        if (np.min(t) <= t_max and np.min(t) >= t_min): return np.min(t)
        elif (np.max(t) <= t_max and np.max(t) >= t_min): return np.max(t)
        else: return INF

@njit(cache=True)
def hit_triangle(v0, v1, v2, ro, rd, t_min, t_max):
    """Task 1. Smallest t in [t_min, t_max] where the ray hits the triangle, or INF.

    Use the Moller-Trumbore algorithm. Both sides of the triangle count.
    """
    e1 = v1 - v0
    e2 = v2 - v0 
    h = np.cross(rd, e2)
    a = np.dot(e1, h)
    if (np.abs(a) < 1e-12): return INF
    s = ro - v0
    u = np.dot(s, h)/a
    q = np.cross(s, e1)
    v = np.dot(rd, q)/a
    if u+v > 1 or u < 0 or u > 1 or v < 0: return INF
    t = np.dot(e2, q)/a

    return t if t <= t_max and t >= t_min else INF

@njit(cache=True)
def hit_aabb(bmin, bmax, ro, inv_d, t_min, t_max):
    """Task 1. True when the ray hits the box [bmin, bmax] within [t_min, t_max].

    inv_d is 1.0 / rd, computed once per ray by the caller.
    """
    t0 = (bmin - ro)*inv_d
    t1 = (bmax - ro)*inv_d

    for k in range(3):
        if(t0[k] > t1[k]):
            temp = t0[k]
            t0[k] = t1[k]
            t1[k] = temp

        if min(t_max, t1[k]) < max(t_min, t0[k]): return False
    
    return True
    



# Task 2: materials

@njit(cache=True)
def scatter_lambertian(n, state):
    """Task 2. Cosine-weighted unit direction around the unit normal n.

    Check: python check.py 2

    Cause: the return line of scatter_lambertian, return normalize(dir) if ... else n, mixes two arrays in one ternary. Inside trace_path, n has a different array layout, so Numba types the result as Optional. That Optional is what failed at np.cross(rd, ...) in hit_triangle.
    """
    dir = n + rng_unit_vector(state)
    sq = dot(dir, dir)
    if sq < 1e-12:
        return n.copy()
    return dir / math.sqrt(sq)



@njit(cache=True)
def scatter_metal(d, n, roughness, state):
    """Task 2. Return (ok, direction): the mirror direction of d perturbed by roughness.

    ok is False when the perturbed direction points into the surface.
    """
    s = rng_in_unit_sphere(state)
    r = reflect(d, n)
    direction = normalize(r + roughness*s)
    ok = bool(np.dot(direction, n) > 0)
    return (ok, direction)

@njit(cache=True)
def scatter_dielectric(d, n, front_face, ior, state):
    """Task 2. Reflected or refracted unit direction for glass with refractive index ior.

    front_face is True when the ray arrives from outside the object.
    """
    costi = min(np.dot(-d, n), 1.0)
    ior = 1/ior if front_face else ior
    D = (ior**2)*(1 - costi**2)
    if D>1: return reflect(d, n)
    else:
        F0 = ((1 - ior)/(1 + ior))**2
        F = F0 + (1 - F0)*(1 - costi)**5
        if(rng_next(state) < F): return reflect(d, n)
        else:
            # print("REFRACT!")
            t = ior*(d + costi*n) - np.sqrt(1 - D)*n
            return t



# Task 5: signed distance functions (used by hit_primitive below)

@njit(cache=True)
def hit_sdf(params, ro, rd, t_min, t_max):
    """Task 5. Sphere tracing of sdf_eval(p, params): t of the hit, or INF.

    Start at t_min, stop with a hit when the distance is below 1e-5, with a
    miss when t exceeds t_max or after 256 steps.
    Check: python check.py 5
    """
    raise NotImplementedError('Task 5: hit_sdf')


@njit(cache=True)
def sdf_normal(p, params):
    """Task 5. Unit normal of the SDF at p by central differences with h = 1e-5."""
    raise NotImplementedError('Task 5: sdf_normal')


# Provided: primitive dispatch and the brute-force search

@njit(cache=True)
def hit_primitive(scene, i, ro, rd, t_min, t_max):
    """Provided. t of the hit between the ray and primitive i, or INF."""
    j = scene.ref[i]
    k = scene.kind[i]
    if k == SPHERE:
        return hit_sphere(scene.spheres[j, 0:3], scene.spheres[j, 3], ro, rd, t_min, t_max)
    if k == TRIANGLE:
        tri = scene.tris[j]
        return hit_triangle(tri[0], tri[1], tri[2], ro, rd, t_min, t_max)
    return hit_sdf(scene.sdfs[j], ro, rd, t_min, t_max)


@njit(cache=True)
def hit_brute(scene, ro, rd, t_min, t_max):
    """Provided. (prim, t) of the closest hit by testing every primitive; prim is -1 on a miss."""
    best = -1
    for i in range(scene.kind.shape[0]):
        t = hit_primitive(scene, i, ro, rd, t_min, t_max)
        if t < t_max:
            t_max = t
            best = i
    return best, t_max


# Task 4: bounding volume hierarchy

@njit(cache=True)
def build_bvh(bounds, leaf_size):
    """Task 4. Binned SAH BVH over boxes bounds (N, 2, 3) with 12 bins.

    Return (node_min, node_max, node_left, node_first, node_count, order):
      node_min, node_max  (M, 3) float  box of every node; node 0 is the root
      node_left           (M,) int      first child j (children j and j + 1), or -1 for a leaf
      node_first, node_count (M,) int   a leaf holds order[first : first + count]
      order               (N,) int      permutation of the primitive indices
    A node with at most leaf_size primitives is a leaf. Otherwise split at the
    bin boundary with the lowest SAH cost if that cost is lower than the number
    of primitives of the node, else make it a leaf.
    Check: python check.py 4
    """

    N = bounds.shape[0]
    size = 2*N-1
    node_min = np.zeros(size, 3)
    node_max = np.zeros(size, 3)
    node_left = -np.ones(size)
    node_first =  np.zeros(size)
    node_count =  np.zeros(size)
    order = np.array(range(N))
    centroids = 0.5 * (bounds[:, 0] + bounds[:, 1])

    node_min[0] = np.array(bounds[:, 0, 0].max(),
                           bounds[:, 0, 1].max(),
                           bounds[:, 0, 2].max())

    node_max[0] = np.array(bounds[:, 1, 0].max(),
                           bounds[:, 1, 1].max(),
                           bounds[:, 1, 2].max())
    next_free = 1
    stack = [(0, 0, N)]

    while(len(stack) > 0):
        i, start, end = stack.pop()
        n = end - start

        if n <= leaf_size:
            make_leaf(i, start, n)
            continue

        best_cost = INF
        best_axis = -1
        best_boudnary = -1

        pdx, pdy, pdz  = node_max[i] - node_min[i]
        sap = 2*(pdx*pdy + pdx*pdz + pdy*pdz)

        # bins = [[]*12]
        for axis in [0, 1, 2]:
            cmin = min(centroids[p][axis] for p in range(start, end))
            cmax = max(centroids[p][axis] for p in range(start, end))
            range = cmax - cmin
            if range < EPS:
                continue

            lcum_bin_count = np.zeros(12, np.int64)
            lcum_bin_min   = np.full((12, 3),  INF)
            lcum_bin_max   = np.full((12, 3), -INF)
            for p in range(start, end):
                b = np.floor(12 * (centroids[p][axis] - cmin) / range)
                lcum_bin_count[b] += 1
                lcum_bin_min[b] = np.minimum(lcum_bin_min[b], bounds[p, 0])
                lcum_bin_max[b] = np.maximum(lcum_bin_max[b], bounds[p, 1])

            rcum_bin_count = np.zeros(12, np.int64)
            rcum_bin_min   = np.full((12, 3),  INF)
            rcum_bin_max   = np.full((12, 3), -INF)
            
            for p in reversed(range(start, end)):
                b = np.floor(12 * (centroids[p][axis] - cmin) / range)
                rcum_bin_count[b] += 1
                rcum_bin_min[b] = np.minimum(rcum_bin_min[b], bounds[p, 0])
                rcum_bin_max[b] = np.maximum(rcum_bin_max[b], bounds[p, 1])

            for i in range(12):
                if lcum_bin_count[i] <= 0 or rcum_bin_count[i] <= 0: continue 
                ldx, ldy, ldz = lcum_bin_max[i] - lcum_bin_min[i] 
                sal = 2*(ldx*ldy + ldx*ldz + ldy*ldz)
                rdx, rdy, rdz = rcum_bin_max[i] - rcum_bin_min[i]
                sar = 2*(rdx*rdy + rdx*rdz + rdy*rdz)

                cur_cost = (lcum_bin_count[i]*sal + rcum_bin_count[i]*sar)/sap
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_axis = axis
                    best_boudnary = i

        if best_cost == INF or best_cost > n:
            make_leaf(i, start, n)
            continue

        # --- split ---
        l = start
        r = end - 1

        cmin = min(centroids[p][best_axis] for p in range(start, end))
        cmax = max(centroids[p][best_axis] for p in range(start, end))
        range = cmax - cmin
        while l < r:
            if ((int) min(np.floor(12 * (centroids[l][axis] - cmin) / range), 11)) < b:
                l += 1
            else:
                temp = order[l]
                order[l] = order[r]
                order[r] = temp
                r -= 1
        
        m = l
        L = next_free
        R = L + 1
        next_free += 2
        node_left[i] = L
        node_min[L] = lcum_bin_min[best_boudnary]
        node_max[L] = lcum_bin_max[best_boudnary]
        node_min[R] = rcum_bin_min[best_boudnary]
        node_max[R] = rcum_bin_max[best_boudnary]


    def make_leaf(i, start, n):
        node_first[i] = start;  node_count[i] = n



@njit(cache=True)
def hit_bvh(scene, ro, rd, t_min, t_max):
    """Task 4. (prim, t) of the closest hit through the BVH in scene, or (-1, t_max).

    The BVH arrays are scene.node_min, scene.node_max, scene.node_left,
    scene.node_first, scene.node_count, and scene.order. Use an explicit stack.
    """
    raise NotImplementedError('Task 4: hit_bvh')


# Provided: scene queries used by Task 3

@njit(cache=True)
def hit_scene(scene, ro, rd, t_min, t_max):
    """Provided. Closest hit (prim, t); uses the BVH when the scene has one."""
    if scene.node_left.shape[0] > 0:
        return hit_bvh(scene, ro, rd, t_min, t_max)
    return hit_brute(scene, ro, rd, t_min, t_max) # Eventually calls hit_triange/sphere/bvd


@njit(cache=True)
def surface_normal(scene, prim, p):
    """Provided. Outward unit normal of primitive prim at the surface point p."""
    j = scene.ref[prim]
    k = scene.kind[prim]
    if k == SPHERE:
        return (p - scene.spheres[j, 0:3]) / scene.spheres[j, 3]
    if k == TRIANGLE:
        tri = scene.tris[j]
        return normalize(cross(tri[1] - tri[0], tri[2] - tri[0]))
    return sdf_normal(p, scene.sdfs[j])


# Task 3: path tracing

@njit(cache=True)
def trace_path(ro, rd, scene, state, max_depth, use_roulette):
    """Task 3. Radiance (3,) carried by the camera ray (ro, rd).

    Follow the iterative algorithm of the PDF: closest hit with hit_scene,
    background on a miss, emission on an emitter, scattering by material type,
    throughput update, and Russian roulette from depth 3 when use_roulette is True.
    Material row: scene.materials[scene.mat_id[prim]]; background: scene.background.
    Check: python run.py --scene cornell --size 128 --spp 16 --out cornell.png

    scene.py:35
    self.materials.append([kind, *albedo, roughness, ior, *emission])
    """
    L = np.zeros(3)
    beta = np.ones(3)
    for depth in range(max_depth):
        prim, t = hit_scene(scene, ro, rd, EPS, INF)
        if (prim == -1):  
            # print("background")
            L += beta * scene.background
            return L
        else:
            row = scene.materials[scene.mat_id[prim]]
            material = int(row[M_TYPE])
            if material == EMISSIVE:
                L += beta * row[M_EMISSION:M_EMISSION+3]
                return L

            p = ro + t*rd
            n = surface_normal(scene, prim, p)
            front = np.dot(rd, n) < 0
            n = n if front else -n
            
            if material == LAMBERTIAN:
                rd = scatter_lambertian(n, state)
                beta = beta * row[M_ALBEDO:M_ALBEDO+3]
                # print(beta)
            elif material == METAL:
                reflected, rd = scatter_metal(rd, 
                                            n,
                                            row[M_ROUGHNESS],
                                            state)
                if not reflected: break
                else:
                    beta = beta * row[M_ALBEDO:M_ALBEDO+3]
            elif material == DIELECTRIC:
                rd = scatter_dielectric(rd, 
                                   n,
                                   front,
                                   row[M_IOR],
                                   state)
                # its attenuation is (1, 1, 1), no change to beta

        ro = p
        if use_roulette and depth >= 3:
            q = min(0.95, np.max(beta))
            if rng_next(state) > q: return L
            beta = beta/q
    return L
            
