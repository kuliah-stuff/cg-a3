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
    size = 2 * N - 1                      # a binary tree with N leaves has at most 2N-1 nodes
    node_min = np.zeros((size, 3))
    node_max = np.zeros((size, 3))
    node_left = -np.ones(size, np.int64)  # int arrays: these hold indices
    node_first = np.zeros(size, np.int64)
    node_count = np.zeros(size, np.int64)
    order = np.arange(N)
    centroids = 0.5 * (bounds[:, 0] + bounds[:, 1])

    # root box: min over the min corners, max over the max corners
    for d in range(3):
        node_min[0, d] = bounds[:, 0, d].min()
        node_max[0, d] = bounds[:, 1, d].max()

    next_free = 1
    # explicit stack of (node index, start, end); at most `size` entries are ever needed
    stack_node = np.zeros(size, np.int64)
    stack_start = np.zeros(size, np.int64)
    stack_end = np.zeros(size, np.int64)
    sp = 1
    stack_node[0] = 0
    stack_start[0] = 0
    stack_end[0] = N

    while sp > 0:
        sp -= 1
        i = stack_node[sp]
        start = stack_start[sp]
        end = stack_end[sp]
        n = end - start

        if n <= leaf_size:
            node_first[i] = start
            node_count[i] = n
            continue

        pdx = node_max[i, 0] - node_min[i, 0]
        pdy = node_max[i, 1] - node_min[i, 1]
        pdz = node_max[i, 2] - node_min[i, 2]
        sap = 2.0 * (pdx * pdy + pdx * pdz + pdy * pdz)
        if sap < 1e-30:
            sap = 1e-30                   # avoid dividing by zero for a degenerate node

        best_cost = INF
        best_axis = -1
        best_boundary = -1
        best_cmin = 0.0
        best_extent = 1.0
        best_lmin = np.zeros(3)
        best_lmax = np.zeros(3)
        best_rmin = np.zeros(3)
        best_rmax = np.zeros(3)

        for axis in range(3):
            cmin = INF
            cmax = -INF
            for k in range(start, end):
                c = centroids[order[k], axis]
                if c < cmin: cmin = c
                if c > cmax: cmax = c
            extent = cmax - cmin          # renamed from `range`, which shadowed the builtin
            if extent < 1e-12:
                continue                  # all centroids coincide on this axis: cannot bin

            # 1) per-bin count and tight box (each bin only sees its own primitives)
            bin_count = np.zeros(12, np.int64)
            bin_min = np.full((12, 3), INF)
            bin_max = np.full((12, 3), -INF)
            for k in range(start, end):
                p = order[k]
                b = min(int(12 * (centroids[p, axis] - cmin) / extent), 11)
                bin_count[b] += 1
                for d in range(3):
                    bin_min[b, d] = min(bin_min[b, d], bounds[p, 0, d])
                    bin_max[b, d] = max(bin_max[b, d], bounds[p, 1, d])

            # 2) left sweep: entry b describes bins 0..b-1, i.e. everything left of boundary b
            left_count = np.zeros(12, np.int64)
            left_min = np.full((12, 3), INF)
            left_max = np.full((12, 3), -INF)
            run_count = 0
            run_min = np.full(3, INF)
            run_max = np.full(3, -INF)
            for b in range(1, 12):
                run_count += bin_count[b - 1]
                for d in range(3):
                    run_min[d] = min(run_min[d], bin_min[b - 1, d])
                    run_max[d] = max(run_max[d], bin_max[b - 1, d])
                left_count[b] = run_count
                left_min[b] = run_min
                left_max[b] = run_max

            # 3) right sweep: entry b describes bins b..11, i.e. everything right of boundary b
            right_count = np.zeros(12, np.int64)
            right_min = np.full((12, 3), INF)
            right_max = np.full((12, 3), -INF)
            run_count = 0
            run_min = np.full(3, INF)
            run_max = np.full(3, -INF)
            for b in range(11, 0, -1):
                run_count += bin_count[b]
                for d in range(3):
                    run_min[d] = min(run_min[d], bin_min[b, d])
                    run_max[d] = max(run_max[d], bin_max[b, d])
                right_count[b] = run_count
                right_min[b] = run_min
                right_max[b] = run_max

            # 4) SAH cost of each of the 11 boundaries
            for b in range(1, 12):        # `b`, not `i`: `i` is the current node index
                if left_count[b] == 0 or right_count[b] == 0:
                    continue
                ldx = left_max[b, 0] - left_min[b, 0]
                ldy = left_max[b, 1] - left_min[b, 1]
                ldz = left_max[b, 2] - left_min[b, 2]
                sal = 2.0 * (ldx * ldy + ldx * ldz + ldy * ldz)
                rdx = right_max[b, 0] - right_min[b, 0]
                rdy = right_max[b, 1] - right_min[b, 1]
                rdz = right_max[b, 2] - right_min[b, 2]
                sar = 2.0 * (rdx * rdy + rdx * rdz + rdy * rdz)

                cur_cost = 1.0 + (left_count[b] * sal + right_count[b] * sar) / sap
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_axis = axis
                    best_boundary = b
                    best_cmin = cmin
                    best_extent = extent
                    # keep the child boxes now: these arrays are overwritten on the next axis
                    best_lmin = left_min[b].copy()
                    best_lmax = left_max[b].copy()
                    best_rmin = right_min[b].copy()
                    best_rmax = right_max[b].copy()

        # keep the node as a leaf when no split exists or splitting is not cheaper
        if best_axis < 0 or best_cost >= n:
            node_first[i] = start
            node_count[i] = n
            continue

        # --- split: partition order[start:end] in place, left-bin primitives first ---
        l = start
        r = end - 1
        while l <= r:
            p = order[l]                  # `l` is a position in `order`, not a primitive id
            b = min(int(12 * (centroids[p, best_axis] - best_cmin) / best_extent), 11)
            if b < best_boundary:
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
        node_min[L] = best_lmin
        node_max[L] = best_lmax
        node_min[R] = best_rmin
        node_max[R] = best_rmax

        stack_node[sp] = R
        stack_start[sp] = m
        stack_end[sp] = end
        sp += 1
        stack_node[sp] = L
        stack_start[sp] = start
        stack_end[sp] = m
        sp += 1

    return (node_min[:next_free], node_max[:next_free], node_left[:next_free],
            node_first[:next_free], node_count[:next_free], order)


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
            
