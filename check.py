"""Public checks for Assignment 3. Run from the starter directory:

    python check.py 1      # Task 1
    python check.py        # every task

These checks cover only a few cases; test your functions on inputs of
your own as well.
"""

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import configure_numba  # noqa: E402

configure_numba('--no-jit' in sys.argv)

import numpy as np  # noqa: E402

import student  # noqa: E402
from framework import INF  # noqa: E402

RESULTS = []


def check(name, condition, detail=''):
    RESULTS.append(bool(condition))
    print(f'  [{"pass" if condition else "FAIL"}] {name}' + (f': {detail}' if detail and not condition else ''))


def vec(*v):
    return np.array(v, float)


def state(seed=1):
    return np.array([seed], np.uint64)


def task1():
    print('Task 1: ray-primitive intersection')
    c, r = vec(0, 0, 0), 1.0
    t = student.hit_sphere(c, r, vec(0, 0, -3), vec(0, 0, 1), 1e-4, INF)
    check('sphere hit from outside at t = 2', abs(t - 2.0) < 1e-9, f'got {t}')
    t = student.hit_sphere(c, r, vec(0, 0, 0), vec(0, 0, 1), 1e-4, INF)
    check('sphere hit from inside at t = 1', abs(t - 1.0) < 1e-9, f'got {t}')
    t = student.hit_sphere(c, r, vec(0, 2, -3), vec(0, 0, 1), 1e-4, INF)
    check('sphere missed', t == INF, f'got {t}')
    t = student.hit_sphere(c, r, vec(0, 0, -3), vec(0, 0, 1), 1e-4, 1.5)
    check('sphere beyond t_max', t == INF, f'got {t}')
    a, b, cc = vec(-1, -1, 0), vec(1, -1, 0), vec(0, 1, 0)
    t = student.hit_triangle(a, b, cc, vec(0, 0, -2), vec(0, 0, 1), 1e-4, INF)
    check('triangle hit at t = 2', abs(t - 2.0) < 1e-9, f'got {t}')
    t = student.hit_triangle(a, b, cc, vec(0, 0, 2), vec(0, 0, -1), 1e-4, INF)
    check('triangle hit from the back at t = 2', abs(t - 2.0) < 1e-9, f'got {t}')
    t = student.hit_triangle(a, b, cc, vec(0.9, 0.9, -2), vec(0, 0, 1), 1e-4, INF)
    check('triangle missed outside an edge', t == INF, f'got {t}')
    t = student.hit_triangle(a, b, cc, vec(0, 0, -2), vec(1, 0, 0), 1e-4, INF)
    check('triangle missed by a parallel ray', t == INF, f'got {t}')
    d = vec(1, 1, 1) / math.sqrt(3)
    hit = student.hit_aabb(vec(1, 1, 1), vec(2, 2, 2), vec(0, 0, 0), 1.0 / d, 1e-4, INF)
    check('box hit along the diagonal', hit)
    hit = student.hit_aabb(vec(1, 1, 1), vec(2, 2, 2), vec(0, 0, 0), 1.0 / d, 1e-4, 1.0)
    check('box beyond t_max', not hit)
    d = normalize(vec(1, -1, 1))
    hit = student.hit_aabb(vec(1, 1, 1), vec(2, 2, 2), vec(0, 0, 0), 1.0 / d, 1e-4, INF)
    check('box missed', not hit)


def normalize(v):
    return v / np.linalg.norm(v)


def task2():
    print('Task 2: materials (statistics over 20,000 samples)')
    n = normalize(vec(0.3, 1.0, -0.2))
    st = state(3)
    dirs = np.array([student.scatter_lambertian(n, st) for _ in range(20000)])
    cos = dirs @ n
    check('Lambertian directions have unit length', np.allclose(np.linalg.norm(dirs, axis=1), 1, atol=1e-9))
    check('Lambertian directions lie above the surface', np.all(cos >= -1e-9))
    check('Lambertian mean of cos(theta) is close to 2/3', abs(cos.mean() - 2 / 3) < 0.01, f'got {cos.mean():.4f}')
    d = normalize(vec(1.0, -1.0, 0.0))
    up = vec(0, 1, 0)
    ok, r = student.scatter_metal(d, up, 0.0, state())
    check('metal with roughness 0 is a mirror', ok and np.allclose(r, normalize(vec(1, 1, 0)), atol=1e-9), f'got {ok}, {r}')
    ok, r = student.scatter_metal(normalize(vec(1.0, -0.05, 0.0)), up, 1.0, state(5))
    oks = [student.scatter_metal(normalize(vec(1.0, -0.05, 0.0)), up, 1.0, st)[0] for _ in range(5000)]
    check('rough metal at a grazing angle absorbs some rays', 0.05 < 1 - np.mean(oks) < 0.95, f'absorbed fraction {1 - np.mean(oks):.3f}')
    d = vec(0, -1, 0)
    outs = np.array([student.scatter_dielectric(d, up, True, 1.5, st) for _ in range(20000)])
    reflected = outs[:, 1] > 0
    check('glass at normal incidence reflects about 4% of the rays', abs(reflected.mean() - 0.04) < 0.006, f'got {reflected.mean():.4f}')
    check('glass at normal incidence refracts straight through', np.allclose(outs[~reflected], d, atol=1e-9))
    d = normalize(vec(1.0, -0.3, 0.0))
    outs = np.array([student.scatter_dielectric(d, up, False, 1.5, st) for _ in range(200)])
    check('total internal reflection inside glass', np.allclose(outs, normalize(vec(1.0, 0.3, 0.0)), atol=1e-9))
    # print("SNELL CHECK")
    d = normalize(vec(1.0, -1.0, 0.0))
    outs = np.array([student.scatter_dielectric(d, up, True, 1.5, st) for _ in range(2000)])
    refr = outs[outs[:, 1] < 0]
    sin_t = np.abs(refr[:, 0])
    check("refracted rays follow Snell's law", len(refr) > 0 and np.allclose(sin_t, math.sin(math.pi / 4) / 1.5, atol=1e-9), f"got #refracted: {len(refr)}, sin_t: {sin_t}")


def task3():
    print('Task 3: path tracing')
    import scene as sc
    from framework import LAMBERTIAN, METAL
    sky = (0.5, 0.7, 1.0)
    b = sc.SceneBuilder(background=sky)
    b.quad((-50, 0, -50), (-50, 0, 50), (50, 0, 50), (50, 0, -50), b.material(LAMBERTIAN, (0.6, 0.5, 0.4)))
    world = b.build()
    st = state(9)
    miss = student.trace_path(vec(0, 1, 0), vec(0, 1, 0), world, st, 50, True)
    check('a ray that misses returns the background', np.allclose(miss, sky), f'got {miss}')
    hit = student.trace_path(vec(0, 1, 0), vec(0, -1, 0), world, st, 50, True)
    check('one bounce on a Lambertian floor under the sky gives albedo * sky', np.allclose(hit, vec(0.3, 0.35, 0.4)), f'got {hit}')
    hit = student.trace_path(vec(0, 1, 0), vec(0, -1, 0), world, st, 1, True)
    check('max_depth 1 stops before the sky is reached', np.allclose(hit, 0), f'got {hit}')
    b = sc.SceneBuilder(background=sky)
    b.quad((-50, 0, -50), (-50, 0, 50), (50, 0, 50), (50, 0, -50), b.material(METAL, (0.9, 0.9, 0.9), roughness=0.0))
    hit = student.trace_path(vec(0, 1, 0), normalize(vec(0.2, -1, 0)), b.build(), st, 50, True)
    check('a mirror floor reflects the sky', np.allclose(hit, 0.9 * np.array(sky)), f'got {hit}')


def task4():
    print('Task 4: bounding volume hierarchy')
    import scene as sc
    rng = np.random.default_rng(4)
    lo = rng.uniform(-5, 5, (300, 3))
    bounds = np.stack([lo, lo + rng.uniform(0.05, 0.5, (300, 3))], axis=1)
    node_min, node_max, node_left, node_first, node_count, order = student.build_bvh(bounds, 2)
    check('order is a permutation', sorted(order.tolist()) == list(range(300)))
    leaves = node_left < 0
    check('every primitive is in exactly one leaf', node_count[leaves].sum() == 300)
    check('children of inner nodes exist', np.all(node_left[~leaves] + 1 < len(node_left)))
    ok = True
    for i in np.nonzero(leaves)[0]:
        prims = order[node_first[i]:node_first[i] + node_count[i]]
        ok &= np.all(bounds[prims, 0] >= node_min[i] - 1e-12) and np.all(bounds[prims, 1] <= node_max[i] + 1e-12)
    check('leaf boxes contain their primitives', ok)
    world, _ = sc.cornell(2)
    world = sc.without_sdf(world)
    bvh = sc.with_bvh(world)
    same = 0
    for _ in range(500):
        ro = rng.uniform(0.05, 0.95, 3)
        rd = normalize(rng.normal(size=3))
        p1, t1 = student.hit_bvh(bvh, ro, rd, 1e-4, INF)
        p2, t2 = student.hit_brute(world, ro, rd, 1e-4, INF)
        same += abs(t1 - t2) < 1e-9
    check('hit_bvh agrees with the brute-force search on 500 rays', same == 500, f'{same} of 500 agree')


def task5():
    print('Task 5: signed distance functions')
    from framework import sdf_eval
    params = vec(0, 0, 0, 0.5, 0.5, 0.5, 0.0, 0.6)
    t = student.hit_sdf(params, vec(0.45, 0.45, -3), vec(0, 0, 1), 1e-4, INF)
    check('box face hit at t = 2.5', abs(t - 2.5) < 1e-4, f'got {t}')
    t = student.hit_sdf(params, vec(0, 0, -3), vec(0, 0, 1), 1e-4, INF)
    check('a ray along the axis passes through the hole', t == INF, f'got {t}')
    t = student.hit_sdf(params, vec(0, 3, -3), vec(0, 0, 1), 1e-4, INF)
    check('SDF missed', t == INF, f'got {t}')
    n = student.sdf_normal(vec(0.45, 0.45, -0.5), params)
    check('normal on the face z = -0.5 is (0, 0, -1)', np.allclose(n, vec(0, 0, -1), atol=1e-4), f'got {n}')
    p = 0.6 * normalize(vec(1, 1, 1))
    n = student.sdf_normal(p, params)
    check('normal on the spherical cut points to the centre', abs(sdf_eval(p, params)) < 1e-9
          and np.allclose(n, -normalize(vec(1, 1, 1)), atol=1e-4), f'got {n}')


TASKS = {'1': task1, '2': task2, '3': task3, '4': task4, '5': task5}


def main():
    chosen = [a for a in sys.argv[1:] if a in TASKS] or list(TASKS)
    for key in chosen:
        try:
            TASKS[key]()
        except NotImplementedError as error:
            print(f'  Incomplete student implementation: {error}')
            RESULTS.append(False)
    print(f'{sum(RESULTS)} of {len(RESULTS)} checks passed')


if __name__ == '__main__':
    main()
