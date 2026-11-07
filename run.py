"""Command-line renderer for Assignment 3.

Examples (run from the starter directory):
    python run.py --scene cornell --size 128 --spp 16 --out cornell.png
    python run.py --scene cornell --size 64 --spp 4 --no-bvh --stats
    python run.py --scene materials --size 320 --spp 64 --out materials.png --npy materials.npy
    python run.py --scene cornell --size 32 --spp 1 --no-jit
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Assignment 3 path tracer')
    parser.add_argument('--scene', default='cornell',
                        help='cornell, spheres, materials, or sdf (default cornell)')
    parser.add_argument('--size', type=int, default=128, help='image height in pixels (default 128)')
    parser.add_argument('--spp', type=int, default=16, help='samples per pixel (default 16)')
    parser.add_argument('--max-depth', type=int, default=50, help='maximum path length (default 50)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('--mesh-level', type=int, default=4,
                        help='subdivision level of the triangle ball in cornell (default 4)')
    parser.add_argument('--no-bvh', action='store_true', help='test every primitive for every ray')
    parser.add_argument('--no-roulette', action='store_true', help='disable Russian roulette')
    parser.add_argument('--no-jit', action='store_true', help='run as plain Python (slow, for debugging)')
    parser.add_argument('--stats', action='store_true', help='print primitive counts and timings')
    parser.add_argument('--out', help='write the tone-mapped image to this PNG file')
    parser.add_argument('--npy', help='write the linear radiance (height, width, 3) to this .npy file')
    return parser.parse_args()


def configure_numba(no_jit):
    """Disable the JIT, or cache compiled code per version of the source files."""
    if no_jit:
        import warnings
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        # the random number generator relies on uint64 wrap-around, which NumPy reports
        warnings.filterwarnings('ignore', message='overflow encountered', category=RuntimeWarning)
        return
    digest = hashlib.sha1()
    for name in ('student.py', 'framework.py', 'scene.py'):
        digest.update((HERE / name).read_bytes())
    os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '__pycache__' / f'numba-{digest.hexdigest()[:12]}'))


def probe_tasks():
    """Return the first unfinished task needed for rendering, and whether Task 5 is done."""
    import numpy as np
    import student
    calls = [
        ('Task 1', lambda: student.hit_sphere(np.zeros(3), 1.0, np.array([0., 0, -3]), np.array([0., 0, 1]), 1e-4, 1e30)),
        ('Task 1', lambda: student.hit_triangle(np.array([-1., -1, 0]), np.array([1., -1, 0]), np.array([0., 1, 0]),
                                                np.array([0., 0, -3]), np.array([0., 0, 1]), 1e-4, 1e30)),
        ('Task 2', lambda: student.scatter_lambertian(np.array([0., 1, 0]), np.ones(1, np.uint64))),
        ('Task 2', lambda: student.scatter_metal(np.array([0., -1, 0]), np.array([0., 1, 0]), 0.1, np.ones(1, np.uint64))),
        ('Task 2', lambda: student.scatter_dielectric(np.array([0., -1, 0]), np.array([0., 1, 0]), True, 1.5, np.ones(1, np.uint64))),
    ]
    for _, call in calls:
        try:
            call()
        except NotImplementedError as error:
            return str(error), None, None
    sdf_done = True
    try:
        student.hit_sdf(np.array([0., 0, 0, .5, .5, .5, 0, .6]), np.array([0., 0, -3]), np.array([0., 0, 1]), 1e-4, 1e30)
        student.sdf_normal(np.array([0., 0, -0.5]), np.array([0., 0, 0, .5, .5, .5, 0, .6]))
    except NotImplementedError:
        sdf_done = False
    bvh_done = True
    try:
        student.build_bvh(np.array([[[0., 0, 0], [1., 1, 1]]]), 2)
    except NotImplementedError:
        bvh_done = False
    return None, sdf_done, bvh_done


def main():
    args = parse_args()
    configure_numba(args.no_jit)
    import numpy as np
    import scene as sc

    missing, sdf_done, bvh_done = probe_tasks()
    if missing:
        print(f'Incomplete student implementation: {missing}. No image was rendered.')
        return
    try:
        world, (cam, half_h) = sc.load_scene(args.scene, args.mesh_level)
    except ValueError as error:
        print(error)
        raise SystemExit(1)
    if not sdf_done:
        print('Task 5 is not implemented yet: the SDF object is left out of the scene.')
        world = sc.without_sdf(world)
    t0 = time.perf_counter()
    if not args.no_bvh:
        if not bvh_done:
            print('Incomplete student implementation: Task 4: build_bvh. Use --no-bvh until Task 4 is done.')
            return
        world = sc.with_bvh(world)
    t_build = time.perf_counter() - t0

    import student
    try:
        student.hit_scene(world, cam[0].copy(), -cam[3], 1e-4, 1e30)
        student.trace_path(cam[0].copy(), -cam[3], world, np.ones(1, np.uint64), 1, True)
    except NotImplementedError as error:
        print(f'Incomplete student implementation: {error}. No image was rendered.')
        return

    height = args.size
    width = round(args.size * (4 / 3 if args.scene in ('spheres', 'materials') else 1))
    try:
        t0 = time.perf_counter()
        sc.render(world, cam, half_h, 2, 2, 1, args.max_depth, args.seed, not args.no_roulette)
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        image = sc.render(world, cam, half_h, width, height, args.spp, args.max_depth, args.seed,
                          not args.no_roulette)
        t_render = time.perf_counter() - t0
    except NotImplementedError as error:
        print(f'Incomplete student implementation: {error}. No image was rendered.')
        return

    if not np.all(np.isfinite(image)):
        print('Warning: the image contains NaN or infinite values; check your normals and directions.')
    print(f'Rendered {args.scene} at {width}x{height}, {args.spp} spp, seed {args.seed}; '
          f'mean radiance {np.nanmean(image):.4f}')
    if args.stats:
        nodes = len(world.node_left)
        print(f'primitives {len(world.kind)}, BVH nodes {nodes if nodes else "none"}')
        print(f'BVH build {t_build:.3f} s, compile and warm-up {t_compile:.1f} s, render {t_render:.2f} s')
    if args.out:
        sc.save_png(image, args.out)
        print(f'Wrote {args.out}')
    if args.npy:
        np.save(args.npy, image)
        print(f'Wrote {args.npy}')


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    main()
