"""Per-scene random configuration of the large-scale run, and replay of one of its scenes.

Each scene index draws its own configuration (number of Gaussians, scene radius, scale and
opacity ranges, camera distance, field of view, number of cameras, resolution, scene type)
from a generator seeded with "<seed>:<index>", so any scene can be rebuilt on its own from the
run's seed and list of scene types.

Two runs are referred to throughout the repository:
  - the all-camera run (`soc-cluster.sh`): seed 2000, all nine scene types (the defaults here)
  - the earlier camera-0 runs (weeks 9-12): seed 1000, the original eight scene types
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, Sequence, Tuple

from gsplat_mt.harness import build_argparser, build_scene_and_cameras
from gsplat_mt.scene import ORIGINAL_MODES, SCENE_MODES

RESOLUTIONS = [(64, 64), (120, 90), (128, 128), (256, 192), (256, 256)]
WORLD_SCALES = [0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0]
N_MIN, N_MAX, MAX_CAMERAS = 10, 5000, 8

ALLCAM_SEED, ALLCAM_MODES = 2000, SCENE_MODES
CAMERA0_SEED, CAMERA0_MODES = 1000, ORIGINAL_MODES


def loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_config(scene_idx: int, args: argparse.Namespace) -> Dict[str, object]:
    """Deterministic per-scene configuration (independent of the scene's own seed).

    Reads `args.seed`, `args.modes`, `args.n_min`, `args.n_max` and `args.max_cameras`.
    """
    rng = random.Random(f"{args.seed}:{scene_idx}")
    n = int(round(loguniform(rng, args.n_min, args.n_max)))
    width, height = rng.choice(RESOLUTIONS)
    radius = loguniform(rng, 0.3, 2.0)
    scale_c = loguniform(rng, 0.01, 0.3)
    op_lo = rng.uniform(0.05, 0.9)
    dist_min = radius * rng.uniform(1.2, 8.0)
    return {
        "mode": rng.choice(args.modes),
        "n": n,
        "width": width,
        "height": height,
        "n_cameras": rng.randint(1, args.max_cameras),
        "mean_radius": radius,
        "scale_min": scale_c * 0.5,
        "scale_max": scale_c * 1.5,
        "opacity_min": op_lo,
        "opacity_max": rng.uniform(op_lo, 1.0),
        "dist_min": dist_min,
        "dist_max": dist_min * rng.uniform(1.0, 1.6),
        "fov_deg": rng.uniform(30.0, 90.0),
    }


def scene_args(cfg: Dict[str, object], args: argparse.Namespace) -> argparse.Namespace:
    """Namespace with the fields `harness.run_one_scene` reads, set from `cfg`."""
    ns = build_argparser().parse_args([])
    ns.seed = args.seed
    ns.world_scales = args.world_scales
    ns.n_gaussians_min = ns.n_gaussians_max = cfg["n"]
    for k in (
        "width",
        "height",
        "n_cameras",
        "mean_radius",
        "scale_min",
        "scale_max",
        "opacity_min",
        "opacity_max",
        "dist_min",
        "dist_max",
        "fov_deg",
    ):
        setattr(ns, k, cfg[k])
    return ns


def run_namespace(seed: int = ALLCAM_SEED, modes: Sequence[str] = ALLCAM_MODES) -> argparse.Namespace:
    """The sampling arguments of a large-scale run with the runner's default ranges."""
    return argparse.Namespace(
        seed=seed, modes=list(modes), n_min=N_MIN, n_max=N_MAX, max_cameras=MAX_CAMERAS, world_scales=WORLD_SCALES
    )


def replay_scene(
    scene_idx: int, run: argparse.Namespace = None, device: str = "cuda"
) -> Tuple[Dict[str, object], argparse.Namespace, object, list]:
    """(cfg, scene_args, scene, cameras) of scene `scene_idx` of the large-scale run `run`
    (a `run_namespace`, or the runner's own parsed arguments; default: the all-camera run)."""
    run = run or run_namespace()
    cfg = sample_config(scene_idx, run)
    ns = scene_args(cfg, run)
    scene, cameras = build_scene_and_cameras(scene_idx, cfg["mode"], ns, device)
    return cfg, ns, scene, cameras


def add_run_args(p: argparse.ArgumentParser, seed: int = ALLCAM_SEED, modes: Sequence[str] = ALLCAM_MODES) -> None:
    """--seed and --modes of the large-scale run whose scenes a tool replays."""
    p.add_argument("--seed", type=int, default=seed, help=f"seed of the run being replayed (default {seed})")
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(modes),
        choices=SCENE_MODES,
        help="scene types of the run being replayed, in the same order (default: "
        + ("all nine" if list(modes) == list(SCENE_MODES) else "the original eight")
        + "); the camera-0 runs used --seed 1000 and the original eight",
    )
