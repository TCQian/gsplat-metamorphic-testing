"""Pixel-level diagnosis of an MR1 (packed vs unpacked) difference on large-scale scenes.

Replays scene(s) as runners/run_large_scale.py built them (by default the
all-camera run; `--seed 1000 --modes` with the original eight types for the
camera-0 runs), renders with packed=True and packed=False on the camera that
shows the difference (`--camera`, as in diagnose_mr4.py), and bisects (delta
debugging) a smallest set of Gaussians that still reproduces the worst pixel's
difference, falling back to the alpha image when the rgb image has nothing over
tolerance. It prints the kept Gaussians' depths and alphas from each render's own
gsplat meta and ends with `CLASS:` and `PATTERN:` lines read off those values.

`--out FILE` (with --scenes-from) writes one JSON line per scene as it goes,
`--resume` continues a file after a timeout, and `--summarize FILE...` prints the
pattern counts.

Usage (from the repository root):
    python diagnosis/diagnose_mr1.py --scene-idx 42081
    python diagnosis/diagnose_mr1.py --scenes-from results/allcam_100000_findings.csv --where mr1_verdict=isolated_discontinuity --out results/bisect/mr1.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for `import gsplat_mt`

from diagnosis.diagnose_mr4 import (  # noqa: E402, F401 - render_mode is re-exported for other tools
    add_batch_args,
    add_camera_arg,
    bisect_finding,
    load,
    render_mode,
    run_batch,
    summarize,
)
from gsplat_mt.scene import SCENE_MODES  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-idx", type=int, default=None)
    p.add_argument("--scenes-from", type=str, default=None)
    p.add_argument("--where", type=str, default="mr1_verdict=isolated_discontinuity")
    p.add_argument("--every", type=int, default=1, help="with --scenes-from, bisect every K-th matching scene")
    p.add_argument("--offset", type=int, default=0, help="with --every, start at the OFFSET-th matching scene")
    p.add_argument("--seed", type=int, default=None, help="seed of the run being replayed (default 2000, the all-camera run)")
    p.add_argument("--modes", nargs="+", default=None, choices=SCENE_MODES, help="scene types of the run (default all nine)")
    p.add_argument("--by", choices=["color", "T", "alpha"], default="color")
    add_camera_arg(p)
    add_batch_args(p)
    a = p.parse_args()
    a.large, a.mode = True, None

    if a.summarize:
        summarize(a.summarize)
        return 0
    if a.scenes_from:
        col, val = a.where.split("=", 1)
        with open(a.scenes_from) as f:
            idxs = [int(r["scene_idx"]) for r in csv.DictReader(f) if r.get(col) == val][a.offset :: a.every]
        run_batch("mr1", [(i, 1.0) for i in idxs], a)
        return 0
    if a.scene_idx is None:
        p.error("--scene-idx or --scenes-from is required")
    mode, scene, cameras = load(a, a.scene_idx)
    print(f"scene {a.scene_idx} [{mode}] n={scene.n} cameras={len(cameras)}")
    c, by, _, r = bisect_finding("mr1", scene, cameras, 1.0, a.camera, a.by)
    print(f"\ncamera {c}, {by}: {r['pattern'] if r else 'nothing over tolerance'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
