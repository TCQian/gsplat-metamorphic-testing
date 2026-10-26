"""How gsplat 1.5.3 orders two Gaussians whose depths are equal or one float32 step apart.

Two overlapping Gaussians A (red) and B (blue), same x/y and scale, differ only in
camera-space depth z. Each case is rendered with the Gaussians stored in both
index orders (A first, B first) and with packed=False / packed=True, and the
center pixel is printed, so it is visible whether the front Gaussian is chosen
by depth, by index, or by neither.

Usage:
    python gsplat_mt/probe_ties.py
"""

from __future__ import annotations

import gsplat
import numpy as np
import torch

W = H = 64
Z0 = 2.5


def render(z, order, packed, device="cuda"):
    """Center pixel of A(red) and B(blue) at depths z=(zA, zB), stored in index order `order`."""
    means = torch.tensor([[0.0, 0.0, z[0]], [0.0, 0.0, z[1]]], dtype=torch.float32, device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    idx = torch.tensor(order, device=device)
    rgb, _, _ = gsplat.rasterization(
        means=means[idx],
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(2, 1),
        scales=torch.full((2, 3), 0.2, device=device),
        opacities=torch.tensor([0.8, 0.8], device=device),
        colors=colors[idx],
        viewmats=torch.eye(4, device=device)[None],
        Ks=torch.tensor([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]], device=device)[None],
        width=W,
        height=H,
        sh_degree=None,
        packed=packed,
    )
    return rgb[0, H // 2, W // 2].tolist()


def main() -> int:
    z = np.float32(Z0)
    up, down = float(np.nextafter(z, np.float32(10))), float(np.nextafter(z, np.float32(0)))
    cases = [
        ("equal depth", (float(z), float(z))),
        ("B one float32 step farther than A", (float(z), up)),
        ("B one float32 step nearer than A", (float(z), down)),
    ]
    for name, zz in cases:
        print(f"\n{name}: zA={zz[0]!r} zB={zz[1]!r}")
        for packed in (False, True):
            ab = render(zz, [0, 1], packed)
            ba = render(zz, [1, 0], packed)
            front = lambda c: "A (red) in front" if c[0] > c[2] else "B (blue) in front" if c[2] > c[0] else "equal"
            print(
                f"  packed={packed!s:5}  stored [A,B]: {['%.4f' % v for v in ab]} -> {front(ab)}"
                f" | stored [B,A]: {['%.4f' % v for v in ba]} -> {front(ba)}"
                f" | image changes with storage order: {ab != ba}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
