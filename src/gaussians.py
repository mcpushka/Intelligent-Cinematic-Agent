import numpy as np
import torch
from plyfile import PlyData


class GaussianScene:
    def __init__(self, means, quats, scales, colors, opacities):
        self.means = means
        self.quats = quats
        self.scales = scales
        self.colors = colors
        self.opacities = opacities

        self.center = means.mean(dim=0)
        self.bbox_min = means.min(dim=0).values
        self.bbox_max = means.max(dim=0).values
        self.radius = 0.5 * torch.norm(self.bbox_max - self.bbox_min)
        self.is_indoor = True


def load_gaussian_scene(path: str, device: str = "cuda") -> GaussianScene:
    """Load a simple Gaussian scene from a .ply file.

    This function is intentionally robust to slightly different PLY encodings:
    - Colors are converted to [0, 1] if they appear to be in [0, 255].
    - Scales are clamped to a small minimum to avoid degenerate Gaussians.
    - Quaternions are initialized to identity rotations.
    """
    ply = PlyData.read(path)

    if "chunk" in ply:
        chunk = ply["chunk"].data

        positions = []
        scales = []
        colors = []

        for c in chunk:
            center = [
                0.5 * (c["min_x"] + c["max_x"]),
                0.5 * (c["min_y"] + c["max_y"]),
                0.5 * (c["min_z"] + c["max_z"]),
            ]
            scale = [
                0.5 * (c["min_scale_x"] + c["max_scale_x"]),
                0.5 * (c["min_scale_y"] + c["max_scale_y"]),
                0.5 * (c["min_scale_z"] + c["max_scale_z"]),
            ]
            color = [
                0.5 * (c["min_r"] + c["max_r"]),
                0.5 * (c["min_g"] + c["max_g"]),
                0.5 * (c["min_b"] + c["max_b"]),
            ]

            positions.append(center)
            scales.append(scale)
            colors.append(color)

        positions = torch.tensor(positions, dtype=torch.float32, device=device)
        scales = torch.tensor(scales, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)
        opacities = torch.ones(len(positions), dtype=torch.float32, device=device)
    else:
        raise ValueError("Unsupported PLY format — missing 'chunk' element.")

    # --- Post-process attributes for stable rendering ---
    with torch.no_grad():
        # Colors: make sure they are in [0, 1]
        max_color = colors.max()
        if max_color > 1.5:  # likely 0–255 RGB
            colors = colors / 255.0
        colors = colors.clamp(0.0, 1.0)

        # Scales: avoid zero / negative scales
        scales = torch.clamp(scales, min=1e-3)

        # Quaternions: identity rotation, normalized, format (x, y, z, w)
        quats = torch.zeros((positions.shape[0], 4), dtype=torch.float32, device=device)
        quats[:, 3] = 1.0  # w = 1

    return GaussianScene(positions, quats, scales, colors, opacities)
