import numpy as np
import torch
from plyfile import PlyData
from dataclasses import dataclass
from typing import Optional


SH_C0 = 0.28209479177387814  # constant used for SH -> RGB


@dataclass
class GaussianScene:
    """Container for a Gaussian Splat scene in torch tensors."""
    means: torch.Tensor      # [N, 3]
    quats: torch.Tensor      # [N, 4]  (wxyz)
    scales: torch.Tensor     # [N, 3]
    opacities: torch.Tensor  # [N]
    colors: torch.Tensor     # [N, 3], RGB in [0,1]

    bbox_min: torch.Tensor   # [3]
    bbox_max: torch.Tensor   # [3]
    center: torch.Tensor     # [3]
    radius: float            # scalar

    @property
    def is_indoor(self) -> bool:
        """Heuristic to guess indoor vs outdoor based on extents."""
        extents = self.bbox_max - self.bbox_min
        # Simple heuristic: indoor scenes usually not too huge in XY.
        xy_extent = float(max(extents[0], extents[2]))
        z_extent = float(extents[1])
        # If horizontal extent is moderate and vertical is small, call it indoor.
        return xy_extent < 100.0 and z_extent < 20.0


def _load_ply_numpy(path: str):
    """Load Gaussian-splat PLY file into numpy arrays.

    This follows the de-facto 3DGS PLY layout:
    - x, y, z
    - scale_0, scale_1, scale_2 (log-scales, so we exp them)
    - rot_0..rot_3 (quaternion components)
    - f_dc_0..f_dc_2 (SH DC term -> base color)
    - opacity (logit, so we pass through sigmoid)
    If SH fields are missing, we fall back to standard RGB fields if present.
    """
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # Scales: stored as log, need exp
    if {"scale_0", "scale_1", "scale_2"}.issubset(v.data.dtype.names):
        scales = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
        ).astype(np.float32)
        scales = np.exp(scales)
    else:
        # Fallback: small isotropic scales
        scales = np.full_like(positions, 0.01, dtype=np.float32)

    # Rotations: quaternion wxyz
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(v.data.dtype.names):
        quats = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        ).astype(np.float32)
    else:
        # Identity rotation (no orientation data)
        quats = np.zeros((positions.shape[0], 4), dtype=np.float32)
        quats[:, 0] = 1.0

    # Colors:
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(v.data.dtype.names):
        # As in viser example: color = 0.5 + SH_C0 * f_dc
        colors = 0.5 + SH_C0 * np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
        ).astype(np.float32)
        colors = np.clip(colors, 0.0, 1.0)
    elif {"red", "green", "blue"}.issubset(v.data.dtype.names):
        colors = np.stack(
            [v["red"], v["green"], v["blue"]], axis=-1
        ).astype(np.float32) / 255.0
    else:
        # Fallback: white
        colors = np.ones_like(positions, dtype=np.float32)

    # Opacity: stored as logit -> sigmoid
    if "opacity" in v.data.dtype.names:
        opacity_raw = v["opacity"].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_raw))
    else:
        opacities = np.ones((positions.shape[0],), dtype=np.float32)

    return positions, quats, scales, opacities, colors


def load_gaussian_scene(path: str, device: Optional[str] = None) -> GaussianScene:
    """Load Gaussian scene from PLY and wrap it into GaussianScene."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    positions, quats, scales, opacities, colors = _load_ply_numpy(path)

    means_t = torch.from_numpy(positions).to(device)
    quats_t = torch.from_numpy(quats).to(device)
    scales_t = torch.from_numpy(scales).to(device)
    opacities_t = torch.from_numpy(opacities).to(device)
    colors_t = torch.from_numpy(colors).to(device)

    bbox_min = means_t.min(dim=0).values
    bbox_max = means_t.max(dim=0).values
    center = (bbox_min + bbox_max) * 0.5
    radius = float(torch.norm(bbox_max - bbox_min) * 0.5)

    return GaussianScene(
        means=means_t,
        quats=quats_t,
        scales=scales_t,
        opacities=opacities_t,
        colors=colors_t,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=center,
        radius=radius,
    )
