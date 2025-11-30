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
    plydata = PlyData.read(path)

    element_names = {el.name for el in plydata.elements}

    if "chunk" in element_names:
        c = plydata["chunk"]
        cn = set(c.data.dtype.names)

        required = {
            "min_x", "min_y", "min_z",
            "max_x", "max_y", "max_z",
            "min_scale_x", "min_scale_y", "min_scale_z",
            "max_scale_x", "max_scale_y", "max_scale_z",
        }
        if not required.issubset(cn):
            raise ValueError(
                f"SuperSplat 'chunk' element missing fields, got {cn}"
            )

        min_x = c["min_x"].astype(np.float32)
        min_y = c["min_y"].astype(np.float32)
        min_z = c["min_z"].astype(np.float32)
        max_x = c["max_x"].astype(np.float32)
        max_y = c["max_y"].astype(np.float32)
        max_z = c["max_z"].astype(np.float32)

        x = 0.5 * (min_x + max_x)
        y = 0.5 * (min_y + max_y)
        z = 0.5 * (min_z + max_z)
        positions = np.stack([x, y, z], axis=-1)

        min_sx = c["min_scale_x"].astype(np.float32)
        min_sy = c["min_scale_y"].astype(np.float32)
        min_sz = c["min_scale_z"].astype(np.float32)
        max_sx = c["max_scale_x"].astype(np.float32)
        max_sy = c["max_scale_y"].astype(np.float32)
        max_sz = c["max_scale_z"].astype(np.float32)

        sx = 0.5 * (min_sx + max_sx)
        sy = 0.5 * (min_sy + max_sy)
        sz = 0.5 * (min_sz + max_sz)
        scales = np.stack([sx, sy, sz], axis=-1)

        if {
            "min_r", "min_g", "min_b",
            "max_r", "max_g", "max_b",
        }.issubset(cn):
            min_r = c["min_r"].astype(np.float32)
            min_g = c["min_g"].astype(np.float32)
            min_b = c["min_b"].astype(np.float32)
            max_r = c["max_r"].astype(np.float32)
            max_g = c["max_g"].astype(np.float32)
            max_b = c["max_b"].astype(np.float32)

            r = 0.5 * (min_r + max_r)
            g = 0.5 * (min_g + max_g)
            b = 0.5 * (min_b + max_b)
            colors = np.stack([r, g, b], axis=-1)

            if colors.max() > 1.0:  # вдруг 0..255
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
        else:
            colors = np.ones_like(positions, dtype=np.float32)

        n = positions.shape[0]
        opacities = np.ones((n,), dtype=np.float32)

        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0  # w=1, x=y=z=0 (identity rotation)

        return (
            positions.astype(np.float32),
            quats,
            scales.astype(np.float32),
            opacities,
            colors.astype(np.float32),
        )

    if "vertex" in element_names:
        v = plydata["vertex"]
        names = set(v.data.dtype.names)

        if {"x", "y", "z"}.issubset(names):
            positions = np.stack(
                [v["x"], v["y"], v["z"]], axis=-1
            ).astype(np.float32)

            # Scales
            if {"scale_0", "scale_1", "scale_2"}.issubset(names):
                scales = np.stack(
                    [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
                ).astype(np.float32)
                scales = np.exp(scales)
            else:
                scales = np.full_like(positions, 0.01, dtype=np.float32)

            # Rotations
            if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
                quats = np.stack(
                    [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
                ).astype(np.float32)
            else:
                quats = np.zeros((positions.shape[0], 4), dtype=np.float32)
                quats[:, 0] = 1.0

            # Colors
            if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
                colors = 0.5 + SH_C0 * np.stack(
                    [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
                ).astype(np.float32)
                colors = np.clip(colors, 0.0, 1.0)
            elif {"red", "green", "blue"}.issubset(names):
                colors = np.stack(
                    [v["red"], v["green"], v["blue"]], axis=-1
                ).astype(np.float32) / 255.0
            else:
                colors = np.ones_like(positions, dtype=np.float32)

            # Opacity
            if "opacity" in names:
                opacity_raw = v["opacity"].astype(np.float32)
                opacities = 1.0 / (1.0 + np.exp(-opacity_raw))
            else:
                opacities = np.ones((positions.shape[0],), dtype=np.float32)

            return positions, quats, scales, opacities, colors

    raise ValueError(
        f"Unsupported PLY layout in {path}. "
        f"Available elements: {[el.name for el in plydata.elements]}"
    )




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
