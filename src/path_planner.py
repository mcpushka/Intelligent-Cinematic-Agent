import math
from typing import List, Optional

import torch
from .gaussians import GaussianScene


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Construct a right-handed look-at view matrix (world to camera)."""
    z_axis = eye - target
    z_axis = z_axis / (torch.norm(z_axis) + 1e-8)

    x_axis = torch.cross(up, z_axis, dim=0)
    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)

    y_axis = torch.cross(z_axis, x_axis, dim=0)

    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    t = -R @ eye

    view = torch.eye(4, device=eye.device, dtype=eye.dtype)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def generate_orbit_keyframes(
    scene: GaussianScene,
    n_keyframes: int = 120,
    orbit_height_factor: float = 0.2,
    orbit_radius_scale: float = 1.2,
    cam_y: Optional[float] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    """Generate view matrices for a camera orbiting around the scene center."""
    device = scene.means.device
    center = scene.center
    extents = scene.bbox_max - scene.bbox_min

    radius = orbit_radius_scale * max(extents[0], extents[2])
    height = cam_y if cam_y is not None else float(center[1] + orbit_height_factor * extents[1])

    rng = torch.Generator().manual_seed(seed)
    offset_theta = torch.rand(1, generator=rng).to(device).item() * 2.0 * math.pi

    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    views: List[torch.Tensor] = []
    for i in range(n_keyframes):
        theta = offset_theta + 2.0 * math.pi * (i / n_keyframes)
        eye = torch.tensor([
            center[0] + radius * math.cos(theta),
            height,
            center[2] + radius * math.sin(theta)
        ], device=device)
        views.append(look_at(eye, center, up))

    return views


def resample_catmull_rom(
    points: torch.Tensor,
    num_samples: int,
    loop: bool = True
) -> torch.Tensor:
    """Perform Catmull-Rom interpolation to resample points into a smooth path."""
    device = points.device
    K = points.shape[0]

    if loop:
        extended = torch.cat([points[-1:], points, points[:2]], dim=0)
    else:
        extended = torch.cat([points[:1], points, points[-1:], points[-1:]], dim=0)

    def catmull_rom(p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    samples = []
    for u in torch.linspace(0, K, num_samples, device=device):
        i = int(torch.floor(u).item())
        t = u - i
        p0 = extended[i + 0]
        p1 = extended[i + 1]
        p2 = extended[i + 2]
        p3 = extended[i + 3]
        samples.append(catmull_rom(p0, p1, p2, p3, t))

    return torch.stack(samples)


def build_camera_path(
    scene: GaussianScene,
    duration_sec: float,
    fps: int = 24,
    n_keyframes: int = 120,
    cam_y: Optional[float] = None,
    seed: int = 42
) -> torch.Tensor:
    """High-level function to generate a smooth camera path for rendering."""
    keyframes = generate_orbit_keyframes(
        scene, n_keyframes=n_keyframes, cam_y=cam_y, seed=seed
    )

    cam_positions = torch.stack([
        torch.inverse(view_mat)[:3, 3] for view_mat in keyframes
    ])  # [K, 3]

    positions = resample_catmull_rom(cam_positions, int(duration_sec * fps))

    up = torch.tensor([0.0, 1.0, 0.0], device=scene.means.device)

    view_mats = torch.stack([
        look_at(pos, scene.center, up) for pos in positions
    ])  # [F, 4, 4]

    return view_mats
