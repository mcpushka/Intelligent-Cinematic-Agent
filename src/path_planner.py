import math
from typing import List, Optional

import torch
from .gaussians import GaussianScene


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
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
    seed: int = 42,
) -> List[torch.Tensor]:
    device = scene.means.device
    center = scene.center
    bbox_min = scene.bbox_min
    bbox_max = scene.bbox_max
    extents = bbox_max - bbox_min

    horizontal_extent = float(max(extents[0], extents[2]))
    radius = orbit_radius_scale * horizontal_extent

    # height: from cam_y or calculated
    height = cam_y if cam_y is not None else float(center[1] + orbit_height_factor * extents[1])

    # Add random start angle (for variation)
    rng = torch.Generator()
    rng.manual_seed(seed)
    offset_theta = torch.rand(1, generator=rng).to(device).item() * 2.0 * math.pi


    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)


    keyframes = []
    for i in range(n_keyframes):
        theta = offset_theta + 2.0 * math.pi * (i / n_keyframes)
        eye = torch.tensor(
            [
                center[0] + radius * math.cos(theta),
                height,
                center[2] + radius * math.sin(theta),
            ],
            device=device,
            dtype=torch.float32,
        )

        view = look_at(eye, center, up)
        keyframes.append(view)
    return keyframes


def resample_catmull_rom(
    points: torch.Tensor,
    num_samples: int,
    loop: bool = True,
) -> torch.Tensor:
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
            (2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    samples = []
    for i in range(num_samples):
        u = (i / num_samples) * K
        idx = int(math.floor(u))
        t = torch.tensor(u - idx, device=device, dtype=torch.float32)

        p0, p1, p2, p3 = extended[idx:idx + 4]
        samples.append(catmull_rom(p0, p1, p2, p3, t))
    return torch.stack(samples)


def build_camera_path(
    scene: GaussianScene,
    duration_sec: float,
    fps: int = 24,
    n_keyframes: int = 120,
    cam_y: Optional[float] = None,
    seed: int = 42,
) -> torch.Tensor:
    keyframes = generate_orbit_keyframes(
        scene,
        n_keyframes=n_keyframes,
        cam_y=cam_y,
        seed=seed,
    )

    key_positions = []
    for view in keyframes:
        cam_to_world = torch.inverse(view)
        key_positions.append(cam_to_world[:3, 3])
    key_positions = torch.stack(key_positions)

    num_frames = int(duration_sec * fps)
    positions = resample_catmull_rom(key_positions, num_frames, loop=True)

    up = torch.tensor([0.0, 1.0, 0.0], device=scene.means.device, dtype=torch.float32)

    center = scene.center

    return torch.stack([look_at(pos, center, up) for pos in positions])
