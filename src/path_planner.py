import math
from typing import List, Tuple

import torch

from .gaussians import GaussianScene


def look_at(eye: torch.Tensor,
            target: torch.Tensor,
            up: torch.Tensor) -> torch.Tensor:
    """Build a 4x4 world-to-camera matrix (view matrix) from eye, target, up.

    The result is standard pinhole view: R|t where camera looks along -Z.
    """
    # Ensure float32 tensors on same device
    eye = eye.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    up = up.to(dtype=torch.float32)

    z_axis = eye - target
    z_axis = z_axis / (torch.norm(z_axis) + 1e-8)

    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)

    y_axis = torch.cross(z_axis, x_axis)

    R = torch.stack([x_axis, y_axis, z_axis], dim=0)  # [3, 3]
    t = -R @ eye.view(3, 1)                           # [3, 1]

    view = torch.eye(4, device=eye.device, dtype=torch.float32)
    view[:3, :3] = R
    view[:3, 3] = t.squeeze(-1)
    return view


def generate_orbit_keyframes(
    scene: GaussianScene,
    n_keyframes: int = 120,
    orbit_height_factor: float = 0.2,
    orbit_radius_scale: float = 1.2,
) -> List[torch.Tensor]:
    """Generate keyframe view matrices that orbit around scene center.

    This is our basic "exploration strategy":
      - camera flies around the whole scene,
      - stays outside the bounding volume (obstacle avoidance),
      - keeps the scene center in view (cinematic framing).
    """
    device = scene.means.device
    center = scene.center
    bbox_min = scene.bbox_min
    bbox_max = scene.bbox_max
    extents = bbox_max - bbox_min

    # Horizontal radius: a bit larger than max horizontal distance from center
    horizontal_extent = float(max(extents[0], extents[2]))
    radius = orbit_radius_scale * horizontal_extent

    # Height: slightly above the "middle", to avoid being inside the geometry
    height = float(center[1] + orbit_height_factor * extents[1])

    up = torch.tensor([0.0, 1.0, 0.0], device=device)

    keyframes: List[torch.Tensor] = []
    for i in range(n_keyframes):
        theta = 2.0 * math.pi * (i / n_keyframes)
        eye = torch.tensor(
            [
                center[0] + radius * math.cos(theta),
                height,
                center[2] + radius * math.sin(theta),
            ],
            device=device,
        )
        view = look_at(eye, center, up)
        keyframes.append(view)
    return keyframes


def resample_catmull_rom(
    points: torch.Tensor,
    num_samples: int,
    loop: bool = True,
) -> torch.Tensor:
    """Catmull-Rom interpolation for camera positions.

    Args:
        points: [K, 3] keyframe positions.
        num_samples: number of positions to sample along the loop.
        loop: if True, treat points as closed loop.

    Returns:
        [num_samples, 3] interpolated positions.
    """
    device = points.device
    K = points.shape[0]

    if loop:
        # Wrap indices for closed curve
        extended = torch.cat(
            [points[-1:].clone(), points, points[:2].clone()], dim=0
        )  # [K+3, 3]
    else:
        extended = torch.cat(
            [points[:1].clone(), points, points[-1:].clone(), points[-1:].clone()],
            dim=0,
        )

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
    for i in range(num_samples):
        u = (i / num_samples) * K
        idx = int(math.floor(u))
        t = torch.tensor(u - idx, device=device)
        idx0 = idx
        p0 = extended[idx0 + 0]
        p1 = extended[idx0 + 1]
        p2 = extended[idx0 + 2]
        p3 = extended[idx0 + 3]
        pos = catmull_rom(p0, p1, p2, p3, t)
        samples.append(pos)
    return torch.stack(samples, dim=0)


def build_camera_path(
    scene: GaussianScene,
    duration_sec: float,
    fps: int = 24,
    n_keyframes: int = 120,
) -> torch.Tensor:
    """High-level helper: build a smooth camera path for Video 1.

    Steps:
      1. Create orbit keyframes (exploration + coverage).
      2. Resample with Catmull-Rom to get smooth motion.
      3. Create view matrices along the path (world-to-camera).
    """
    device = scene.means.device
    keyframes = generate_orbit_keyframes(scene, n_keyframes=n_keyframes)

    # Extract keyframe positions from view matrices (invert to get camera pose)
    key_positions = []
    for view in keyframes:
        # Inverse of view matrix gives camera-to-world transform.
        cam_to_world = torch.inverse(view)
        cam_pos = cam_to_world[:3, 3]
        key_positions.append(cam_pos)
    key_positions = torch.stack(key_positions, dim=0)  # [K, 3]

    num_frames = int(duration_sec * fps)

    # Smooth path via Catmull-Rom
    positions = resample_catmull_rom(key_positions, num_frames, loop=True)  # [F, 3]

    # Build view matrices along path (always look at scene center)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    center = scene.center

    view_mats = []
    for i in range(num_frames):
        eye = positions[i]
        view = look_at(eye, center, up)
        view_mats.append(view)

    return torch.stack(view_mats, dim=0)  # [F, 4, 4]
