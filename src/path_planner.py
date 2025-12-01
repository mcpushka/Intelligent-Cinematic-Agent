import math
from typing import List, Optional, Tuple

import torch
from .gaussians import GaussianScene


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Construct a world-to-camera view matrix using a simple look-at convention."""
    eye = eye.float()
    target = target.float()
    up = up.float()

    z = eye - target
    z_norm = torch.norm(z) + 1e-8
    z = z / z_norm

    x = torch.cross(up, z, dim=0)
    x_norm = torch.norm(x) + 1e-8
    x = x / x_norm

    y = torch.cross(z, x, dim=0)

    R = torch.stack([x, y, z], dim=0)
    t = -R @ eye
    view = torch.eye(4, device=eye.device, dtype=torch.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def _build_occupancy_grid(
    scene: GaussianScene,
    grid_resolution: int = 32,
    influence_radius: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Build a coarse 3D occupancy grid from point positions.

    Returns:
        occ_grid: [R, R, R] bool tensor where True means occupied.
        grid_min: [3] tensor, world-space minimum corner.
        cell_size: scalar cell size.
    """
    device = scene.means.device
    bbox_min = scene.bbox_min
    bbox_max = scene.bbox_max

    extents = bbox_max - bbox_min
    # Small padding to keep us slightly away from the walls.
    padding = 0.05 * extents
    grid_min = bbox_min - padding
    grid_max = bbox_max + padding

    cube_extents = grid_max - grid_min
    max_extent = torch.max(cube_extents)
    cell_size = max_extent / grid_resolution

    occ_grid = torch.zeros(
        (grid_resolution, grid_resolution, grid_resolution),
        dtype=torch.bool,
        device=device,
    )

    # Vectorized marking of occupied cells (one cell per Gaussian mean).
    # This avoids a very slow Python loop over all points and keeps path
    # planning fast even for large scenes.
    local = (scene.means - grid_min) / cell_size  # [N, 3] in grid coords
    idx = local.long()
    # Clamp indices to valid range.
    idx = torch.clamp(idx, 0, grid_resolution - 1)
    if idx.numel() > 0:
        x, y, z = idx.unbind(dim=1)
        occ_grid[x, y, z] = True

    return occ_grid, grid_min, float(cell_size)


def _grid_to_world(
    idx: torch.Tensor, grid_min: torch.Tensor, cell_size: float
) -> torch.Tensor:
    """Convert integer grid indices [3] to world space center position [3]."""
    return grid_min + (idx.float() + 0.5) * cell_size


def _astar_path(
    occ: torch.Tensor,
    start_idx: torch.Tensor,
    goal_idx: torch.Tensor,
) -> List[torch.Tensor]:
    """Very small A* implementation on a 3D grid with 6-connectivity."""
    import heapq

    R = occ.shape[0]

    def neighbors(idx):
        x, y, z = idx
        for dx, dy, dz in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < R and 0 <= ny < R and 0 <= nz < R:
                if not occ[nx, ny, nz]:
                    yield (nx, ny, nz)

    def heuristic(a, b):
        # Manhattan distance works well on a grid.
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    start = tuple(int(v) for v in start_idx.tolist())
    goal = tuple(int(v) for v in goal_idx.tolist())

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [torch.tensor(p, device=occ.device, dtype=torch.long) for p in path]

        for nb in neighbors(current):
            tentative_g = g + 1
            if tentative_g < g_score.get(nb, 1e9):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_set, (f, tentative_g, nb))

    # Fallback: no path found, just return straight line between start and goal.
    return [
        torch.round(
            start_idx + (goal_idx - start_idx) * t
        ).long()
        for t in torch.linspace(0.0, 1.0, steps=16, device=occ.device)
    ]


def build_camera_path(
    scene: GaussianScene,
    duration_sec: float,
    fps: int = 24,
    n_keyframes: int = 120,
    cam_y: Optional[float] = None,
    seed: int = 42,
) -> torch.Tensor:
    """Plan a camera path *inside* the scene with simple obstacle avoidance.

    - We build a coarse occupancy grid based on Gaussian means.
    - We run a grid-based A* from a free start cell to a free goal cell.
    - We convert the resulting grid path to world coordinates.
    - We interpolate along this path to get a smooth trajectory of `duration_sec * fps` frames.
    - The camera looks forward along its local path direction (no need to be realistic).
    """
    device = scene.means.device
    torch.manual_seed(seed)

    # 1) Build occupancy grid.
    occ, grid_min, cell_size = _build_occupancy_grid(scene, grid_resolution=32)
    R = occ.shape[0]

    # 2) Pick random free start/goal cells.
    free_indices = torch.nonzero(~occ, as_tuple=False)
    if free_indices.shape[0] < 2:
        # Extremely dense scene – fall back to orbit around center.
        from math import pi

        center = scene.center
        extents = scene.bbox_max - scene.bbox_min
        radius = float(max(extents[0], extents[2]) * 0.6 + 1e-2)
        height = (
            cam_y
            if cam_y is not None
            else float(center[1] + 0.2 * extents[1])
        )
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        frames = int(duration_sec * fps)
        views = []
        for i in range(frames):
            theta = 2.0 * pi * (i / frames)
            eye = torch.tensor(
                [
                    center[0] + radius * math.cos(theta),
                    height,
                    center[2] + radius * math.sin(theta),
                ],
                device=device,
            )
            views.append(look_at(eye, center, up))
        return torch.stack(views)

    rng = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(free_indices.shape[0], generator=rng, device=device)
    start_idx = free_indices[perm[0]]
    goal_idx = free_indices[perm[-1]]

    # 3) A* path on the grid.
    grid_path_idx = _astar_path(occ, start_idx, goal_idx)

    # 4) Convert grid path to world positions.
    world_points = torch.stack(
        [_grid_to_world(idx, grid_min, cell_size) for idx in grid_path_idx],
        dim=0,
    )  # [K, 3]

    K = world_points.shape[0]
    total_frames = int(duration_sec * fps)
    if K < 2:
        world_points = world_points.repeat(2, 1)
        K = 2

    # 5) Interpolate along the polyline.
    # Map each frame to a position along the path index axis [0, K-1].
    positions = []
    for u in torch.linspace(0, K - 1, total_frames, device=device):
        i0 = torch.clamp(torch.floor(u).long(), 0, K - 2)
        i1 = i0 + 1
        t = (u - i0.float()).unsqueeze(-1)
        p = (1.0 - t) * world_points[i0] + t * world_points[i1]
        positions.append(p)
    positions = torch.stack(positions, dim=0)  # [F, 3]

    # 6) Build view matrices.
    #    Чтобы гарантировать, что камера всегда "видит" сцену и кадр не будет
    #    чёрным, смотрим в центр сцены, а не строго вперёд по траектории.
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    center = scene.center
    view_mats: List[torch.Tensor] = []
    for i in range(total_frames):
        eye = positions[i]
        target = center
        view_mats.append(look_at(eye, target, up))

    return torch.stack(view_mats, dim=0)
