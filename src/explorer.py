from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

from .gaussians import GaussianScene
from .path_planner import build_camera_path


@dataclass
class ExplorationResult:
    """Stores the result of scene exploration planning."""
    view_mats: torch.Tensor  # [F, 4, 4]
    metadata: Dict[str, Any]


class SceneExplorer:
    """Simple scene exploration agent with optional normalization and camera config."""

    def __init__(
        self,
        scene: GaussianScene,
        normalize: bool = False,
        cam_y: Optional[float] = None,
        seed: int = 42
    ):
        # Optional normalization (in-place)
        if normalize:
            center = scene.center
            radius = scene.radius

            scene.means = (scene.means - center) / radius
            scene.bbox_min = (scene.bbox_min - center) / radius
            scene.bbox_max = (scene.bbox_max - center) / radius

            scene.center = (scene.bbox_min + scene.bbox_max) * 0.5
            scene.radius = torch.norm(scene.bbox_max - scene.bbox_min) * 0.5

        self.scene = scene
        self.cam_y = cam_y
        self.seed = seed

    def plan_panorama_tour(
        self,
        duration_sec: float,
        fps: int = 24,
        n_keyframes: int = 120
    ) -> ExplorationResult:
        """Plan a generic 360° panorama tour."""
        view_mats = build_camera_path(
            self.scene,
            duration_sec=duration_sec,
            fps=fps,
            n_keyframes=n_keyframes,
            cam_y=self.cam_y,
            seed=self.seed,
        )

        metadata = {
            "fps": fps,
            "duration_sec": duration_sec,
            "num_frames": view_mats.shape[0],
            "cam_y": self.cam_y,
        }

        return ExplorationResult(view_mats=view_mats, metadata=metadata)
