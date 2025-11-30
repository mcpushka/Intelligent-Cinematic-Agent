from dataclasses import dataclass
from typing import Dict, Any

import torch

from .gaussians import GaussianScene
from .path_planner import build_camera_path


@dataclass
class ExplorationResult:
    """Stores the result of scene exploration planning."""
    view_mats: torch.Tensor  # [F, 4, 4]
    metadata: Dict[str, Any]


class SceneExplorer:
    """Simple scene exploration agent.

    Responsibilities:
      - Initialize a camera path automatically (no hardcoded start/end).
      - Ensure the path covers the whole scene (orbit).
      - Provide metadata useful for reporting/visualization.
    """

    def __init__(self, scene: GaussianScene):
        self.scene = scene

    def plan_panorama_tour(
        self,
        duration_sec: float,
        fps: int = 24,
        n_keyframes: int = 120,
    ) -> ExplorationResult:
        """Plan a generic 360° panorama tour."""
        view_mats = build_camera_path(
            self.scene, duration_sec=duration_sec, fps=fps, n_keyframes=n_keyframes
        )

        # Very crude "coverage" metric: number of viewpoints
        num_frames = view_mats.shape[0]

        metadata: Dict[str, Any] = {
            "num_frames": num_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "is_indoor": self.scene.is_indoor,
        }
        return ExplorationResult(view_mats=view_mats, metadata=metadata)
