import math
import os
from typing import Tuple

import torch
import imageio
from tqdm import tqdm

from gsplat import rasterization

from .gaussians import GaussianScene


def make_intrinsics(
    width: int,
    height: int,
    fov_deg: float = 60.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Build a simple pinhole intrinsic matrix K.

    We define f based on vertical field of view.
    """
    fov_rad = math.radians(fov_deg)
    fy = 0.5 * height / math.tan(0.5 * fov_rad)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return K


class GsplatRenderer:
    """Renderer using gsplat.rasterization."""

    def __init__(
        self,
        scene: GaussianScene,
        width: int = 1280,
        height: int = 720,
        fov_deg: float = 60.0,
        device: str = "cuda",
    ):
        self.scene = scene
        self.width = width
        self.height = height
        self.device = device

        self.K = make_intrinsics(width, height, fov_deg=fov_deg, device=device)

    def _prepare_scene_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Return tensors in the format expected by gsplat.rasterization."""
        means = self.scene.means
        quats = self.scene.quats
        scales = self.scene.scales
        opacities = self.scene.opacities
        colors = self.scene.colors
        return means, quats, scales, opacities, colors

    @torch.no_grad()
    def render_video(
        self,
        view_mats: torch.Tensor,  # [F, 4, 4]
        out_path: str,
        fps: int = 24,
        batch_size: int = 8,
        radius_clip: float = 0.0,
    ):
        """Render all frames along the path and write an MP4 video.

        Args:
            view_mats: [F, 4, 4] world-to-camera matrices.
            out_path: output file path (.mp4).
            fps: frames per second.
            batch_size: how many cameras to render at once.
            radius_clip: passed to gsplat to skip tiny far gaussians.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        means, quats, scales, opacities, colors = self._prepare_scene_tensors()
        means = means.to(self.device)
        quats = quats.to(self.device)
        scales = scales.to(self.device)
        opacities = opacities.to(self.device)
        colors = colors.to(self.device)
        
        # Debug info
        print(f"[INFO] Rendering {view_mats.shape[0]} frames")
        print(f"[INFO] Gaussians: {means.shape[0]}")
        print(f"[INFO] Means range: [{means.min().item():.3f}, {means.max().item():.3f}]")
        print(f"[INFO] Scales range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")
        print(f"[INFO] Opacities range: [{opacities.min().item():.3f}, {opacities.max().item():.3f}]")
        print(f"[INFO] Colors range: [{colors.min().item():.3f}, {colors.max().item():.3f}]")
        
        # Check first camera position
        first_cam_pos = torch.inverse(view_mats[0])[:3, 3]
        print(f"[INFO] First camera position: {first_cam_pos}")

        writer = imageio.get_writer(out_path, fps=fps)

        F = view_mats.shape[0]
        K_batched = self.K[None, :, :]  # [1, 3, 3]

        for start in tqdm(range(0, F, batch_size), desc="Rendering"):
            end = min(F, start + batch_size)
            vmats_batch = view_mats[start:end].to(self.device)  # [B, 4, 4]

            # Expand K to match number of cameras in the batch.
            Ks = K_batched.expand(vmats_batch.shape[0], -1, -1)

            render_colors, render_alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=vmats_batch,
                Ks=Ks,
                width=self.width,
                height=self.height,
                radius_clip=radius_clip,
                packed=True,
                render_mode="RGB",
            )
            # render_colors: [B, H, W, 3] in [0,1] (float32)
            frames = render_colors.clamp(0.0, 1.0).cpu().numpy()

            for b in range(frames.shape[0]):
                # Convert to uint8 for video writing
                frame = (frames[b] * 255.0).astype("uint8")
                
                # Debug: check if frame is completely black (only on first frame of first batch)
                if start == 0 and b == 0:
                    frame_mean = frame.mean()
                    frame_max = frame.max()
                    print(f"[INFO] First frame stats: mean={frame_mean:.2f}, max={frame_max}")
                    if frame_mean < 1.0:
                        print(f"[WARNING] First frame is very dark (mean={frame_mean:.2f})")
                
                writer.append_data(frame)

        writer.close()
        print(f"[✓] Finished rendering {F} frames to {out_path}")

