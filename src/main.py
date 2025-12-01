import argparse
import os
from typing import List
from typing import Tuple

import torch

from .gaussians import load_gaussian_scene
from .explorer import SceneExplorer
from .renderer import GsplatRenderer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICV Assignment 4"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="List of .ply scene files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store rendered videos.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration of the panorama video in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Video resolution, e.g. 1280x720.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (cuda or cpu). If None, auto-detect.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gsplat",
        help="Rendering backend (only 'gsplat' is supported in this implementation).",
    )
    parser.add_argument(
        "--radius-clip",
        type=float,
        default=0.0,
        help="radius_clip parameter for gsplat.rasterization (0 disables).",
    )
    return parser.parse_args()

def parse_resolution(res_str: str) -> Tuple[int, int]:
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format: {res_str}")
    w = int(parts[0])
    h = int(parts[1])
    return w, h

def auto_configure_camera(scene, scene_cfg):
    center = scene.center.cpu().numpy()
    bbox_min = scene.bbox_min.cpu().numpy()
    bbox_max = scene.bbox_max.cpu().numpy()

    extent_x = bbox_max[0] - bbox_min[0]
    extent_z = bbox_max[2] - bbox_min[2]
    cam_y = bbox_min[1] + 2.0

    if extent_z >= extent_x:
        start = [center[0], cam_y, bbox_min[2] - 0.1 * extent_z]
        end = [center[0], cam_y, bbox_max[2] + 0.1 * extent_z]
    else:
        start = [bbox_min[0] - 0.1 * extent_x, cam_y, center[2]]
        end = [bbox_max[0] + 0.1 * extent_x, cam_y, center[2]]

    scene_cfg["straight_path_waypoints_xyz"] = [start, end]
    scene_cfg["straight_start_cam_y"] = cam_y
    scene_cfg["normalize_scene"] = True
    scene_cfg["random_seed"] = 42

def process_scene(
    scene_path: str,
    output_dir: str,
    duration: float,
    fps: int,
    width: int,
    height: int,
    device: str,
    radius_clip: float,
):
    print(f"\n=== Processing scene: {scene_path} ===")
    scene = load_gaussian_scene(scene_path, device=device)

    # Fix camera logic to prevent black render
    scene_cfg = {}
    auto_configure_camera(scene, scene_cfg)

    explorer = SceneExplorer(
        scene,
        normalize=scene_cfg.get("normalize_scene", False),
        cam_y=scene_cfg.get("straight_start_cam_y", None),
        seed=scene_cfg.get("random_seed", 42)
    )

    exploration_result = explorer.plan_panorama_tour(
        duration_sec=duration,
        fps=fps
    )

    view_mats = exploration_result.view_mats


    scene_name = os.path.splitext(os.path.basename(scene_path))[0]
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)

    video_path = os.path.join(scene_output_dir, "panorama_tour.mp4")

    renderer = GsplatRenderer(
        scene,
        width=width,
        height=height,
        fov_deg=60.0,
        device=device,
    )
    renderer.render_video(
        view_mats=view_mats,
        out_path=video_path,
        fps=fps,
        batch_size=8,
        radius_clip=radius_clip,
    )
    print(f"[OK] Saved panorama tour to: {video_path}")

def main():
    args = parse_args()

    if args.backend.lower() != "gsplat":
        raise ValueError("Only backend 'gsplat' is supported in this implementation.")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    width, height = parse_resolution(args.resolution)

    print(f"Using device: {device}")
    print(f"Target resolution: {width}x{height}, duration={args.duration}s, fps={args.fps}")

    for scene_path in args.scenes:
        process_scene(
            scene_path=scene_path,
            output_dir=args.output_dir,
            duration=args.duration,
            fps=args.fps,
            width=width,
            height=height,
            device=device,
            radius_clip=args.radius_clip,
        )

if __name__ == "__main__":
    main()
