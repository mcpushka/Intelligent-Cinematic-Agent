import numpy as np
import torch
from plyfile import PlyData


class GaussianScene:
    def __init__(self, means, quats, scales, colors, opacities):
        self.means = means
        self.quats = quats
        self.scales = scales
        self.colors = colors
        self.opacities = opacities

        self.center = means.mean(dim=0)
        self.bbox_min = means.min(dim=0).values
        self.bbox_max = means.max(dim=0).values
        self.radius = 0.5 * torch.norm(self.bbox_max - self.bbox_min)
        self.is_indoor = True


def load_gaussian_scene(path: str, device: str = "cuda") -> GaussianScene:
    """Load a Gaussian scene from a .ply file.
    
    Supports both standard Gaussian Splatting format and custom 'chunk' format.
    """
    ply = PlyData.read(path)
    
    # Try standard Gaussian Splatting format first
    if "vertex" in ply:
        vertex_data = ply["vertex"].data
        
        # Extract positions
        positions = torch.stack([
            torch.tensor(vertex_data["x"], dtype=torch.float32),
            torch.tensor(vertex_data["y"], dtype=torch.float32),
            torch.tensor(vertex_data["z"], dtype=torch.float32),
        ], dim=1).to(device)
        
        # Extract scales (log scale, need to exp)
        if "scale_0" in vertex_data.dtype.names:
            scales = torch.stack([
                torch.tensor(vertex_data["scale_0"], dtype=torch.float32),
                torch.tensor(vertex_data["scale_1"], dtype=torch.float32),
                torch.tensor(vertex_data["scale_2"], dtype=torch.float32),
            ], dim=1).to(device)
            scales = torch.exp(scales)  # Convert from log scale
        else:
            # Fallback: use small default scale
            scales = torch.ones((len(positions), 3), dtype=torch.float32, device=device) * 0.01
        
        # Extract rotations (quaternions from rot_0, rot_1, rot_2, rot_3)
        if "rot_0" in vertex_data.dtype.names:
            quats = torch.stack([
                torch.tensor(vertex_data["rot_0"], dtype=torch.float32),
                torch.tensor(vertex_data["rot_1"], dtype=torch.float32),
                torch.tensor(vertex_data["rot_2"], dtype=torch.float32),
                torch.tensor(vertex_data["rot_3"], dtype=torch.float32),
            ], dim=1).to(device)
            # Normalize quaternions
            quat_norm = torch.norm(quats, dim=1, keepdim=True) + 1e-8
            quats = quats / quat_norm
        else:
            # Identity quaternion (x, y, z, w) format
            quats = torch.zeros((len(positions), 4), dtype=torch.float32, device=device)
            quats[:, 3] = 1.0  # w = 1
        
        # Extract opacities (sigmoid activation)
        if "opacity" in vertex_data.dtype.names:
            opacities = torch.tensor(vertex_data["opacity"], dtype=torch.float32).to(device)
            opacities = torch.sigmoid(opacities)  # Convert from logit
        else:
            opacities = torch.ones(len(positions), dtype=torch.float32, device=device) * 0.9
        
        # Extract colors (spherical harmonics or RGB)
        if "f_dc_0" in vertex_data.dtype.names:
            # Spherical harmonics - use DC component (first 3 values)
            colors = torch.stack([
                torch.tensor(vertex_data["f_dc_0"], dtype=torch.float32),
                torch.tensor(vertex_data["f_dc_1"], dtype=torch.float32),
                torch.tensor(vertex_data["f_dc_2"], dtype=torch.float32),
            ], dim=1).to(device)
            # Apply sigmoid and scale to [0, 1]
            colors = torch.sigmoid(colors)
        elif "red" in vertex_data.dtype.names:
            # Direct RGB values
            colors = torch.stack([
                torch.tensor(vertex_data["red"], dtype=torch.float32),
                torch.tensor(vertex_data["green"], dtype=torch.float32),
                torch.tensor(vertex_data["blue"], dtype=torch.float32),
            ], dim=1).to(device)
            # Normalize if in [0, 255] range
            if colors.max() > 1.5:
                colors = colors / 255.0
            colors = colors.clamp(0.0, 1.0)
        else:
            # Default white color
            colors = torch.ones((len(positions), 3), dtype=torch.float32, device=device)
    
    # Try custom 'chunk' format
    elif "chunk" in ply:
        chunk = ply["chunk"].data
        
        positions = []
        scales = []
        colors = []
        
        for c in chunk:
            center = [
                0.5 * (c["min_x"] + c["max_x"]),
                0.5 * (c["min_y"] + c["max_y"]),
                0.5 * (c["min_z"] + c["max_z"]),
            ]
            scale = [
                0.5 * (c["min_scale_x"] + c["max_scale_x"]),
                0.5 * (c["min_scale_y"] + c["max_scale_y"]),
                0.5 * (c["min_scale_z"] + c["max_scale_z"]),
            ]
            color = [
                0.5 * (c["min_r"] + c["max_r"]),
                0.5 * (c["min_g"] + c["max_g"]),
                0.5 * (c["min_b"] + c["max_b"]),
            ]
            
            positions.append(center)
            scales.append(scale)
            colors.append(color)
        
        positions = torch.tensor(positions, dtype=torch.float32, device=device)
        scales = torch.tensor(scales, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)
        opacities = torch.ones(len(positions), dtype=torch.float32, device=device) * 0.9
        
        # Identity quaternion
        quats = torch.zeros((len(positions), 4), dtype=torch.float32, device=device)
        quats[:, 3] = 1.0
    else:
        raise ValueError(f"Unsupported PLY format. Available elements: {list(ply.elements)}")
    
    # --- Post-process attributes for stable rendering ---
    with torch.no_grad():
        # Colors: ensure [0, 1] range
        colors = colors.clamp(0.0, 1.0)
        
        # Scales: ensure reasonable size (not too small, not too large)
        scales = torch.clamp(scales, min=1e-4, max=1.0)
        
        # Opacities: ensure visibility
        opacities = torch.clamp(opacities, min=0.1, max=1.0)
        
        # Quaternions: ensure normalized (x, y, z, w) format for gsplat
        quat_norm = torch.norm(quats, dim=1, keepdim=True) + 1e-8
        quats = quats / quat_norm
    
    print(f"[INFO] Loaded {len(positions)} Gaussians")
    print(f"[INFO] Position range: [{positions.min().item():.3f}, {positions.max().item():.3f}]")
    print(f"[INFO] Scale range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")
    print(f"[INFO] Opacity range: [{opacities.min().item():.3f}, {opacities.max().item():.3f}]")
    print(f"[INFO] Color range: [{colors.min().item():.3f}, {colors.max().item():.3f}]")
    
    return GaussianScene(positions, quats, scales, colors, opacities)
