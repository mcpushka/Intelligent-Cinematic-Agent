import numpy as np
import torch
from plyfile import PlyData


def _unpack_supersplat_data(vertex_data, chunk_data, device, max_gaussians=200000):
    """Unpack SuperSplat packed format using chunk metadata for ranges.
    
    Args:
        max_gaussians: Maximum number of gaussians to load (downsample if more).
                       Set to None to load all (may cause OOM on smaller GPUs).
    """
    n_gaussians = len(vertex_data)
    n_chunks = len(chunk_data)
    
    # Downsample if too many gaussians (to avoid GPU OOM)
    if max_gaussians is not None and n_gaussians > max_gaussians:
        step = n_gaussians // max_gaussians
        indices = np.arange(0, n_gaussians, step)[:max_gaussians]
        print(f"[INFO] Downsampling from {n_gaussians} to {len(indices)} gaussians (step={step})")
    else:
        indices = np.arange(n_gaussians)
    
    # Get packed data as uint32 (with downsampling)
    packed_pos = np.array(vertex_data["packed_position"], dtype=np.uint32)[indices]
    packed_rot = np.array(vertex_data["packed_rotation"], dtype=np.uint32)[indices]
    packed_scale = np.array(vertex_data["packed_scale"], dtype=np.uint32)[indices]
    packed_color = np.array(vertex_data["packed_color"], dtype=np.uint32)[indices]
    
    n_gaussians = len(indices)  # Update count after downsampling
    
    # Determine which chunk each gaussian belongs to (based on original indices)
    original_n_gaussians = len(vertex_data)
    gaussians_per_chunk = original_n_gaussians // n_chunks
    all_chunk_indices = np.repeat(np.arange(n_chunks), gaussians_per_chunk)
    if len(all_chunk_indices) < original_n_gaussians:
        all_chunk_indices = np.concatenate([all_chunk_indices, np.full(original_n_gaussians - len(all_chunk_indices), n_chunks - 1)])
    chunk_indices = all_chunk_indices[indices]  # Apply same downsampling
    
    # Build chunk bounds arrays
    chunk_min_x = np.array([c["min_x"] for c in chunk_data], dtype=np.float32)
    chunk_max_x = np.array([c["max_x"] for c in chunk_data], dtype=np.float32)
    chunk_min_y = np.array([c["min_y"] for c in chunk_data], dtype=np.float32)
    chunk_max_y = np.array([c["max_y"] for c in chunk_data], dtype=np.float32)
    chunk_min_z = np.array([c["min_z"] for c in chunk_data], dtype=np.float32)
    chunk_max_z = np.array([c["max_z"] for c in chunk_data], dtype=np.float32)
    
    chunk_min_scale_x = np.array([c["min_scale_x"] for c in chunk_data], dtype=np.float32)
    chunk_max_scale_x = np.array([c["max_scale_x"] for c in chunk_data], dtype=np.float32)
    chunk_min_scale_y = np.array([c["min_scale_y"] for c in chunk_data], dtype=np.float32)
    chunk_max_scale_y = np.array([c["max_scale_y"] for c in chunk_data], dtype=np.float32)
    chunk_min_scale_z = np.array([c["min_scale_z"] for c in chunk_data], dtype=np.float32)
    chunk_max_scale_z = np.array([c["max_scale_z"] for c in chunk_data], dtype=np.float32)
    
    chunk_min_r = np.array([c["min_r"] for c in chunk_data], dtype=np.float32)
    chunk_max_r = np.array([c["max_r"] for c in chunk_data], dtype=np.float32)
    chunk_min_g = np.array([c["min_g"] for c in chunk_data], dtype=np.float32)
    chunk_max_g = np.array([c["max_g"] for c in chunk_data], dtype=np.float32)
    chunk_min_b = np.array([c["min_b"] for c in chunk_data], dtype=np.float32)
    chunk_max_b = np.array([c["max_b"] for c in chunk_data], dtype=np.float32)
    
    # Unpack position (10-10-10-2 format: x, y, z packed into 32 bits)
    pos_x_norm = ((packed_pos >> 0) & 0x3FF).astype(np.float32) / 1023.0
    pos_y_norm = ((packed_pos >> 10) & 0x3FF).astype(np.float32) / 1023.0
    pos_z_norm = ((packed_pos >> 20) & 0x3FF).astype(np.float32) / 1023.0
    
    # Denormalize using chunk bounds
    ci = chunk_indices
    pos_x = chunk_min_x[ci] + pos_x_norm * (chunk_max_x[ci] - chunk_min_x[ci])
    pos_y = chunk_min_y[ci] + pos_y_norm * (chunk_max_y[ci] - chunk_min_y[ci])
    pos_z = chunk_min_z[ci] + pos_z_norm * (chunk_max_z[ci] - chunk_min_z[ci])
    
    positions = torch.tensor(np.stack([pos_x, pos_y, pos_z], axis=1), dtype=torch.float32, device=device)
    
    # Unpack rotation (8-8-8-8 format: quaternion components)
    rot_x = ((packed_rot >> 0) & 0xFF).astype(np.float32) / 127.5 - 1.0
    rot_y = ((packed_rot >> 8) & 0xFF).astype(np.float32) / 127.5 - 1.0
    rot_z = ((packed_rot >> 16) & 0xFF).astype(np.float32) / 127.5 - 1.0
    rot_w = ((packed_rot >> 24) & 0xFF).astype(np.float32) / 127.5 - 1.0
    
    quats = torch.tensor(np.stack([rot_x, rot_y, rot_z, rot_w], axis=1), dtype=torch.float32, device=device)
    quat_norm = torch.norm(quats, dim=1, keepdim=True) + 1e-8
    quats = quats / quat_norm
    
    # Unpack scale (10-10-10-2 format)
    scale_x_norm = ((packed_scale >> 0) & 0x3FF).astype(np.float32) / 1023.0
    scale_y_norm = ((packed_scale >> 10) & 0x3FF).astype(np.float32) / 1023.0
    scale_z_norm = ((packed_scale >> 20) & 0x3FF).astype(np.float32) / 1023.0
    
    # Denormalize scales using chunk bounds (these are in log space)
    scale_x = chunk_min_scale_x[ci] + scale_x_norm * (chunk_max_scale_x[ci] - chunk_min_scale_x[ci])
    scale_y = chunk_min_scale_y[ci] + scale_y_norm * (chunk_max_scale_y[ci] - chunk_min_scale_y[ci])
    scale_z = chunk_min_scale_z[ci] + scale_z_norm * (chunk_max_scale_z[ci] - chunk_min_scale_z[ci])
    
    # Convert from log scale
    scales = torch.tensor(np.stack([scale_x, scale_y, scale_z], axis=1), dtype=torch.float32, device=device)
    scales = torch.exp(scales)
    
    # Unpack color (8-8-8-8 format: r, g, b, opacity)
    color_r = ((packed_color >> 0) & 0xFF).astype(np.float32) / 255.0
    color_g = ((packed_color >> 8) & 0xFF).astype(np.float32) / 255.0
    color_b = ((packed_color >> 16) & 0xFF).astype(np.float32) / 255.0
    opacity = ((packed_color >> 24) & 0xFF).astype(np.float32) / 255.0
    
    # Denormalize colors using chunk bounds
    color_r = chunk_min_r[ci] + color_r * (chunk_max_r[ci] - chunk_min_r[ci])
    color_g = chunk_min_g[ci] + color_g * (chunk_max_g[ci] - chunk_min_g[ci])
    color_b = chunk_min_b[ci] + color_b * (chunk_max_b[ci] - chunk_min_b[ci])
    
    colors = torch.tensor(np.stack([color_r, color_g, color_b], axis=1), dtype=torch.float32, device=device)
    opacities = torch.tensor(opacity, dtype=torch.float32, device=device)
    
    print(f"[INFO] Unpacked {n_gaussians} gaussians from {n_chunks} chunks")
    
    return positions, scales, colors, quats, opacities


def _load_chunk_format(chunk_data, device):
    """Load data from chunk-only format (uses chunk centers as gaussians)."""
    positions = []
    scales = []
    colors = []
    
    for c in chunk_data:
        center = [
            0.5 * (c["min_x"] + c["max_x"]),
            0.5 * (c["min_y"] + c["max_y"]),
            0.5 * (c["min_z"] + c["max_z"]),
        ]
        # Use chunk extent as scale (larger for visibility)
        scale = [
            0.25 * abs(c["max_x"] - c["min_x"]),
            0.25 * abs(c["max_y"] - c["min_y"]),
            0.25 * abs(c["max_z"] - c["min_z"]),
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
    opacities = torch.ones(len(positions), dtype=torch.float32, device=device) * 0.95
    
    # Identity quaternion
    quats = torch.zeros((len(positions), 4), dtype=torch.float32, device=device)
    quats[:, 3] = 1.0
    
    return positions, scales, colors, quats, opacities


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
    
    # Debug: print available elements and their fields
    print(f"[INFO] PLY elements: {list(ply.elements)}")
    
    # Check for packed format with chunk metadata (SuperSplat format)
    # NOTE: Using chunk-only format for stability and speed on limited GPU memory
    if "chunk" in ply:
        chunk_data = ply["chunk"].data
        print(f"[INFO] Using 'chunk' format with {len(chunk_data)} chunks (faster, more stable)")
        positions, scales, colors, quats, opacities = _load_chunk_format(chunk_data, device)
    
    # Try 'chunk' format only
    elif "chunk" in ply:
        chunk_data = ply["chunk"].data
        print(f"[INFO] Using 'chunk' format with {len(chunk_data)} chunks")
        positions, scales, colors, quats, opacities = _load_chunk_format(chunk_data, device)
    
    # Try standard Gaussian Splatting format (unpacked vertex data)
    elif "vertex" in ply:
        vertex_data = ply["vertex"].data
        available_fields = vertex_data.dtype.names
        print(f"[INFO] Available vertex fields: {available_fields}")
        
        # Extract positions - try different field name variations
        if "x" in available_fields and "y" in available_fields and "z" in available_fields:
            positions = torch.stack([
                torch.tensor(vertex_data["x"], dtype=torch.float32),
                torch.tensor(vertex_data["y"], dtype=torch.float32),
                torch.tensor(vertex_data["z"], dtype=torch.float32),
            ], dim=1).to(device)
        elif "X" in available_fields and "Y" in available_fields and "Z" in available_fields:
            positions = torch.stack([
                torch.tensor(vertex_data["X"], dtype=torch.float32),
                torch.tensor(vertex_data["Y"], dtype=torch.float32),
                torch.tensor(vertex_data["Z"], dtype=torch.float32),
            ], dim=1).to(device)
        else:
            raise ValueError(f"Could not find position fields (x,y,z or X,Y,Z) in PLY. Available: {available_fields}")
        
        # Extract scales (log scale, need to exp)
        if "scale_0" in available_fields:
            scales = torch.stack([
                torch.tensor(vertex_data["scale_0"], dtype=torch.float32),
                torch.tensor(vertex_data["scale_1"], dtype=torch.float32),
                torch.tensor(vertex_data["scale_2"], dtype=torch.float32),
            ], dim=1).to(device)
            scales = torch.exp(scales)  # Convert from log scale
        else:
            # Fallback: use small default scale
            print(f"[WARNING] No scale fields found, using default scale")
            scales = torch.ones((len(positions), 3), dtype=torch.float32, device=device) * 0.01
        
        # Extract rotations (quaternions from rot_0, rot_1, rot_2, rot_3)
        if "rot_0" in available_fields:
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
            print(f"[WARNING] No rotation fields found, using identity quaternions")
            quats = torch.zeros((len(positions), 4), dtype=torch.float32, device=device)
            quats[:, 3] = 1.0  # w = 1
        
        # Extract opacities (sigmoid activation)
        if "opacity" in available_fields:
            opacities = torch.tensor(vertex_data["opacity"], dtype=torch.float32).to(device)
            opacities = torch.sigmoid(opacities)  # Convert from logit
        else:
            print(f"[WARNING] No opacity field found, using default opacity 0.9")
            opacities = torch.ones(len(positions), dtype=torch.float32, device=device) * 0.9
        
        # Extract colors (spherical harmonics or RGB)
        if "f_dc_0" in available_fields:
            # Spherical harmonics - use DC component (first 3 values)
            colors = torch.stack([
                torch.tensor(vertex_data["f_dc_0"], dtype=torch.float32),
                torch.tensor(vertex_data["f_dc_1"], dtype=torch.float32),
                torch.tensor(vertex_data["f_dc_2"], dtype=torch.float32),
            ], dim=1).to(device)
            # Apply sigmoid and scale to [0, 1]
            colors = torch.sigmoid(colors)
        elif "red" in available_fields:
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
            print(f"[WARNING] No color fields found, using default white color")
            colors = torch.ones((len(positions), 3), dtype=torch.float32, device=device)
    
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
