import numpy as np
import torch
from plyfile import PlyData

class GaussianScene:
    def __init__(self, means, scales, colors, opacities):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.colors = torch.tensor(colors, dtype=torch.float32)
        self.opacities = torch.tensor(opacities, dtype=torch.float32)
        self.center = self.means.mean(dim=0)
        self.bbox_min = self.means.min(dim=0).values
        self.bbox_max = self.means.max(dim=0).values
        self.radius = torch.norm(self.bbox_max - self.bbox_min) * 0.5
        self.is_indoor = True # hardcoded, or infer from scene name


def load_gaussian_scene(path, device='cuda'):
    plydata = PlyData.read(path)

    if 'chunk' in plydata:
        # Gaussian Splatting .ply format
        chunk = plydata['chunk'].data

        # Position: center of bounding box
        positions = np.array([[
            0.5 * (chunk['min x'][0] + chunk['max x'][0]),
            0.5 * (chunk['min y'][0] + chunk['max y'][0]),
            0.5 * (chunk['min z'][0] + chunk['max z'][0]),
        ]], dtype=np.float32)

        # Uniform scale from average scale
        avg_scale = 0.5 * (chunk['min_scale'][0] + chunk['max_scale'][0])
        scales = np.array([[avg_scale, avg_scale, avg_scale]], dtype=np.float32)

        # Average color
        colors = np.array([[
            0.5 * (chunk['min r'][0] + chunk['max r'][0]),
            0.5 * (chunk['min g'][0] + chunk['max g'][0]),
            0.5 * (chunk['min b'][0] + chunk['max b'][0]),
        ]], dtype=np.float32)

        # Fully opaque
        opacities = np.ones((1,), dtype=np.float32)

    else:
        # Standard vertex-based .ply format
        v = plydata['vertex'].data
        positions = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
        colors = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)

        opacities = v['opacity'].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacities))  # convert logit to alpha

    return GaussianScene(positions, scales, colors, opacities)