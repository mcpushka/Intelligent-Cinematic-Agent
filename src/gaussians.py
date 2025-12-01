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

        self.center = self.means.mean(dim=0)
        self.bbox_min = self.means.min(dim=0).values
        self.bbox_max = self.means.max(dim=0).values
        self.radius = 0.5 * torch.norm(self.bbox_max - self.bbox_min)
        self.is_indoor = True  # можно сделать зависимым от имени файла

def load_gaussian_scene(path: str, device: str = 'cuda') -> GaussianScene:
    plydata = PlyData.read(path)
    v = plydata['vertex'].data

    def to_tensor(name):
        return torch.from_numpy(np.asarray([row[name] for row in v], dtype=np.float32)).to(device)

    means = torch.stack([to_tensor("x"), to_tensor("y"), to_tensor("z")], dim=-1)
    scales = torch.stack([to_tensor("scale_0"), to_tensor("scale_1"), to_tensor("scale_2")], dim=-1)
    quats = torch.stack([to_tensor("rot_0"), to_tensor("rot_1"), to_tensor("rot_2"), to_tensor("rot_3")], dim=-1)
    colors = torch.stack([to_tensor("f_dc_0"), to_tensor("f_dc_1"), to_tensor("f_dc_2")], dim=-1)
    opacities = to_tensor("opacity")

    return GaussianScene(means, quats, scales, colors, opacities)
