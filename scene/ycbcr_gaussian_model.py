import numpy as np
import torch
from simple_knn._C import distCUDA2
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.ycbcr_utils import rgb_to_ycbcr


class YCbCrGaussianModel(GaussianModel):
    """Gaussian model whose three SH channels represent Y, Cb and Cr."""

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if isinstance(pcd, BasicPointCloud):
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            initial_rgb = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd._xyz)).float().cuda()
            initial_rgb = torch.tensor(np.asarray(pcd._rgb)).float().cuda()

        fused_ycbcr = rgb_to_ycbcr(initial_rgb, channel_dim=-1)
        fused_color = RGB2SH(fused_ycbcr)
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device="cuda",
        )
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.detach().clone().float()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotations = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rotations[:, 0] = 1
        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
