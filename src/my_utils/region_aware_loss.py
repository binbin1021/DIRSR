# src/my_utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GaussianBlur(nn.Module):
    """
    高斯模糊模块：用于软化 Mask 和生成平滑背景参考
    """
    def __init__(self, kernel_size=7, sigma=2.0, channels=3):
        super(GaussianBlur, self).__init__()
        # 生成 1D 高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.
        variance = sigma**2.
        
        gaussian_kernel = (1./(2.*3.14159*variance)) *\
                          torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # 重塑为卷积核权重
        kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.register_buffer('weight', kernel.repeat(channels, 1, 1, 1))
        self.padding = kernel_size // 2

    def forward(self, x):
        c = x.shape[1]
        # 动态适配通道数
        if c > self.weight.shape[0]:
            # 如果输入通道变多了(比如 batch 维度导致)，这里简单防御一下
            weight = self.weight.repeat(c // self.weight.shape[0] + 1, 1, 1, 1)[:c]
        else:
            weight = self.weight[:c, :, :, :]
            
        return F.conv2d(x, weight, padding=self.padding, groups=c)


class SobelFilter(nn.Module):
    """
    Sobel 算子：用于提取图像梯度 (边缘)
    """
    def __init__(self):
        super(SobelFilter, self).__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)

    def forward(self, x):
        b, c, h, w = x.shape
        wx = self.weight_x.repeat(c, 1, 1, 1)
        wy = self.weight_y.repeat(c, 1, 1, 1)
        
        grad_x = F.conv2d(x, wx, padding=1, groups=c)
        grad_y = F.conv2d(x, wy, padding=1, groups=c)
        
        # 返回梯度模长
        grad_magnitude = torch.abs(grad_x) + torch.abs(grad_y)
        return grad_magnitude


class PhysicsInformedRegionAwareLoss(nn.Module):
    """
    语义引导的双流奖励机制 Loss (Semantic-Guided Dual-Stream Reward)
    """
    def __init__(self, w_fg_edge=1.0, w_fg_percep=0.01, w_bg_smooth=1.0, w_bg_tv=0.1, device='cuda'):
        super(PhysicsInformedRegionAwareLoss, self).__init__()
        self.device = device
        self.w_fg_edge = w_fg_edge
        self.w_fg_percep = w_fg_percep
        self.w_bg_smooth = w_bg_smooth
        self.w_bg_tv = w_bg_tv
        
        # 基础工具
        self.gaussian_blur = GaussianBlur(kernel_size=7, sigma=2.0).to(device)
        self.sobel_filter = SobelFilter().to(device)
        
        # VGG 感知损失工具 (冻结参数)
        vgg = models.vgg19(pretrained=True).features
        self.vgg_features = nn.Sequential(*list(vgg.children())[:35]).to(device).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()

    def get_mask_weight(self, mask):
        """Mask 软化"""
        return self.gaussian_blur(mask)

    def forward(self, sr, hr, binary_mask):
        """
        sr: [B, C, H, W] in [-1, 1] usually
        hr: [B, C, H, W] in [-1, 1] usually
        binary_mask: [B, 1, H, W] in [0, 1]
        """
        
        # 1. 范围归一化: 确保输入到 Loss 计算的是 [0, 1] 范围
        # 如果输入是 [-1, 1] (Diffusion 输出)，则转换；如果是 [0, 1] 则不变
        if sr.min() < -0.5: sr = (sr + 1.0) / 2.0
        if hr.min() < -0.5: hr = (hr + 1.0) / 2.0
        
        # 截断防御
        sr = torch.clamp(sr, 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        # 2. Mask 对齐
        if binary_mask.shape[-2:] != sr.shape[-2:]:
            binary_mask = F.interpolate(binary_mask, size=sr.shape[-2:], mode='nearest')

        # 3. 计算权重图
        M = self.get_mask_weight(binary_mask) # Soft Foreground Mask
        M_bg = 1.0 - M                        # Soft Background Mask
        
        # ================= 前景流 (锐化 & 结构) =================
        # A. 梯度 Loss
        grad_sr = self.sobel_filter(sr)
        grad_hr = self.sobel_filter(hr)
        loss_edge = F.l1_loss(grad_sr * M, grad_hr * M)
        
        # B. 感知 Loss
        feat_sr = self.vgg_features(sr)
        feat_hr = self.vgg_features(hr).detach()
        # 下采样 Mask 以匹配 Feature Map
        M_vgg = F.interpolate(M, size=feat_sr.shape[2:], mode='bilinear')
        loss_percep = F.mse_loss(feat_sr * M_vgg, feat_hr * M_vgg)

        # ================= 背景流 (平滑 & 降噪) =================
        # C. 平滑参考 Loss (拟合 GT 的模糊版)
        hr_smooth_target = self.gaussian_blur(hr).detach()
        loss_smooth_ref = self.mse_loss(sr * M_bg, hr_smooth_target * M_bg)
        
        # D. TV Loss (去伪影)
        diff_i = torch.abs(sr[:, :, :, 1:] - sr[:, :, :, :-1])
        diff_j = torch.abs(sr[:, :, 1:, :] - sr[:, :, :-1, :])
        # Mask 也要切片对齐
        loss_tv = torch.mean(diff_i * M_bg[:, :, :, 1:]) + torch.mean(diff_j * M_bg[:, :, 1:, :])

        # ================= 总 Loss =================
        total_loss = (self.w_fg_edge * loss_edge) +\
                     (self.w_fg_percep * loss_percep) +\
                     (self.w_bg_smooth * loss_smooth_ref) +\
                     (self.w_bg_tv * loss_tv)
                     
        return total_loss