"""Standalone, warm-startable IDE with learned DCNv1 offsets.

The legacy computational definitions below are copied verbatim from the audited
``source_read_20261007/de_net.py``.  Only its unused BasicSR registry import is
omitted, so deployment requires PyTorch and torchvision, not the source tree.

Use ``IDE(num_in_ch=3, num_degradation=2)`` for the RGB LR
interface.  Outputs remain ``[noise_score, blur_score]`` with shape ``[B, 2]``.
``load_legacy_checkpoint`` permits exactly the two newly added offset tensors
to be absent.  ``load_model``/``load_new_checkpoint`` require every tensor.
Neither loader silently strips prefixes or guesses checkpoint wrapper keys.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def sobel_edge(x):
    Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(x[:, :1], Gx, padding=1)
    edge_y = F.conv2d(x[:, :1], Gy, padding=1)
    return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


class FrequencyEnhancedUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)
    def forward(self, x):
        fft = torch.fft.fft2(x, norm='ortho')
        fft_mag = torch.abs(fft)
        fft_feat = torch.cat([x, fft_mag], dim=1)
        return self.conv(fft_feat)


class NoiseBranch(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.rand_path = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(),
            FrequencyEnhancedUnit(base),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU()
        )
        self.pattern_path = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base, 5, padding=2, groups=base),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        r = self.rand_path(x)
        p = self.pattern_path(x)
        return self.fuse(r + p)


class BaseBlurBranch(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU()
        )
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(base, base, 3, padding=1),
            nn.Conv2d(base, base, 5, padding=2),
            nn.Conv2d(base, base, 7, padding=3),
        ])
        self.deform_conv = DeformConv2d(base, base, 3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.out_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.initial(x)
        ms_feat = sum([conv(x) for conv in self.multi_scale]) / len(self.multi_scale)
        offset = torch.zeros(ms_feat.size(0), 18, ms_feat.size(2), ms_feat.size(3), device=ms_feat.device)
        df_feat = self.deform_conv(ms_feat, offset)
        attn_map = self.attn(df_feat)
        feat = df_feat * attn_map
        return self.out_head(feat)


class IDEBase(nn.Module):
    """Infrared-Aware IDEBase with Noise + Blur branches, compatible with DIRSR interface"""
    def __init__(self, num_in_ch=1, **kwargs):
        super().__init__()
        self.total_in_ch = num_in_ch + 1  # e.g. IR + Sobel edge
        self.noise_branch = NoiseBranch(in_ch=self.total_in_ch)
        self.blur_branch = BaseBlurBranch(in_ch=self.total_in_ch)
        self.num_degradation = 2  # fixed: noise and blur

    def forward(self, x):  # x: [B, C, H, W]
        edge = sobel_edge(x)
        x_aug = torch.cat([x, edge], dim=1)
        sn = self.noise_branch(x_aug)  # [B, 1]
        sb = self.blur_branch(x_aug)  # [B, 1]
        return torch.cat([sn, sb], dim=1)  # [B, 2]

    def set_train(self):
        # 全部参与训练，或你可自定义只训练部分模块
        for p in self.parameters():
            p.requires_grad = True

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=True)

    def load_and_expand_model(self, path, num_degradation):
        self.load_model(path)
        # 不再支持动态扩展，函数保留做兼容
        assert num_degradation == 2, "This IDEBase version supports only two degradations (noise & blur)."


# SHA256 of the complete, unchanged audited source file (including comments).
LEGACY_SOURCE_SHA256 = "8fe8b06fc7a7d6fac90bf8565c3045f40a41f812659a41af868d856874e57500"
OFFSET_STATE_KEYS = frozenset({
    "blur_branch.offset_predictor.weight",
    "blur_branch.offset_predictor.bias",
})
Checkpoint = Union[str, Path, Mapping]


def _read_tensor_state_dict(checkpoint: Checkpoint, map_location="cpu"):
    """Read an explicit tensor state dict without key rewriting or unwrapping."""
    if isinstance(checkpoint, (str, Path)):
        state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    elif isinstance(checkpoint, Mapping):
        state = checkpoint
    else:
        raise TypeError("checkpoint must be a path or a tensor state-dict mapping")
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint must contain a nonempty tensor state dict")
    invalid = [key for key, value in state.items()
               if not isinstance(key, str) or not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(
            "Expected a raw tensor state dict; select a training-checkpoint "
            "state-dict field explicitly before loading. Invalid keys: "
            + repr(invalid)
        )
    return state


def _check_keys_and_shapes(state, expected):
    """Validate the entire checkpoint before copying any parameters."""
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = {
        key: {"checkpoint": tuple(state[key].shape),
              "expected": tuple(expected[key].shape)}
        for key in set(expected) & set(state)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Checkpoint incompatibility: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )


class BlurBranch(BaseBlurBranch):
    """Original blur branch with an unbounded, learned DCNv1 offset predictor."""

    def __init__(self, in_ch=2, base=32):
        super().__init__(in_ch=in_ch, base=base)
        self.offset_predictor = nn.Conv2d(base, 18, 3, padding=1)
        self.reset_offsets_to_zero()

    def reset_offsets_to_zero(self):
        nn.init.zeros_(self.offset_predictor.weight)
        nn.init.zeros_(self.offset_predictor.bias)

    def forward(self, x):
        x = self.initial(x)
        ms_feat = sum([conv(x) for conv in self.multi_scale]) / len(self.multi_scale)
        offset = self.offset_predictor(ms_feat)
        df_feat = self.deform_conv(ms_feat, offset)
        attn_map = self.attn(df_feat)
        feat = df_feat * attn_map
        return self.out_head(feat)


class IDE(IDEBase):
    """Legacy IDE with exactly 5,202 new learned-offset parameters at base=32."""

    def __init__(self, num_in_ch=3, num_degradation=2):
        if num_degradation != 2:
            raise ValueError("IDE supports only noise and blur")
        # Construct each original component once; inherit the unchanged joint
        # forward and set_train methods from IDEBase.
        nn.Module.__init__(self)
        self.total_in_ch = num_in_ch + 1
        self.noise_branch = NoiseBranch(in_ch=self.total_in_ch)
        self.blur_branch = BlurBranch(in_ch=self.total_in_ch)
        self.num_degradation = 2

    def load_legacy_checkpoint(self, checkpoint: Checkpoint, map_location="cpu"):
        """Warm-start all legacy tensors; only the two new tensors start at zero."""
        state = _read_tensor_state_dict(checkpoint, map_location=map_location)
        current = self.state_dict()
        expected_legacy = {key: value for key, value in current.items()
                           if key not in OFFSET_STATE_KEYS}
        _check_keys_and_shapes(state, expected_legacy)
        # Validation precedes mutation; loading a legacy checkpoint deliberately
        # discards any learned offsets this instance may have held previously.
        self.blur_branch.reset_offsets_to_zero()
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != OFFSET_STATE_KEYS or result.unexpected_keys:
            raise RuntimeError(f"Unexpected legacy-loading result: {result}")
        return {
            "loaded_keys": len(state),
            "loaded_numel": sum(value.numel() for value in state.values()),
            "new_zero_initialized_keys": sorted(OFFSET_STATE_KEYS),
            "legacy_source_sha256": LEGACY_SOURCE_SHA256,
        }

    def load_new_checkpoint(self, checkpoint: Checkpoint, map_location="cpu"):
        """Strictly reload a learned-offset model; no parameter may be missing."""
        state = _read_tensor_state_dict(checkpoint, map_location=map_location)
        _check_keys_and_shapes(state, self.state_dict())
        return self.load_state_dict(state, strict=True)

    def load_model(self, path):
        """DIRSR-style interface for strictly loading a NEW architecture state."""
        return self.load_new_checkpoint(path)

    def load_and_expand_model(self, path, num_degradation):
        if num_degradation != 2:
            raise ValueError("IDE supports only noise and blur")
        return self.load_new_checkpoint(path)
