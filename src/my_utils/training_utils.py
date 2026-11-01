import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session.
    """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--lambda_dual", default=2.0, type=float)
    parser.add_argument("--base_config", default="./configs/sr.yaml", type=str)
    parser.add_argument("--w_fg_edge", type=float, default=1.0, 
                        help="Weight for foreground edge loss (Structure Stream)")
    parser.add_argument("--w_fg_percep", type=float, default=0.01, 
                        help="Weight for foreground perceptual loss (Structure Stream)")
    parser.add_argument("--w_bg_smooth", type=float, default=1.0, 
                        help="Weight for background smooth reference loss (Denoising Stream)")
    parser.add_argument("--w_bg_tv", type=float, default=0.1, 
                        help="Weight for background TV loss (Denoising Stream)")

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_val", default=True, action="store_false")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--de_net_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--neg_prob", default=0.05, type=float)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution infrared image with high-fidelity thermal edges, realistic emissivity textures, preserved fine thermal gradients, sharp and natural details.")
    parser.add_argument("--neg_prompt", type=str, default="stripe noise, banding, nonuniformity, fixed-pattern noise, hot pixels, ghosting, thermal texture loss, oversmoothing")


    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=50000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "piecewise_constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=0.1, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


# @DATASET_REGISTRY.register(suffix='basicsr')
class PairedDataset(data.Dataset):
    """
    修改后的 Dataset，用于读取已生成的成对数据 (HR, LR) 和 Mask。
    包含越界保护和强制 Scale 检查。
    """
    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        # 1. 路径设置
        self.gt_folder = opt['dataroot_gt']    # HR
        self.lq_folder = opt['dataroot_lq']    # LR
        self.mask_folder = opt.get('dataroot_mask', None) # Mask

        # 2. 扫描文件 (使用 sorted 确保一一对应)
        self.paths_gt = sorted(list(self._scan_folder(self.gt_folder)))
        self.paths_lq = sorted(list(self._scan_folder(self.lq_folder)))
        
        # 3. 校验数量
        assert len(self.paths_gt) == len(self.paths_lq), \
            f"GT count ({len(self.paths_gt)}) != LQ count ({len(self.paths_lq)})"

        if self.mask_folder:
            self.paths_mask = sorted(list(self._scan_folder(self.mask_folder)))
            if len(self.paths_mask) != len(self.paths_gt):
                print(f"[Warning] Mask count ({len(self.paths_mask)}) != GT count. Will use dummy masks.")
                self.paths_mask = None
        else:
            self.paths_mask = None

        # 【修改】强制从配置读取 scale，没有则报错，防止默认值误导
        if 'scale' in opt:
            self.scale = opt['scale']
        else:
            raise ValueError("❌ Config Error: 'scale' is missing in your YAML config! Please add 'scale: 2' (or 4).")
        
        print(f"✅ PairedDataset initialized. Scale: {self.scale}, Count: {len(self.paths_gt)}")

    def _scan_folder(self, folder):
        # 递归扫描常见图片格式
        return Path(folder).rglob('*.*')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # 1. Load Images & Config
        # 【修改】直接使用 self.scale
        scale = self.scale
        
        gt_path = str(self.paths_gt[index])
        lq_path = str(self.paths_lq[index])
        
        # Load GT
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # Load LQ
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Load Mask
        if self.paths_mask:
            mask_path = str(self.paths_mask[index])
            mask_bytes = self.file_client.get(mask_path, 'mask')
            img_mask = imfrombytes(mask_bytes, flag='grayscale', float32=True) # [H, W]
            if img_mask.ndim == 2:
                img_mask = np.expand_dims(img_mask, axis=2) # [H, W, 1]
        else:
            img_mask = np.ones_like(img_gt)[:, :, :1]

        # ------------------------------------------------------------
        # 核心逻辑 (含越界保护)
        # ------------------------------------------------------------
        h_gt, w_gt = img_gt.shape[0:2]
        crop_size = self.opt.get('crop_size', 512)

        # ------------------ Phase 1: Train ------------------
        if self.opt['phase'] == 'train':
            # A. 随机翻转
            if self.opt.get('use_hflip', True) and random.random() < 0.5:
                img_gt = cv2.flip(img_gt, 1)
                img_lq = cv2.flip(img_lq, 1)
                img_mask = cv2.flip(img_mask, 1)
                if img_mask.ndim == 2: img_mask = np.expand_dims(img_mask, axis=2)

            # B. Padding
            if h_gt < crop_size or w_gt < crop_size:
                pad_h = max(0, crop_size - h_gt)
                pad_w = max(0, crop_size - w_gt)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_mask = cv2.copyMakeBorder(img_mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                if img_mask.ndim == 2: img_mask = np.expand_dims(img_mask, axis=2)
                
                # LQ Padding
                pad_h_lq = int(pad_h / scale)
                pad_w_lq = int(pad_w / scale)
                img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h_lq, 0, pad_w_lq, cv2.BORDER_REFLECT_101)
                
                # 更新宽高
                h_gt, w_gt = img_gt.shape[0:2]

            # C. Random Crop
            if h_gt > crop_size or w_gt > crop_size:
                lq_crop_size = crop_size // scale
                h_lq, w_lq = img_lq.shape[0:2]
                
                # 只有当 LQ 足够大才进行随机 Crop，否则 Padding 已经处理了
                if h_lq > lq_crop_size and w_lq > lq_crop_size:
                    top_lq = random.randint(0, h_lq - lq_crop_size)
                    left_lq = random.randint(0, w_lq - lq_crop_size)
                    
                    top_gt = top_lq * scale
                    left_gt = left_lq * scale
                    
                    # 【修改】强制越界修正 (Bounds Check)
                    # 无论 Scale 怎么设置，这里都强制把坐标拉回 GT 尺寸范围内
                    if top_gt + crop_size > img_gt.shape[0]:
                        top_gt = img_gt.shape[0] - crop_size
                    if left_gt + crop_size > img_gt.shape[1]:
                        left_gt = img_gt.shape[1] - crop_size
                    
                    # 防止负数
                    top_gt = max(0, top_gt)
                    left_gt = max(0, left_gt)

                    # 执行裁剪
                    img_lq = img_lq[top_lq : top_lq + lq_crop_size, left_lq : left_lq + lq_crop_size, ...]
                    img_gt = img_gt[top_gt : top_gt + crop_size, left_gt : left_gt + crop_size, ...]
                    img_mask = img_mask[top_gt : top_gt + crop_size, left_gt : left_gt + crop_size, ...]
                    
                    if img_mask.ndim == 2: img_mask = np.expand_dims(img_mask, axis=2)

        # ------------------ Phase 2: Validation ------------------
        else:
            align_factor = scale * 8 
            new_h_gt = h_gt - (h_gt % align_factor)
            new_w_gt = w_gt - (w_gt % align_factor)
            
            new_h_lq = new_h_gt // scale
            new_w_lq = new_w_gt // scale
            
            if new_h_gt != h_gt or new_w_gt != w_gt:
                top_gt = (h_gt - new_h_gt) // 2
                left_gt = (w_gt - new_w_gt) // 2
                top_lq = top_gt // scale
                left_lq = left_gt // scale
                
                img_gt = img_gt[top_gt : top_gt + new_h_gt, left_gt : left_gt + new_w_gt, ...]
                img_mask = img_mask[top_gt : top_gt + new_h_gt, left_gt : left_gt + new_w_gt, ...]
                img_lq = img_lq[top_lq : top_lq + new_h_lq, left_lq : left_lq + new_w_lq, ...]
                
                if img_mask.ndim == 2: img_mask = np.expand_dims(img_mask, axis=2)

        # ---------------- To Tensor & Final Check ---------------- #
        # 【修改】最后的保险：如果 img_gt 是空的，打印错误并返回全黑图
        if img_gt.size == 0 or img_lq.size == 0:
            print(f"❌ Error: Empty Crop detected! Path: {gt_path}, Scale used: {scale}")
            # 返回 dummy 数据
            img_gt = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
            img_lq = np.zeros((crop_size//scale, crop_size//scale, 3), dtype=np.float32)
            img_mask = np.zeros((crop_size, crop_size, 1), dtype=np.float32)

        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_mask = img2tensor([img_mask], bgr2rgb=False, float32=True)[0]

        return {
            'lq': img_lq, 
            'gt': img_gt, 
            'mask': img_mask, 
            'lq_path': lq_path, 
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths_gt)


def randn_cropinput(lq, gt, base_size=[64, 128, 256, 512]):
    cur_size_h = random.choice(base_size)
    cur_size_w = random.choice(base_size)
    init_h = lq.size(-2)//2
    init_w = lq.size(-1)//2
    lq = lq[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    gt = gt[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    assert lq.size(-1)>=64
    assert lq.size(-2)>=64
    return [lq, gt]


def degradation_proc(configs, batch, device, val=False, use_usm=False, resize_lq=True, random_size=False):
    """
    简化版的处理函数：
    1. 从 Dataset 获取 crop 好的 LR, GT, Mask
    2. 移动到 GPU
    3. 归一化到 [-1, 1]
    """
    # 1. 获取数据 (Dataset 此时返回的是 [0, 1])
    lq = batch['lq'].to(device)
    gt = batch['gt'].to(device)
    
    # 2. 获取 Mask
    if 'mask' in batch:
        mask = batch['mask'].to(device)
    else:
        mask = torch.ones_like(gt)[:, :1, :, :]
        
    # 3. 如果需要把 LR resize 到和 GT 一样大 (某些模型架构需要 concat)
    ori_lq = lq # 备份原始尺寸 LR
    if resize_lq:
        lq = F.interpolate(
            lq, 
            size=(gt.size(-2), gt.size(-1)), 
            mode='bicubic', 
            align_corners=False
        )

    # 4. 转换范围 [0, 1] -> [-1, 1] (Diffusion 标准)
    lq = (lq * 2.0) - 1.0
    gt = (gt * 2.0) - 1.0
    
    # Mask 保持 [0, 1] 用于 Loss 计算，不需要转到 [-1, 1]
    
    # 更新 batch 里的 mask，以便 train.py 里能取到
    batch['mask'] = mask

    return lq, gt, ori_lq