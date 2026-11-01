import os
from pathlib import Path
import sys
import gc
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

# 确保路径能找到项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import diffusers
from diffusers.utils.import_utils import is_xformers_available

# 注意：如果你想用 Tiled 推理，保持 dirsr_tile；如果想用标准推理，改回 dirsr
from dirsr_tile import DIRSR 
from my_utils.testing_utils import parse_args_paired_testing, PlainDataset
from utils.wavelet_color import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference

# =========================================================================
# 辅助函数
# =========================================================================

def preprocess_tensor_image(img_tensor):
    # img_tensor: [3, H, W], range [0, 1]
    # RAM 需要 resize 到 384x384
    img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False)[0]
    img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])(img_tensor)
    return img_tensor

def get_validation_prompt(args, image, model, device='cuda'):
    # image: [3, H, W] tensor
    validation_prompt = ""
    lq = preprocess_tensor_image(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        res = inference(lq, model)
        # 注意：如果 model.generate_image_embeds 也是 no_grad 的话
        ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]},"
    return validation_prompt, ram_encoder_hidden_states

def load_tag_model(args, device='cuda'):
    os.environ['DIRSR_BERT_PATH'] = str(Path(args.bert_path).expanduser().resolve())
    # 【修改点 1】路径修正为训练环境的路径
    print("Loading RAM and DAPE...")
    model = ram(pretrained=args.ram_path,
                pretrained_condition=args.dape_path, 
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    return model

# =========================================================================
# 主函数
# =========================================================================

def main(args):
    config = OmegaConf.load(args.base_config)

    # 1. 确定权重路径
    if args.pretrained_path is None:
        from huggingface_hub import hf_hub_download
        pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
    else:
        pretrained_path = args.pretrained_path

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    # 2. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 3. 初始化各个模型
    # DIRSR
    print("Initializing DIRSR...")
    net_sr = DIRSR(lora_rank_unet=args.lora_rank_unet, 
                    lora_rank_vae=args.lora_rank_vae, 
                    sd_path=sd_path, 
                    pretrained_path=pretrained_path, 
                    args=args) # 确保传入 args
    net_sr.set_eval()

    # Degradation Encoder
    print("Initializing DE Net...")
    from ide import IDE as DegradationEncoder
    net_de = DegradationEncoder(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()
    net_de.requires_grad_(False)

    # RAM (Tagging)
    print("Initializing RAM...")
    ram_model = load_tag_model(args)
    ram_model.eval().to(accelerator.device)
    
    # 4. 优化 (Xformers / Checkpointing)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            print("Warning: xformers not available, skipping.")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 5. 数据集加载
    dataset_val = PlainDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare with accelerator
    net_sr, net_de = accelerator.prepare(net_sr, net_de)

    # Cast types
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
      
    print(f"Start Inference... Total images: {len(dataset_val)}")
    
    # =========================================================================
    # 推理循环
    # =========================================================================
    for step, batch_val in enumerate(dl_val):
        lr_path = batch_val['lr_path'][0]
        (path, name) = os.path.split(lr_path)

        # 读取数据
        im_lr = batch_val['lr'].cuda()
        
        # 【修改点 2】红外单通道兼容：如果是单通道，复制为 3 通道
        # 很多红外数据集读进来是 [B, 1, H, W]，但模型需要 3 通道
        if im_lr.shape[1] == 1:
            im_lr = im_lr.repeat(1, 3, 1, 1)

        im_lr = im_lr.to(memory_format=torch.contiguous_format).float()    

        # Resize (Upsample)
        ori_h, ori_w = im_lr.shape[2:]
        target_h, target_w = ori_h * config.sf, ori_w * config.sf
        
        im_lr_resize = F.interpolate(
            im_lr,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False 
        )

        # 归一化到 [-1, 1]
        im_lr_resize = im_lr_resize.contiguous() 
        im_lr_resize_norm = im_lr_resize * 2 - 1.0
        im_lr_resize_norm = torch.clamp(im_lr_resize_norm, -1.0, 1.0)
        
        # 【修改点 3】Smart Padding (Align to 64 with Reflect)
        resize_h, resize_w = im_lr_resize_norm.shape[2:]
        pad_base = 64
        pad_h = (math.ceil(resize_h / pad_base)) * pad_base - resize_h
        pad_w = (math.ceil(resize_w / pad_base)) * pad_base - resize_w
        
        if pad_h > 0 or pad_w > 0:
            im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
        
        B = im_lr_resize.size(0)
        
        with torch.no_grad():
            # 获取退化评分
            deg_score = net_de(im_lr) # im_lr 已经是 3 通道了
            
            # 获取 Prompt
            pos_tag_prompt = []                
            neg_tag_prompt = []                
            
            for i in range(B):
                tag_prompt, _ = get_validation_prompt(args, image=im_lr[i], model=ram_model, device=accelerator.device)

                full_pos = f"{args.pos_prompt}, {tag_prompt}"   
                full_neg = args.neg_prompt                      

                pos_tag_prompt.append(full_pos)
                neg_tag_prompt.append(full_neg)

            # DIRSR 推理
            # 这里的调用方式假设 DIRSR 支持 pos_prompt/neg_prompt 参数
            # 同时也支持 mixed_precision 自动转换
            x_tgt_pred = accelerator.unwrap_model(net_sr)(
                im_lr_resize_norm, 
                deg_score, 
                pos_prompt=pos_tag_prompt, 
                neg_prompt=neg_tag_prompt
            )
            
            # 【修改点 3】Un-Padding (裁剪回原始 Resize 后的尺寸)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            
            # 转回 [0, 1]
            out_img = (x_tgt_pred * 0.5 + 0.5).clamp(0, 1).cpu().detach()

        # 保存图片
        output_pil = transforms.ToPILImage()(out_img[0])

        # 颜色校正 (Color Fix)
        if args.align_method == 'nofix':
            pass
        else:
            # 这里的 im_lr_resize 还是 [0, 1] 范围的 tensor (interpolation 后未 norm)
            im_lr_pil = transforms.ToPILImage()(im_lr_resize[0].cpu().detach().clamp(0, 1))
            
            if args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(output_pil, im_lr_pil)
            elif args.align_method == 'adain':
                output_pil = adain_color_fix(output_pil, im_lr_pil)

        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.output_dir, fname+'.png')
        output_pil.save(outf)
        
        if step % 10 == 0:
            print(f"[{step}/{len(dataset_val)}] Saved: {outf}")

    print("Inference Finished!")
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 解析参数 (使用 testing_utils 里的解析器)
    args = parse_args_paired_testing()
    main(args)