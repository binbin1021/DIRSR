import os
from glob import glob
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


# ================== 直接在这里改配置 ================== #
CONFIG = {
    # I/O
    "input_dir": None,
    "output_dir": None,
    "keep_color": False,   # False: 强制灰度输出（推荐红外）；True: 保持RGB

    # Random seed (可复现)
    "seed": 2025,

    # 固定链路：blur -> downsample -> noise
    # Blur config
    "blur_type": "random",   # "none" | "motion" | "defocus" | "random"

    # 非常微弱的湍流模糊：先于原有运动/离焦模糊，在 HR 分辨率上施加。
    # 平滑位移场的正/反向重采样平均，近似空间变化的局部模糊；非物理标定。
    "add_turbulence": True,
    "turbulence_p": 1.0,
    "turbulence_min_rms": 0.01,   # 每个方向的位移 RMS，单位为 HR 像素
    "turbulence_max_rms": 0.03,
    "turbulence_min_cell": 64,    # 更平缓的位移场，单位为 HR 像素
    "turbulence_max_cell": 96,
    "turbulence_strength": 0.01,  # 仅 1% 的模糊分量；量化后可能完全不变

    # motion blur (适当减小了模糊程度，防止太糊)
    "motion_min_len": 3,     # 原 20 -> 改为 3
    "motion_max_len": 15,    # 原 60 -> 改为 15
    "motion_min_angle": 0.0,
    "motion_max_angle": 180.0,
    "motion_p_no_blur": 0.2, # 增加 20% 概率不模糊，降低整体难度

    # defocus blur
    "defocus_radius": 3,     # 原 15 -> 改为 3 (15太大了，根本看不清)

    # random blur mixing
    "random_p_motion": 0.5,

    # Downsample（新增）
    # 1: 不下采样；2: 下采样2倍；4: 下采样4倍
    "downsample_factor": 4,
    # 下采样插值方式：红外灰度一般用 AREA 效果更像“缩小”
    "downsample_interp": "area",   # "area" | "linear" | "nearest"

    # Noise switches
    "add_shot": True,
    "add_fpn": True,
    "add_stripe": True,

    # --- 降低噪声强度的关键修改 ---
    
    # shot noise (泊松光子噪声)
    # 越大越干净 (方差 = img / scale)。原 150 -> 改为 1000
    "shot_scale": 1000.0,

    # FPN (固定模式底噪)
    # 越小越干净。原 2.0 -> 改为 0.5
    "fpn_sigma": 0.5,

    # stripe (条纹噪声)
    "stripe_vertical": True,
    # 乘性因子：越小条纹越淡。原 0.01 -> 改为 0.002
    "stripe_sigma_gain": 0.002,
    # 加性因子：越小条纹越淡。原 2.0 -> 改为 1.0
    "stripe_sigma_bias": 1.0,
}
# ===================================================== #


# ------------------ 噪声函数 ------------------ #
def add_shot_noise(img: np.ndarray, scale: float = 60.0) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    lam = np.clip(img_f, 0.0, 1.0) * scale + 1e-8
    noisy = np.random.poisson(lam).astype(np.float32) / scale
    noisy = np.clip(noisy, 0.0, 1.0) * 255.0
    return noisy.astype(np.uint8)


def add_fpn(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    h, w = img.shape[:2]
    fpn = np.random.randn(h, w).astype(np.float32) * sigma
    if img.ndim == 3:
        fpn = fpn[..., None]
    out = img.astype(np.float32) + fpn
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def add_stripe_noise(
    img: np.ndarray,
    sigma_gain: float = 0.05,
    sigma_bias: float = 10.0,
    vertical: bool = True,
) -> np.ndarray:
    h, w = img.shape[:2]

    if vertical:
        gain = 1.0 + np.random.randn(w).astype(np.float32) * sigma_gain
        bias = np.random.randn(w).astype(np.float32) * sigma_bias
        gain = gain[None, :]
        bias = bias[None, :]
    else:
        gain = 1.0 + np.random.randn(h).astype(np.float32) * sigma_gain
        bias = np.random.randn(h).astype(np.float32) * sigma_bias
        gain = gain[:, None]
        bias = bias[:, None]

    if img.ndim == 3:
        gain = gain[..., None]
        bias = bias[..., None]

    out = img.astype(np.float32) * gain + bias
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


# ------------------ Blur 函数 ------------------ #
def turbulence_blur(
    img_u8: np.ndarray,
    rng: np.random.Generator,
    displacement_rms: float = 0.02,
    cell_size: int = 80,
    strength: float = 0.01,
) -> np.ndarray:
    """微弱湍流模糊的图像域近似，支持灰度/RGB。

    构造空间平滑的二维位移场，将正、反位移结果平均以避免单向漂移，
    再与原图按 strength 混合。使用独立 rng，不直接消耗原有模糊/噪声
    的全局随机数。这里模拟空间变化的模糊效果，不是标定的大气模型。
    """
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("turbulence_strength must be in [0, 1]")
    if not np.isfinite(displacement_rms) or displacement_rms < 0.0:
        raise ValueError("turbulence displacement RMS must be nonnegative")
    if cell_size < 1:
        raise ValueError("turbulence cell_size must be positive")
    if strength == 0.0 or displacement_rms == 0.0:
        return img_u8

    h, w = img_u8.shape[:2]
    grid_h = max(2, int(np.ceil(h / cell_size)) + 1)
    grid_w = max(2, int(np.ceil(w / cell_size)) + 1)

    def smooth_field():
        coarse = rng.standard_normal((grid_h, grid_w)).astype(np.float32)
        field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        field -= field.mean()
        field_std = float(field.std())
        if field_std > 1e-8:
            field *= displacement_rms / field_std
        else:
            field.fill(0.0)
        return field

    dx, dy = smooth_field(), smooth_field()
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                               np.arange(h, dtype=np.float32))
    img_f = img_u8.astype(np.float32)
    plus = cv2.remap(img_f, map_x + dx, map_y + dy,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    minus = cv2.remap(img_f, map_x - dx, map_y - dy,
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    blurred = 0.5 * (plus + minus)
    out = img_f + strength * (blurred - img_f)
    return np.rint(np.clip(out, 0.0, 255.0)).astype(np.uint8)


def apply_turbulence(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    if not cfg.get("add_turbulence", False):
        return img
    strength = cfg.get("turbulence_strength", 0.01)
    if strength == 0.0:
        return img
    probability = cfg.get("turbulence_p", 1.0)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("turbulence_p must be in [0, 1]")
    if rng.random() >= probability:
        return img
    min_rms = cfg.get("turbulence_min_rms", 0.01)
    max_rms = cfg.get("turbulence_max_rms", 0.03)
    min_cell = cfg.get("turbulence_min_cell", 64)
    max_cell = cfg.get("turbulence_max_cell", 96)
    if not (0.0 <= min_rms <= max_rms):
        raise ValueError("invalid turbulence RMS range")
    if not (1 <= min_cell <= max_cell):
        raise ValueError("invalid turbulence cell-size range")
    return turbulence_blur(img, rng,
                           displacement_rms=float(rng.uniform(min_rms, max_rms)),
                           cell_size=int(rng.integers(min_cell, max_cell + 1)),
                           strength=strength)


def create_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(1, int(length))
    ksize = length if length % 2 == 1 else length + 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    center = (ksize / 2, ksize / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (ksize, ksize))

    s = kernel.sum()
    if s != 0:
        kernel /= s
    return kernel


def motion_blur(img_u8: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    kernel = create_motion_kernel(length, angle_deg)
    return cv2.filter2D(img_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


def create_disk_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    ksize = radius * 2 + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel


def defocus_blur(img_u8: np.ndarray, radius: int) -> np.ndarray:
    kernel = create_disk_kernel(radius)
    return cv2.filter2D(img_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


# ------------------ Downsample（新增） ------------------ #
def downsample_image(img_u8: np.ndarray, factor: int, interp: str = "area") -> np.ndarray:
    """
    factor: 1/2/4... 这里你只用 1/2/4
    interp: "area"(推荐缩小), "linear", "nearest"
    """
    if factor is None or factor == 1:
        return img_u8
    if factor not in (2, 4):
        raise ValueError("downsample_factor must be 1, 2, or 4")

    h, w = img_u8.shape[:2]
    new_w = max(1, w // factor)
    new_h = max(1, h // factor)

    interp_map = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    if interp not in interp_map:
        raise ValueError("downsample_interp must be 'area', 'linear', or 'nearest'")

    resized = cv2.resize(img_u8, (new_w, new_h), interpolation=interp_map[interp])
    return resized.astype(np.uint8)


# ------------------ I/O ------------------ #
def list_images(input_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(input_dir, e)))
    return sorted(paths)


def read_image_any(path: str, force_gray: bool = True) -> np.ndarray:
    img = Image.open(path)
    if force_gray:
        img = img.convert("L")
        return np.array(img, dtype=np.uint8)
    else:
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)


def save_image_u8(arr: np.ndarray, path: str, force_gray: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if force_gray:
        im = Image.fromarray(arr if arr.ndim == 2 else arr[..., 0]).convert("L")
    else:
        if arr.ndim == 3 and arr.shape[2] == 3:
            im = Image.fromarray(arr).convert("RGB")
        else:
            im = Image.fromarray(arr).convert("L")
    im.save(path)


# ------------------ 固定链路：BLUR -> DOWNSAMPLE -> NOISE ------------------ #
def apply_blur(img: np.ndarray, cfg: dict) -> np.ndarray:
    bt = cfg["blur_type"]

    if bt == "none":
        return img

    if bt == "motion":
        if random.random() < cfg["motion_p_no_blur"]:
            return img
        length = np.random.randint(cfg["motion_min_len"], cfg["motion_max_len"] + 1)
        angle = np.random.uniform(cfg["motion_min_angle"], cfg["motion_max_angle"])
        return motion_blur(img, length=length, angle_deg=angle)

    if bt == "defocus":
        return defocus_blur(img, radius=cfg["defocus_radius"])

    if bt == "random":
        if random.random() < cfg["random_p_motion"]:
            if random.random() < cfg["motion_p_no_blur"]:
                return img
            length = np.random.randint(cfg["motion_min_len"], cfg["motion_max_len"] + 1)
            angle = np.random.uniform(cfg["motion_min_angle"], cfg["motion_max_angle"])
            return motion_blur(img, length=length, angle_deg=angle)
        else:
            return defocus_blur(img, radius=cfg["defocus_radius"])

    raise ValueError(f"Unknown blur_type: {bt}")


def apply_noise(img: np.ndarray, cfg: dict) -> np.ndarray:
    out = img

    if cfg["add_shot"]:
        out = add_shot_noise(out, scale=cfg["shot_scale"])

    if cfg["add_fpn"]:
        out = add_fpn(out, sigma=cfg["fpn_sigma"])

    if cfg["add_stripe"]:
        out = add_stripe_noise(
            out,
            sigma_gain=cfg["stripe_sigma_gain"],
            sigma_bias=cfg["stripe_sigma_bias"],
            vertical=cfg["stripe_vertical"],
        )
    return out


def process_folder(cfg: dict):
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    # 独立随机流：新增位移场不直接改变原有 random/np.random 的抽样状态。
    turbulence_rng = np.random.default_rng(int(cfg["seed"]) + 314159)

    paths = list_images(cfg["input_dir"])
    if not paths:
        print(f"[Warning] No images found in {cfg['input_dir']}")
        return

    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"Found {len(paths)} images.")

    for p in tqdm(paths, desc="Degrading (blur->downsample->noise)"):
        name = os.path.basename(p)
        out_path = os.path.join(cfg["output_dir"], name)

        img = read_image_any(p, force_gray=(not cfg["keep_color"]))

        # 1) blur：先加非常微弱的湍流，再沿用原运动/离焦模糊
        img = apply_turbulence(img, cfg, turbulence_rng)
        img = apply_blur(img, cfg)

        # 2) downsample（可选：1/2/4）
        img = downsample_image(
            img,
            factor=cfg["downsample_factor"],
            interp=cfg["downsample_interp"],
        )

        # 3) noise
        img = apply_noise(img, cfg)

        save_image_u8(img, out_path, force_gray=(not cfg["keep_color"]))

    print("Done.")


if __name__ == "__main__":
    from make_lr import main
    main()
