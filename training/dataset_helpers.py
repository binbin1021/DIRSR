"""Paired HR/LR/mask datasets with portable optional split lists."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from my_utils.training_utils import (
    FileClient, PairedDataset, degradation_proc, imfrombytes, img2tensor,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validation_crop_geometry(gt_hw, lq_hw, scale):
    gt_h, gt_w = map(int, gt_hw)
    lq_h, lq_w = map(int, lq_hw)
    scale = int(scale)
    if scale < 1 or (gt_h, gt_w) != (lq_h * scale, lq_w * scale):
        raise ValueError(f"Non-paired dimensions: GT={gt_hw}, LR={lq_hw}, scale={scale}")
    height, width = lq_h - lq_h % 8, lq_w - lq_w % 8
    if not height or not width:
        raise ValueError(f"LR too small for native alignment: {lq_hw}")
    top, left = (lq_h - height) // 2, (lq_w - width) // 2
    return {
        "source_gt_hw": [gt_h, gt_w], "source_lq_hw": [lq_h, lq_w],
        "lq_xyxy": [left, top, left + width, top + height],
        "gt_mask_xyxy": [v * scale for v in [left, top, left + width, top + height]],
        "output_gt_hw": [height * scale, width * scale], "output_lq_hw": [height, width],
    }

class StrictPairedDataset(PairedDataset):
    """Keep native training processing while requiring real matched masks."""
    def __init__(self, options, triplets):
        self.opt = copy.deepcopy(options)
        self.file_client = None
        self.io_backend_opt = self.opt["io_backend"]
        if self.io_backend_opt.get("type") != "disk":
            raise ValueError("Only disk paired data are supported")
        self.gt_folder = self.opt["dataroot_gt"]
        self.lq_folder = self.opt["dataroot_lq"]
        self.mask_folder = self.opt["dataroot_mask"]
        self.scale = int(self.opt["scale"])
        self.paths_gt = [str(t[0]) for t in triplets]
        self.paths_lq = [str(t[1]) for t in triplets]
        self.paths_mask = [str(t[2]) for t in triplets]

class AlignedValidationDataset(StrictPairedDataset):
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)
        gt_path, lq_path, mask_path = (
            self.paths_gt[index], self.paths_lq[index], self.paths_mask[index])
        gt = imfrombytes(self.file_client.get(gt_path, "gt"), float32=True)
        lq = imfrombytes(self.file_client.get(lq_path, "lq"), float32=True)
        mask = imfrombytes(self.file_client.get(mask_path, "mask"), flag="grayscale", float32=True)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        geometry = validation_crop_geometry(gt.shape[:2], lq.shape[:2], self.scale)
        lx0, ly0, lx1, ly1 = geometry["lq_xyxy"]
        gx0, gy0, gx1, gy1 = geometry["gt_mask_xyxy"]
        gt, mask, lq = gt[gy0:gy1, gx0:gx1], mask[gy0:gy1, gx0:gx1], lq[ly0:ly1, lx0:lx1]
        return {
            "lq": img2tensor([lq], bgr2rgb=True, float32=True)[0],
            "gt": img2tensor([gt], bgr2rgb=True, float32=True)[0],
            "mask": img2tensor([mask], bgr2rgb=False, float32=True)[0],
            "lq_path": lq_path, "gt_path": gt_path,
        }

def _triplets(options, manifest):
    roots = [Path(options[key]).resolve() for key in ["dataroot_gt", "dataroot_lq", "dataroot_mask"]]
    for root in roots:
        if not root.is_dir():
            raise ValueError("Paired-data directory missing: " + str(root))
    if manifest:
        names = []
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.is_absolute():
                p = p.resolve().relative_to(roots[0])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Manifest paths must stay inside the HR root: " + line)
            names.append(p)
    else:
        names = [p.relative_to(roots[0]) for p in sorted(roots[0].rglob("*"))
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate entries in paired-data selection")
    records, triplets = [], []
    for name in names:
        paths = tuple(root / name for root in roots)
        if any(not p.is_file() for p in paths):
            raise ValueError("Missing same-relative-name HR/LR/mask triplet: " + str(name))
        sizes = []
        for p in paths:
            with Image.open(p) as image:
                sizes.append((image.height, image.width))
        if sizes[0] != sizes[2]:
            raise ValueError("HR/mask dimensions differ: " + str(name))
        validation_crop_geometry(sizes[0], sizes[1], options["scale"])
        records.append({"name": name.as_posix(), "hr_sha256": _sha256(paths[0]),
                        "lr_sha256": _sha256(paths[1]), "mask_sha256": _sha256(paths[2]),
                        "hr_size": sizes[0], "lr_size": sizes[1]})
        triplets.append(paths)
    return triplets, records

def make_datasets(config, manifest_dir):
    native = copy.deepcopy(config["datasets"])
    native["train"]["phase"], native["validation"]["phase"] = "train", "val"
    for opt in native.values():
        if int(opt["scale"]) != 4 or int(opt.get("crop_size", 512)) % 32:
            raise ValueError("This DIRSR entry uses x4 and crops divisible by 32")
    selected, records = {}, {}
    for split, key in [("train", "train"), ("val", "validation")]:
        selected[split], records[split] = _triplets(native[key], config.get(split + "_manifest"))
    train_paths = {p[0].resolve() for p in selected["train"]}
    val_paths = {p[0].resolve() for p in selected["val"]}
    train_hashes = {r["hr_sha256"] for r in records["train"]}
    val_hashes = {r["hr_sha256"] for r in records["val"]}
    if train_paths & val_paths or train_hashes & val_hashes:
        raise ValueError("Training and validation contain overlapping HR paths or content")
    train = StrictPairedDataset(native["train"], selected["train"])
    val = AlignedValidationDataset(native["validation"], selected["val"])
    identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    metadata = {
        "train_count": len(train), "val_count": len(val),
        "pool_count": len(train) + len(val), "content_sha256": identity,
        "records": records, "native_train_options": native["train"],
        "native_val_options": native["validation"],
        "mask_policy": "Mandatory real same-relative-name HR-size masks",
        "train_processing": "Native PairedDataset processing and degradation_proc",
        "validation_processing": "Aligned center crop in LR coordinates, divisible by 8",
    }
    directory = Path(manifest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (directory / (split + ".txt")).write_text("\n".join(r["name"] for r in records[split]) + "\n")
    (directory / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return train, val, metadata

__all__ = ["make_datasets", "degradation_proc", "validation_crop_geometry"]
