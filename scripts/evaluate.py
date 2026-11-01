#!/usr/bin/env python3
"""Evaluate filename-matched SR/HR images with pyiqa PSNR, SSIM, LPIPS and DISTS."""
import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRIC_NAMES = ("psnr", "ssim", "lpips", "dists")


def image_files(directory):
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return {p.name: p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def matched_pairs(sr_dir, hr_dir, limit=None):
    sr, hr = image_files(sr_dir), image_files(hr_dir)
    if not sr or not hr:
        raise ValueError("Both directories must contain supported image files.")
    if set(sr) != set(hr):
        missing_sr = sorted(set(hr) - set(sr))
        missing_hr = sorted(set(sr) - set(hr))
        raise ValueError("Filenames must match exactly (including extensions). "
                         f"Missing SR: {missing_sr[:10]}; missing HR: {missing_hr[:10]}.")
    names = sorted(sr)
    if limit is not None:
        names = names[:limit]
    return [(name, sr[name], hr[name]) for name in names]


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr", "--sr-dir", "-i", required=True, type=Path,
                        help="SR output directory; images are evaluated as stored.")
    parser.add_argument("--hr", "--hr-dir", "-r", required=True, type=Path,
                        help="Paired HR reference directory.")
    parser.add_argument("--output-dir", "--output", type=Path,
                        default=Path("results/evaluation"),
                        help="Destination for summary.json, summary.csv and per_image.csv.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--metrics", nargs="+", choices=METRIC_NAMES,
                        default=list(METRIC_NAMES), help="Default: all four metrics.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N sorted matched pairs.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    args.metrics = list(dict.fromkeys(args.metrics))
    try:
        sr_dir, hr_dir = args.sr.expanduser().resolve(), args.hr.expanduser().resolve()
        pairs = matched_pairs(sr_dir, hr_dir, args.limit)
        import numpy as np
        from PIL import Image
        import torch
        import pyiqa
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
        options = {name: ({"test_y_channel": True, "color_space": "ycbcr", "crop_border": 0}
                          if name in ("psnr", "ssim") else {}) for name in args.metrics}
        metrics = {name: pyiqa.create_metric(name, device=device, **options[name]).eval()
                   for name in args.metrics}

        def read_tensor(path):
            with Image.open(path) as image:
                if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                    raise ValueError(f"Expected an 8-bit image, got {image.mode}: {path}")
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

        rows = []
        with torch.inference_mode():
            for index, (name, sr_path, hr_path) in enumerate(pairs, 1):
                sr_tensor, hr_tensor = read_tensor(sr_path), read_tensor(hr_path)
                if sr_tensor.shape != hr_tensor.shape:
                    raise ValueError(f"Shape mismatch for {name}: {tuple(sr_tensor.shape)} "
                                     f"vs {tuple(hr_tensor.shape)}. No resize or crop is applied.")
                row = {"filename": name, "sr_file": str(sr_path), "hr_file": str(hr_path),
                       "width": int(sr_tensor.shape[-1]), "height": int(sr_tensor.shape[-2])}
                row.update({key: float(metric(sr_tensor, hr_tensor).item())
                            for key, metric in metrics.items()})
                rows.append(row)
                print(f"[{index}/{len(pairs)}] {name}", flush=True)
        means = {key: sum(row[key] for row in rows) / len(rows) for key in args.metrics}
        summary = {"n_images": len(rows), "metrics": means,
                   "protocol": {"sr_dir": str(sr_dir), "hr_dir": str(hr_dir),
                                "matching": "exact filename, including extension; sorted",
                                "device": str(device), "input": "stored 8-bit images -> RGB float32 [0,1]",
                                "precision": "float32 inputs; pyiqa native metric arithmetic; no autocast",
                                "psnr_ssim": "YCbCr Y; crop_border=0 (no extra border crop)",
                                "lpips_dists": "RGB; pyiqa defaults",
                                "metric_options": options, "limit": args.limit,
                                "pyiqa_version": importlib.metadata.version("pyiqa"),
                                "torch_version": torch.__version__}}
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["n_images", *args.metrics])
            writer.writeheader()
            writer.writerow({"n_images": len(rows), **means})
        print(json.dumps(json_safe({"n_images": len(rows), **means}), indent=2))
        print(f"Saved evaluation to {output}")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
