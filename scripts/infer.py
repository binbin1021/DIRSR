#!/usr/bin/env python3
"""DIRSR x4 inference with the learned-offset IDE."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from _common import ROOT, path, read_config, python_path, environment, missing_assets, show

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.json")
    parser.add_argument("--lr-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--sr-path")
    parser.add_argument("--ide-path")
    parser.add_argument("--ram-path")
    parser.add_argument("--dape-path")
    parser.add_argument("--sd-path")
    parser.add_argument("--bert-path")
    parser.add_argument("--pos-prompt")
    parser.add_argument("--neg-prompt")
    parser.add_argument("--align", choices=["nofix", "wavelet", "adain"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration and missing files without importing ML modules or writing output")
    args = parser.parse_args()
    try:
        cfg = read_config(args.config)
        overrides = {"lr_dir": args.lr_dir, "output_dir": args.output_dir,
                     "parent_checkpoint": args.sr_path, "ide_checkpoint": args.ide_path,
                     "ram_checkpoint": args.ram_path, "dape_checkpoint": args.dape_path,
                     "sd_path": args.sd_path, "bert_path": args.bert_path,
                     "pos_prompt": args.pos_prompt, "neg_prompt": args.neg_prompt,
                     "align_method": args.align, "seed": args.seed}
        cfg.update({key: value for key, value in overrides.items() if value is not None})
        lr, output = path(cfg["lr_dir"]), path(cfg["output_dir"])
        missing = missing_assets(cfg)
        if not lr.is_dir():
            missing.append(str(lr))
        source = ROOT / 'src/inference_dirsr.py'
        if not source.is_file():
            missing.append(str(source))
        data = {"sf": 4, "validation": {"lr_path": str(lr), "io_backend": {"type": "disk"}}}
        python = python_path(args.python)
        def command(config):
            return [
                python, str(source), "--base_config", str(config),
                "--sd_path", str(path(cfg["sd_path"])), "--de_net_path", str(path(cfg["ide_checkpoint"])),
                "--pretrained_path", str(path(cfg["parent_checkpoint"])),
                "--ram_path", str(path(cfg["ram_checkpoint"])), "--dape_path", str(path(cfg["dape_checkpoint"])),
                "--bert_path", str(path(cfg["bert_path"])), "--ide_variant", "learned_offset",
                "--output_dir", str(output), "--scale", "4",
                "--lora_rank_unet", "32", "--lora_rank_vae", "16",
                "--pos_prompt", cfg["pos_prompt"], "--neg_prompt", cfg["neg_prompt"],
                "--align_method", cfg["align_method"], "--seed", str(cfg["seed"]),
                "--mixed_precision", "no", "--report_to", "tensorboard",
            ]
        if args.dry_run:
            show(command("<temporary-data-config.json>"), cfg, missing, True, data_config=data)
            return 0
        if missing:
            raise ValueError("Required files/directories are missing:\n  " + "\n  ".join(missing))
        images = sorted(p for p in lr.iterdir() if p.is_file() and p.suffix in {".png", ".jpg", ".jpeg"})
        if not images:
            raise ValueError("No lowercase PNG/JPG/JPEG inputs in " + str(lr))
        stems = [p.stem for p in images]
        if len(stems) != len(set(stems)):
            raise ValueError("Duplicate input stems would overwrite one output PNG.")
        if lr == output or any((output / (p.stem + ".png")).exists() for p in images):
            raise ValueError("Use a separate output folder with no existing output filenames.")
        with tempfile.TemporaryDirectory(prefix="dirsr-infer-") as temp:
            config = Path(temp) / "data.json"
            config.write_text(json.dumps(data, indent=2) + "\n")
            show(command(config), cfg, [], False, image_count=len(images))
            return subprocess.call(command(config), cwd=ROOT, env=environment(args.gpu))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    raise SystemExit(main())
