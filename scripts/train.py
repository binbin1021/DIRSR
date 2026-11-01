#!/usr/bin/env python3
"""Continue DIRSR SR training with a frozen learned-offset IDE and DAPE."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from _common import ROOT, path, read_config, python_path, environment, missing_assets, show

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--initial-sr", help="Native SR checkpoint, e.g. model_28501.pkl or model_38501.pkl")
    parser.add_argument("--initial-step", type=int, help="Cumulative cycle number of the initial SR checkpoint")
    parser.add_argument("--ide-path")
    parser.add_argument("--ram-path")
    parser.add_argument("--dape-path")
    parser.add_argument("--sd-path")
    parser.add_argument("--bert-path")
    for name in ["train-hr", "train-lr", "train-mask", "val-hr", "val-lr", "val-mask"]:
        parser.add_argument("--" + name)
    parser.add_argument("--train-manifest", help="Optional list of HR paths relative to the training HR root")
    parser.add_argument("--val-manifest", help="Optional list of HR paths relative to the validation HR root")
    parser.add_argument("--steps", type=int, help="Additional synchronized training cycles")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", help="Complete saved states/step_* directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report missing inputs without importing ML modules, creating a run or training")
    args = parser.parse_args()
    try:
        cfg = read_config(args.config)
        overrides = {
            "experiment_dir": args.output_dir, "parent_checkpoint": args.initial_sr,
            "parent_step": args.initial_step, "ide_checkpoint": args.ide_path,
            "ram_checkpoint": args.ram_path, "dape_checkpoint": args.dape_path,
            "sd_path": args.sd_path, "bert_path": args.bert_path,
            "train_manifest": args.train_manifest, "val_manifest": args.val_manifest,
            "additional_steps": args.steps, "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate, "seed": args.seed, "physical_gpu": args.gpu,
        }
        if args.initial_sr and args.initial_step is None:
            match = re.fullmatch(r"model_(\d+)", Path(args.initial_sr).stem)
            if match:
                overrides["parent_step"] = int(match.group(1))
            else:
                raise ValueError("--initial-step is required when the initial checkpoint is not named model_<step>.pkl")
        cfg.update({key: value for key, value in overrides.items() if value is not None})
        for prefix, split in [("train", "train"), ("val", "validation")]:
            for suffix, key in [("hr", "dataroot_gt"), ("lr", "dataroot_lq"), ("mask", "dataroot_mask")]:
                value = getattr(args, prefix + "_" + suffix)
                if value is not None:
                    cfg["datasets"][split][key] = value
        for key in ["additional_steps", "batch_size", "gradient_accumulation_steps", "val_samples",
                    "checkpoint_interval", "full_state_interval", "val_interval", "log_interval"]:
            if int(cfg[key]) < 1:
                raise ValueError(key + " must be positive")
        if cfg["mixed_precision"] != "no" or int(cfg["num_processes"]) != 1:
            raise ValueError("This training entry preserves the single-process FP32 protocol.")
        missing = missing_assets(cfg, train=True)
        source = ROOT / "training/train_sr.py"
        if not source.is_file():
            missing.append(str(source))
        output = path(cfg["experiment_dir"])
        runtime_config = output / "config.json"
        python = python_path(args.python)
        command = [python, str(source), "--config", str(runtime_config)]
        if args.resume:
            state = path(args.resume)
            command.extend(["--resume", str(state)])
            if not (state / "complete.json").is_file():
                missing.append(str(state / "complete.json"))
        if args.dry_run:
            show(command, cfg, missing, True)
            return 0
        if missing:
            raise ValueError("Required files/directories are missing:\n  " + "\n  ".join(missing))
        if not args.resume and output.exists() and any(output.iterdir()):
            raise ValueError("Existing run directory: use --resume or choose a new output directory.")
        if args.resume and runtime_config.exists() and json.loads(runtime_config.read_text()) != cfg:
            runtime_config = output / "config.requested.json"
            command[command.index("--config") + 1] = str(runtime_config)
        output.mkdir(parents=True, exist_ok=True)
        runtime_config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        show(command, cfg, [], False)
        return subprocess.call(command, cwd=ROOT, env=environment(cfg["physical_gpu"]))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    raise SystemExit(main())
