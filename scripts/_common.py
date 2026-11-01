"""Portable standard-library-only command configuration."""
from pathlib import Path
import copy
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent

def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def python_path(value):
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.absolute())
    found = shutil.which(value)
    if found:
        return found
    raise ValueError("Python executable not found: " + value)

def read_config(value):
    p = path(value)
    return json.loads(p.read_text(encoding="utf-8"))

def resolve_train_config(config):
    cfg = copy.deepcopy(config)
    cfg["project_root"] = str(ROOT)
    for key in ["experiment_dir", "parent_checkpoint", "ide_checkpoint", "sd_path",
                "ram_checkpoint", "dape_checkpoint", "bert_path"]:
        cfg[key] = str(path(cfg[key]))
    for key in ["train_manifest", "val_manifest"]:
        if cfg.get(key):
            cfg[key] = str(path(cfg[key]))
    for split in ["train", "validation"]:
        opt = cfg["datasets"][split]
        for key in ["dataroot_gt", "dataroot_lq", "dataroot_mask"]:
            opt[key] = str(path(opt[key]))
    return cfg

def environment(gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env

def missing_assets(cfg, train=False):
    keys = ["parent_checkpoint", "ide_checkpoint", "ram_checkpoint", "dape_checkpoint", "sd_path", "bert_path"]
    missing = [str(path(cfg[k])) for k in keys if not path(cfg[k]).exists()]
    for item in ["scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                 "text_encoder/config.json", "unet/config.json", "vae/config.json"]:
        p = path(cfg["sd_path"]) / item
        if not p.is_file():
            missing.append(str(p))
    for item in ["vocab.txt", "tokenizer_config.json"]:
        p = path(cfg["bert_path"]) / item
        if not p.is_file():
            missing.append(str(p))
    if train:
        for split in ["train", "validation"]:
            for key in ["dataroot_gt", "dataroot_lq", "dataroot_mask"]:
                p = path(cfg["datasets"][split][key])
                if not p.is_dir():
                    missing.append(str(p))
        for key in ["train_manifest", "val_manifest"]:
            if cfg.get(key) and not path(cfg[key]).is_file():
                missing.append(str(path(cfg[key])))
    return list(dict.fromkeys(missing))

def show(command, cfg, missing, dry_run, **extra):
    import shlex
    result = {"dry_run": dry_run, "command": shlex.join(command),
              "missing": missing, "config": cfg}
    result.update(extra)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
