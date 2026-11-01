"""Independent, resumable infrared IDE continuation with learned DCNv1 offsets.

The official MM-RealSR synthesis and its degree-ranking/anchor objective are
reused, but this 70k-iteration standalone continuation is a new experiment,
not an assertion about how the historical checkpoint was trained.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from ide_arch import IDEBase, IDE, OFFSET_STATE_KEYS
from loss_and_checks import degree_loss, VIEW_NAMES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_json(path, value):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def preserved_rng(seed=None):
    state = rng_state()
    try:
        if seed is not None:
            seed_all(seed)
        yield
    finally:
        restore_rng(state)


class IndexedNativeDataset(Dataset):
    """Seed native crop/kernel sampling per epoch/index, including on resume."""
    def __init__(self, native_dataset, seed):
        self.native = native_dataset
        self.seed = seed

    def __len__(self):
        return len(self.native)

    def __getitem__(self, key):
        epoch, index = key
        seed = (self.seed + epoch * 1000003 + index * 9176) % (2**32)
        # Only CPU generators are touched inside DataLoader worker processes.
        py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            return self.native[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)


class ResumableBatchSampler(Sampler):
    def __init__(self, count, batch_size, seed, epoch, start_batch):
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.count, generator=generator).tolist()
        stop = self.count // self.batch_size * self.batch_size
        for start in range(self.start_batch * self.batch_size, stop, self.batch_size):
            yield [(self.epoch, i) for i in indices[start:start + self.batch_size]]

    def __len__(self):
        return self.count // self.batch_size - self.start_batch


def five_views(synth):
    assert tuple(synth.view_order) == VIEW_NAMES
    if synth.lq.ndim != 5 or synth.lq.shape[1] != 3 or synth.anchor.shape[1] != 2:
        raise RuntimeError(f"Unexpected native shapes: {synth.lq.shape}, {synth.anchor.shape}")
    views = torch.cat([synth.lq, synth.anchor], dim=1)
    if not torch.isfinite(views).all() or views.min() < 0 or views.max() > 1:
        raise RuntimeError("IDE inputs must be finite and within [0,1]")
    return views


def update_ema(ema, model, decay):
    with torch.no_grad():
        state = model.state_dict()
        for name, value in ema.state_dict().items():
            if torch.is_floating_point(value):
                value.mul_(decay).add_(state[name], alpha=1 - decay)
            else:
                value.copy_(state[name])


@torch.no_grad()
def evaluate(model, cache, device, source_batch_size=6):
    model.eval()
    totals = {}
    count = 0
    for start in range(0, len(cache), source_batch_size):
        views = cache[start:start + source_batch_size].to(device)
        scores = model(views.flatten(0, 1)).view(views.shape[0], 5, 2)
        loss, metrics = degree_loss(scores)
        metrics = {key: float(value) for key, value in metrics.items()}
        metrics["loss"] = float(loss)
        for key, value in metrics.items():
            # RMSE is converted back to MSE for correct aggregation below.
            v = value**2 if key == "anchor_rmse" or key.startswith("anchor_rmse_") else value
            totals[key] = totals.get(key, 0) + v * views.shape[0]
        count += views.shape[0]
    result = {k: v / count for k, v in totals.items()}
    for key in result:
        if key == "anchor_rmse" or key.startswith("anchor_rmse_"):
            result[key] = math.sqrt(result[key])
    result["images"] = count
    return result


def make_dataset(dataset_class, options, manifest):
    opt = copy.deepcopy(options["datasets"]["train"])
    opt["dataroot_gt"] = "/"
    opt["meta_info"] = str(manifest)
    return dataset_class(opt)


def validation_cache(dataset_class, options, synth_class, cfg, root, device):
    fingerprint = {"val_list_sha256": sha256(root / "data_manifest/val.txt"),
                   "options_sha256": sha256(cfg["native_options"]),
                   "native_source_sha256": options["_native_mmreal"]["native_source_sha256"],
                   "validation_seed": cfg["validation_seed"], "views": list(VIEW_NAMES)}
    path = root / "validation_cache.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("Validation cache fingerprint mismatch")
        return saved["views"]
    dataset = make_dataset(dataset_class, options, root / "data_manifest/val.txt")
    synth = synth_class(options, device, queue_size=cfg["queue_size"], use_queue=False)
    batches = []
    with preserved_rng(cfg["validation_seed"]):
        for index in range(len(dataset)):
            # The fixed views, rather than fresh random degradations, select checkpoints.
            seed_all(cfg["validation_seed"] + index)
            synth.feed_data(default_collate([dataset[index]]))
            batches.append(five_views(synth).cpu())
            if (index + 1) % 25 == 0:
                print(f"Prepared validation: {index + 1}/{len(dataset)}", flush=True)
    views = torch.cat(batches)
    atomic_save(path, {"fingerprint": fingerprint, "views": views})
    return views


def smoke_check(dataset_class, options, synth_class, cfg, root, device):
    """Use a disposable copy: smoke optimizer updates never enter the real run."""
    with preserved_rng(cfg["seed"] + 99):
        dataset = make_dataset(dataset_class, options, root / "data_manifest/train.txt")
        synth = synth_class(options, device, queue_size=cfg["queue_size"], use_queue=False)
        synth.feed_data(default_collate([dataset[0]]))
        views = five_views(synth)
        model = IDE(num_in_ch=3).to(device)
        loading = model.load_legacy_checkpoint(cfg["warm_start"])
        old = IDEBase(num_in_ch=3).to(device)
        old.load_model(cfg["warm_start"])
        model.eval()
        old.eval()
        with torch.no_grad():
            init_difference = (model(views.flatten(0, 1)) - old(views.flatten(0, 1))).abs().max().item()
        if init_difference != 0:
            raise RuntimeError(f"Legacy/init parity failed: {init_difference}")
        if any(model.state_dict()[key].count_nonzero() for key in OFFSET_STATE_KEYS):
            raise RuntimeError("Offsets must start at zero")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
        scores = model(views.flatten(0, 1)).view(views.shape[0], 5, 2)
        loss, metrics = degree_loss(scores)
        loss.backward()
        grad = model.blur_branch.offset_predictor.weight.grad
        if grad is None or not torch.isfinite(grad).all() or not grad.count_nonzero():
            raise RuntimeError("Offset predictor did not get a finite nonzero gradient")
        gradient_nonzero = int(grad.count_nonzero())
        optimizer.step()
        changed = model.blur_branch.offset_predictor.weight
        if not torch.isfinite(changed).all() or not changed.count_nonzero():
            raise RuntimeError("Offset predictor did not update")
        temporary = root / "smoke_checkpoint.pt"
        atomic_save(temporary, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        state = torch.load(temporary, map_location=device, weights_only=False)
        reloaded = IDE(num_in_ch=3).to(device)
        reloaded.load_new_checkpoint(state["model"])
        opt2 = torch.optim.Adam(reloaded.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
        opt2.load_state_dict(state["optimizer"])
        model.eval()
        reloaded.eval()
        with torch.no_grad():
            reload_difference = (model(views.flatten(0, 1)) - reloaded(views.flatten(0, 1))).abs().max().item()
        if reload_difference != 0 or len(opt2.state) != len(optimizer.state):
            raise RuntimeError("Model/optimizer checkpoint reload failed")
        temporary.unlink()
        result = {"passed": True, "device": str(device), "source_images": 1,
                  "input_shape": list(views.shape), "input_min": float(views.min()),
                  "input_max": float(views.max()), "init_max_difference": init_difference,
                  "offset_gradient_nonzero": gradient_nonzero,
                  "updated_offset_weight_nonzero": int(changed.count_nonzero()),
                  "reload_max_difference": reload_difference, "loading": loading,
                  "loss": float(loss), "metrics": {k: float(v) for k, v in metrics.items()}}
        atomic_json(root / "smoke_results.json", result)
        print("GPU synthesis/backward/checkpoint smoke check passed", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/ide.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    project = Path(__file__).resolve().parents[2]
    cfg["project_root"] = str(project)
    for key in ("mmreal_root", "native_options", "experiment_dir", "warm_start"):
        candidate = Path(cfg[key]).expanduser()
        cfg[key] = str(candidate.resolve() if candidate.is_absolute() else (project / candidate).resolve())
    if not cfg.get("warm_start_sha256"):
        cfg["warm_start_sha256"] = sha256(cfg["warm_start"])
    root = Path(cfg["experiment_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(cfg["torch_cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real training was not started")
    if sha256(cfg["warm_start"]) != cfg["warm_start_sha256"]:
        raise RuntimeError("Warm-start checkpoint SHA256 mismatch")
    if tuple(cfg["view_order"]) != VIEW_NAMES or cfg["output_order"] != ["noise", "blur"]:
        raise RuntimeError("View/score order mismatch")
    sys.path.insert(0, cfg["project_root"])
    from mmreal_synthesis import load_native_components, NativeSynthesizer
    dataset_class, options = load_native_components(cfg["mmreal_root"], cfg["native_options"])
    seed_all(cfg["seed"])
    if args.smoke_only:
        smoke_check(dataset_class, options, NativeSynthesizer, cfg, root, device)
        return
    if not args.resume and (root / "checkpoints/latest.pt").exists():
        raise RuntimeError("Existing training checkpoint: use --resume to continue safely")
    if not args.resume:
        smoke_check(dataset_class, options, NativeSynthesizer, cfg, root, device)
    cache = validation_cache(dataset_class, options, NativeSynthesizer, cfg, root, device)
    seed_all(cfg["seed"])
    model = IDE(num_in_ch=3).to(device)
    model.load_legacy_checkpoint(cfg["warm_start"])
    model.set_train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"])
    synth = NativeSynthesizer(options, device, queue_size=cfg["queue_size"], use_queue=True)
    native_train = make_dataset(dataset_class, options, root / "data_manifest/train.txt")
    dataset = IndexedNativeDataset(native_train, cfg["seed"])
    batches_per_epoch = len(dataset) // cfg["source_batch_size"]
    completed, best_loss, best_step = 0, float("inf"), 0
    active_seconds = 0.0
    provenance = {"config": cfg, "config_sha256": sha256(args.config),
                  "warm_start_sha256": sha256(cfg["warm_start"]),
                  "train_list_sha256": sha256(root / "data_manifest/train.txt"),
                  "val_list_sha256": sha256(root / "data_manifest/val.txt"),
                  "native_options_sha256": sha256(cfg["native_options"]),
                  "native_synthesis_audit": options["_native_mmreal"],
                  "python": sys.version, "torch": str(torch.__version__),
                  "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                  "note": "New independent IDE continuation; original SR/DAPE and weights untouched"}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("config_sha256", "warm_start_sha256", "train_list_sha256", "val_list_sha256", "native_options_sha256", "native_synthesis_audit", "code_sha256"):
            if saved["provenance"][key] != provenance[key]:
                raise RuntimeError(f"Resume provenance mismatch: {key}")
        model.load_new_checkpoint(saved["model"])
        ema.load_new_checkpoint(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        synth.load_queue_state(saved["queue"])
        completed, best_loss, best_step = saved["step"], saved["best_loss"], saved["best_step"]
        active_seconds = saved["active_seconds"]
        restore_rng(saved["rng"])
    else:
        atomic_json(root / "provenance.json", provenance)
        baseline = evaluate(model, cache, device, cfg["source_batch_size"])
        baseline.update({"step": 0, "variant": "legacy_equals_zero_initialized", "selection": False})
        append_json(root / "validation.jsonl", baseline)
        atomic_json(root / "baseline_validation.json", baseline)
        print("Initial held-out validation: " + json.dumps(baseline), flush=True)

    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"Signal {signum}: saving after current iteration", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    began = time.monotonic()
    status = {"state": "running", "pid": os.getpid(), "step": completed, "total_iters": cfg["total_iters"],
              "source_train_images": len(dataset), "internal_val_images": len(cache),
              "best_step": best_step, "best_loss": None if math.isinf(best_loss) else best_loss}

    def save_checkpoint():
        state = {"step": completed, "model": model.state_dict(), "ema": ema.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "queue": synth.queue_state(), "rng": rng_state(), "best_loss": best_loss,
                 "best_step": best_step, "active_seconds": active_seconds + time.monotonic() - began,
                 "provenance": provenance}
        atomic_save(root / "checkpoints/latest.pt", state)
        # Raw tensor dictionaries are directly compatible with load_new_checkpoint.
        atomic_save(root / "checkpoints/latest_raw.pth", model.state_dict())
        atomic_save(root / "checkpoints/latest_ema.pth", ema.state_dict())

    def validate():
        nonlocal best_loss, best_step
        metrics = evaluate(ema, cache, device, cfg["source_batch_size"])
        metrics.update({"step": completed, "variant": "learned_offset_ema", "selection": True})
        append_json(root / "validation.jsonl", metrics)
        if metrics["loss"] < best_loss:
            best_loss, best_step = metrics["loss"], completed
            atomic_save(root / "checkpoints/best_ema.pth", ema.state_dict())
            atomic_json(root / "checkpoints/best_metrics.json", metrics)
        print("Held-out validation: " + json.dumps(metrics), flush=True)

    atomic_json(root / "status.json", status)
    try:
        while completed < cfg["total_iters"] and not stop["requested"]:
            epoch, start_batch = divmod(completed, batches_per_epoch)
            sampler = ResumableBatchSampler(len(dataset), cfg["source_batch_size"], cfg["seed"], epoch, start_batch)
            # DataLoader's bookkeeping uses its own generator and cannot consume synthesis RNG.
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg["num_workers"],
                                pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
            model.train()
            for data in loader:
                step_start = time.monotonic()
                synth.feed_data(data)
                views = five_views(synth)
                optimizer.zero_grad(set_to_none=True)
                scores = model(views.flatten(0, 1)).view(views.shape[0], 5, 2)
                loss, metrics = degree_loss(scores)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at iteration {completed + 1}")
                loss.backward()
                offset_grad = model.blur_branch.offset_predictor.weight.grad
                if offset_grad is None or not torch.isfinite(offset_grad).all():
                    raise RuntimeError(f"Invalid offset gradient at iteration {completed + 1}")
                optimizer.step()
                update_ema(ema, model, cfg["ema_decay"])
                scheduler.step()
                completed += 1
                if completed == 1 or completed % cfg["log_every"] == 0:
                    row = {"step": completed, "loss": float(loss), "lr": optimizer.param_groups[0]["lr"],
                           "seconds_per_step": time.monotonic() - step_start,
                           "elapsed_seconds": active_seconds + time.monotonic() - began,
                           "offset_weight_abs_mean": float(model.blur_branch.offset_predictor.weight.detach().abs().mean()),
                           "offset_grad_abs_mean": float(offset_grad.detach().abs().mean()),
                           **{k: float(v) for k, v in metrics.items()}}
                    append_json(root / "training.jsonl", row)
                    status.update({"step": completed, "last_loss": row["loss"], "elapsed_seconds": row["elapsed_seconds"],
                                   "best_loss": None if math.isinf(best_loss) else best_loss, "best_step": best_step,
                                   "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z")})
                    atomic_json(root / "status.json", status)
                if completed == 1 or completed % cfg["print_every"] == 0:
                    print(f"Iter {completed}/{cfg['total_iters']} loss={float(loss):.8f} "
                          f"offset_grad={float(offset_grad.detach().abs().mean()):.3e}", flush=True)
                if completed % cfg["validation_every"] == 0 or completed == cfg["total_iters"]:
                    validate()
                if completed == 1 or completed % cfg["checkpoint_every"] == 0:
                    save_checkpoint()
                if stop["requested"] or completed >= cfg["total_iters"]:
                    break
        save_checkpoint()
        if completed == cfg["total_iters"]:
            atomic_save(root / "checkpoints/final_raw.pth", model.state_dict())
            atomic_save(root / "checkpoints/final_ema.pth", ema.state_dict())
        status.update({"state": "completed" if completed == cfg["total_iters"] else "stopped_with_checkpoint",
                       "step": completed, "best_step": best_step, "best_loss": None if math.isinf(best_loss) else best_loss,
                       "elapsed_seconds": active_seconds + time.monotonic() - began})
        atomic_json(root / "status.json", status)
    except BaseException as exc:
        status.update({"state": "failed", "step": completed, "error": repr(exc)})
        atomic_json(root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
