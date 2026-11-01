"""Single-process shuffling with restorable permutation and sample cursor.

Use both torchdata sampler classes.  A normal PyTorch BatchSampler hides its
underlying random iterator state from StatefulDataLoader, leaving only a count
that cannot recover a shuffled epoch after a process restart.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader.sampler import BatchSampler, RandomSampler


def make_stateful_train_loader(dataset, *, batch_size: int, seed: int,
                               pin_memory: bool = True) -> DataLoader:
    """Return the CPU loader for Accelerate's use_stateful_dataloader=True.

    This training entry is one process, zero workers, no replacement and no dropped
    examples.  The generator belongs solely to the sampler; model/metric RNG
    use cannot change the sequence of future shuffled epochs.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sampler = RandomSampler(dataset, replacement=False, generator=generator)
    batch_sampler = BatchSampler(sampler, batch_size=int(batch_size), drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0,
                      pin_memory=pin_memory)


def assert_permutation_state(state: dict[str, Any]) -> dict[str, int]:
    """Reject count-only checkpoints and return the saved cursor evidence."""
    batch = state.get("_sampler_iter_state")
    random_state = batch.get("sampler_iter_state") if isinstance(batch, dict) else None
    if not isinstance(random_state, dict):
        raise ValueError("Loader checkpoint lacks the random sampler iterator state")
    generator_state = random_state.get("generator")
    if not torch.is_tensor(generator_state) or generator_state.dtype != torch.uint8:
        raise ValueError("Loader checkpoint lacks its permutation generator state")
    samples = int(batch["samples_yielded"])
    if int(random_state["yielded"]) != samples:
        raise ValueError("Batch and random-sampler cursors disagree")
    batches = int(state["_num_yielded"])
    if int(state["_sampler_iter_yielded"]) != batches:
        raise ValueError("Loader and batch-sampler cursors disagree")
    return {"samples_yielded": samples, "batches_yielded": batches,
            "generator_state_bytes": generator_state.numel()}


def cpu_resume_smoke() -> dict[str, Any]:
    """Verify actual Accelerate prefetch snapshots without initializing CUDA.

    Compare an uninterrupted epoch and its following epoch with process-style
    loader reconstruction at several synchronized cut positions.  Deliberately
    disturb the global RNG before loading to test the sampler's independence.
    """
    from accelerate import Accelerator, DataLoaderConfiguration

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU sampler smoke must start without CUDA initialized")
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(
        use_stateful_dataloader=True))
    dataset = torch.arange(41)

    def build():
        return accelerator.prepare_data_loader(make_stateful_train_loader(
            dataset, batch_size=1, seed=42, pin_memory=False))

    def values(iterator):
        return [int(batch.item()) for batch in iterator]

    reference = build()
    epoch0, epoch1 = values(iter(reference)), values(iter(reference))
    if sorted(epoch0) != list(range(len(dataset))) or sorted(epoch1) != list(range(len(dataset))):
        raise AssertionError("Uninterrupted epoch is not a full permutation")
    evidence = []
    for cut in (4, 20, 40):
        original = build()
        iterator = iter(original)
        prefix = [int(next(iterator).item()) for _ in range(cut)]
        snapshot = copy.deepcopy(original.state_dict())
        cursors = assert_permutation_state(snapshot)
        if cursors["samples_yielded"] != cut or cursors["batches_yielded"] != cut:
            raise AssertionError("Accelerate snapshot includes a prefetched extra batch")
        torch.manual_seed(991 + cut)
        _ = torch.rand(17)
        resumed = build()
        resumed.load_state_dict(snapshot)
        suffix = values(iter(resumed))
        next_epoch = values(iter(resumed))
        if prefix + suffix != epoch0 or next_epoch != epoch1:
            raise AssertionError("Resume duplicated, skipped, or reordered examples")
        evidence.append({"cut": cut, "remaining": len(suffix), **cursors})
    if torch.cuda.is_initialized():
        raise AssertionError("CPU test unexpectedly initialized CUDA")
    return {"passed": True, "dataset_size": len(dataset), "cuts": evidence,
            "next_epoch_exact": True, "cuda_initialized": False}


if __name__ == "__main__":
    import json
    print(json.dumps(cpu_resume_smoke(), indent=2))
