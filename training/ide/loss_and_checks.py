"""MM-RealSR degree loss and targeted checks for the learned-offset IDE.

Running this file directly executes only fabricated-score tests on the CPU.
Model checks are explicit functions for the trainer's smoke test; the optimizer
step check intentionally performs one update on the supplied model/optimizer.
"""

import json

import torch
import torch.nn.functional as F


VIEW_NAMES = (
    "base", "stronger_blur", "stronger_noise", "minimum_anchor", "maximum_anchor"
)
SCORE_NAMES = ("noise", "blur")
MARGIN = 0.05
RANK_WEIGHT = 0.01
ANCHOR_WEIGHT = 0.0001
# Tuple entries are (lower-severity view, higher-severity view, score column).
RANK_PAIRS = {"noise": (0, 2, 0), "blur": (0, 1, 1)}


def degree_loss(scores):
    """Return weighted loss and detached diagnostics for scores[B, 5, 2].

    Noise ranking compares base/noise-stronger; blur ranking compares
    base/blur-stronger.  No ordering is imposed on the other score column of
    either stronger view.  The anchor loss is the SUM of four batch-mean MSEs:
    each head at minimum(target 0) and maximum(target 1).  ``anchor_rmse``
    averages the four MSEs before its square root, so sqrt(anchor / 4).
    """
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a Tensor")
    if scores.ndim != 3 or scores.shape[1:] != (5, 2) or scores.shape[0] == 0:
        raise ValueError(f"Expected nonempty scores[B,5,2], got {tuple(scores.shape)}")
    if not scores.is_floating_point():
        raise TypeError("scores must have floating-point dtype")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores contain nonfinite values")

    noise_gap = scores[:, 2, 0] - scores[:, 0, 0]
    blur_gap = scores[:, 1, 1] - scores[:, 0, 1]
    ranking_target = -torch.ones_like(scores[:, 0, 0])
    rank_noise = F.margin_ranking_loss(
        scores[:, 0, 0], scores[:, 2, 0], ranking_target, margin=MARGIN
    )
    rank_blur = F.margin_ranking_loss(
        scores[:, 0, 1], scores[:, 1, 1], ranking_target, margin=MARGIN
    )
    min_noise_mse = scores[:, 3, 0].square().mean()
    min_blur_mse = scores[:, 3, 1].square().mean()
    max_noise_mse = (scores[:, 4, 0] - 1).square().mean()
    max_blur_mse = (scores[:, 4, 1] - 1).square().mean()
    anchor = min_noise_mse + min_blur_mse + max_noise_mse + max_blur_mse
    loss = RANK_WEIGHT * (rank_noise + rank_blur) + ANCHOR_WEIGHT * anchor
    metrics = {
        "rank_noise": rank_noise,
        "rank_blur": rank_blur,
        "anchor": anchor,
        "strict_pairacc_noise": (noise_gap > 0).float().mean(),
        "strict_pairacc_blur": (blur_gap > 0).float().mean(),
        "margin_pairacc_noise": (noise_gap >= MARGIN).float().mean(),
        "margin_pairacc_blur": (blur_gap >= MARGIN).float().mean(),
        "anchor_rmse": (anchor / 4).sqrt(),
        "anchor_rmse_noise": ((min_noise_mse + max_noise_mse) / 2).sqrt(),
        "anchor_rmse_blur": ((min_blur_mse + max_blur_mse) / 2).sqrt(),
    }
    return loss, {name: value.detach() for name, value in metrics.items()}


def check_initialization_matches_legacy(model, legacy_checkpoint, images):
    """Assert old tensors match and zero-offset predictions match the old IDE.

    Call before any update, after ``model.load_legacy_checkpoint``.  ``images``
    is an FP32 RGB tensor [B,3,H,W] on the model's device.  No optimizer step is
    performed, and the supplied model's train/eval state is restored.
    """
    from ide_arch import IDEBase, OFFSET_STATE_KEYS, _read_tensor_state_dict

    if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
        raise ValueError("Initialization check requires nonempty RGB images[B,3,H,W]")
    if images.dtype != torch.float32:
        raise ValueError("Initialization check uses FP32, matching the IDE protocol")
    state = _read_tensor_state_dict(legacy_checkpoint)
    current = model.state_dict()
    if set(current) != set(state) | set(OFFSET_STATE_KEYS):
        raise AssertionError("The learned model differs from legacy by more than offset keys")
    for name, value in state.items():
        target = current[name]
        if not torch.equal(target.detach().cpu(), value.detach().cpu()):
            raise AssertionError(f"Legacy tensor was not preserved: {name}")
    predictor = model.blur_branch.offset_predictor
    for name, parameter in predictor.named_parameters():
        if not parameter.requires_grad:
            raise AssertionError(f"Offset parameter is frozen: {name}")
        if not bool(torch.isfinite(parameter).all()) or bool(torch.count_nonzero(parameter)):
            raise AssertionError(f"Offset parameter is not exactly initialized to zero: {name}")

    reference = IDEBase(num_in_ch=3, num_degradation=2).to(
        device=images.device, dtype=images.dtype
    ).eval()
    reference.load_state_dict(state, strict=True)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            legacy_scores = reference(images)
            new_scores = model(images)
        if not bool(torch.isfinite(new_scores).all()):
            raise AssertionError("New model produced nonfinite scores at initialization")
        torch.testing.assert_close(new_scores, legacy_scores, atol=0, rtol=0)
    finally:
        model.train(was_training)
    return {
        "legacy_tensors_preserved": True,
        "offset_parameters_exactly_zero": True,
        "initial_scores_bitwise_equal": torch.equal(new_scores, legacy_scores),
        "initial_score_maxabs": float((new_scores - legacy_scores).abs().max()),
        "legacy_parameters": sum(value.numel() for value in state.values()),
        "learned_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def check_optimizer_step(model, optimizer, views):
    """Perform one explicit smoke update and verify offsets actually learn.

    ``views`` follows VIEW_NAMES and has FP32 shape [B,5,3,H,W] in [0,1].
    This function mutates the supplied model and optimizer by ONE step; use
    dedicated smoke instances, or account for that step in training history.
    """
    if views.ndim != 5 or views.shape[0] == 0 or views.shape[1:3] != (5, 3):
        raise ValueError("Expected nonempty RGB views[B,5,3,H,W]")
    if views.dtype != torch.float32 or not bool(torch.isfinite(views).all()):
        raise ValueError("Smoke views must be finite FP32 tensors")
    if bool((views < 0).any()) or bool((views > 1).any()):
        raise ValueError("Smoke views must be in [0,1]")
    predictor = model.blur_branch.offset_predictor
    before = {name: parameter.detach().clone()
              for name, parameter in predictor.named_parameters()}
    selected = {id(parameter) for group in optimizer.param_groups
                for parameter in group["params"]}
    for name, parameter in predictor.named_parameters():
        if not parameter.requires_grad or id(parameter) not in selected:
            raise AssertionError(f"Offset parameter is not selected for training: {name}")

    optimizer.zero_grad(set_to_none=True)
    scores = model(views.flatten(0, 1)).reshape(views.shape[0], 5, 2)
    loss, metrics = degree_loss(scores)
    if not bool(torch.isfinite(loss)):
        raise AssertionError("Smoke loss is nonfinite")
    loss.backward()
    gradient_report = {}
    for name, parameter in predictor.named_parameters():
        grad = parameter.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"Missing/nonfinite offset gradient: {name}")
        gradient_report[name + "_grad_nonzero"] = int(torch.count_nonzero(grad))
        gradient_report[name + "_grad_maxabs"] = float(grad.abs().max())
    if not sum(gradient_report[name + "_grad_nonzero"] for name in before):
        raise AssertionError("All offset gradients are zero")
    optimizer.step()

    update_report = {}
    for name, parameter in predictor.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise AssertionError(f"Nonfinite updated offset parameter: {name}")
        update_report[name + "_changed"] = not torch.equal(before[name], parameter.detach())
        update_report[name + "_nonzero"] = int(torch.count_nonzero(parameter))
    if not any(update_report[name + "_changed"] for name in before):
        raise AssertionError("Optimizer did not update the offset parameters")
    if not sum(update_report[name + "_nonzero"] for name in before):
        raise AssertionError("Updated offset parameters are still all zero")

    # Observe the tensor actually passed by the real forward path, rather than
    # assuming nonzero parameters necessarily imply nonzero sampled offsets.
    observed = []
    handle = predictor.register_forward_hook(
        lambda module, inputs, output: observed.append(output.detach())
    )
    try:
        with torch.no_grad():
            after_scores = model(views.flatten(0, 1))
    finally:
        handle.remove()
    if len(observed) != 1:
        raise AssertionError(f"Expected one offset-predictor call, observed {len(observed)}")
    offset = observed[0]
    if not bool(torch.isfinite(offset).all()) or not bool(torch.count_nonzero(offset)):
        raise AssertionError("Updated forward produced nonfinite/all-zero offsets")
    if not bool(torch.isfinite(after_scores).all()):
        raise AssertionError("Updated forward produced nonfinite degree scores")

    return {
        "optimizer_steps_performed": 1,
        "loss": float(loss.detach()),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "offset_gradients": gradient_report,
        "offset_update": update_report,
        "updated_offset_tensor": {
            "shape": list(offset.shape),
            "nonzero_count": int(torch.count_nonzero(offset)),
            "abs_mean": float(offset.abs().mean()),
            "abs_max": float(offset.abs().max()),
        },
    }


def run_cpu_loss_tests():
    """Check view/column mapping, reductions, loss weights and gradients."""
    good = torch.tensor([
        [0.2, 0.3], [0.2, 0.6], [0.8, 0.3], [0.0, 0.0], [1.0, 1.0]
    ], dtype=torch.float64).unsqueeze(0)

    def close(actual, expected):
        expected = torch.as_tensor(expected, device="cpu", dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)

    loss, metrics = degree_loss(good)
    close(loss, 0.0)
    for name in ("rank_noise", "rank_blur", "anchor", "anchor_rmse"):
        close(metrics[name], 0.0)
    for name in ("strict_pairacc_noise", "strict_pairacc_blur",
                 "margin_pairacc_noise", "margin_pairacc_blur"):
        close(metrics[name], 1.0)

    # Neither the opposite score column nor the wrong stronger view may be used.
    for wrong in (good[:, :, [1, 0]], good[:, [0, 2, 1, 3, 4], :]):
        wrong_loss, wrong_metrics = degree_loss(wrong)
        close(wrong_loss, 0.001)
        close(wrong_metrics["rank_noise"], 0.05)
        close(wrong_metrics["rank_blur"], 0.05)
        close(wrong_metrics["strict_pairacc_noise"], 0.0)
        close(wrong_metrics["strict_pairacc_blur"], 0.0)

    bad = good.clone()
    bad[0, 2, 0] = 0.1
    bad[0, 1, 1] = 0.25
    bad[0, 3, :] = torch.tensor([0.2, 0.4], dtype=torch.float64)
    bad[0, 4, :] = torch.tensor([0.6, 0.8], dtype=torch.float64)
    bad.requires_grad_(True)
    loss, metrics = degree_loss(bad)
    close(metrics["rank_noise"], 0.15)
    close(metrics["rank_blur"], 0.10)
    close(metrics["anchor"], 0.40)
    close(metrics["anchor_rmse"], 0.1 ** 0.5)
    close(loss, 0.00254)
    assert loss.requires_grad and all(not value.requires_grad for value in metrics.values())
    loss.backward()
    expected_grad = torch.zeros_like(bad)
    expected_grad[0, 0, :] = 0.01
    expected_grad[0, 2, 0] = -0.01
    expected_grad[0, 1, 1] = -0.01
    expected_grad[0, 3, :] = torch.tensor([0.00004, 0.00008], dtype=torch.float64)
    expected_grad[0, 4, :] = torch.tensor([-0.00008, -0.00004], dtype=torch.float64)
    close(bad.grad, expected_grad)

    batch_loss, batch_metrics = degree_loss(torch.cat([good, bad.detach()], dim=0))
    close(batch_loss, 0.00127)
    close(batch_metrics["anchor"], 0.2)
    close(batch_metrics["strict_pairacc_noise"], 0.5)
    close(batch_metrics["margin_pairacc_blur"], 0.5)

    boundary = good.clone()
    boundary[0, 0, :] = 0
    boundary[0, 2, 0] = MARGIN
    boundary[0, 1, 1] = 0
    _, boundary_metrics = degree_loss(boundary)
    close(boundary_metrics["margin_pairacc_noise"], 1.0)
    close(boundary_metrics["strict_pairacc_blur"], 0.0)
    close(boundary_metrics["margin_pairacc_blur"], 0.0)

    for invalid in (torch.zeros(1, 2, 5), torch.zeros(0, 5, 2),
                    torch.zeros(1, 5, 2, dtype=torch.int64), good * float("nan")):
        try:
            degree_loss(invalid)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid score tensor accepted")
    return {"device": "cpu", "fabricated_loss_tests": "passed",
            "view_order": list(VIEW_NAMES), "score_order": list(SCORE_NAMES),
            "margin": MARGIN, "rank_weight": RANK_WEIGHT,
            "anchor_weight": ANCHOR_WEIGHT,
            "checks": ["view_and_column_mapping", "four_anchor_terms", "batch_mean",
                       "strict_vs_margin_accuracy", "detached_metrics", "exact_gradients",
                       "invalid_scores_rejected"]}


if __name__ == "__main__":
    print(json.dumps(run_cpu_loss_tests(), indent=2))
