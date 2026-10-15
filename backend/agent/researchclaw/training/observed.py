"""Observe a single-process PyTorch experiment, not an LLM-written report."""

from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import sys
import traceback
from pathlib import Path


def tensor_hash(tensor):
    import torch

    values = tensor.detach().cpu().contiguous()
    if not torch.isfinite(values).all().item():
        raise ValueError("Non-finite model parameter")
    payload = values.reshape(-1).view(dtype=torch.uint8).numpy().tobytes()
    return hashlib.sha256(str((tuple(values.shape), str(values.dtype))).encode() + payload).hexdigest()


def _state_tensors(value):
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_tensors(item)


def _saved_hashes(state):
    hashes = set()
    for value in _state_tensors(state):
        try:
            hashes.add(tensor_hash(value))
        except (ValueError, RuntimeError, TypeError):
            continue  # Non-finite or non-hashable tensors cannot equal a finite trained parameter.
    return hashes


def verify_checkpoints(final_checkpoint, optimizers, evidence):
    """Every trained optimizer must be saved in some outputs/ checkpoint; model.pt must be one of them.

    Independent runs (e.g. one model per seed) each create their own optimizer, while
    outputs/model.pt holds only the final deliverable, so it cannot contain them all.
    """
    import torch

    trained = [{tensor_hash(parameter) for group in optimizer.param_groups
                for parameter in group["params"] if parameter.requires_grad} for optimizer in optimizers]
    trained = [(index, hashes) for index, hashes in enumerate(trained) if hashes]
    state = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise ValueError("outputs/model.pt must contain a model state_dict")
    saved = {tensor_hash(value) for value in state.values() if torch.is_tensor(value)}
    matches = {index: final_checkpoint.as_posix() for index, hashes in trained if hashes.issubset(saved)}
    final_matched = bool(matches)
    del state, saved

    root = final_checkpoint.parent
    others = sorted(path for pattern in ("*.pt", "*.pth") for path in root.rglob(pattern)
                    if path != final_checkpoint and path.is_file() and not path.is_symlink())
    unreadable = []
    for path in others:
        if len(matches) == len(trained):
            break
        try:
            saved = _saved_hashes(torch.load(path, map_location="cpu", weights_only=True))
        except Exception as exc:
            unreadable.append({"path": path.as_posix(), "error": str(exc)[:200]})
            continue
        for index, hashes in trained:
            if index not in matches and hashes.issubset(saved):
                matches[index] = path.as_posix()

    evidence["trained_optimizers"] = len(trained)
    evidence["checkpoint_matches"] = [{"optimizer": index, "parameters": len(hashes), "checkpoint": matches.get(index)}
                                      for index, hashes in trained]
    if unreadable:
        evidence["unreadable_checkpoints"] = unreadable
    evidence["checkpoint_verified"] = bool(trained) and final_matched and len(matches) == len(trained)


def main():
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    evidence_path = Path(sys.argv[1]).resolve()
    evidence = {"optimizer_steps": 0, "parameter_update_observed": False, "cuda_update_observed": False,
                "losses": [], "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "checkpoint_verified": False, "scope": "single_process_pytorch_execution_not_scientific_validation",
                "loss_semantics": "Scalars passed to Tensor.backward; may be AMP-scaled. See experiment logs for unscaled losses."}
    snapshots = {}
    optimizers = {}
    original_backward = torch.Tensor.backward

    def backward(tensor, *args, **kwargs):
        if tensor.numel() == 1:
            loss = float(tensor.detach().item())
            if not math.isfinite(loss):
                raise ValueError("Non-finite training loss")
            evidence["last_loss"] = loss
            if len(evidence["losses"]) < 1000:
                evidence["losses"].append(loss)
        return original_backward(tensor, *args, **kwargs)

    def before_step(optimizer, args, kwargs):
        optimizers[id(optimizer)] = optimizer
        if not evidence["parameter_update_observed"] or (evidence["cuda_visible_devices"] and not evidence["cuda_update_observed"]):
            snapshots[id(optimizer)] = [
                (parameter, tensor_hash(parameter))
                for group in optimizer.param_groups for parameter in group["params"]
                if parameter.grad is not None
            ]

    def after_step(optimizer, args, kwargs):
        evidence["optimizer_steps"] += 1
        for parameter, previous in snapshots.pop(id(optimizer), []):
            if previous != tensor_hash(parameter):
                evidence["parameter_update_observed"] = True
                if parameter.is_cuda:
                    evidence["cuda_update_observed"] = True
        with evidence_path.with_suffix(".jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps({"step": evidence["optimizer_steps"], "loss": evidence.get("last_loss"),
                                     "parameter_update_observed": evidence["parameter_update_observed"]}) + "\n")

    handles = [register_optimizer_step_pre_hook(before_step), register_optimizer_step_post_hook(after_step)]
    torch.Tensor.backward = backward
    exit_code = 0
    try:
        sys.argv = ["main.py"]
        sys.path.insert(0, str(Path.cwd()))
        try:
            runpy.run_path("main.py", run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        checkpoint = Path("outputs/model.pt")
        if checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0:
            verify_checkpoints(checkpoint, list(optimizers.values()), evidence)
            evidence["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if evidence["cuda_update_observed"]:
            from researchclaw.training.jobs import gpu_evidence
            evidence["gpu_process"] = gpu_evidence(evidence["cuda_visible_devices"])
    except BaseException as exc:
        evidence["error"] = str(exc)
        traceback.print_exc()
        exit_code = 1
    finally:
        torch.Tensor.backward = original_backward
        for handle in handles:
            handle.remove()
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
