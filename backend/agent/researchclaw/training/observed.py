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
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                state = state.get("model_state_dict", state.get("state_dict", state))
            if not isinstance(state, dict):
                raise ValueError("outputs/model.pt must contain a model state_dict")
            saved = {tensor_hash(value) for value in state.values() if torch.is_tensor(value)}
            trained = {tensor_hash(parameter) for optimizer in optimizers.values()
                       for group in optimizer.param_groups for parameter in group["params"] if parameter.requires_grad}
            evidence["checkpoint_verified"] = bool(trained) and trained.issubset(saved)
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
