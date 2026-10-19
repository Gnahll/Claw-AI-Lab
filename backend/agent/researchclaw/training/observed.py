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


# Some layers (e.g. BayarConv2d) renormalise their own weights in place on every forward
# pass, so evaluating a model after loading its checkpoint shifts a few tensors by float
# rounding. Such a tensor may match its saved copy numerically instead of bit-for-bit.
NUMERIC_RTOL = 1e-5
NUMERIC_ATOL = 1e-8


def _saved_tensors(state):
    hashes, tensors = set(), {}
    for value in _state_tensors(state):
        try:
            hashes.add(tensor_hash(value))
        except (ValueError, RuntimeError, TypeError):
            continue  # Non-finite or non-hashable tensors cannot equal a finite trained parameter.
        tensors.setdefault((tuple(value.shape), value.dtype), []).append(value)
    return hashes, tensors


def _match(parameters, saved):
    """Return the number of numerically matched tensors, or None when the file does not hold them.

    All but max(1, 1%) of the tensors must match exactly; the rest within NUMERIC_RTOL/ATOL.
    """
    import torch

    hashes, tensors = saved
    missing = [parameter for digest, parameter in parameters if digest not in hashes]
    if len(missing) > max(1, len(parameters) // 100):
        return None
    for parameter in missing:
        values = parameter.detach().cpu()
        if not any(torch.allclose(values, candidate, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
                   for candidate in tensors.get((tuple(values.shape), values.dtype), [])):
            return None
    return len(missing)


def _find(states, saved):
    """Return (state name, numeric matches) of the first trained state the file holds, or None."""
    for name, parameters in states.items():
        count = _match(parameters, saved)
        if count is not None:
            return name, count
    return None


def verify_checkpoints(final_checkpoint, optimizers, evidence, last_steps=()):
    """Every trained optimizer must be saved in some outputs/ checkpoint; model.pt must be one of them.

    Independent runs (e.g. one model per seed) each create their own optimizer, while
    outputs/model.pt holds only the final deliverable, so it cannot contain them all.
    A model counts as saved in the state right after its last optimizer step
    (``last_steps``, aligned with ``optimizers``) or at process exit: scripts often save
    the final model and then reload the best-validation checkpoint for testing.
    """
    import torch

    last_steps = dict(enumerate(last_steps))
    trained = []
    for index, optimizer in enumerate(optimizers):
        parameters = [parameter for group in optimizer.param_groups
                      for parameter in group["params"] if parameter.requires_grad]
        if not parameters:
            continue
        states = {"process_exit": [(tensor_hash(parameter), parameter) for parameter in parameters]}
        snapshot = last_steps.get(index)
        if snapshot is not None and len(snapshot) == len(parameters):
            hashed = [(tensor_hash(tensor), tensor) for tensor in snapshot]
            if [digest for digest, _ in hashed] != [digest for digest, _ in states["process_exit"]]:
                states["last_optimizer_step"] = hashed
        trained.append((index, states))
    state = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise ValueError("outputs/model.pt must contain a model state_dict")
    saved = _saved_tensors({key: value for key, value in state.items() if torch.is_tensor(value)})
    matches = {}
    for index, states in trained:
        found = _find(states, saved)
        if found is not None:
            matches[index] = (final_checkpoint.as_posix(), *found)
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
            saved = _saved_tensors(torch.load(path, map_location="cpu", weights_only=True))
        except Exception as exc:
            unreadable.append({"path": path.as_posix(), "error": str(exc)[:200]})
            continue
        for index, states in trained:
            if index not in matches:
                found = _find(states, saved)
                if found is not None:
                    matches[index] = (path.as_posix(), *found)
        del saved

    evidence["trained_optimizers"] = len(trained)
    evidence["checkpoint_matches"] = [{"optimizer": index, "parameters": len(states["process_exit"]),
                                       "checkpoint": matches.get(index, (None,))[0],
                                       "state": matches.get(index, (None, None))[1],
                                       "numeric_matches": matches.get(index, (None, None, 0))[2]}
                                      for index, states in trained]
    if any(match[2] for match in matches.values()):
        evidence["checkpoint_numeric_tolerance"] = {"rtol": NUMERIC_RTOL, "atol": NUMERIC_ATOL}
    if unreadable:
        evidence["unreadable_checkpoints"] = unreadable
    missing = [index for index, _ in trained if index not in matches]
    if not trained:
        evidence["checkpoint_problem"] = "no optimizer updated any trainable parameter"
    elif not final_matched:
        evidence["checkpoint_problem"] = (
            "outputs/model.pt matches none of the trained models, neither right after its last optimizer "
            "step nor at process exit; save model.pt from the trained weights of the delivered model")
    elif missing:
        evidence["checkpoint_problem"] = (
            f"trained model(s) {missing} (in order of creation) are not saved in any outputs/*.pt checkpoint; "
            "each independently trained model, e.g. one per seed, needs its own checkpoint")
    evidence["checkpoint_verified"] = bool(trained) and final_matched and not missing


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
    last_steps = {}
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
        remember_last_step(optimizer)

    def remember_last_step(optimizer):
        """Keep each optimizer's trainable parameters as they are right after its latest step."""
        parameters = [parameter for group in optimizer.param_groups
                      for parameter in group["params"] if parameter.requires_grad]
        snapshot = last_steps.get(id(optimizer))
        if snapshot is None:
            # A new optimizer usually means the previous model (e.g. the previous seed) is done;
            # move earlier copies off the GPU so at most one extra copy of each live model stays there.
            for other in last_steps.values():
                other[:] = [tensor.cpu() for tensor in other]
        if snapshot is None or len(snapshot) != len(parameters) or any(
                kept.device != parameter.device or kept.shape != parameter.shape
                for kept, parameter in zip(snapshot, parameters)):
            last_steps[id(optimizer)] = [parameter.detach().clone() for parameter in parameters]
        else:
            with torch.no_grad():
                for kept, parameter in zip(snapshot, parameters):
                    kept.copy_(parameter)

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
            verify_checkpoints(checkpoint, list(optimizers.values()), evidence,
                               [last_steps.get(key) for key in optimizers])
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
