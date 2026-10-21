"""Observe a PyTorch experiment process, not an LLM-written report.

Multi-GPU runs are launched by torchrun with one copy of this wrapper per GPU (DDP). Each
rank records its own evidence; rank 0 owns outputs/model.pt and its checkpoint check."""

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


def verify_checkpoints(final_checkpoint, optimizers, evidence, saved_live=()):
    """Every trained optimizer must be saved in some outputs/ checkpoint; model.pt must be one of them.

    Independent runs (e.g. one model per seed) each create their own optimizer, while
    outputs/model.pt holds only the final deliverable, so it cannot contain them all.
    ``saved_live`` (aligned with ``optimizers``) lists the files torch.save wrote straight
    from that model's parameters after its last optimizer step and nobody changed since.
    Those count as saved even though the parameters changed afterwards: scripts reload the
    best-validation checkpoint for testing, and layers such as BayarConv2d renormalise their
    weights in place on every forward pass.
    """
    import torch

    root = final_checkpoint.parent.resolve()
    saved_live = dict(enumerate(saved_live))
    trained = [[(tensor_hash(parameter), parameter) for group in optimizer.param_groups
                for parameter in group["params"] if parameter.requires_grad] for optimizer in optimizers]
    trained = [(index, parameters) for index, parameters in enumerate(trained) if parameters]
    matches = {}
    for index, _ in trained:
        live = sorted(path for path in saved_live.get(index) or () if path.is_relative_to(root))
        if final_checkpoint.resolve() in live:
            matches[index] = (final_checkpoint.as_posix(), "saved_from_trained_model", 0)
        elif live:
            matches[index] = (final_checkpoint.parent.joinpath(live[0].relative_to(root)).as_posix(),
                              "saved_from_trained_model", 0)
    final_matched = any(match[0] == final_checkpoint.as_posix() for match in matches.values())

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise ValueError("outputs/model.pt must contain a model state_dict")
    if not final_matched:
        saved = _saved_tensors({key: value for key, value in state.items() if torch.is_tensor(value)})
        for index, parameters in trained:
            if index not in matches:
                count = _match(parameters, saved)
                if count is not None:
                    matches[index] = (final_checkpoint.as_posix(), "process_exit", count)
                    final_matched = True
        del saved
    del state

    others = sorted(path for pattern in ("*.pt", "*.pth") for path in final_checkpoint.parent.rglob(pattern)
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
        for index, parameters in trained:
            if index not in matches:
                count = _match(parameters, saved)
                if count is not None:
                    matches[index] = (path.as_posix(), "process_exit", count)
        del saved

    evidence["trained_optimizers"] = len(trained)
    evidence["checkpoint_matches"] = [{"optimizer": index, "parameters": len(parameters),
                                       "checkpoint": matches.get(index, (None,))[0],
                                       "state": matches.get(index, (None, None))[1],
                                       "numeric_matches": matches.get(index, (None, None, 0))[2]}
                                      for index, parameters in trained]
    if any(match[2] for match in matches.values()):
        evidence["checkpoint_numeric_tolerance"] = {"rtol": NUMERIC_RTOL, "atol": NUMERIC_ATOL}
    if unreadable:
        evidence["unreadable_checkpoints"] = unreadable
    missing = [index for index, _ in trained if index not in matches]
    if not trained:
        evidence["checkpoint_problem"] = "no optimizer updated any trainable parameter"
    elif not final_matched:
        evidence["checkpoint_problem"] = (
            "outputs/model.pt matches none of the trained models: it was neither written by torch.save from "
            "a trained model after its last optimizer step nor equal to a model's parameters at process exit")
    elif missing:
        evidence["checkpoint_problem"] = (
            f"trained model(s) {missing} (in order of creation) are not saved in any outputs/*.pt checkpoint; "
            "each independently trained model, e.g. one per seed, needs its own checkpoint")
    evidence["checkpoint_verified"] = bool(trained) and final_matched and not missing


def rank_evidence_path(evidence_path, rank):
    """Rank 0 writes the main evidence file; rank r > 0 writes <stem>.rank<r>.json next to it."""
    evidence_path = Path(evidence_path)
    return evidence_path if rank == 0 else evidence_path.with_name(f"{evidence_path.stem}.rank{rank}{evidence_path.suffix}")


def main():
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item for item in visible.split(",") if item]
    evidence_path = rank_evidence_path(Path(sys.argv[1]).resolve(), rank)
    evidence = {"optimizer_steps": 0, "parameter_update_observed": False, "cuda_update_observed": False,
                "losses": [], "pid": os.getpid(), "cuda_visible_devices": visible,
                "rank": rank, "world_size": world_size,
                "gpu_uuid": devices[local_rank] if world_size > 1 and local_rank < len(devices) else visible,
                "checkpoint_verified": False, "scope": "pytorch_execution_not_scientific_validation",
                "loss_semantics": "Scalars passed to Tensor.backward; may be AMP-scaled. See experiment logs for unscaled losses."}
    snapshots = {}
    optimizers = {}
    steps = {}
    saves = {}
    original_backward = torch.Tensor.backward
    original_save = torch.save

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
        steps[id(optimizer)] = steps.get(id(optimizer), 0) + 1

    def save(obj, f, *args, **kwargs):
        """torch.save, also recording which trained models the file is written from directly."""
        pointers = {(tensor.data_ptr(), tuple(tensor.shape)) for tensor in _state_tensors(obj) if tensor.numel()}
        live = {}
        for key, optimizer in optimizers.items():
            parameters = [parameter for group in optimizer.param_groups
                          for parameter in group["params"] if parameter.requires_grad and parameter.numel()]
            if parameters and all((parameter.data_ptr(), tuple(parameter.shape)) in pointers
                                  for parameter in parameters):
                live[key] = steps.get(key, 0)
        result = original_save(obj, f, *args, **kwargs)
        if isinstance(f, (str, os.PathLike)):
            path = Path(f).resolve()
            stat = path.stat()
            saves[path] = {"stat": (stat.st_size, stat.st_mtime_ns), "models": live}
        return result

    def saved_live(key):
        """Files last written by torch.save from this model after its last step, unchanged since."""
        found = []
        for path, record in saves.items():
            if record["models"].get(key) == steps.get(key) and path.is_file():
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) == record["stat"]:
                    found.append(path)
        return found

    handles = [register_optimizer_step_pre_hook(before_step), register_optimizer_step_post_hook(after_step)]
    torch.Tensor.backward = backward
    torch.save = save
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
        if rank != 0:
            evidence["checkpoint_verified"] = None  # rank 0 saves and verifies the shared model
        elif checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0:
            verify_checkpoints(checkpoint, list(optimizers.values()), evidence,
                               [saved_live(key) for key in optimizers])
            evidence["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if evidence["cuda_update_observed"]:
            from researchclaw.training.jobs import gpu_evidence
            evidence["gpu_process"] = gpu_evidence(evidence["gpu_uuid"])
    except BaseException as exc:
        evidence["error"] = str(exc)
        traceback.print_exc()
        exit_code = 1
    finally:
        torch.Tensor.backward = original_backward
        torch.save = original_save
        for handle in handles:
            handle.remove()
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
