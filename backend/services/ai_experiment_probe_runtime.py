"""Measure a proposed mechanism on ordinary training samples before the full budget."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

_report = None
_step = 0
_last = 0
_slots = ()
_values = {}
_samples = {}
_invalid = False
_THRESHOLD = 1e-4


def configure(report, *, start_step, pilot_steps, changed_slots):
    global _report, _last, _slots
    report = Path(report)
    if report.is_file() and json.loads(report.read_text()).get("passed") is True:
        return
    _report = report
    _last = start_step + pilot_steps
    _slots = tuple(changed_slots)


def begin(step):
    global _step
    _step = step


def active():
    return _report is not None and 0 < _step <= _last


def observe(name, current, reference):
    global _invalid
    if not active() or (name != "control" and name not in _slots):
        return
    with torch.no_grad():
        current, reference = (torch.as_tensor(x).detach().float() for x in (current, reference))
        current, reference = torch.broadcast_tensors(current, reference)
        stride = max(1, current.numel() // 4096)
        current = current.reshape(-1)[::stride]
        reference = reference.reshape(-1)[::stride]
        difference = (current-reference).abs().mean()
        scale = torch.maximum(current.abs().mean(), reference.abs().mean()).clamp_min(1e-6)
        relative = float(difference/scale)
        absolute = float(difference)
    if not all(map(math.isfinite,(relative, absolute))):
        _invalid = True
        return
    entry = _values.setdefault(name, [0., 0.])
    entry[0] = max(entry[0], relative)
    entry[1] = max(entry[1], absolute)
    _samples[name] = _samples.get(name, 0)+1


def finish(step, device):
    global _report
    if not active() or step != _last:
        return
    names = (*_slots, "control")
    stats = torch.tensor([_values.get(name, [0., 0.]) for name in names], device=device, dtype=torch.float64)
    samples = torch.tensor([_samples.get(name, 0) for name in names], device=device, dtype=torch.int64)
    invalid = torch.tensor(int(_invalid), device=device, dtype=torch.int64)
    if dist.is_initialized():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        dist.all_reduce(samples, op=dist.ReduceOp.SUM)
    rows = {name:{"max_relative_l1":r,"max_absolute_l1":a,"observations":n}
            for name,(r,a),n in zip(names,stats.tolist(),samples.tolist())}
    changed = all(rows[name]["max_relative_l1"] > _THRESHOLD and rows[name]["max_absolute_l1"] > 1e-8
                  for name in _slots)
    separable = rows["control"]["max_relative_l1"] > _THRESHOLD and rows["control"]["max_absolute_l1"] > 1e-8
    passed = changed and separable and not invalid.item()
    if not dist.is_initialized() or dist.get_rank() == 0:
        output = {"passed":passed,"step":step,"finish_before_optimizer_step":True,
                  "changed_signal":changed,"control_separable":separable,"nonfinite_signal":bool(invalid.item()),"metrics":rows,
                  "relative_l1_threshold":_THRESHOLD}
        temp = _report.with_suffix(".tmp")
        temp.write_text(json.dumps(output, indent=2, allow_nan=False))
        os.replace(temp,_report)
    if dist.is_initialized():
        dist.barrier()
    _report = None
    if not passed:
        raise RuntimeError("MechanismCheckFailed: real training samples did not separate the proposed formula or its control")
