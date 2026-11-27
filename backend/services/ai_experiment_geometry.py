"""AI-authored formula insertion points for OSGK and spectral fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from ai_experiment_cirb import validate_expression

OSGK = "osgk_depth_order"
SPECTRAL = "gsf_spectral"

A_SLOTS = {
    "neighbor_affinity": ("common = similarity - SHAPE_STRENGTH * penalty * valid_pair",
                          "similarity - 2.0 * penalty * valid_pair", "similarity penalty valid_pair distance",
                          "Affinity of the RGB center and depth neighbor, with geometric shape consistency."),
    "ordinal_bias": ("ordinal = (0.5 * distance).clamp(-3.0, 3.0)",
                     "(0.5 * distance).clamp(-3.0, 3.0)", "distance similarity penalty valid_pair",
                     "Signed near/far weighting before the masked softmax; retain the sign of depth disparity."),
    "near_far_context": ("context = (near - far) * both_sides", "near - far", "near far r",
                        "Feature relation between near/far geometry and the instance stream."),
    "residual_gain": ("added=(instance.float()+delta*torch.minimum(torch.ones_like(rms),cap/rms)).to(instance.dtype)",
                     "torch.minimum(cap/rms, (rms/rms))", "delta rms cap",
                     "Spatial gain of the instance residual, capped by the original safe RMS limit."),
}
G_SLOTS = {
    "graph_detail": ("detail=weighted-unweighted", "weighted-unweighted", "shared weighted unweighted r z",
                     "Signal sent from graph diffusion to the zero-initialized readout. Preserve all 64 channels, typically using shared, weighted and unweighted."),
    "residual_gain": ("coefficient=torch.rsqrt(1.+relative_squared)",
                     "1.0/(1.0+relative_squared).sqrt()", "relative_squared update cap",
                     "Spatial gain of the spectral residual, capped to preserve the frozen base. relative_squared is squared update norm divided by squared cap."),
}


def slots_for(family):
    slots = {OSGK:A_SLOTS, SPECTRAL:G_SLOTS}[family]
    fields = ("line", "default", "variables", "meaning")
    return {name: dict(zip(fields, row)) for name, row in slots.items()}


def verify_numerics(family, formulas):
    import torch
    grid = torch.linspace(-1.3, 1.3, 24).reshape(2, 3, 4)
    positive = grid.abs() + .1
    grad = grid.clone().requires_grad_(True)
    if family == OSGK:
        inputs = {
            "neighbor_affinity": {"similarity":grad, "penalty":positive, "valid_pair":grid>.1, "distance":grid},
            "ordinal_bias": {"distance":grid, "similarity":grad, "penalty":positive, "valid_pair":grid>.1},
            "near_far_context": {"near":grad, "far":.33*grid, "r":grid},
            "residual_gain": {"delta":grad, "rms":positive, "cap":positive*.05},
        }
        expected = grid.shape
        require_grad = {"near_far_context"}
    else:
        inputs = {
            "graph_detail": {"shared":grid, "weighted":grid*.9, "unweighted":grid*.7,
                             "r":grid[:,:1], "z":grid[:,1:]},
            "residual_gain": {"relative_squared":positive,"update":grad,"cap":positive*.05},
        }
        expected = grid.shape
        require_grad = set()
    slots = slots_for(family)
    for slot, formula in formulas.items():
        checked = validate_expression(slot, formula, slots)
        result = eval(compile(checked, "<checked-geometry-expression>", "eval"),
                      {"__builtins__": {}, "torch": torch}, inputs[slot])
        if (not isinstance(result, torch.Tensor) or result.shape != expected
                or not result.is_floating_point() or not result.isfinite().all()):
            raise ValueError(f"{slot} has invalid output shape or values")
        if slot == "ordinal_bias" and torch.any(result * grid < 0):
            raise ValueError("Ordinal bias reversed the marked near/far directions")
        if slot in require_grad:
            vector, = torch.autograd.grad(result.sum(), grad, allow_unused=True)
            if vector is None or not vector.isfinite().all() or not vector.abs().any():
                raise ValueError(f"{slot} lost the learning signal")


def _edit(family, file, formulas, profile):
    src = file.read_text(encoding="utf-8")
    slots = slots_for(family)
    replacements = (
        {
            "neighbor_affinity": lambda x: f"common = ({x}).clamp(-12., 12.)",
            "ordinal_bias": lambda x: f"ordinal = ({x}).clamp(-3., 3.)",
            "near_far_context": lambda x: f"context = ({x}) * both_sides",
            "residual_gain": lambda x: f"added=(instance.float()+delta*torch.minimum(({x}).clamp(0.,2.),cap/rms)).to(instance.dtype)",
        } if family == OSGK else {
            "graph_detail": lambda x: f"detail=({x})",
            "residual_gain": lambda x: f"coefficient=torch.minimum(({x}).clamp(0.,1.), (1.+relative_squared).sqrt().reciprocal())",
        }
    )
    for name, formula in formulas.items():
        old = slots[name]["line"]
        if src.count(old) != 1:
            raise ValueError(f"Unexpected source for {profile}: {name}")
        expr = validate_expression(name, formula, slots)
        src = src.replace(old, replacements[name](expr))
    compile(src, str(file), "exec")
    file.write_text(src, encoding="utf-8")


def _digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _apply_osgk(source, staged, formulas):
    shutil.copytree(source / "experiment", staged / "experiment",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source / "dispatch.py", staged / "dispatch.py")
    shutil.copy2(source / "setup.json", staged / "setup.json")
    method = staged / "experiment" / "osgf.py"
    _edit(OSGK, method, formulas, "depth order")
    return {"experiment/osgf.py":_digest(method)}


def _apply_spectral(source, staged, formulas, revision):
    setup = json.loads((source / "setup.json").read_text(encoding="utf-8"))
    hashes = {}
    if (source / "model").is_dir():
        (staged / "model").symlink_to((source / "model").resolve(), target_is_directory=True)
    new_name = setup["method_version"] + "_ai_" + revision
    protocol = hashlib.sha256((setup["protocol_digest"] + revision).encode()).hexdigest()
    for name in setup["copied_source_sha256"]:
        origin = source / name
        new = staged / name
        new.parent.mkdir(parents=True, exist_ok=True)
        if origin.name == "plan.json":
            plan = json.loads(origin.read_text(encoding="utf-8"))
            plan.update(method_version=new_name, protocol_digest=protocol)
            new.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            shutil.copy2(origin, new)
            if new.name == "binding.py":
                _edit(SPECTRAL, new, formulas, new.parent.name)
        hashes[name] = _digest(new)
    setup.update(method_version=new_name, protocol_digest=protocol, copied_source_sha256=hashes)
    (staged / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(source / "dispatch.py", staged / "dispatch.py")
    f = staged / "supervisor.py"
    monitor = (source / "supervisor.py").read_text(encoding="utf-8")
    original = 'assert state==[f"{uid}, 0" for uid in setup["gpu_uuids"]], state'
    if monitor.count(original) != 1:
        raise ValueError("Unexpected spectral resource guard")
    monitor = monitor.replace(original, '''rows=[line.split(", ") for line in state]
            assert [uid for uid,_ in rows]==setup["gpu_uuids"] and all(int(mem)<=16 for _,mem in rows), state
            active=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
            assert not set(setup["gpu_uuids"]) & {u.strip() for u in active.splitlines()}, "GPU busy"''')
    compile(monitor, str(f), "exec")
    f.write_text(monitor, encoding="utf-8")
    return {name:digest for name,digest in hashes.items() if name.endswith("/binding.py")}


def prepare(source, destination, family, revision, decision):
    if destination.exists():
        marker = json.loads((destination / "ai_revision.json").read_text(encoding="utf-8"))
        if marker.get("revision") != revision or marker.get("source") != source.name:
            raise ValueError("Existing run differs from this revision")
        return
    formulas = decision["formulas"]
    verify_numerics(family, formulas)
    with tempfile.TemporaryDirectory(prefix=".ai-geometry-", dir=destination.parent) as temp:
        staged = Path(temp) / "trial"
        staged.mkdir()
        files = (_apply_osgk(source, staged, formulas) if family == OSGK
                 else _apply_spectral(source, staged, formulas, revision))
        audit = {"family": family, "revision":revision, "source":source.name,
                 "decision":decision, "method_file_sha256":files}
        (staged / "ai_revision.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.rename(destination)
