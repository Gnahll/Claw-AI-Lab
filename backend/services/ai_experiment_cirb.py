"""Trusted changes for the fixed-candidate CIRB mask-boundary experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

VARIANTS = {
    "protect_good_masks": {
        "description": "将回归保护损失加倍，验证现有高质量实例的退化是否抵消了边界收益",
        "old": "+.5*(1-current.mean())+guard)",
        "new": "+.5*(1-current.mean())+2.*guard)",
    },
    "reduce_boundary_displacement": {
        "description": "将掩码边缘最大改变量由 0.80 降为 0.40，验证跨实例泄漏是否由修正过大引起",
        "old": "delta=.80*x.tanh()*boundary*present",
        "new": "delta=.40*x.tanh()*boundary*present",
    },
    "emphasize_iou": {
        "description": "提高直接优化实例 IoU 的损失权重，验证边缘像素损失与整实例覆盖目标是否冲突",
        "old": "+.5*(1-current.mean())+guard)",
        "new": "+1.5*(1-current.mean())+guard)",
    },
}


def prepare(source: Path, destination: Path, variant: str, decision: dict) -> None:
    if destination.exists():
        registered = json.loads((destination / "ai_revision.json").read_text(encoding="utf-8"))
        if registered.get("variant") != variant or registered.get("source") != source.name:
            raise ValueError("Existing experiment does not match this revision")
        return
    change = VARIANTS[variant]
    before = json.loads((source / "setup.json").read_text(encoding="utf-8"))
    if before.get("method_version") != "cirb_fixed_score_conditional_mask_boundary_v1":
        raise ValueError("Unsupported CIRB source experiment")
    with tempfile.TemporaryDirectory(prefix=".ai-cirb-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "trial"
        staged.mkdir()
        shutil.copytree(source / "experiment", staged / "experiment",
                        ignore=shutil.ignore_patterns("outputs", "__pycache__"))
        for filename in ("dispatch.py", "supervisor.py"):
            shutil.copy2(source / filename, staged / filename)
        model = staged / "experiment" / "directed_boundary.py"
        content = model.read_text(encoding="utf-8")
        if content.count(change["old"]) != 1:
            raise ValueError("The baseline method's expected source expression has changed")
        content = content.replace(change["old"], change["new"])
        compile(content, str(model), "exec")
        model.write_text(content, encoding="utf-8")
        setup = dict(before)
        setup["module_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        setup["protocol_digest"] = hashlib.sha256(
            (setup["protocol_digest"] + setup["module_sha256"] + variant).encode()
        ).hexdigest()
        setup["method_version"] = before["method_version"] + "_" + variant
        (staged / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
        (staged / "ai_revision.json").write_text(json.dumps({
            "variant": variant, "source": source.name, "decision": decision,
            "method_sha256": setup["module_sha256"], "method_change": change["description"],
            "short_steps_per_condition": setup["target_steps"], "long_steps_per_condition": setup["long_steps"],
            "validation_scenes": len(setup["scene_ids"]), "gpu_uuids": setup["gpu_uuids"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.rename(destination)
