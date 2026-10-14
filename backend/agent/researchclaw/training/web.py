"""Bounded, explicitly authorized web acceptance jobs using the managed adapter."""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from researchclaw.data.coverage import file_sha256, validate_splits
from researchclaw.training import jobs


def allowed_training_origin(origin: str) -> bool:
    try:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return False
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            return False
        return parsed.hostname in {"localhost", "127.0.0.1", "::1"} or origin in {
            value.strip() for value in os.environ.get("CLAW_TRAINING_ALLOWED_ORIGINS", "").split(",") if value.strip()
        }
    except ValueError:
        return False


class WebTrainingService:
    def __init__(self, runs_dir: Path, repo_root: Path):
        self.root = (runs_dir / "training-jobs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checks = self.root / "preflights"
        self.checks.mkdir(exist_ok=True)
        self.defaults = jobs.read_json(repo_root / "examples/cotransformers-pilot.json")
        web_defaults = jobs.read_json(repo_root / "examples/cotransformers-web.json")
        self.defaults["split_manifest"] = os.environ.get("CLAW_TRAINING_SPLIT_MANIFEST") or web_defaults["split_manifest"]
        self.default_topic = web_defaults["topic"]
        self.lock = threading.RLock()

    def _identifier(self, identifier: str) -> str:
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid training job/preflight identifier")
        return identifier

    def _directory(self, identifier: str) -> Path:
        path = self.root / self._identifier(identifier)
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise ValueError("Training job path escapes the managed job root")
        return path

    def _spec(self, incoming: dict) -> dict:
        if not isinstance(incoming, dict):
            raise ValueError("A training specification is required")
        spec = dict(self.defaults)
        spec.update(incoming)
        jobs.validate_spec(spec)
        if spec.get("epochs", 1) != 1 or not 1 <= spec.get("steps_per_epoch", 3) <= 3:
            raise ValueError("Web acceptance is limited to one truncated epoch and at most three optimizer steps")
        for key in ("source_dir", "dataset_root", "initial_checkpoint", "split_manifest"):
            supplied = Path(spec[key]).expanduser()
            if not supplied.is_absolute():
                raise ValueError(f"{key} requires an absolute server path")
            candidate = supplied.resolve(strict=True)
            configured = Path(self.defaults[key]).expanduser().resolve(strict=True)
            root = configured.parent if key == "initial_checkpoint" else configured
            allowed = candidate == configured if key == "split_manifest" else candidate.is_relative_to(root)
            if not allowed:
                raise ValueError(f"{key} is outside the server-configured training resources")
            spec[key] = str(candidate)
        return spec

    def _preflight(self, data: dict) -> dict:
        topic = data.get("topic", self.default_topic)
        if not isinstance(topic, str) or not topic.strip() or len(topic) > 4000:
            raise ValueError("Provide a research topic of at most 4000 characters")
        spec = self._spec(data.get("spec", {}))
        manifest_path = Path(spec["split_manifest"])
        manifest_digest = file_sha256(manifest_path)
        manifest = jobs.read_json(manifest_path)
        counts = validate_splits(manifest, root=Path(spec["dataset_root"]), require_approval=False)
        if file_sha256(manifest_path) != manifest_digest:
            raise ValueError("Split manifest changed during preflight")
        if spec.get("steps_per_epoch", 3) > counts["train"]:
            raise ValueError("Optimizer steps exceed available training samples")
        if file_sha256(Path(spec["initial_checkpoint"])) != spec["initial_checkpoint_sha256"]:
            raise ValueError("Initial checkpoint hash mismatch")
        source_files = jobs.source_hashes(Path(spec["source_dir"]))
        identifier = uuid.uuid4().hex
        check = {
            "id": identifier, "topic": topic.strip(), "spec": spec, "counts": counts,
            "manifest_sha256": manifest_digest, "source_files": source_files,
            "protocol_status": manifest.get("protocol_status"),
            "previously_evaluated_groups": manifest.get("previously_evaluated_groups", []),
            "test_is_blind": False, "checked_at": time.time(),
            "protocol": "coverage_finetuning_pilot_not_paper_reproduction",
        }
        jobs.write_json(self.checks / f"{identifier}.json", check)
        return check

    def _prepare(self, data: dict) -> str:
        if data.get("protocolApproved") is not True:
            raise ValueError("Explicit approval of the reviewed COVERAGE pilot split is required")
        identifier = self._identifier(data.get("preflightId", ""))
        check = jobs.read_json(self.checks / f"{identifier}.json")
        job_dir = self._directory(identifier)
        if (job_dir / "web_request.json").is_file():
            return identifier
        spec = self._spec(check["spec"])
        if time.time() - check["checked_at"] > 86400:
            raise ValueError("Preflight expired; check the inputs again")
        if file_sha256(Path(spec["split_manifest"])) != check["manifest_sha256"]:
            raise ValueError("Split manifest changed; review a new preflight before approval")
        if jobs.source_hashes(Path(spec["source_dir"])) != check["source_files"]:
            raise ValueError("Model source changed; review a new preflight")
        manifest = jobs.read_json(Path(spec["split_manifest"]))
        validate_splits(manifest, root=Path(spec["dataset_root"]), require_approval=False)
        if file_sha256(Path(spec["split_manifest"])) != check["manifest_sha256"]:
            raise ValueError("Split manifest changed during approval")
        manifest["protocol_status"] = "approved"
        manifest["web_approval"] = {
            "preflight_id": identifier, "approved_at": time.time(),
            "reviewed_manifest_sha256": check["manifest_sha256"],
            "scope": "COVERAGE pilot fine-tuning, not original-paper reproduction or a blind benchmark",
        }
        approval_dir = self.checks / identifier
        approval_dir.mkdir(exist_ok=False)
        jobs.write_json(approval_dir / "approved-splits.json", manifest)
        spec["split_manifest"] = str(approval_dir / "approved-splits.json")
        jobs.write_json(approval_dir / "spec.json", spec)
        jobs.prepare_job(approval_dir / "spec.json", job_dir)
        jobs.write_json(job_dir / "web_request.json", {
            "topic": check["topic"], "counts": check["counts"],
            "protocol_approved": True, "preflight_id": identifier, "created_at": time.time(),
        })
        return identifier

    def _verify(self, job_dir: Path) -> dict:
        try:
            from researchclaw.training.evidence import import_training_stage, verify_completed_job

            result = verify_completed_job(job_dir)
            for stage, refinement in ((14, False), (15, True)):
                imported = import_training_stage(job_dir / f"pipeline/stage-{stage}", job_dir, refinement=refinement)
                if imported.error:
                    raise ValueError(f"S{stage} result import failed: {imported.error}")
            verification = {"status": "passed", "verified_at": time.time(), "result": result, "pipeline_imported": True}
        except Exception as exc:
            verification = {"status": "failed", "verified_at": time.time(), "error": str(exc)}
            runs = job_dir / "pipeline/stage-14/runs"
            runs.mkdir(parents=True, exist_ok=True)
            jobs.write_json(runs / "run_report.json", {
                "success": False, "has_results": False, "results": {},
                "training_job": str(job_dir), "verification_error": str(exc),
            })
        jobs.write_json(job_dir / "web_verification.json", verification)
        return verification

    def _job(self, identifier: str, *, verify: bool = False) -> dict:
        job_dir = self._directory(identifier)
        request = jobs.read_json(job_dir / "web_request.json")
        state = jobs.job_status(job_dir)
        verification_path = job_dir / "web_verification.json"
        if verify or (state["status"] == "completed" and not verification_path.exists()):
            verification = self._verify(job_dir)
        elif verification_path.is_file():
            verification = jobs.read_json(verification_path)
        else:
            verification = {"status": "pending"}
        log_path = job_dir / "worker.log"
        log = ""
        if log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 12000))
                log = stream.read(12000).decode("utf-8", errors="replace")
        spec = jobs.read_json(job_dir / "spec.json")
        predictions = []
        if verification["status"] == "passed":
            results = jobs.read_json(job_dir / "test_results.json")
            predictions = [{"id": record["id"], "prediction": record["prediction"], "target": record["target"]}
                           for record in results["records"]]
        return {
            "id": identifier, "job_dir": str(job_dir), "topic": request["topic"],
            "counts": request["counts"], "state": state, "verification": verification,
            "expected_steps": spec.get("epochs", 1) * spec.get("steps_per_epoch", 3),
            "can_resume": state["status"] in {"failed", "interrupted"} and (job_dir / "resume.json").is_file(),
            "log": log, "predictions": predictions,
        }

    def snapshot(self) -> dict:
        collected = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if re.fullmatch(r"[a-f0-9]{32}", path.name) and (path / "web_request.json").is_file():
                try:
                    collected.append(self._job(path.name))
                except Exception as exc:
                    collected.append({"id": path.name, "topic": path.name, "state": {"status": "unreadable", "error": str(exc)}})
        collected.sort(key=lambda item: item.get("state", {}).get("time", 0), reverse=True)
        return {"defaults": self.defaults, "default_topic": self.default_topic, "jobs": collected}

    def handle(self, data: dict) -> dict:
        with self.lock:
            action = data.get("command", "")
            payload = {"action": action, "request_id": data.get("requestId", ""), "ok": True}
            try:
                if action == "training_preflight":
                    payload["preflight"] = self._preflight(data)
                elif action == "training_prepare":
                    payload["last_job_id"] = self._prepare(data)
                elif action in {"training_start", "training_resume"}:
                    if data.get("gpuAllocated") is not True:
                        raise ValueError("Confirm administrator/scheduler allocation of this GPU; idle does not mean allocated")
                    identifier = self._identifier(data.get("jobId", ""))
                    job_dir = self._directory(identifier)
                    jobs.read_json(job_dir / "web_request.json")
                    if os.environ.get("CLAW_WEB_TRAINING_DISABLED") == "1":
                        raise ValueError("Web GPU training is disabled by the server administrator")
                    gpu_uuid = data.get("gpuUuid", "")
                    if not isinstance(gpu_uuid, str) or not re.fullmatch(r"GPU-[a-fA-F0-9-]+", gpu_uuid):
                        raise ValueError("Provide the explicitly allocated physical GPU UUID")
                    jobs.start_job(job_dir, gpu_uuid, resume=action == "training_resume")
                    jobs.write_json(job_dir / "web_gpu_approval.json", {"gpu_uuid": gpu_uuid, "approved_at": time.time()})
                    payload["last_job_id"] = identifier
                elif action == "training_verify":
                    identifier = self._identifier(data.get("jobId", ""))
                    if jobs.job_status(self._directory(identifier))["status"] != "completed":
                        raise ValueError("Wait for training and test completion before evidence verification")
                    job = self._job(identifier, verify=True)
                    if job["verification"]["status"] != "passed":
                        raise ValueError(job["verification"]["error"])
                    payload["last_job_id"] = identifier
                elif action == "training_gpus":
                    payload["gpus"] = [{"uuid": row[0], "name": row[1], "memory_mib": row[2], "utilization": row[3]}
                                       for row in jobs.gpu_rows("uuid,name,memory.used,utilization.gpu")]
                elif action != "training_list":
                    raise ValueError("Unknown managed training command")
            except Exception as exc:
                payload.update(ok=False, error=str(exc))
            payload.update(self.snapshot())
            return {"type": "training_update", "payload": payload}

    def artifact(self, identifier: str, filename: str) -> Path:
        job_dir = self._directory(identifier)
        jobs.read_json(job_dir / "web_request.json")
        allowed = {"web_verification.json", "test_results.json", "events.jsonl", "worker.log", "state.json",
                   "pipeline/stage-14/runs/run_report.json", "pipeline/stage-14/training_limitations.md",
                   "pipeline/stage-15/refinement_disabled.json"}
        if filename.startswith("pipeline/") and jobs.read_json(job_dir / "web_verification.json")["status"] != "passed":
            raise ValueError("Imported pipeline artifacts require passed evidence verification")
        if filename not in allowed:
            if not filename.startswith("test_predictions/") or not filename.endswith(".png"):
                raise ValueError("Artifact is not exposed by the training acceptance endpoint")
            verification = jobs.read_json(job_dir / "web_verification.json")
            if verification["status"] != "passed":
                raise ValueError("Prediction previews require passed evidence verification")
            results = jobs.read_json(job_dir / "test_results.json")
            masks = {f"test_predictions/{record[kind]}" for record in results["records"] for kind in ("prediction", "target")}
            if filename not in masks:
                raise ValueError("Unknown test prediction")
        path = (job_dir / filename).resolve(strict=True)
        if not path.is_relative_to(job_dir.resolve()) or not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("Artifact path or size is not permitted")
        return path
