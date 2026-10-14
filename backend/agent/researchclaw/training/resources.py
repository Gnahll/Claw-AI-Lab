"""Explicit resource ownership for the original research stages."""

from __future__ import annotations

import functools
import os
import re
import signal
import subprocess
import threading
from contextlib import ExitStack
from pathlib import Path

from researchclaw.training.jobs import job_lock, require_idle_gpu


def validate_gpu_uuid(value: str) -> str:
    if value and not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise ValueError("请填写已获授权的完整 GPU UUID；留空仅使用 CPU")
    return value


def with_execution_resources(stage):
    def decorate(function):
        @functools.wraps(function)
        def execute(stage_dir, run_dir, config, adapters, **kwargs):
            from researchclaw.pipeline.executor import StageResult
            from researchclaw.pipeline.stages import Stage, StageStatus

            if stage in (Stage.EXPERIMENT_RUN, Stage.ITERATIVE_REFINE) and getattr(config.experiment, "training_job_dir", ""):
                return function(stage_dir, run_dir, config, adapters, **kwargs)

            old_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
            previous_handler = None
            with ExitStack() as resources:
                try:
                    gpu_uuid = validate_gpu_uuid(getattr(config.experiment, "gpu_uuid", ""))
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                    if gpu_uuid:
                        lock_dir = Path.home() / ".cache" / "researchclaw" / "gpu-locks"
                        lock_dir.mkdir(parents=True, exist_ok=True)
                        lock_path = lock_dir / f"{gpu_uuid}.lock"
                        resources.enter_context(job_lock(lock_path))
                        require_idle_gpu(gpu_uuid)
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    if old_visibility is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_visibility
                    return StageResult(stage=stage, status=StageStatus.FAILED, artifacts=(), error=str(exc))
                try:
                    if threading.current_thread() is threading.main_thread():
                        previous_handler = signal.getsignal(signal.SIGTERM)
                        signal.signal(signal.SIGTERM, _terminate)
                    return function(stage_dir, run_dir, config, adapters, **kwargs)
                finally:
                    if previous_handler is not None:
                        signal.signal(signal.SIGTERM, previous_handler)
                    if old_visibility is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_visibility
        return execute
    return decorate


def _terminate(signum, frame):
    raise SystemExit(128 + signum)
