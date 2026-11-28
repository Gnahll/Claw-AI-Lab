"""Recover an unambiguous notation error in a run-owned stage input."""
from __future__ import annotations

import codecs
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time


MAX_BYTES = 2 * 1024 * 1024
# Exact input artifacts. Checkpoints, weights, metrics, decisions and approvals
# are intentionally outside the writer's scope.
INPUTS = {"schedule.json", "codebase_candidates.json", "setup.json",
          "status.json", "mechanism_plan.json"}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("non-standard JSON constant")


def _verify(raw, expected):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                       parse_constant=_reject_constant)
    if not isinstance(value, expected):
        raise ValueError("unexpected JSON root")


def _candidate(raw, exc, expected):
    # All transformations splice the original bytes. Never serialize the
    # decoded data back to disk, infer missing values, or edit string contents.
    if raw.startswith(codecs.BOM_UTF8):
        op, content = "remove_utf8_bom", raw[len(codecs.BOM_UTF8):]
    elif match := re.fullmatch(rb"```json[ \t]*\r?\n(.*?)\r?\n```[ \t]*(?:\r?\n)?", raw, re.S):
        op, content = "unwrap_json_fence", match[1]
    elif (exc.doc[exc.pos:exc.pos + 1] in ("}", "]")
          or re.match(r",[ \t\r\n]*[}\]]", exc.doc[exc.pos:])):
        end = exc.pos + int(exc.doc[exc.pos:exc.pos + 1] == ",")
        before = exc.doc[:end].rstrip(" \n\t\r")
        if not before.endswith(","):
            return None
        left = before[:-1].rstrip(" \n\t\r")
        if not left or left[-1] in "[:,{":
            return None
        idx = len(before[:-1].encode("utf-8"))
        op, content = "remove_trailing_comma", raw[:idx] + raw[idx + 1:]
    else:
        return None
    _verify(content, expected)
    return op, content


def _atomic(path, data, mode=0o600):
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def suggest(client, diagnostic):
    """Allow the configured AI to approve a single prevalidated byte edit."""
    reply = client.chat(
        [{"role": "user", "content": json.dumps(diagnostic)}],
        system=("Return JSON with action equal to the supplied operation or 'hold'. "
                "Only syntax may change. No replacement text or paths."),
        json_mode=True, max_tokens=100,
    )
    return json.loads(reply.content)


def _snapshot(run, path):
    # Reject symlinks below the trusted run directory, links to other files,
    # hidden paths, special files, and paths outside this run.
    root = run.resolve(strict=True)
    rel = path.absolute().relative_to(run.absolute())
    if any(part.startswith(".") for part in rel.parts):
        raise ValueError("hidden or parent path")
    target = root / rel
    if target.resolve(strict=True) != target:
        raise ValueError("linked artifact")
    st = target.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > MAX_BYTES:
        raise ValueError("ineligible file")
    with target.open("rb") as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) != st.st_size:
        raise ValueError("artifact changed during read")
    return target, raw, (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode)


YAML_INPUTS = {"search_plan.yaml", "exp_plan.yaml"}


def _yaml_candidate(raw):
    import yaml
    match = re.fullmatch(rb"```(?:yaml|yml)[ \t]*\r?\n(.+?\n)```[ \t]*(?:\r?\n)?", raw, re.S)
    if match is None:
        return None
    body = match[1]  # Includes the last payload newline for YAML block scalars.
    if any(getattr(event, "anchor", None) is not None for event in yaml.parse(body, Loader=yaml.SafeLoader)):
        return None

    class UniqueLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            keys = set()
            for key, _ in node.value:
                if key.tag != "tag:yaml.org,2002:str" or key.value in keys:
                    raise ValueError("duplicate or non-string YAML key")
                keys.add(key.value)
            return super().construct_mapping(node, deep=deep)

    value = yaml.load(body, Loader=UniqueLoader)
    return ("unwrap_yaml_fence", body) if isinstance(value, dict) else None


def _pinned_plan_allows(path, output):
    if path.name != "exp_plan.yaml":
        return True
    binding = path.with_name("candidate.json")
    if binding.is_symlink():
        return False
    if not binding.exists():
        return True
    record = json.loads(binding.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    return isinstance(record, dict) and record.get("plan_sha256") == hashlib.sha256(output).hexdigest()


class FormatRecoveryPending(RuntimeError):
    def __init__(self, next_check):
        self.next_check = next_check
        super().__init__("Format recovery is waiting for the configured AI")


AI_ATTEMPTS = 3
BACKOFF_SECONDS = 60


def _claim(directory):
    for number in range(1, AI_ATTEMPTS + 1):
        attempt = directory if number == 1 else directory / f"retry-{number}"
        try:
            attempt.mkdir(mode=0o700)
            return attempt, number
        except FileExistsError:
            if attempt.is_symlink() or attempt.resolve() != attempt:
                return None
            record = attempt / "audit.json"
            if record.is_symlink():
                return None
            if not record.exists():
                pending_until = attempt.stat().st_mtime + 120
                if time.time() < pending_until:
                    raise FormatRecoveryPending(pending_until)
                return None
            saved = json.loads(record.read_text(encoding="utf-8"))
            if saved.get("status") == "choosing":
                pending_until = record.stat().st_mtime + 120
                if time.time() < pending_until:
                    raise FormatRecoveryPending(pending_until)
                return None
            if saved.get("reason") != "ai_unavailable":
                return None
            if number == AI_ATTEMPTS:
                return None
            # Also handle the first release, which recorded ai_unavailable
            # without a retry time.
            next_check = saved.get("next_check", record.stat().st_mtime + BACKOFF_SECONDS)
            if time.time() < next_check:
                raise FormatRecoveryPending(next_check)
    return None


def repair(run, path, error, *, choose=None, expected=(dict, list)):
    """Return True only after a verified replacement; retry via the caller."""
    yaml_input = path.name in YAML_INPUTS
    if not yaml_input and (path.name not in INPUTS or not isinstance(error, json.JSONDecodeError)):
        return False
    try:
        target, raw, identity = _snapshot(run, path)
        original = raw.decode("utf-8")
        # Path.read_text performs newline translation. Reproduce the parser
        # error on the original bytes to obtain the correct byte offset.
        source = error if yaml_input else error.doc
        if original.replace("\r\n", "\n").replace("\r", "\n") != source.replace("\r\n", "\n").replace("\r", "\n"):
            return False
        if yaml_input:
            import yaml
            try:
                parsed = yaml.safe_load(original)
                if isinstance(parsed, dict):
                    return False
                reason = {"error": "expected a YAML mapping", "line": 1}
            except yaml.YAMLError as exc:
                mark = getattr(exc, "problem_mark", None)
                reason = {"error": type(exc).__name__, "line": mark.line + 1 if mark else None}
            try:
                candidate = _yaml_candidate(raw)
                if candidate and not _pinned_plan_allows(target, candidate[1]):
                    candidate = None
                    reason["error"] = "frozen candidate digest mismatch"
            except (yaml.YAMLError, ValueError, RecursionError):
                candidate = None
        else:
            try:
                json.loads(original)
                return False
            except json.JSONDecodeError as exc:
                try:
                    candidate = _candidate(raw, exc, expected)
                except (ValueError, RecursionError):
                    candidate = None
                reason = {"error": exc.msg, "line": exc.lineno, "column": exc.colno}
        audit = run.resolve() / ".artifact-format-recovery"
        audit.mkdir(mode=0o700, exist_ok=True)
        if audit.is_symlink() or audit.resolve() != audit:
            return False
        rel = str(target.relative_to(run.resolve()))
        digest = hashlib.sha256(raw).hexdigest()
        # A persistent exclusive claim bounds repeat attempts and survives
        # concurrent workers or a crash between backup and replacement.
        key = hashlib.sha256((rel + "|" + digest).encode()).hexdigest()
        claim = _claim(audit / key)
        if claim is None:
            return False
        attempt, number = claim
        event = dict(path=rel, before_sha256=digest, parser=reason,
                     status="manual_review", reason="no_unique_verified_edit")

        def record():
            _atomic(attempt / "audit.json", json.dumps(event, indent=2).encode())

        if not candidate:
            record()
            return False
        operation, output = candidate
        event.update(operation=operation, after_sha256=hashlib.sha256(output).hexdigest(),
                     status="choosing", reason="awaiting_choice", selected_by="ai", attempt=number)
        record()
        try:
            if choose is None:
                raise ConnectionError("No configured AI client")
            answer = choose(dict(reason, operation=operation))
        except (ValueError, TypeError):
            answer = None
        except Exception:
            next_check = time.time() + BACKOFF_SECONDS * 2 ** (number - 1)
            pending = number < AI_ATTEMPTS
            event.update(status="await_ai" if pending else "manual_review",
                         reason="ai_unavailable", next_check=next_check)
            record()
            if pending:
                raise FormatRecoveryPending(next_check)
            return False
        if answer != {"action": operation}:
            event.update(status="manual_review", reason="ai_held_or_invalid_response")
            record()
            return False
        _atomic(attempt / "original.bytes", raw)
        event.update(status="backed_up", reason="before_replace")
        record()
        if _snapshot(run, path) != (target, raw, identity) or not _pinned_plan_allows(target, output):
            event.update(status="manual_review", reason="source_changed")
            record()
            return False
        _atomic(target, output, stat.S_IMODE(identity[-1]))
        event.update(status="repaired", reason="retry_original_reader")
        record()
        return True
    except (OSError, ValueError, RecursionError):
        return False
