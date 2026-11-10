"""Observe four research jobs and restore their bridge after a clean crash.

The bridge already retries failed layer processes. This guard never relaunches a
failed experiment stage, so code and protocol failures cannot cause retry loops.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import os
import re
import subprocess
import sys
import time


def processes() -> dict[int, str]:
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            result[int(path.name)] = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, PermissionError):
            pass
    return result


def load_env(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        env[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))
    env['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'
    env['REQUESTS_CA_BUNDLE'] = env['SSL_CERT_FILE']
    return env


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    runs = repo / 'backend/runs'
    guard_config = json.loads((runs / 'experiment_guard_config.json').read_text(encoding='utf-8'))
    command = guard_config['bridge_command']
    assert command[0] == str(repo / '.venv/bin/python')
    assert command[2] == str(repo / 'backend/services/agent_bridge.py')
    assert command[command.index('--runs-dir') + 1] == str(runs)
    with (runs / 'experiment_guard.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        previous_path = runs / 'experiment_guard_status.json'
        try:
            previous = json.loads(previous_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            previous = {}
        table = processes()
        bridge_matches = [(pid, cmd) for pid, cmd in table.items()
                          if str(repo / 'backend/services/agent_bridge.py') in cmd
                          and '--runs-dir ' + str(runs) in cmd]
        active_jobs = [(pid, cmd) for pid, cmd in table.items()
                       if str(repo) in cmd and ('researchclaw run' in cmd
                                                or 'torch.distributed.run' in cmd
                                                or 'model.training.train' in cmd
                                                or re.search(r'python(?:\d+(?:\.\d+)?)? main\.py', cmd))]
        restarted = False
        alerts = []
        if not bridge_matches:
            if active_jobs:
                alerts.append('Agent Bridge stopped while research jobs remain; waiting to avoid duplicate processes')
            else:
                log = repo / 'logs/agent_bridge.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('ab') as output:
                    child = subprocess.Popen(command, cwd=repo / 'backend/services',
                                             env=load_env(repo / '.env'),
                                             stdin=subprocess.DEVNULL,
                                             stdout=output, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                restarted = True
                bridge_matches = [(child.pid, 'restarted')]
                alerts.append(f'Agent Bridge restarted as PID {child.pid}')

        tasks_by_project: dict[str, list[dict]] = {}
        for queue in (runs / 'queues').glob('*.json'):
            try:
                value = json.loads(queue.read_text(encoding='utf-8'))
                tasks = value if isinstance(value, list) else value.get('tasks', [])
            except (OSError, ValueError):
                continue
            for task in tasks:
                if task.get('status') in ('pending', 'assigned') and task.get('project_id'):
                    tasks_by_project.setdefault(task['project_id'], []).append({
                        'queue': queue.stem, 'status': task['status'],
                        'attempt': task.get('attempt', 0),
                        'not_before': task.get('not_before', 0),
                    })
        project_rows = []
        for meta_path in sorted((runs / 'projects').glob('*/project_meta.json')):
            project_id = meta_path.parent.name
            matching_jobs = [(pid, cmd) for pid, cmd in active_jobs if project_id in cmd]
            candidates = [meta_path.parent / name for name in (
                'run-ppt-four-scale-revision3-20261110',
                'run-ppt-four-scale-revision2-20261110',
                'run-ppt-four-scale-20261110', 'run-integrated')]
            candidates = [candidate for candidate in candidates if candidate.is_dir()]
            run = next((candidate for candidate in candidates
                        if any(str(candidate) in cmd for _, cmd in matching_jobs)),
                       candidates[0] if candidates else meta_path.parent / 'run-integrated')
            summary_path = run / 'pipeline_summary.json'
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                summary = {}
            matching = [pid for pid, _ in matching_jobs]
            queued = tasks_by_project.get(project_id, [])
            state = 'running' if matching else 'queued' if queued else 'stopped'
            active_stage = None
            active_stage_name = None
            if matching:
                logs = sorted((path for path in run.glob('*.log') if path.is_file()),
                              key=lambda path: path.stat().st_mtime, reverse=True)
                for log in logs:
                    stages = re.findall(r'Stage (\d+)/\d+ ([A-Z_]+) — running',
                                        log.read_text(encoding='utf-8', errors='replace'))
                    if stages:
                        active_stage, active_stage_name = int(stages[-1][0]), stages[-1][1]
                        break
            row = {'project_id': project_id, 'state': state, 'pids': matching,
                   'queue': queued, 'run_dir': str(run),
                   'last_stage': active_stage or summary.get('final_stage'),
                   'active_stage_name': active_stage_name,
                   'last_status': 'running' if matching else 'queued' if queued else summary.get('final_status'),
                   'previous_summary_status': summary.get('final_status')}
            project_rows.append(row)
            if state == 'stopped' and summary.get('final_stage', 0) < 14:
                alerts.append(f'{project_id}: stopped before short engineering training; inspect stage decision')
        snapshot = {'checked_utc': datetime.now(timezone.utc).isoformat(),
                    'host': os.uname().nodename, 'bridge_pids': [pid for pid, _ in bridge_matches],
                    'bridge_restarted': restarted, 'projects': project_rows,
                    'alerts': alerts}
        atomic_json(previous_path, snapshot)
        previous_alerts = previous.get('alerts', [])
        if alerts != previous_alerts:
            with (runs / 'experiment_guard_alerts.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps({'timestamp_utc': snapshot['checked_utc'],
                                         'alerts': alerts}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
