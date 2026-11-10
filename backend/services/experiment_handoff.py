"""One-shot, bounded handoff from failed old plans to four-scale S9–S13.

Runs from cron. It never starts S14 training, never edits a failed method into
an apparent success, and never starts a second process for the same project.
"""
from __future__ import annotations

from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

import yaml


def all_processes() -> list[str]:
    commands = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            commands.append((path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        except (OSError, PermissionError):
            pass
    return commands


def load_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    for line in (repo / '.env').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        env[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))
    assert env.get('RESEARCHCLAW_API_KEY')
    env['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'
    env['REQUESTS_CA_BUNDLE'] = env['SSL_CERT_FILE']
    env['PYTHONUNBUFFERED'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    return env


def launch(repo: Path, run: Path, config: Path, phase: str, env: dict[str, str], gpu_ids: list[int]) -> None:
    if phase == 'stage9':
        marker = run / 'stage9_replan_manifest.json'
        logfile = run / 'stage9_replan.log'
        from_stage = to_stage = 'EXPERIMENT_DESIGN'
    else:
        marker = run / 'stage10_to_13_manifest.json'
        logfile = run / 'stage10_to_13.log'
        from_stage, to_stage = 'CODEBASE_SEARCH', 'RESOURCE_PLANNING'
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    assert not marker.exists() and not logfile.exists()
    command = [str(repo / '.venv/bin/researchclaw'), 'run', '--config', str(config),
               '--output', str(run), '--from-stage', from_stage,
               '--to-stage', to_stage, '--auto-approve']
    with logfile.open('wb') as output:
        process = subprocess.Popen(command, cwd=repo / 'backend', env=env,
                                   stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    marker.write_text(json.dumps({'pid': process.pid, 'command': command,
                                  'gpu_indices': gpu_ids if phase != 'stage9' else [],
                                  'started_unix': time.time(),
                                  'stop_after_stage': to_stage}, indent=2) + '\n')
    print(f'{run.parent.name}: started {phase} PID {process.pid}', flush=True)


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    runs = repo / 'backend/runs'
    with (runs / 'experiment_handoff.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        commands = all_processes()
        tasks_by_project: dict[str, int] = {}
        for queue in (runs / 'queues').glob('*.json'):
            try:
                value = json.loads(queue.read_text(encoding='utf-8'))
                tasks = value if isinstance(value, list) else value.get('tasks', [])
            except (OSError, ValueError):
                continue
            for task in tasks:
                if task.get('status') in ('pending', 'assigned') and task.get('project_id'):
                    key = task['project_id']
                    tasks_by_project[key] = tasks_by_project.get(key, 0) + 1
        gpu_rows = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        gpu_map = {uuid.strip(): int(index) for index, uuid in
                   (row.split(',', 1) for row in gpu_rows.splitlines())}
        compute_rows = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
            text=True, capture_output=True, check=False)
        busy_uuids = {row.strip() for row in compute_rows.stdout.splitlines()} if compute_rows.returncode == 0 else set(gpu_map)
        env = load_env(repo)
        for meta_path in sorted((runs / 'projects').glob('*/project_meta.json')):
            project_id = meta_path.parent.name
            revised3 = meta_path.parent / 'run-ppt-four-scale-revision3-20261110'
            revised2 = meta_path.parent / 'run-ppt-four-scale-revision2-20261110'
            revised = revised3 if revised3.is_dir() else revised2
            run = revised if revised.is_dir() else meta_path.parent / 'run-ppt-four-scale-20261110'
            if not revised.is_dir() and not (run / 'stage-08').is_dir():
                continue
            if revised.is_dir():
                launch_manifest = revised / 'launch_manifest.json'
                try:
                    original_command = json.loads(launch_manifest.read_text(encoding='utf-8'))['command']
                except (OSError, ValueError, KeyError):
                    original_command = []
                if 'RESOURCE_PLANNING' in original_command:
                    continue
            if tasks_by_project.get(project_id):
                continue
            if any('researchclaw run' in cmd and project_id in cmd for cmd in commands):
                continue
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
            config_path = Path(metadata['config_path'])
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            if config['experiment']['existing_project_dir'] != '/data_tan/3D_Segment/3D_Segment+CGGA_PPTDualStream':
                continue
            uuids = config['experiment']['gpu_uuid'].split(',')
            if any(uuid not in gpu_map for uuid in uuids):
                continue
            gpu_ids = [gpu_map[uuid] for uuid in uuids]
            if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
                continue
            decision_path = run / 'stage-09/decision.json'
            try:
                decision = json.loads(decision_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                decision = {}
            if decision.get('status') == 'done':
                if (run / 'stage10_to_13_manifest.json').exists():
                    continue
                plan = run / 'stage-09/exp_plan.yaml'
                if not plan.is_file():
                    continue
                plan_text = plan.read_text(encoding='utf-8')
                if 'PPTDualStream' not in plan_text and 'PPTFourScaleFusion' not in plan_text:
                    (run / 'handoff_blocked.txt').write_text('S9 plan does not identify the verified four-scale baseline.\n')
                    continue
                if any(uuid in busy_uuids for uuid in uuids):
                    continue
                launch(repo, run, config_path, 'stage10_to_13', env.copy(), gpu_ids)
            elif not revised.is_dir() and not (run / 'stage9_replan_manifest.json').exists():
                source = run / 'stage-09'
                backup = run / 'stage-09_quota_failure_20261110'
                if source.exists() and not backup.exists():
                    shutil.copytree(source, backup)
                launch(repo, run, config_path, 'stage9', env.copy(), gpu_ids)


if __name__ == '__main__':
    main()
