"""Content-addressed evidence reuse for an unchanged, already verified S14 run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict[str, str]:
    skipped = {'outputs', 'datasets', 'checkpoints', 'codebases', '__pycache__', 'repair_backup'}
    return {p.relative_to(root).as_posix(): hashlib.sha256(
                p.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for p in sorted(root.rglob('*.py'))
            if not any(part.startswith('.') or part in skipped for part in p.relative_to(root).parts[:-1])}


def protocol_manifest(run_dir: Path) -> dict[str, str]:
    names = ('exp_plan.yaml', 'experiment_spec.md')
    result = {}
    for stage in ('stage-09', 'stage-11'):
        for name in names:
            path = run_dir / stage / name
            if path.is_file():
                result[f'{stage}/{name}'] = file_hash(path)
    return result


def config_contract(config) -> dict:
    return json.loads(json.dumps({name: getattr(config.experiment, name, None) for name in (
        'require_training', 'metric_key', 'metric_direction', 'runs_per_condition',
        'optional_experiments', 'datasets_dir', 'checkpoints_dir', 'codebases_dir')}))


def create_receipt(workspace: Path, run_dir: Path, report_path: Path, config) -> dict:
    report = json.loads(report_path.read_text(encoding='utf-8'))
    training = report.get('training', {})
    if (report.get('success') is not True or report.get('smoke') or not report.get('require_training')
            or not training.get('parameter_update_observed') or not training.get('checkpoint_verified')
            or training.get('optimizer_steps', 0) < 1):
        raise ValueError('Only successful supervised training can create a verified-run receipt')
    sources = source_manifest(workspace)
    if sources != report.get('source_manifest'):
        raise ValueError('Experiment source changed during supervised execution')
    if 'main.py' not in sources or not (workspace / 'outputs/model.pt').is_file():
        raise ValueError('Verified run is missing source or final weights')
    artifacts = {'results.json': file_hash(workspace / 'results.json')}
    for path in sorted((workspace / 'outputs').rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors'} or 'pred' in rel.as_posix().lower():
            if not path.resolve().is_relative_to(workspace.resolve()):
                raise ValueError('Verified artifacts must remain inside the executed workspace')
            artifacts[rel.as_posix()] = file_hash(path)
    if not any('pred' in name.lower() for name in artifacts):
        raise ValueError('Verified run is missing saved test predictions')
    return {'schema': 1, 'workspace': str(workspace.resolve()), 'report_path': str(report_path.resolve()),
            'report_sha256': file_hash(report_path), 'sources': sources, 'artifacts': artifacts,
            'protocol': protocol_manifest(run_dir), 'config_contract': config_contract(config)}


def validate_receipt(receipt: dict, experiment_dir: Path, run_dir: Path, config) -> Path:
    if receipt.get('schema') != 1 or not receipt.get('sources') or not receipt.get('artifacts'):
        raise ValueError('Missing verified-run manifest')
    workspace = Path(receipt['workspace'])
    # A receipt can only refer to this project's executed workspace, not another run.
    if not workspace.resolve().is_relative_to((run_dir / 'stage-14').resolve()):
        raise ValueError('Verified workspace is outside this project S14')
    report = Path(receipt['report_path'])
    if not report.resolve().is_relative_to((run_dir / 'stage-14').resolve()):
        raise ValueError('Verified execution report is outside S14')
    if file_hash(report) != receipt['report_sha256']:
        raise ValueError('Supervised execution evidence changed')
    if source_manifest(workspace) != receipt['sources'] or source_manifest(experiment_dir) != receipt['sources']:
        raise ValueError('Executed experiment source changed; refusing automatic retraining')
    if protocol_manifest(run_dir) != receipt['protocol'] or config_contract(config) != receipt['config_contract']:
        raise ValueError('Experiment protocol/configuration changed')
    for name, expected in receipt['artifacts'].items():
        path = workspace / name
        if not path.resolve().is_relative_to(workspace.resolve()) or file_hash(path) != expected:
            raise ValueError(f'Verified artifact missing or changed: {name}')
    # Recheck the provenance and all required measurements, not just a file's existence.
    supervised = json.loads(report.read_text(encoding='utf-8'))
    if not supervised.get('success') or supervised.get('smoke') or not supervised.get('require_training'):
        raise ValueError('Receipt does not describe a verified full training run')
    return workspace
