"""Mechanical publication checks; never manufacture evidence or missing figures."""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path


def _json_lines(path: Path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def manuscript_errors(text: str) -> list[str]:
    # Ignore code blocks and normalize numbered top-level manuscript headings.
    body = re.sub(r'```[\s\S]*?```', '', text)
    headings = []
    for value in re.findall(r'^#{1,2}\s+(.+)$', body, re.M):
        value = re.sub(r'^\d+(?:\.\d+)*[.)]?\s+', '', value).strip().lower()
        value = re.sub(r'\s*\{[^}]*\}\s*$', '', value)
        headings.append(value)
    counts = Counter(headings)
    return [f'Duplicate manuscript section: {key} ({counts[key]})'
            for key in ('title', 'abstract', 'introduction', 'method', 'methods', 'conclusion') if counts[key] > 1]


def audit_package(dest: Path) -> dict:
    def read(name):
        try: return json.loads((dest / name).read_text(encoding='utf-8'))
        except (OSError, ValueError): return {}
    paper = dest / 'paper_final.md'
    text = paper.read_text(encoding='utf-8') if paper.is_file() else ''
    errors = manuscript_errors(text)
    if not text.strip(): errors.append('Manuscript missing or empty')
    for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text):
        if link.startswith(('http://', 'https://', 'data:')): continue
        path = dest / link.split(' "')[0].strip('<>')
        if not path.is_file(): errors.append('Missing figure: ' + link)
    compile_report = read('compile_report.json')
    if compile_report.get('success') is not True or not (dest / 'paper.pdf').is_file():
        errors.append('Compiled PDF missing or compilation did not succeed')
    verification = read('verification_report.json').get('summary', {})
    if verification.get('fully_verified') is not True:
        errors.append('Citation verification incomplete')
    report = {'schema': 1, 'ready': not errors, 'status': 'artifact_checks_passed' if not errors else 'requires_revision',
              'errors': errors, 'scope': 'Mechanical artifact checks only; not scientific acceptance'}
    (dest / 'artifact_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def collect_training_evidence(source: Path, destination: Path) -> dict:
    """Small reproducible summaries, not a copy of multi-GB logs or weights."""
    conditions = json.loads((source / 'results.json').read_text())['conditions']
    roots = [source]
    settings = source / 'supervisor_config.json'
    if settings.is_file():
        predecessor = json.loads(settings.read_text()).get('predecessor')
        if predecessor and Path(predecessor).is_dir(): roots.insert(0, Path(predecessor))
    result = {'scope': 'saved logs only; missing evidence is not zero', 'conditions': {}, 'warnings': []}
    for condition in conditions:
        epochs = {}; counters = []; provenance = []
        for root in roots:
            # Each project may have a different output layout. Associate by exact condition directory.
            paths = sorted((root / 'outputs').glob('**/' + condition + '/epochs.jsonl'))
            for path in paths:
                provenance.append(str(path))
                for row in _json_lines(path):
                    if isinstance(row.get('epoch'), int): epochs[row['epoch']] = row
        # Diagnostics describe only this completed continuation; do not mix schedules.
        for path in sorted((source / 'outputs').glob('**/' + condition + '/steps.jsonl')):
            provenance.append(str(path))
            for row in _json_lines(path):
                losses = row.get('losses', {})
                active = losses.get('fusion_active_window_scales'); bypassed = losses.get('fusion_bypassed_window_scales')
                if isinstance(active, (int, float)) and isinstance(bypassed, (int, float)) and active + bypassed > 0:
                    counters.append({'step': row.get('step'), 'active_fraction': active / (active + bypassed),
                                     'decorrelation': losses.get('L_decorr'), 'relation': losses.get('L_inst_comp')})
        stats = None
        if counters:
            mean = sum(x['active_fraction'] for x in counters) / len(counters)
            stats = {'observations': len(counters), 'positive_observations': sum(x['active_fraction'] > 0 for x in counters),
                     'mean_active_fraction': mean, 'max_active_fraction': max(x['active_fraction'] for x in counters)}
            if mean < .01:
                result['warnings'].append(f'{condition}: mean reported projection-active fraction is {mean:.6%}; '
                    'numerical guards bypassed most interactions. Review mechanism execution before attributing gains to it. '
                    'This is a diagnostic flag, not proof that the run is invalid.')
        result['conditions'][condition] = {'epochs': [epochs[k] for k in sorted(epochs)],
                                          'projection_activity': stats, 'activity_trace': counters, 'sources': provenance}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'training_evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def plot_training_evidence(run_dir: Path, stage_dir: Path) -> list[str]:
    path = run_dir / 'stage-15/training_evidence.json'
    if not path.is_file(): return []
    evidence = json.loads(path.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    charts = stage_dir / 'charts'; charts.mkdir(exist_ok=True)
    artifacts = []
    rows = evidence['conditions']
    if any(x['epochs'] for x in rows.values()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for name, row in rows.items():
            e = row['epochs']
            for ax, key in zip(axes, ['train_loss', 'instance']):
                pairs = [(x['epoch'], x.get('train_loss') if key == 'train_loss' else x.get('validation', {}).get('instance')) for x in e]
                pairs = [(x, y) for x, y in pairs if isinstance(y, (int, float))]
                if pairs: ax.plot(*zip(*pairs), marker='o', label=name)
        for ax, title in zip(axes, ['Recorded training objective (terms may differ)', 'Shared validation instance loss']):
            ax.set(title=title, xlabel='Recorded epoch', ylabel='Loss'); ax.grid(alpha=.2)
            if ax.lines: ax.legend()
        fig.tight_layout(); fig.savefig(charts / 'recorded_training_history.png', dpi=180); plt.close(fig)
        artifacts.append('charts/recorded_training_history.png')
    for name, row in rows.items():
        if not row['activity_trace']: continue
        fig, ax = plt.subplots(figsize=(8, 3))
        trace = row['activity_trace']; ax.plot([x['step'] for x in trace], [100*x['active_fraction'] for x in trace], lw=.7)
        ax.set(title=name + ': recorded projection activity', xlabel='Recorded step', ylabel='Active window-scales (%)')
        fig.tight_layout(); filename = 'projection_activity_' + re.sub(r'[^a-zA-Z0-9_-]', '_', name) + '.png'
        fig.savefig(charts / filename, dpi=180); plt.close(fig); artifacts.append('charts/' + filename)
    return artifacts
