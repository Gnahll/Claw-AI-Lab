"""Completion, provenance and durable queue regression tests (CPU only)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'backend/agent'
sys.path.insert(0, str(AGENT))
spec = importlib.util.spec_from_file_location('external_results', AGENT / 'researchclaw/pipeline/external_results.py')
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
sys.modules['researchclaw.pipeline.external_results'] = ex
spec = importlib.util.spec_from_file_location('bridge_test', ROOT / 'backend/services/agent_bridge.py')
bridge = importlib.util.module_from_spec(spec); sys.modules[spec.name] = bridge; spec.loader.exec_module(bridge)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / 'projects/project'; self.project.mkdir(parents=True)
        self.source = self.project / 'run-training'; self.source.mkdir()
        self.context = self.project / 'run-context'; self.context.mkdir()
        (self.context / 'stage-09').mkdir(); (self.context / 'stage-09/exp_plan.yaml').write_text('conditions: [baseline, method]')
        self.config = self.root / 'config.yaml'; self.config.write_text('test: true')
        ex.write(self.project / 'project_meta.json', {'topic': 'Test', 'config_path': str(self.config)})
        self.policy = {'enabled': True, 'source_run': str(self.source), 'context_run': str(self.context),
                       'config_path': str(self.config), 'conditions': ['baseline', 'method'],
                       'target_steps': 11070, 'target_epochs': 5, 'artifacts': ['final.pt'], 'analysis_scope': 'Subset only, one seed.'}
        self.result = {'conditions': {k: {'training': {'completed_steps': 11070, 'unique_steps': 11070},
                                        'evaluation': {'iou_50_90': {'AP': .2}}} for k in ['baseline', 'method']}}
        ex.write(self.source / 'results.json', self.result)
        ex.write(self.source / 'queue_status.json', {'state': 'completed', 'target_steps': 11070, 'epochs': 5})
        (self.source / 'exit_code.txt').write_text('0'); (self.source / 'final.pt').write_bytes(b'fixture weights')
        ex.write(self.project / 'external_execution.json', {'run_dir': str(self.source), 'target_steps': 11070, 'target_epochs': 5})
        ex.write(self.project / 'external_handoff_policy.json', self.policy)

    def test_import_review_and_idempotence(self):
        run = ex.prepare(self.project, self.policy)
        self.assertEqual(ex.prepare(self.project, self.policy), run)
        self.assertEqual(ex.read(run / 'checkpoint.json')['last_completed_stage'], 14)
        ex.review(run, run / 'stage-15')
        self.assertFalse(ex.read(run / 'stage-15/refinement_log.json')['retrained'])
        self.assertEqual((run / 'stage-15/results.json').read_bytes(), (self.source / 'results.json').read_bytes())

    def test_three_result_schemas(self):
        for evaluation in [{'iou_50_90': {'AP': .2}}, {'iou_50_90': 20.0}, {'AP_50_95_points': 20.0}]:
            for row in self.result['conditions'].values(): row['evaluation'] = evaluation
            ex.write(self.source / 'results.json', self.result)
            ex._validate_source(self.project, self.policy)

    def test_incomplete_condition_rejected(self):
        self.result['conditions']['method']['training']['completed_steps'] = 100
        ex.write(self.source / 'results.json', self.result)
        with self.assertRaises(ValueError): ex.prepare(self.project, self.policy)

    def test_failure_and_missing_evaluation_rejected(self):
        (self.source / 'exit_code.txt').write_text('1')
        with self.assertRaises(ValueError): ex.prepare(self.project, self.policy)
        (self.source / 'exit_code.txt').write_text('0')
        self.result['conditions']['method']['evaluation'] = {}
        ex.write(self.source / 'results.json', self.result)
        with self.assertRaises(ValueError): ex.prepare(self.project, self.policy)

    def test_missing_artifact_and_cross_project_rejected(self):
        with self.assertRaises(ValueError): ex.prepare(self.project, {**self.policy, 'artifacts': ['missing.pt']})
        with self.assertRaises(ValueError): ex.prepare(self.project, {**self.policy, 'source_run': str(self.root)})

    def test_changes_after_import_rejected(self):
        run = ex.prepare(self.project, self.policy)
        (self.source / 'final.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError): ex.validate_import(run)

    def test_result_tampering_rejected(self):
        run = ex.prepare(self.project, self.policy)
        (run / 'stage-14/runs/results.json').write_text('{}')
        with self.assertRaises(ValueError): ex.validate_import(run)

    def test_poll_queues_exactly_once_across_reload(self):
        queue = bridge.TaskQueue('coding_to_execution', self.root / 'queue.json')
        state = SimpleNamespace(agent_package_dir=str(AGENT), projects_dir=lambda: self.root / 'projects',
                                queues={'coding_to_execution': queue})
        bridge._advance_external_runs(state); bridge._advance_external_runs(state)
        self.assertEqual(len(queue.tasks), 1)
        queue.load(); bridge._advance_external_runs(state)
        self.assertEqual(len(queue.tasks), 1)
        self.assertEqual(queue.tasks[0].target_layer, 'execution')
        self.assertTrue(Path(queue.tasks[0].run_dir, 'external_result_import.json').exists())

    def test_disabled_policy_never_queues(self):
        ex.write(self.project / 'external_handoff_policy.json', {**self.policy, 'enabled': False})
        queue = bridge.TaskQueue('coding_to_execution', self.root / 'queue.json')
        state = SimpleNamespace(agent_package_dir=str(AGENT), projects_dir=lambda: self.root / 'projects', queues={'coding_to_execution': queue})
        bridge._advance_external_runs(state)
        self.assertEqual(queue.tasks, [])


if __name__ == '__main__': unittest.main()
