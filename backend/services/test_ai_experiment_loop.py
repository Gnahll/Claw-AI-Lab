from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_experiment_loop as loop
from ai_experiment_cirb import VARIANTS, prepare


class ExperimentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.run = self.project / "run-original"
        (self.run / "experiment").mkdir(parents=True)
        for name in ("dispatch.py", "supervisor.py"):
            (self.run / name).write_text("# experiment entry point\n")
        (self.run / "experiment" / "directed_boundary.py").write_text(
            "def objective(current,guard):\n    return (1+.5*(1-current.mean())+guard)\n"
            "def update(x,boundary,present):\n    delta=.80*x.tanh()*boundary*present\n    return delta\n"
        )
        self.write(self.run / "setup.json", {
            "method_version": "cirb_fixed_score_conditional_mask_boundary_v1",
            "protocol_digest": "seed", "target_steps": 1350,
            "long_steps": 11070, "scene_ids": list(range(312)), "gpu_uuids": ["A", "B", "C", "D"],
        })
        self.policy = {
            "enabled": True, "family": "cirb_mask_boundary", "initial_run": self.run.name,
            "initial_unit": "initial", "scenes": 312, "min_ap_gain": .5, "max_ap25_drop": .1,
            "max_retries_per_run": 3, "variants": list(VARIANTS), "api_retry_seconds": 600,
        }
        self.write(self.project / loop.POLICY, self.policy)
        self.calls = []
        self.started = []

    def write(self, path, data):
        path.write_text(json.dumps(data))

    def state(self):
        return loop.read(self.project / loop.STATE)

    def tick(self, action=None, active=False):
        def model(_, payload):
            self.calls.append(payload)
            return {"action": action, "diagnosis": "observed problem", "hypothesis": "one changed factor"}
        loop.tick(self.project, self.project, model=model, active=lambda _: active,
                  launch=lambda unit, run: self.started.append((unit, run)))

    def result(self, ap):
        def metric(a):
            return {"AP":a,"AP25":42.,"AP50":25.,"AP75":6.}
        self.write(self.run / "status.json", {"phase":"completed","counts":{
            "depth":{"scenes":312},"brightness":{"scenes":312}
        }})
        self.write(self.run / "comparison.json", {"scenes_each":312,"metrics":{
            "depth":metric(ap), "brightness":metric(10.), "frozen_anchor":metric(10.),
            "original_baseline":metric(10.),
        }})

    def test_running_is_observed(self):
        self.tick(active=True)
        self.assertEqual(self.state()["status"], "running")
        self.assertEqual(self.calls, [])

    def test_gate_failure_revises_and_preserves_original(self):
        self.result(9)
        self.tick("protect_good_masks")
        self.assertEqual(self.state()["status"], "preparing")
        self.tick()
        variant = self.started[0][1]
        new = loop.read(variant / "setup.json")
        self.assertEqual(new["target_steps"], 1350)
        self.assertEqual(len(new["scene_ids"]), 312)
        self.assertIn("+2.*guard", (variant / "experiment" / "directed_boundary.py").read_text())
        self.assertNotIn("+2.*guard", (self.run / "experiment" / "directed_boundary.py").read_text())
        self.assertEqual(self.state()["tested"], ["protect_good_masks"])

    def test_interruption_requeues_same_checkpoint(self):
        self.tick("resume_checkpoint")
        s = self.state()
        self.assertEqual(s["status"], "launching")
        s["next_check"] = 0
        self.write(self.project / loop.STATE, s)
        self.tick()
        self.assertEqual(self.started[0][1], self.run)

    def test_api_invalid_action_waits_without_launched_job(self):
        self.result(9)
        self.tick("fabricated_action")
        self.assertEqual(self.state()["status"], "await_ai")
        self.assertEqual(self.started, [])

    def test_pass_requires_finished_long_handoff(self):
        self.result(10.6)
        self.write(self.run / "promotion.json", {"status":"long_completed"})
        self.tick()
        self.assertEqual(self.state()["status"], "passed")
        self.assertEqual(self.calls, [])

    def test_busy_and_nonrecoverable_error_use_different_actions(self):
        s = {"tested":[], "retries":0}
        self.assertEqual(set(loop.plan_actions(self.policy,s,["gpu_busy"],"interrupted")), {"await_resources"})
        self.assertEqual(set(loop.plan_actions(self.policy,s,["checkpoint_error"],"interrupted")), {"needs_attention"})

    def test_reject_run_escape(self):
        with self.assertRaises(ValueError):
            loop.child(self.project, "run-../escape")


if __name__ == "__main__":
    unittest.main()
