from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_experiment_loop as loop
from ai_experiment_cirb import proposal, validate_expression
from ai_experiment_geometry import OSGK, SPECTRAL, slots_for, verify_numerics
from ai_experiment_pilot import revision_plan
import ai_experiment_probe_runtime as probe


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
            "def objective(current,guard,parts,wb):\n            targets.append((parts*wb).sum(1).mean()+.5*(1-current.mean())+guard)\n"
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
            "max_retries_per_run": 3, "api_retry_seconds": 600,
        }
        self.write(self.project / loop.POLICY, self.policy)
        self.calls = []
        self.started = []

    def write(self, path, data):
        path.write_text(json.dumps(data))

    def state(self):
        return loop.read(self.project / loop.STATE)

    def tick(self, action=None, active=False, edits=None):
        def model(_, payload):
            self.calls.append(payload)
            return {"action": action, "diagnosis": "observed problem", "hypothesis": "one changed factor",
                    "parent": "initial", "edits": edits,
                    "experiment":{"observable":next(iter(edits or {}),""),"budget":"screen",
                                  "question":"Which signal?","expected":"activity differs","budget_reason":"screen"}}
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
        self.tick("revise_method", edits={"mask_delta":"x.tanh() * boundary.square() * .55"})
        self.assertEqual(self.state()["status"], "preparing")
        with patch("ai_experiment_pilot.install"):
            self.tick()
        variant = self.started[0][1]
        new = loop.read(variant / "setup.json")
        self.assertEqual(new["target_steps"], 1350)
        self.assertEqual(len(new["scene_ids"]), 312)
        self.assertIn("boundary.square()", (variant / "experiment" / "directed_boundary.py").read_text())
        self.assertNotIn("boundary.square()", (self.run / "experiment" / "directed_boundary.py").read_text())
        self.assertEqual(len(self.state()["tested"]), 1)
        self.assertEqual(self.state()["formulas"]["mask_delta"], "x.tanh() * boundary.square() * 0.55")

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

    def test_does_not_execute_arbitrary_python_or_repeat(self):
        for code in ("__import__('os')", "torch.load(x)", "x.__class__", "[x for x in x]"):
            with self.assertRaises(ValueError):
                validate_expression("mask_delta", code)
        request = {"parent":"initial","edits":{"mask_delta":"0.4*x.tanh()*boundary"}}
        _, key = proposal(request, {}, [])
        with self.assertRaises(ValueError):
            proposal(request, {}, [key])

    def test_numerically_invalid_revision_does_not_start(self):
        self.result(9)
        self.tick("revise_method", edits={"mask_delta":"boundary"})
        self.assertEqual(self.state()["status"], "await_ai")
        self.assertEqual(self.started, [])

    def test_accepts_structured_diagnosis(self):
        body = {"action":"resume_checkpoint","diagnosis":{"observation":"timeout"},"hypothesis":["retry same checkpoint"]}
        data = loop.validated_choice(body, {"resume_checkpoint":"resume"})
        self.assertIn("timeout", data["diagnosis"])
        self.assertIn("retry same checkpoint", data["hypothesis"])

    def test_each_geometry_family_has_its_own_slots(self):
        for family, name, code in (
            (OSGK,"near_far_context","(near-far)/(1+(near-far).abs())"),
            (SPECTRAL,"graph_detail","(weighted-unweighted)/(1+shared.abs())"),
        ):
            formulas, _ = proposal({"parent":"initial","edits":{name:code}}, {}, [], slots_for(family))
            verify_numerics(family, formulas)
            self.assertNotIn("mask_delta", slots_for(family))

    def test_osgk_requires_paired_result_receipts(self):
        self.result(10.6)
        metrics = loop.read(self.run/"comparison.json")["metrics"]
        result = {"scenes_each":312,"same_batches":True,
                  "metrics":{"osgf32":metrics["depth"],"four_scale_baseline":metrics["brightness"]}}
        folder = self.run/"short"
        folder.mkdir()
        self.write(folder/"comparison.json", result)
        self.assertIsNone(loop.terminal_result(self.run,312,OSGK))
        for name in result["metrics"]:
            dest = folder/name
            dest.mkdir()
            self.write(dest/"pooled312.json",{"scenes":312})
        table = loop.terminal_result(self.run,312,OSGK)
        self.assertTrue(loop.is_pass(table,dict(self.policy,family=OSGK)))

    def test_spectral_gate_checks_frozen_anchor_in_addition_to_control(self):
        def values(ap):
            return {"AP":ap,"AP25":42.,"AP50":25.,"AP75":6.}
        self.write(self.run/"setup.json",{"frozen_metrics":values(10.2)})
        self.write(self.run/"status.json",{"phase":"completed","counts":{
            "geometry_spectral":{"scenes":312},"local_correction":{"scenes":312}
        }})
        self.write(self.run/"full_validation_result.json",{"scenes_each":312,"comparisons":{
            "geometry_spectral":values(10.6),"local_correction":values(10.)
        }})
        table = loop.terminal_result(self.run,312,SPECTRAL)
        self.assertFalse(loop.is_pass(table,dict(self.policy,family=SPECTRAL)))

    def test_budget_and_indistinguishable_spectral_control(self):
        reply = {"parent":"initial","experiment":{"observable":"graph_detail","budget":"compare",
                 "question":"separate graph evidence","expected":"counterfactual differs","budget_reason":"matched"}}
        with self.assertRaisesRegex(ValueError,"same graph signal"):
            revision_plan(reply,{"graph_detail":"shared-unweighted"},{},SPECTRAL)
        a = revision_plan(reply,{"graph_detail":"(weighted-unweighted)+shared"},{},SPECTRAL)
        self.assertEqual(a["steps_per_arm"],900)

    def test_real_signal_gate_rejects_same_control(self):
        import importlib
        import torch
        probe_test = importlib.reload(probe)
        dest = self.project/"pilot.json"
        probe_test.configure(dest,start_step=0,pilot_steps=1,changed_slots=["new"])
        probe_test.begin(1)
        value=torch.ones(4)
        probe_test.observe("new", value*1.2,value)
        probe_test.observe("control",value,value)
        with self.assertRaisesRegex(RuntimeError,"MechanismCheckFailed"):
            probe_test.finish(1,torch.device("cpu"))
        self.assertFalse(loop.read(dest)["control_separable"])

    def test_reject_run_escape(self):
        with self.assertRaises(ValueError):
            loop.child(self.project, "run-../escape")


if __name__ == "__main__":
    unittest.main()
