"""Native pilot review at the paired A/B/GSF dispatch boundary."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_experiment_loop as loop
import ai_experiment_promotion as gate
from ai_experiment_pilot import install_promotion_hook


class PairedPromotionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name)
        self.run = self.project / "run-test"
        self.run.mkdir()

    def put(self, name, obj):
        file = self.run / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(obj))

    def prepare(self, family):
        filename, table, names = loop.FAMILIES[family]
        self.policy = {"enabled":True, "family":family, "initial_run":self.run.name, "initial_unit":"pilot",
                       "scenes":2, "min_ap_gain":0.5, "max_ap25_drop":0.1, "max_retries_per_run":2}
        loop.write(self.project / loop.POLICY, self.policy)
        metrics = {n:{key:(2.0 if n == names[0] else 1.0) for key in ("AP","AP25","AP50","AP75")} for n in names}
        setup = {"scene_ids":["s1","s2"], "target_steps":8, "short_endpoint":8,
                 "long_steps":200, "frozen_metrics":metrics[names[-1]], "protocol_digest":"protocol"}
        self.put("setup.json", setup)
        self.put("preflight_mechanism.json", {"passed":True, "nonfinite_signal":False})
        self.put("mechanism_plan.json", {"pilot_steps":8, "steps_per_arm":8})
        (self.run / "binding.py").write_text("pass")
        (self.run / "dispatch.py").write_text("pass")
        self.put("ai_revision.json", {"revision":"test","method_file_sha256":{"binding.py":gate._hash(self.run/"binding.py")}})
        self.put("status.json", {"phase":"completed","counts":{n:{"scenes":2} for n in names[:2]}})
        report = {"scenes_each":2, table:metrics.copy(), "same_batches":True, "step":8, "stage_updates":8, "weights":{}}
        for arm in names[:2]:
            data = arm.encode()
            digest = hashlib.sha256(data).hexdigest()
            if family == "osgk_depth_order":
                path = Path("short")/arm/"training"/arm
                self.put(Path("short")/arm/"pooled312.json",
                         {"step":8,"scenes":2,"AP_points":metrics[arm],"checkpoint_sha256":digest})
            elif family == "gsf_spectral":
                path = Path(arm)/"training"/arm
                report["weights"][arm] = {"sha256":digest}
                self.put("training_audit.json", {"matched_actual_samples_frames_and_lr":True,"adapter_updates_per_condition":8})
            else:
                path = Path(arm)/"training"
                self.put("training_audit.json", {"matched_data_and_lr":True,"optimizer_updates_each":8})
                for scene in setup["scene_ids"]:
                    self.put(Path(arm)/"scenes"/scene/"verified.json",
                             {"scene":scene,"checkpoint_sha256":digest,"protocol":"protocol"})
            (self.run/path).mkdir(parents=True, exist_ok=True)
            (self.run/path/"last_resume.pt").write_bytes(data)
            rank_key = "rank_records" if family == "osgk_depth_order" else "ranks"
            sample_key = "sample_indices" if family == "cirb_mask_boundary" else "sample_ids"
            with (self.run/path/"steps.jsonl").open("w") as ledger:
                for step in range(1, 9):
                    ranks = [{sample_key:[i+step], "frame_ids":[step,step+1]} for i in range(4)]
                    ledger.write(json.dumps({"step":step, rank_key:ranks}) + "\n")
        self.put(filename, report)

    def test_review_boundary_for_all_three_families(self):
        for family in loop.FAMILIES:
            with self.subTest(family=family):
                self.run = self.project / ("run-"+family)
                self.run.mkdir()
                self.prepare(family)
                self.assertFalse(gate.may_start_long(self.run, family))
                model = Mock(return_value={"action":"approve_long","diagnosis":"verified paired budget","hypothesis":"continue"})
                self.assertEqual(gate.review_request(self.project,self.run,family,model)["action"], "approve_long")
                self.assertTrue(gate.may_start_long(self.run,family))
                gate.review_request(self.project,self.run,family,model)
                model.assert_called_once()
                weight = next(self.run.rglob("last_resume.pt"))
                weight.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    gate.may_start_long(self.run,family)

    def test_rejection_and_model_retry(self):
        family = "cirb_mask_boundary"
        self.prepare(family)
        self.assertFalse(gate.may_start_long(self.run,family))
        model = Mock(side_effect=[TimeoutError(), {}, {"action":"revise_method","diagnosis":"control concern","hypothesis":"compare"}])
        for _ in range(2):
            self.assertEqual(gate.review_request(self.project,self.run,family,model)["action"], "await_ai")
            self.assertFalse(gate.may_start_long(self.run,family))
        gate.review_request(self.project,self.run,family,model)
        self.assertFalse(gate.may_start_long(self.run,family))
        gate.review_request(self.project,self.run,family,model)
        self.assertEqual(model.call_count,3)

    def test_loop_approves_then_launches_same_run(self):
        family = "cirb_mask_boundary"
        self.prepare(family)
        gate.may_start_long(self.run,family)
        model = Mock(return_value={"action":"approve_long","diagnosis":"pilot passed","hypothesis":"continue paired run"})
        launch = Mock()
        for _ in range(2):
            loop.tick(self.project,self.project,model=model,active=lambda _:False,launch=launch)
        self.assertEqual(launch.call_args.args[1],self.run)
        self.assertTrue(gate.may_start_long(self.run,family))
        model.assert_called_once()

    def test_numeric_failure_or_stale_request(self):
        family = "gsf_spectral"
        self.prepare(family)
        gate.may_start_long(self.run,family)
        setup = loop.read(self.run/"setup.json")
        self.put("setup.json",dict(setup,long_steps=400))
        model = Mock()
        with self.assertRaisesRegex(ValueError, "changed"):
            gate.review_request(self.project,self.run,family,model)
        model.assert_not_called()
        self.policy["min_ap_gain"] = 4
        loop.write(self.project/loop.POLICY,self.policy)
        with self.assertRaisesRegex(ValueError, "gate"):
            gate.may_start_long(self.run,family)

    def test_install_hooks_before_long_entrypoints(self):
        for family in loop.FAMILIES:
            file = self.run/"dispatch.py"
            if family=="osgk_depth_order":
                source="for item in []:\n    if item:\n        if item:\n            if item:\n                if not passed:break\n"
            else:
                source="def main():\n    supervise(r)\n    long=r.parent/'run-达标接续长训'\n    if not passed:return\n    supervise(long)\n"
            file.write_text(source, encoding="utf-8")
            install_promotion_hook(self.run, family)
            hooked=file.read_text(encoding="utf-8")
            install_promotion_hook(self.run,family)
            self.assertEqual(file.read_text(encoding="utf-8"),hooked)
            self.assertIn(f"not may_start_long(",hooked)
            if family!="osgk_depth_order":
                self.assertLess(hooked.index("if not may_start_long("),hooked.index("supervise(long)"))

    def test_verified_budget_requires_matched_four_rank_updates(self):
        family = "cirb_mask_boundary"
        self.prepare(family)
        ledger = self.run/"depth/training/steps.jsonl"
        rows = ledger.read_text().splitlines()
        for description, broken in [
            ("truncation", rows[:-1]),
            ("duplicate", [rows[0], *rows[:-1]]),
            ("missing rank", [json.dumps({"step":1, "ranks":[]}), *rows[1:]]),
            ("different sample", [rows[0].replace('"sample_indices": [1]', '"sample_indices": [99]'), *rows[1:]]),
        ]:
            with self.subTest(description):
                ledger.write_text("\n".join(broken)+"\n")
                with self.assertRaises(ValueError):
                    gate.may_start_long(self.run,family)
                self.assertFalse((self.run/gate.REQUEST).exists())
        ledger.write_text("\n".join(rows)+"\n")
        self.put("mechanism_plan.json", {"pilot_steps":8, "steps_per_arm":900})
        with self.assertRaisesRegex(ValueError, "budgets"):
            gate.may_start_long(self.run,family)

    def test_api_retry_can_reuse_unchanged_checkpoint_hash(self):
        self.prepare("cirb_mask_boundary")
        weight = self.run/"depth/training/last_resume.pt"
        with patch.object(gate, "_hash", wraps=gate._hash) as hash_file:
            for _ in range(2):
                gate.evidence(self.run, "cirb_mask_boundary")
            self.assertEqual(sum(call.args == (weight,) for call in hash_file.call_args_list), 1)
            weight.write_bytes(b"different snapshot")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                gate.evidence(self.run, "cirb_mask_boundary")
            self.assertEqual(sum(call.args == (weight,) for call in hash_file.call_args_list), 2)

if __name__ == "__main__":
    unittest.main()
