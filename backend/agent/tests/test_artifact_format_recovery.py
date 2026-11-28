import codecs
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from researchclaw.pipeline import artifact_format_recovery as fmt


class ArtifactFormatRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run-1"
        self.run.mkdir()
        self.path = self.run / "schedule.json"

    def check(self, raw, **kwargs):
        self.path.write_bytes(raw)
        with self.assertRaises(json.JSONDecodeError) as captured:
            json.loads(self.path.read_text(encoding="utf-8"))
        return fmt.repair(self.run, self.path, captured.exception, **kwargs)

    def test_one_edit_preserves_original_bytes_and_records_choice(self):
        original = '{"名字":"not ,} syntax",\r\n "n":1.00e+01,\r\n}'.encode()
        replies = []
        def ai(info):
            replies.append(info)
            return {"action": info["operation"]}
        self.assertTrue(self.check(original, choose=ai))
        self.assertEqual(self.path.read_bytes(), original.replace(b'01,', b'01'))
        self.assertNotIn("名字", str(replies))
        audit = next((self.run / ".artifact-format-recovery").glob("*/audit.json"))
        self.assertEqual((audit.parent / "original.bytes").read_bytes(), original)
        self.assertEqual(json.loads(audit.read_text())["status"], "repaired")
        self.assertFalse(self.check(original, choose=ai))
        self.assertEqual(len(replies), 1)

    def test_bom_and_exact_fence(self):
        self.assertTrue(self.check(codecs.BOM_UTF8 + b'{"n": 4}'))
        self.assertTrue(self.check(b'```json\r\n["keep ", "1.00"]\r\n```'))
        self.assertEqual(self.path.read_bytes(), b'["keep ", "1.00"]')

    def test_ambiguous_invalid_content_stays_byte_identical(self):
        for raw in [b'{"name":"unfinished', b'{"v":1,"v":3,}', b'[,]',
                    b'{"n":NaN,}', b'{"n": 1,,}', b'explanation\n```json\n{}\n```']:
            with self.subTest(raw=raw):
                self.assertFalse(self.check(raw))
                self.assertEqual(self.path.read_bytes(), raw)
        self.assertFalse(self.check(codecs.BOM_UTF8 + b'[]', expected=dict))

    def test_external_paths_checkpoint_and_symlink(self):
        self.path = Path(self.temp.name) / "schedule.json"
        self.assertFalse(self.check(b'{"a":1,}'))
        self.path = self.run / "checkpoint.json"
        self.assertFalse(self.check(b'{"a":1,}'))
        outside = Path(self.temp.name) / "schedule.json"
        link = self.run / "schedule.json"
        try:
            os.symlink(outside, link)
        except OSError:
            self.skipTest("symlink creation unavailable")
        self.path = link
        self.assertFalse(self.check(b'{"a":1,}'))
        self.assertEqual(outside.read_bytes(), b'{"a":1,}')

    def test_newer_source_and_unvalidated_model_answers_hold(self):
        def race(info):
            self.path.write_bytes(b'{"replacement":3}')
            return {"action": info["operation"]}
        self.assertFalse(self.check(b'{"a":2,}', choose=race))
        self.assertEqual(self.path.read_bytes(), b'{"replacement":3}')
        for i, reply in enumerate([{"action":"hold"}, {"action":"replace"}, {"action":"remove_trailing_comma", "text":"{}"}]):
            raw = ('{"a":%d,}' % i).encode()
            self.assertFalse(self.check(raw, choose=lambda _: reply))
            self.assertEqual(raw, self.path.read_bytes())


    def test_stage_entry_resumes_validation_with_same_schedule(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from researchclaw.pipeline import executor
        from researchclaw.pipeline.stages import Stage
        stage = self.run / "stage-13"
        stage.mkdir()
        schedule = stage / "schedule.json"
        schedule.write_bytes(b'{"steps":450,}')
        ai = Mock()
        ai.chat.return_value.content = '{"action":"remove_trailing_comma"}'
        with patch.object(executor, "_stage_llm", return_value=ai):
            result = executor.execute_stage(Stage.EXPERIMENT_RUN, run_dir=self.run, run_id="test",
                config=SimpleNamespace(research=SimpleNamespace(details="")), adapters=None)
        # The original stage consumes the corrected input and then reports its
        # next independent missing requirement; no training process is started.
        self.assertIn("Missing input: experiment/", result.error)
        self.assertEqual(schedule.read_bytes(), b'{"steps":450}')
        self.assertEqual(ai.chat.call_count, 1)

    def test_experiment_tick_retries_inactive_run_reader_once(self):
        from unittest.mock import patch
        try:
            import fcntl
        except ImportError:
            self.skipTest("experiment service uses Linux file locks")
        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo / "backend/services"))
        self.addCleanup(sys.path.remove, str(repo / "backend/services"))
        import ai_experiment_loop as loop
        project = self.run.parent
        (project / loop.POLICY).write_text("{}")
        (project / loop.STATE).write_text(json.dumps(dict(run=self.run.name, unit="training",
            status="running", history=[])))
        status = self.run / "status.json"
        status.write_text('{"phase":"training",}')
        calls = []
        def current_tick(*args):
            calls.append(True)
            loop.read(status)
        with patch.object(loop, "_tick", side_effect=current_tick):
            loop.tick(project, repo, model=lambda repo, data: {"action":data["parser"]["operation"]},
                      active=lambda unit:False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(status.read_bytes()), {"phase":"training"})
        self.assertEqual(loop.read(project / loop.STATE)["history"][-1]["action"], "format_repaired")


if __name__ == "__main__":
    unittest.main()
