import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'backend/agent/researchclaw/pipeline/artifact_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
 def write(self,path,value):
  p=self.p/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
 def test_duplicate_numbered_manuscript(self):
  self.assertTrue(audit.manuscript_errors('## Abstract\na\n## 1. Abstract\nb'))
 def test_code_examples_are_not_sections(self):
  self.assertFalse(audit.manuscript_errors('## Abstract\na\n```\n## Abstract\n```'))
 def test_normal_subsections(self):
  self.assertFalse(audit.manuscript_errors('## Introduction\na\n### Introduction to dataset\nb\n## Results'))
 def valid(self):
  (self.p/'paper_final.md').write_text('## Abstract\nText\n## Introduction\nText')
  (self.p/'paper.pdf').write_bytes(b'%PDF-test-fixture')
  self.write('compile_report.json',{'success':True});self.write('verification_report.json',{'summary':{'fully_verified':True}})
 def test_good_package(self):
  self.valid();self.assertTrue(audit.audit_package(self.p)['ready'])
 def test_missing_compiler_and_false_completion(self):
  self.valid();self.write('compile_report.json',{'success':False});self.assertFalse(audit.audit_package(self.p)['ready'])
 def test_missing_pdf(self):
  self.valid();(self.p/'paper.pdf').unlink();self.assertFalse(audit.audit_package(self.p)['ready'])
 def test_missing_figure(self):
  self.valid();(self.p/'paper_final.md').write_text('![missing](charts/no.png)');self.assertIn('Missing figure:',str(audit.audit_package(self.p)))
 def test_unverified_citations(self):
  self.valid();self.write('verification_report.json',{'summary':{'fully_verified':False}});self.assertFalse(audit.audit_package(self.p)['ready'])
 def test_mechanism_activity_not_assumed(self):
  source=self.p/'run';out=self.p/'review';source.mkdir();(source/'results.json').write_text(json.dumps({'conditions':{'method':{},'baseline':{}}}))
  d=source/'outputs/engineering/method';d.mkdir(parents=True)
  (d/'steps.jsonl').write_text('\n'.join(json.dumps({'step':i,'losses':{'fusion_active_window_scales':0,'fusion_bypassed_window_scales':24}}) for i in range(5)))
  report=audit.collect_training_evidence(source,out)
  self.assertEqual(report['conditions']['method']['projection_activity']['mean_active_fraction'],0)
  self.assertIsNone(report['conditions']['baseline']['projection_activity'])
  self.assertEqual(len(report['warnings']),1)
if __name__=='__main__':unittest.main()
