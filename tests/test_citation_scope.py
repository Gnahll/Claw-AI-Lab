import ast,json,logging,re,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from researchclaw.literature.verify import CitationResult,VerificationReport,VerifyStatus,parse_bibtex_entries
SOURCE=Path(os.environ.get('CITATION_EXECUTOR_SOURCE',str(Path(__file__).resolve().parents[1]/'backend/agent/researchclaw/pipeline/executor.py')))
tree=ast.parse(SOURCE.read_text());names={'_paper_citation_keys','_bibliography_entries','_execute_citation_verify','_repair_reference_metadata','_check_citation_relevance'}
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
ns={'re':re,'json':json,'logger':logging.getLogger(__name__),'Stage':SimpleNamespace(CITATION_VERIFY=26),'StageStatus':SimpleNamespace(DONE='done',FAILED='failed'),'StageResult':lambda **kw:SimpleNamespace(**kw),'_escape_bib_specials':lambda x:x,'_correct_bibtex_dois':lambda x,y:x}
exec(compile(ast.fix_missing_locations(module),str(SOURCE),'exec'),ns)
def bib(k):return '@article{'+k+',\n title = {Title '+k+'},\n year = {2020}\n}\n'
class ScopeTest(unittest.TestCase):
 def run_case(self,paper,skipped=False):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);pool=bib('a2020')+bib('unused2021');calls=[]
   ns['_read_prior_artifact']=lambda r,n:pool if n=='references.bib' else paper
   def verify(text,**kw):
    keys=[e['key'] for e in parse_bibtex_entries(text)];calls.extend(keys)
    return VerificationReport(total=len(keys),verified=0 if skipped else len(keys),skipped=len(keys) if skipped else 0,results=[CitationResult(k,k,VerifyStatus.SKIPPED if skipped else VerifyStatus.VERIFIED,1,'test') for k in keys])
   with patch('researchclaw.literature.verify.verify_citations',side_effect=verify):
    result=ns['_execute_citation_verify'](d,d,SimpleNamespace(llm=SimpleNamespace()),None)
   return result,json.loads((d/'verification_report.json').read_text()),calls,(d/'references_verified.bib').read_text()
 def test_only_cited_and_preserves_pool(self):
  r,d,c,b=self.run_case('[a2020]');self.assertEqual(c,['a2020']);self.assertEqual(r.status,'done');self.assertIn('unused2021',b);self.assertEqual(d['summary']['uncited_bibliography_total'],1)
 def test_missing_key_fails(self):
  r,d,c,b=self.run_case('[missing2020]');self.assertEqual(c,[]);self.assertEqual(r.status,'failed');self.assertEqual(d['summary']['unresolved_citation_keys'],['missing2020'])
 def test_timeout_not_success(self):
  r,d,c,b=self.run_case('[a2020]',True);self.assertEqual(r.status,'failed');self.assertEqual(d['summary']['pending_citation_keys'],['a2020'])
 def test_no_manuscript_does_not_query_pool(self):
  r,d,c,b=self.run_case('');self.assertEqual(c,[]);self.assertEqual(r.status,'failed')
 def test_explicit_nocite_all(self):
  r,d,c,b=self.run_case(r'\nocite{*}');self.assertEqual(set(c),{'a2020','unused2021'});self.assertEqual(r.status,'done')
 def test_syntax(self):
  text=r'\citep[see][p.2]{a2020,b2021} [@c2022; @d2023, p.4] @e2024 `[@fake2020]` <!-- [bad2020] --> mail@invalid2020.com'
  self.assertEqual(ns['_paper_citation_keys'](text),{'a2020','b2021','c2022','d2023','e2024'})

class MetadataTest(unittest.TestCase):
 def test_note_recovery_and_idempotence(self):
  raw='@misc{dai2017scannet,\n note = {Dai, A., et al. *ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes.* CVPR, 2017. ([arxiv.org](https://arxiv.org/abs/1702.04405?utm_source=openai))}\n}'
  repaired,keys=ns['_repair_reference_metadata'](raw);d=parse_bibtex_entries(repaired)[0]
  self.assertEqual(d['title'],'ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes');self.assertEqual(d['eprint'],'1702.04405');self.assertEqual(d['year'],'2017');self.assertEqual(keys,['dai2017scannet']);self.assertEqual(ns['_repair_reference_metadata'](repaired),(repaired,[]));self.assertIn('note =',repaired)
 def test_title_does_not_include_venue(self):
  raw='@misc{q2022,\n title = {*Geometric Transformer.* CVPR, 2022},\n year = {2022}\n}'
  fixed,_=ns['_repair_reference_metadata'](raw);self.assertEqual(parse_bibtex_entries(fixed)[0]['title'],'Geometric Transformer')
 def test_no_guess_for_plain_note(self):
  raw='@misc{x2020,\n note = {Ambiguous text without explicit title}\n}';self.assertEqual(ns['_repair_reference_metadata'](raw),(raw,[]))
 def test_existing_valid_title_unchanged(self):
  raw=bib('a2020');self.assertEqual(ns['_repair_reference_metadata'](raw),(raw,[]))
 def test_relevance_receives_citation_context(self):
  ns['_safe_json_loads']=lambda text,default:json.loads(text)
  prompts=[]
  llm=SimpleNamespace(chat=lambda messages,**kw:(prompts.append(messages[0]['content']) or SimpleNamespace(content='{"a2020":0.9}')))
  row=CitationResult('a2020','Different task',VerifyStatus.VERIFIED,1,'test')
  ns['_check_citation_relevance'](llm,'Instance segmentation',[row],paper='We contrast reconstruction with segmentation [a2020].')
  self.assertIn('We contrast reconstruction',prompts[0]);self.assertIn('not title similarity alone',prompts[0])
if __name__=='__main__':unittest.main()


