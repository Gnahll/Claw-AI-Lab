"""Install an instrumented, budgeted training entrypoint in a new experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ai_experiment_cirb import SLOTS as B_SLOTS, validate_expression
from ai_experiment_geometry import OSGK, SPECTRAL, slots_for

B_FAMILY = "cirb_mask_boundary"
BUDGETS = {
    OSGK: {"screen": 150, "compare": 300, "confirm": 450},
    B_FAMILY: {"screen": 450, "compare": 900, "confirm": 1350},
    SPECTRAL: {"screen": 450, "compare": 900, "confirm": 1350},
}


def revision_plan(reply, formulas, previous, family):
    slots = B_SLOTS if family == B_FAMILY else slots_for(family)
    before = previous if reply["parent"] == "current" else {}
    changed = sorted(name for name in set(before) | set(formulas) if before.get(name) != formulas.get(name))
    design = reply.get("experiment")
    if not isinstance(design, dict) or design.get("budget") not in BUDGETS[family]:
        raise ValueError("Choose an experiment budget: screen, compare or confirm")
    observation = design.get("observable")
    slot = observation.get("slot") if isinstance(observation, dict) else observation
    if slot not in changed and isinstance(observation,str):
        candidates = [name for name in changed if re.search(r"\b"+name+r"\b",observation)]
        if len(candidates) == 1:
            slot = candidates[0]
    if slot not in changed and len(changed) == 1 and observation:
        slot = changed[0]
    if slot not in changed:
        raise ValueError("experiment.observable must name one changed insertion point, e.g. graph_detail")
    detail = (json.dumps(observation,ensure_ascii=False) if not isinstance(observation,str) else observation)[:700]
    plan = {"parent_formulas":before,"changed_slots":changed,"budget_profile":design["budget"],
            "steps_per_arm":BUDGETS[family][design["budget"]],"pilot_steps":8,"observable":slot,
            "observable_detail":detail}
    for key in ("question","expected","budget_reason"):
        value = design.get(key)
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Experiment design omitted {key}")
        plan[key] = value.strip()[:700]
    if family == SPECTRAL:
        detail = formulas.get("graph_detail", slots["graph_detail"]["default"])
        if re.search(r"\bweighted\b",detail) is None:
            raise ValueError("Both spectral arms would execute the same graph signal; retain a testable use of weighted")
    return plan


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Expected source marker was not found once: "+old)
    return source.replace(old, new)


def _append(source, marker, lines):
    i = source.index(marker)
    pad = re.match(r" *",source[source.rfind("\n",0,i)+1:i]).group(0)
    return _replace(source, marker, marker+"\n"+"\n".join(pad+line for line in lines))


def _write(file, source):
    compile(source,str(file),"exec")
    file.write_text(source,encoding="utf-8")


def _report(slot, expression, old):
    return [f"if _probe.active(): _probe.observe({slot!r}, {expression}, ({old}))"]


def _instrument_b(file, formulas, parent, changed):
    source = "import mechanism_probe as _probe\n"+file.read_text()
    anchor = "            fields,support=neighbor_fields(depth,conf,images,self.use_depth)"
    source = _replace(source,anchor,anchor+"\n            alternate_fields=neighbor_fields(depth,conf,images,False)[0] if _probe.active() else None")
    source = _append(source,"evidence=torch.zeros_like(low)",["counterfactual = torch.zeros_like(low) if _probe.active() else None"])
    message = formulas.get("neighbor_message",B_SLOTS["neighbor_message"]["default"])
    for slot in changed:
        old = parent.get(slot,B_SLOTS[slot]["default"])
        if slot == "edge_affinity":
            marker = next(l.strip() for l in source.splitlines() if l.startswith("        condition="))
            source = _append(source,marker,_report(slot,"condition",old))
        elif slot == "neighbor_message":
            marker = next(l.strip() for l in source.splitlines() if l.startswith("                    evidence=evidence+"))
            source = _append(source,marker,_report(slot,f"({message})",old))
        elif slot == "mask_delta":
            marker = next(l.strip() for l in source.splitlines() if l.startswith("                delta="))
            source = _append(source,marker,_report(slot,"delta",f"({old})*present"))
        elif slot == "training_loss":
            marker = next(l.strip() for l in source.splitlines() if l.startswith("            targets.append"))
            lines = []
            if "classification=(parts*wb).sum(1).mean()" not in source:
                lines.append("classification=(parts*wb).sum(1).mean()")
            lines.extend(_report(slot,"targets[-1]",old))
            source = _append(source,marker,lines)
    alternate = re.sub(r"\bcondition\b","alternate_condition",message)
    marker = next(l.strip() for l in source.splitlines() if l.startswith("                    evidence=evidence+"))
    source = _append(source,marker,[
        "if counterfactual is not None:",
        "    alternate_condition = c[:,:,2*j+1,None,None,None]*alternate_fields[2*j+1][:,None]",
        f"    counterfactual=counterfactual+({alternate})",
    ])
    source = _replace(source,"                x=F.interpolate(evidence.flatten",'''                if counterfactual is not None:
                    _probe.observe("control", evidence, counterfactual)
                x=F.interpolate(evidence.flatten''')
    _write(file,source)


def _instrument_a(file, formulas, parent, changed):
    source = "import mechanism_probe as _probe\n"+file.read_text()
    slots = slots_for(OSGK)
    for slot in changed:
        old = parent.get(slot,slots[slot]["default"])
        if slot in ("neighbor_affinity","ordinal_bias","near_far_context"):
            prefix,variable = {
                "neighbor_affinity":("    common =","common"), "ordinal_bias":("    ordinal =","ordinal"),
                "near_far_context":("            context =","context"),
            }[slot]
            marker = next(l.strip() for l in source.splitlines() if l.startswith(prefix))
            if slot == "near_far_context":
                old = f"({old})*both_sides"
            source = _append(source,marker,_report(slot,variable,old))
        else:
            marker = next(l.strip() for l in source.splitlines() if l.startswith("            added=(instance.float()+delta"))
            source = _append(source,marker,_report(slot,"added.float()-instance.float()",f"delta*torch.minimum(({old}).clamp(0.,2.),cap/rms)"))
    marker = next(l.strip() for l in source.splitlines() if l.startswith("            added=(instance.float()+delta"))
    source = _append(source,marker,['if _probe.active(): _probe.observe("control", added, instance)'])
    _write(file,source)


def _instrument_g(file, formulas, parent, changed):
    source = "import mechanism_probe as _probe\n"+file.read_text()
    slots = slots_for(SPECTRAL)
    detail = formulas.get("graph_detail",slots["graph_detail"]["default"])
    for slot in changed:
        old = parent.get(slot,slots[slot]["default"])
        if slot == "graph_detail":
            marker = next(l.strip() for l in source.splitlines() if l.startswith("            detail="))
            source = _append(source,marker,_report(slot,"detail",old))
        else:
            marker = next(l.strip() for l in source.splitlines() if l.startswith("            coefficient="))
            source = _append(source,marker,_report(slot,"coefficient",old))
    expression = re.sub(r"\bweighted\b","counterfactual_weighted",detail)
    marker = next(l.strip() for l in source.splitlines() if l.startswith("            detail="))
    source = _append(source,marker,[
        "if _probe.active():",
        "    weights = self.weights*support",
        "    order = weights.argsort(dim=1,descending=True,stable=True)",
        "    position = order.argsort(dim=1)",
        "    reverse = (support.sum(dim=1,keepdim=True)-1-position).clamp_min(0)",
        "    counterfactual_weighted = diffuse(shared, weights.gather(1,order.gather(1,reverse))*support)",
        f'    _probe.observe("control", detail, ({expression}))',
    ])
    _write(file,source)

def patch_osgk_checkpoint_handling(dispatch: Path, trainer: Path):
    restore = "def recover_final_receipt(out, arm, ep):\n    import torch\n    path=out/'last_resume.pt'\n    if not path.is_file() or (out/'completion.json').exists():\n        return\n    checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)\n    endpoint=ep['target_steps']\n    if checkpoint.get('step')!=endpoint:\n        return\n    source=Path(ep['source_checkpoint'])\n    assert driver.sha(source)==ep['starting_sha256'], 'Checkpoint lineage mismatch'\n    assert checkpoint['condition']==arm and checkpoint['world_size']==4\n    assert checkpoint['source_sha256']==ep['starting_sha256'] and checkpoint['source_step']==ep['source_step']\n    assert checkpoint['scheduler']['last_epoch']==endpoint-300 and checkpoint['consumed_batches_per_rank']==endpoint*2\n    assert len(checkpoint['rng_per_rank'])==4\n    state=checkpoint['optimizer']['state']\n    assert state and any(int(float(v['step']))==endpoint-300 for v in state.values() if isinstance(v,dict) and 'step' in v)\n    with (out/'steps.jsonl').open() as file:\n        rows=[json.loads(line) for line in file]\n    assert [row['step'] for row in rows]==list(range(ep['source_step']+1,endpoint+1))\n    assert all(len(row['rank_records'])==4 for row in rows)\n    digest=driver.sha(path)\n    receipt={'step':endpoint,'updates':endpoint-ep['source_step'],'checkpoint_sha256':digest,\n             'continuation_seed':checkpoint['continuation_seed'],'source_sha256':ep['starting_sha256'],\n             'optimizer_reset_both_conditions':False,'optimizer_state_count':len(state),\n             'scheduler_epoch':checkpoint['scheduler']['last_epoch'],'recovered_from_durable_checkpoint':True}\n    driver.write(out/'completion.json',receipt)\n\n"
    text = dispatch.read_text()
    text = _replace(text,"def arm_work(arm,previous):",restore+"def arm_work(arm,previous):")
    text = _replace(text,"    out=job/'training'/arm","    out=job/'training'/arm\n    recover_final_receipt(out,arm,ep)")
    _write(dispatch,text)
    text = trainer.read_text()
    text = _replace(text," if source_path==initial_path:assert stat(source_path)==plan['source_checkpoint_stat'] and source['step']==initial_step",
        " if source_path==initial_path:\n  assert source['step']==initial_step\n  if stat(source_path)!=plan['source_checkpoint_stat']:check(sha(source_path)==plan['starting_sha256'],'Starting checkpoint content changed')")
    text = _replace(text,"    check(stat(initial_path)==plan['source_checkpoint_stat'],'Starting checkpoint changed')",
        "    if stat(initial_path)!=plan['source_checkpoint_stat']:check(sha(initial_path)==plan['starting_sha256'],'Starting checkpoint content changed')")
    _write(trainer,text)


def _trainer_hook(file, family):
    source = "import mechanism_probe as _probe\n" + file.read_text()
    if family == B_FAMILY:
        marker = "    for step in range(start+1, count+1):"
        configure = """    mechanism=run/'mechanism_plan.json'
    if not smoke and kind=='depth' and mechanism.is_file():
        settings=json.loads(mechanism.read_text())
        _probe.configure(run/'preflight_mechanism.json',start_step=0,pilot_steps=settings['pilot_steps'],changed_slots=settings['changed_slots'])
"""
        source = _replace(source,marker,configure + marker + "\n        _probe.begin(step)")
        source = _replace(source,"        opt.step()","        _probe.finish(step,trial.DEVICE)\n        opt.step()")
    elif family == OSGK:
        marker = " for step in range(start_step+1,target_steps+1):"
        configure = """ mechanism=job.parents[1]/'mechanism_plan.json'
 if condition=='osgf32' and job.parent.name=='short' and mechanism.is_file():
  settings=json.loads(mechanism.read_text())
  _probe.configure(mechanism.with_name('preflight_mechanism.json'),start_step=initial_step,pilot_steps=settings['pilot_steps'],changed_slots=settings['changed_slots'])
"""
        source = _replace(source,marker,configure+marker+"\n  _probe.begin(step)")
        source = _replace(source,"  optimizer.step();schedule.step()","  _probe.finish(step,device)\n  optimizer.step();schedule.step()")
    else:
        marker = "for step in range(start+1,target+1):"
        configure = """mechanism=ROOT.parent/'mechanism_plan.json'
if condition=='geometry_spectral' and mechanism.is_file():
    settings=json.loads(mechanism.read_text())
    _probe.configure(mechanism.with_name('preflight_mechanism.json'),start_step=0,pilot_steps=settings['pilot_steps'],changed_slots=settings['changed_slots'])
"""
        source = _replace(source,marker,configure+marker+"\n    _probe.begin(step)")
        source = _replace(source,"    optimizer.step();scheduler.step()","    _probe.finish(step,device)\n    optimizer.step();scheduler.step()")
        source = _replace(source,"assert target in (1350,11070)","assert target in (450,900,1350,11070)")
    _write(file,source)


def _set_budgets(path, setup, family, steps):
    if family == OSGK:
        setup["short_endpoint"] = 1200 + steps
        launch = path / "dispatch.py"
        source = launch.read_text()
        source = _replace(source,"stage=run/'short';start=1200;target=1650",
                          "short_endpoint=spec.get('short_endpoint',1650)\nstage=run/'short';start=1200;target=short_endpoint")
        source = _replace(source,"[('short',1200,1650,q/'runs'),('long',1650,11370,run/'short')]",
                          "[('short',1200,short_endpoint,q/'runs'),('long',short_endpoint,11370,run/'short')]")
        _write(launch,source)
    else:
        setup["target_steps"] = steps
        if family == SPECTRAL:
            for arm in ("geometry_spectral", "local_correction"):
                file = path / arm / "plan.json"
                plan = json.loads(file.read_text())
                plan["target_step"] = steps
                plan["training_budget"]["additional_updates_per_condition"] = steps
                file.write_text(json.dumps(plan,indent=2,ensure_ascii=False))


def install_promotion_hook(path: Path, family: str):
    dispatch = path / "dispatch.py"
    source = dispatch.read_text()
    bootstrap = (
        "from pathlib import Path\nimport os, sys\n"
        "_repo = Path(os.path.abspath(sys.argv[1])).parents[3]\n"
        "sys.path.insert(0, os.getenv('CLAW_PLATFORM_SERVICES', str(_repo/'backend/services')))\n"
        "from ai_experiment_promotion import may_start_long\n"
    )
    if family == OSGK:
        marker = "                if not passed:break"
        source = _replace(source, marker, marker + f"\n                if not may_start_long(run, {family!r}):break")
    else:
        # Bind the long destination to this revision; separate trials cannot reuse one another's state.
        source = _replace(source, "long=r.parent/'run-达标接续长训'", "long=r.with_name(r.name+'-long')")
        source = _replace(source, "    supervise(r)", "    if not (r/'long_review_request.json').exists():supervise(r)")
        marker = "    if not passed:return"
        source = _replace(source, marker, marker + f"\n    if not may_start_long(r, {family!r}):return")
    _write(dispatch, bootstrap + source)


def install(path: Path, family: str, decision: dict):
    from shutil import copyfile

    spec = decision["mechanism_plan"]
    file = path / "mechanism_plan.json"
    file.write_text(json.dumps(spec,indent=2,ensure_ascii=False))
    probe_runtime = Path(__file__).with_name("ai_experiment_probe_runtime.py")
    formulas,parent,changed = decision["formulas"],spec["parent_formulas"],spec["changed_slots"]
    setup_path = path / "setup.json"
    setup = json.loads(setup_path.read_text())
    _set_budgets(path,setup,family,spec["steps_per_arm"])
    install_promotion_hook(path, family)
    if family in (B_FAMILY,OSGK):
        exp = path / "experiment"
        copyfile(probe_runtime,exp/"mechanism_probe.py")
        if family == B_FAMILY:
            _instrument_b(exp/"directed_boundary.py",formulas,parent,changed)
            _trainer_hook(exp/"directed_worker.py",family)
            setup["worker_sha256"] = _digest(exp/"directed_worker.py")
            setup["module_sha256"] = _digest(exp/"directed_boundary.py")
        else:
            patch_osgk_checkpoint_handling(path/"dispatch.py",exp/"train_worker.py")
            _instrument_a(exp/"osgf.py",formulas,parent,changed)
            _trainer_hook(exp/"train_worker.py",family)
    else:
        supervisor = path / "supervisor.py"
        _write(supervisor, _replace(supervisor.read_text(), "--max_restarts=1", "--max_restarts=0"))
        for arm in ("geometry_spectral","local_correction"):
            exp = path/arm
            copyfile(probe_runtime,exp/"mechanism_probe.py")
            _instrument_g(exp/"binding.py",formulas,parent,changed)
            _trainer_hook(exp/"train_worker.py",family)
            for name in ("binding.py","train_worker.py","plan.json","mechanism_probe.py"):
                setup["copied_source_sha256"][arm+"/"+name] = _digest(exp/name)
    setup_path.write_text(json.dumps(setup,indent=2,ensure_ascii=False))
    names = {"cirb_mask_boundary":["experiment/directed_boundary.py", "experiment/directed_worker.py"],
             OSGK:["experiment/osgf.py","experiment/train_worker.py"],
             SPECTRAL:["geometry_spectral/binding.py","local_correction/binding.py"]}
    receipt = path/"ai_revision.json"
    evidence = json.loads(receipt.read_text())
    evidence["method_file_sha256"] = {name:_digest(path/name) for name in names[family]}
    evidence["mechanism_plan"] = spec
    receipt.write_text(json.dumps(evidence,indent=2,ensure_ascii=False))


def _digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()
