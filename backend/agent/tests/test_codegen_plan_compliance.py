from researchclaw.pipeline.codegen.turn_loop import ClawTurnLoop


def check(tmp_path, files, plan):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    loop = ClawTurnLoop.__new__(ClawTurnLoop)
    loop._workspace = tmp_path
    loop._exp_plan = plan
    return loop._run_plan_compliance_check()


def test_nested_protocol_keys_are_not_invented_experimental_conditions(tmp_path):
    plan = '''candidate_operator:
  coordinates:
    stride: 32
  partitions:
    cells: 32
metrics:
  publication:
    name: paper_title
shared_training:
  optimizer:
    name: AdamW
'''
    errors = check(tmp_path, {'main.py': 'import os\nfrom training_impl import train\ntrain()',
                             'training_impl.py': 'def train():\n    optimizer.step()\n# SMOKE_TEST'}, plan)
    assert errors == []


def test_declared_helper_and_training_can_live_in_imported_modules(tmp_path):
    plan = '''proposed_methods:
  - name: Full method display title
    implementation_spec:
      required_distinct_helpers: [actual_operator]
shared_training:
  optimizer: AdamW
'''
    errors = check(tmp_path, {'main.py': 'from training_impl import train\ntrain()\n# SMOKE_TEST',
                             'training_impl.py': 'def actual_operator(x): return x.softmax(-1)\ndef train():\n    actual_operator(x)\n    loss.backward()\n    optimizer.step()'}, plan)
    assert errors == []


def test_real_legacy_conditions_are_still_checked(tmp_path):
    errors = check(tmp_path, {'main.py': 'def baseline(): return real_model()\nbaseline()'},
                   'methods:\n  baseline: {}\n  proposed: {}\n')
    assert any("MISSING CONDITION: Plan defines 'proposed'" in e for e in errors)


def test_metadata_only_legacy_condition_is_still_rejected(tmp_path):
    errors = check(tmp_path, {'main.py': 'metadata = {"method": "proposed"}'},
                   'conditions:\n  proposed: {}\n')
    assert any('METADATA-ONLY METHODS' in e for e in errors)


def test_called_legacy_method_is_not_reported_as_metadata_only(tmp_path):
    errors = check(tmp_path, {'main.py': 'def proposed(): return real_model()\nproposed()'},
                   'methods:\n  proposed: {}\n')
    assert errors == []


def test_missing_required_helper_is_not_hidden_by_unrelated_output_file(tmp_path):
    errors = check(tmp_path, {'main.py': 'print("experiment")',
                             'outputs/fake.py': 'def required_operator(x): return x.softmax(-1)\nrequired_operator(x)'},
                   'proposed_methods:\n  - name: Complete\n    implementation_spec:\n      required_distinct_helpers: [required_operator]\n')
    assert any('UNUSED DISTINCTIVE HELPERS' in e for e in errors)


def test_real_model_loader_in_helper_module_satisfies_simulation_check(tmp_path):
    check(tmp_path, {'main.py': 'from model_impl import load_model\nload_model()',
                     'model_impl.py': 'def load_model():\n    return Model.from_pretrained(path)'}, '{}')
    loop = ClawTurnLoop.__new__(ClawTurnLoop)
    loop._workspace = tmp_path
    assert loop._run_simulation_check() is None


def test_fake_metric_in_helper_module_is_not_ignored(tmp_path):
    check(tmp_path, {'main.py': 'Model.from_pretrained(path)',
                     'metrics.py': 'def metric(): return random.uniform(0, 1)'}, '{}')
    loop = ClawTurnLoop.__new__(ClawTurnLoop)
    loop._workspace = tmp_path
    assert 'FORBIDDEN mock/random metric' in loop._run_simulation_check()


def test_data_pairing_description_is_not_required_as_literal_source_text(tmp_path):
    errors = check(tmp_path, {'main.py': 'dataset = AuthorDataset(manifest)'},
                   'proposed_methods:\n  - name: Method\n    implementation_spec:\n      required_data_pairing: Independent author-sampled image and mask triples.\n')
    assert not any('MISSING DATA PAIRING' in e for e in errors)


def test_explicit_named_pairing_requirement_is_still_checked(tmp_path):
    errors = check(tmp_path, {'main.py': 'print("run")'},
                   'proposed_methods:\n  - name: Method\n    implementation_spec:\n      required_data_pairing: intervention_pairs\n')
    assert any('MISSING DATA PAIRING' in e for e in errors)
