"""runs_per_condition: 1 is the user's decision, so its policy text counts as a user quote."""
from types import SimpleNamespace

from researchclaw import single_run
from researchclaw.pipeline.acceptance import quote_found, requirement_text

QUOTE = ("Each experimental condition is trained and evaluated exactly ONCE with one fixed seed, "
         "the same seed for every condition.")


def _config(runs):
    return SimpleNamespace(research=SimpleNamespace(prompt_topic="Topic text."),
                           experiment=SimpleNamespace(runs_per_condition=runs))


def test_single_run_policy_is_quotable(tmp_path):
    text = requirement_text(_config(1), tmp_path)
    assert single_run.SINGLE_RUN_POLICY in text and quote_found(QUOTE, text)


def test_multi_seed_config_adds_nothing(tmp_path):
    assert requirement_text(_config(3), tmp_path) == "Topic text."
