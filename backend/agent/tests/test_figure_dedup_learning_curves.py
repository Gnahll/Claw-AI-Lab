from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.experiment import visualize
from researchclaw.pipeline import executor
from researchclaw.templates.conference import NEURIPS_2025
from researchclaw.templates.converter import markdown_to_latex

PAPER = """# Title

## Results

![Main comparison.](charts/fig_main_comparison.png)

Text about the comparison.

![Breakdown.](charts/fig_metric_breakdown.png)

## Analysis

![Figure 4. Recorded comparison again.](charts/fig_main_comparison.png)

More text.
"""


def test_markdown_figures_keep_first_reference_only():
    deduped, removed = executor._dedupe_markdown_figures(PAPER)
    assert removed == 1
    assert deduped.count("fig_main_comparison.png") == 1
    assert "Main comparison." in deduped and "Figure 4." not in deduped
    assert "fig_metric_breakdown.png" in deduped and "More text." in deduped
    assert executor._dedupe_markdown_figures(deduped) == (deduped, 0)


def test_latex_figures_keep_first_environment_only():
    tex = (
        "\\begin{figure}[htbp]\n\\includegraphics[width=0.8\\textwidth]{figures/a.png}\n\\caption{A}\n\\end{figure}\n"
        "\\begin{figure}[htbp]\n\\includegraphics{figures/b.png}\n\\end{figure}\n"
        "\\begin{figure*}\n\\includegraphics{charts/a.png}\n\\caption{A again}\n\\end{figure*}\n"
    )
    deduped, removed = executor._dedupe_latex_figures(tex)
    assert removed == 1 and "A again" not in deduped and "figures/b.png" in deduped


def test_converter_renders_each_image_once():
    tex = markdown_to_latex(PAPER, NEURIPS_2025, title="T")
    assert tex.count("\\includegraphics") == 2
    assert tex.count("fig_main_comparison.png") == 1
    # Counters reset per document, so a second conversion renders the image again.
    assert markdown_to_latex(PAPER, NEURIPS_2025, title="T").count("\\includegraphics") == 2


def _results(conditions=("baseline", "proposed"), seeds=(42, 123), epochs=4):
    return {
        "runs": [
            {
                "condition": condition, "seed": seed,
                "epochs": [
                    {"epoch": e, "train_loss": 1.0 / e + seed * 1e-4,
                     "validation": {"f1": 0.8 + e * 0.01, "iou": 0.7, "samples": 13}}
                    for e in range(1, epochs + 1)
                ],
            }
            for condition in conditions for seed in seeds
        ]
    }


@pytest.mark.skipif(not visualize.HAS_MATPLOTLIB, reason="matplotlib missing")
def test_learning_curves_from_adopted_results(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "stage-14/runs").mkdir(parents=True)
    (run_dir / "stage-14/runs/results.json").write_text(json.dumps(_results(epochs=1)))
    (run_dir / "stage-15").mkdir()
    (run_dir / "stage-15/results.json").write_text(json.dumps(_results()))

    assert visualize._load_adopted_results(run_dir) == _results()
    out = tmp_path / "charts"
    generated = visualize.generate_all_charts(run_dir, out)
    assert out / "fig_learning_curves.png" in generated
    assert (out / "fig_learning_curves.png").stat().st_size > 1000


@pytest.mark.skipif(not visualize.HAS_MATPLOTLIB, reason="matplotlib missing")
def test_learning_curves_need_two_epochs(tmp_path: Path):
    assert visualize.plot_learning_curves(_results(epochs=1), tmp_path / "x.png") is None
    assert visualize.plot_learning_curves({"runs": "bad"}, tmp_path / "x.png") is None
    flat = {"runs": [{"condition": "a", "epochs": [{"mean_loss": 1.0, "val_acc": 0.5},
                                                   {"mean_loss": 0.5, "val_acc": 0.6}]}]}
    assert visualize.plot_learning_curves(flat, tmp_path / "flat.png") == tmp_path / "flat.png"
