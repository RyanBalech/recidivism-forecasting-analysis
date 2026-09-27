"""Keep the submission's tuning rationale evidence-backed and non-destructive."""
import copy
import runpy

import nbformat as nbf
import pytest

from recidivism.config import ROOT


@pytest.fixture
def builder():
    return runpy.run_path(str(ROOT / "scripts/build_standalone_notebook.py"))


def test_decision_tables_match_saved_results(builder):
    text = builder["modeling_decisions_cell"]().source
    assert "| One-hot | 0.21544, L1 | 0.7324 | 0.2044 | Kept |" in text
    assert "| Ordinal | 0.02154, L2 | 0.7315 | 0.2048 | Rejected |" in text
    assert "| 0: ordinal | 2 | 0.7324 | 0.2043 | Kept |" in text
    assert "| 3: ordinal | 3 | 0.7296 | 0.2054 | Rejected |" in text
    assert "| 4: ordinal | 4 | 0.7297 | 0.2053 | Rejected |" in text
    assert "| 16 | 0.72793 | 0.20533 | 25.4 | Kept |" in text
    assert "| 64 | 0.72797 | 0.20529 | 99.4 | Not selected |" in text


def test_decision_section_discloses_protocol_and_missing_evidence(builder):
    text = builder["modeling_decisions_cell"]().source
    for disclosure in ("52 configurations / 260 fold fits", "60 draws", "117 inner fits",
                       "not retained", "not a controlled encoding-only experiment",
                       "not the absolute AUC winner", "inspected elsewhere"):
        assert disclosure in text


def test_refresh_is_idempotent_and_preserves_code_outputs_and_data(builder):
    code = nbf.v4.new_code_cell("print('saved fit')", execution_count=1,
                              outputs=[nbf.v4.new_output("stream", name="stdout", text="saved fit\n")])
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell("## 4. Check leakage"),
                                        nbf.v4.new_markdown_cell("## 5. Preprocess and train three model families"),
                                        code])
    notebook.metadata["recidivism_embedded_csvs"] = {"preserve": "payload"}
    original_code = copy.deepcopy(code)
    original_metadata = copy.deepcopy(notebook.metadata)
    refresh = builder["refresh_modeling_decisions"]
    refresh(notebook)
    refresh(notebook)
    assert len(notebook.cells) == 4
    assert notebook.cells[1].source.startswith(builder["DECISIONS_HEADING"])
    assert notebook.cells[2].source.startswith("## 5.")
    assert notebook.cells[3] == original_code
    assert notebook.metadata == original_metadata
    nbf.validate(notebook)


def test_submission_has_current_decisions_immediately_before_training(builder):
    notebook = nbf.read(ROOT / "Recidivism_Project_Submission.ipynb", as_version=4)
    training = next(i for i, cell in enumerate(notebook.cells)
                    if cell.cell_type == "markdown" and cell.source.startswith("## 5. Preprocess and train"))
    assert notebook.cells[training - 1].source == builder["modeling_decisions_cell"]().source


def test_stability_reading_matches_saved_audit(builder):
    text = builder["stability_reading_cell"]().source
    for expected in ("| Logistic | 0.0322 | 0.7725 | 12.8% |", "| XGBoost | 0.0351 | 0.7468 | 14.5% |",
                     "| TabICLv2 | 0.0350 | 0.7758 | 12.6% |", "**28 of 28**", "**26 of 28**",
                     "**12.8%** of decisions are contested", "from 0.822 to 0.846",
                     "from −0.090 to −0.119", "47% of the 118 women", "30% of the 1,490 men"):
        assert expected in text
    assert "not recomputed by this notebook" in text


def test_stability_refresh_is_idempotent_and_preserves_code_outputs(builder):
    code = nbf.v4.new_code_cell("stability=1", execution_count=7,
                                outputs=[nbf.v4.new_output("stream", name="stdout", text="refits\n")])
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(builder["STABILITY_HEADING"] + "\n\ntext"),
                                        code, nbf.v4.new_markdown_cell("## 9. Fairness at the actual support rule")])
    notebook.metadata["recidivism_embedded_csvs"] = {"preserve": "payload"}
    original_code = copy.deepcopy(code)
    original_metadata = copy.deepcopy(notebook.metadata)
    refresh = builder["refresh_stability_reading"]
    refresh(notebook)
    refresh(notebook)
    assert len(notebook.cells) == 4
    assert notebook.cells[1] == original_code
    assert notebook.cells[2].source.startswith(builder["STABILITY_READING_HEADING"])
    assert notebook.cells[3].source.startswith("## 9.")
    assert notebook.metadata == original_metadata
    nbf.validate(notebook)


def test_submission_interprets_stability_right_after_its_code(builder):
    notebook = nbf.read(ROOT / "Recidivism_Project_Submission.ipynb", as_version=4)
    heading = next(i for i, cell in enumerate(notebook.cells)
                   if cell.cell_type == "markdown" and cell.source.startswith(builder["STABILITY_HEADING"]))
    assert notebook.cells[heading + 1].cell_type == "code"
    assert notebook.cells[heading + 2].source == builder["stability_reading_cell"]().source
