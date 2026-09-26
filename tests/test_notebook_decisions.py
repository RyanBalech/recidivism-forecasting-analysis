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
