"""Static smoke tests for phase 7 fine-tuning notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINE_TUNING_NOTEBOOK = REPO_ROOT / "notebooks" / "04_fine_tuning.ipynb"
DECODING_NOTEBOOK = REPO_ROOT / "notebooks" / "05_decoding_experiments.ipynb"


def test_fine_tuning_notebook_exists_and_declares_required_qlora_workflow() -> None:
    notebook = _load_notebook(FINE_TUNING_NOTEBOOK)
    text = _notebook_text(notebook)

    assert "FastLanguageModel" in text
    assert "mistral-7b-instruct-v0.3-bnb-4bit" in text
    assert "r = 16" in text
    assert "q_proj" in text
    assert "gate_proj" in text
    assert "num_train_epochs = 3" in text
    assert "learning_rate = 2e-4" in text
    assert "held_out_examples[:5]" in text
    assert "push_to_hub_merged" in text


def test_decoding_notebook_exists_and_scores_required_grid() -> None:
    notebook = _load_notebook(DECODING_NOTEBOOK)
    text = _notebook_text(notebook)

    assert "decoding_grid" in text
    assert "temperature" in text
    assert "0.0" in text
    assert "0.3" in text
    assert "0.7" in text
    assert "top_k" in text
    assert "50" in text
    assert "100" in text
    assert "top_p" in text
    assert "0.9" in text
    assert "0.95" in text
    assert "entity_accuracy" in text
    assert "faithfulness" in text
    assert "select_best_config" in text


def _load_notebook(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        notebook = json.load(handle)
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    return notebook


def _notebook_text(notebook: dict[str, Any]) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

