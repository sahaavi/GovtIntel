"""Tests for checked-in evaluation case fixtures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_QUERIES_PATH = REPO_ROOT / "eval" / "test_queries.json"
GOLD_ANSWERS_PATH = REPO_ROOT / "eval" / "gold_answers.json"


def test_eval_query_and_gold_answer_files_exist() -> None:
    assert TEST_QUERIES_PATH.exists()
    assert GOLD_ANSWERS_PATH.exists()


def test_eval_query_set_has_required_schema_and_coverage() -> None:
    queries = _load_json(TEST_QUERIES_PATH)
    gold_answers = _load_json(GOLD_ANSWERS_PATH)

    assert isinstance(queries, list)
    assert len(queries) >= 30
    assert isinstance(gold_answers, dict)

    seen_ids: set[str] = set()
    for query in queries:
        assert isinstance(query, dict)
        query_id = _required_str(query, "id")
        assert re.fullmatch(r"q\d{3}", query_id)
        assert query_id not in seen_ids
        seen_ids.add(query_id)

        assert _required_str(query, "question").endswith("?")
        assert _required_str(query, "agency_filter")
        assert _required_str(query, "expected_trends") in {
            "increasing",
            "decreasing",
            "stable",
            "mixed",
        }
        assert _required_str(query, "gold_answer_id") == query_id

        contractors = query.get("expected_contractors")
        assert isinstance(contractors, list)
        assert contractors
        assert all(isinstance(contractor, str) and contractor for contractor in contractors)

        valid_ids = query.get("valid_contract_ids")
        assert isinstance(valid_ids, list)
        assert all(isinstance(contract_id, str) and contract_id for contract_id in valid_ids)

        ranges = query.get("expected_dollar_ranges", {})
        assert isinstance(ranges, dict)
        for bounds in ranges.values():
            assert isinstance(bounds, list)
            assert len(bounds) == 2
            assert isinstance(bounds[0], int | float)
            assert isinstance(bounds[1], int | float)
            assert 0 <= bounds[0] <= bounds[1]

        quality_rating = query.get("quality_rating")
        assert isinstance(quality_rating, int)
        assert 1 <= quality_rating <= 5

        gold_answer = gold_answers.get(query_id)
        assert isinstance(gold_answer, str)
        assert len(gold_answer) >= 40

    assert set(gold_answers) == seen_ids


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    assert isinstance(value, str)
    assert value
    return value
