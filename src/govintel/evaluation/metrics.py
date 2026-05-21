"""Deterministic procurement-specific evaluation metrics."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

_AMOUNT_PATTERN = re.compile(
    r"(?P<sign>-)?\$?\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<suffix>thousand|million|billion|[kmb])?\b",
    flags=re.IGNORECASE,
)
_SUFFIX_MULTIPLIERS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}


def entity_accuracy(expected_contractors: Sequence[str], generated_text: str) -> float:
    """Return the fraction of expected contractor names present in generated text."""

    if not expected_contractors:
        return 0.0

    normalized_text = f" {_normalize_for_matching(generated_text)} "
    matches = 0
    for contractor in expected_contractors:
        normalized_contractor = _normalize_for_matching(contractor)
        if normalized_contractor and f" {normalized_contractor} " in normalized_text:
            matches += 1
    return matches / len(expected_contractors)


def dollar_accuracy(
    expected_amounts: Mapping[str, float],
    generated_text: str,
    tolerance: float = 0.05,
) -> float:
    """Return the fraction of expected contractor amounts found within tolerance."""

    if not expected_amounts:
        return 0.0
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    matches = 0
    for contractor, expected_amount in expected_amounts.items():
        if _has_matching_amount(contractor, expected_amount, generated_text, tolerance):
            matches += 1
    return matches / len(expected_amounts)


def citation_correctness(valid_ids: Collection[str], cited_ids: Sequence[str]) -> float:
    """Return supported unique citations divided by unique cited IDs."""

    unique_citations = _unique_normalized_ids(cited_ids)
    if not unique_citations:
        return 0.0

    normalized_valid_ids = {_normalize_contract_id(contract_id) for contract_id in valid_ids}
    supported = sum(1 for citation in unique_citations if citation in normalized_valid_ids)
    return supported / len(unique_citations)


def _has_matching_amount(
    contractor: str,
    expected_amount: float,
    generated_text: str,
    tolerance: float,
) -> bool:
    """Return whether text near contractor contains an amount within tolerance."""

    if expected_amount < 0:
        return False

    pattern = _contractor_pattern(contractor)
    if pattern is None:
        return False

    for match in pattern.finditer(generated_text):
        window = generated_text[match.end() : match.end() + 140]
        for amount in _amounts_in_text(window):
            if _amount_within_tolerance(amount, expected_amount, tolerance):
                return True
    return False


def _amount_within_tolerance(amount: float, expected_amount: float, tolerance: float) -> bool:
    """Return whether amount is within relative tolerance of expected_amount."""

    if expected_amount == 0:
        return abs(amount) <= tolerance
    return abs(amount - expected_amount) / abs(expected_amount) <= tolerance


def _amounts_in_text(text: str) -> list[float]:
    """Extract dollar-like amounts from text."""

    amounts: list[float] = []
    for match in _AMOUNT_PATTERN.finditer(text):
        raw_number = match.group("number").replace(",", "")
        amount = float(raw_number)
        suffix = (match.group("suffix") or "").casefold()
        amount *= _SUFFIX_MULTIPLIERS.get(suffix, 1.0)
        if match.group("sign"):
            amount *= -1
        amounts.append(amount)
    return amounts


def _normalize_for_matching(value: str) -> str:
    """Normalize text for exact token-sequence matching."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _contractor_pattern(contractor: str) -> re.Pattern[str] | None:
    """Build a punctuation-tolerant regex for a contractor name."""

    tokens = re.findall(r"[A-Za-z0-9]+", contractor)
    if not tokens:
        return None
    return re.compile(r"\b" + r"[\W_]+".join(re.escape(token) for token in tokens) + r"\b", re.I)


def _unique_normalized_ids(values: Sequence[str]) -> list[str]:
    """Normalize IDs and preserve first-seen order."""

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        contract_id = _normalize_contract_id(value)
        if contract_id and contract_id not in seen:
            seen.add(contract_id)
            normalized.append(contract_id)
    return normalized


def _normalize_contract_id(value: str) -> str:
    """Normalize a cited award ID or retrieval chunk ID to the award ID."""

    return value.split(":chunk:", maxsplit=1)[0].strip()


__all__ = ["citation_correctness", "dollar_accuracy", "entity_accuracy"]
