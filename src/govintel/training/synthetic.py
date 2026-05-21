"""Synthetic JSONL generation for offline GovIntel fine-tuning."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from govintel.config import Settings, get_settings
from govintel.generation.gemini import GeminiClient
from govintel.ingestion.loader import contracts_table
from govintel.models import ContractAward

DEFAULT_OUTPUT_PATH = Path("data/training/govintel_synthetic.jsonl")
DEFAULT_INSTRUCTION = (
    "Write a grounded GovCon intelligence report snippet for the contractor "
    "and agency in the provided contract context."
)
SYNTHETIC_SYSTEM_PROMPT = (
    "You generate factual procurement intelligence snippets from contract records. "
    "Use only the supplied context, cite at least one source award ID, and do not "
    "invent contractors, agencies, obligations, or dates."
)
_REQUIRED_JSONL_FIELDS = {"instruction", "context", "response"}
_AWARD_ID_LINE_PATTERN = re.compile(r"^\s*(?:- )?Award ID:\s*(?P<award_id>\S+)", re.MULTILINE)


class SnippetGenerator(Protocol):
    """Minimal protocol shared by GeminiClient and deterministic test generators."""

    def generate(self, system: str, user_message: str) -> Awaitable[str]:
        """Generate a snippet from a system prompt and user prompt."""


@dataclass(frozen=True)
class ContractGroup:
    """Contract awards grouped by agency and contractor for one training example."""

    agency: str
    contractor: str
    awards: tuple[ContractAward, ...]


@dataclass(frozen=True)
class TrainingExample:
    """A single supervised fine-tuning row."""

    instruction: str
    context: str
    response: str

    def to_json_record(self) -> dict[str, str]:
        """Return the JSONL record shape consumed by SFT notebooks."""

        return {
            "instruction": self.instruction,
            "context": self.context,
            "response": self.response,
        }


def build_training_example(
    group: ContractGroup,
    *,
    response: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> TrainingExample:
    """Build a JSONL-ready training example from one grouped contract context."""

    return TrainingExample(
        instruction=instruction,
        context=format_contract_context(group),
        response=response.strip(),
    )


def format_contract_context(group: ContractGroup) -> str:
    """Render grouped awards into stable, source-cited fine-tuning context."""

    if not group.awards:
        raise ValueError("contract group must include at least one award")

    total_value = sum(award.award_amount for award in group.awards)
    start_dates = [award.start_date for award in group.awards]
    naics_codes = sorted({award.naics_code for award in group.awards if award.naics_code})
    lines = [
        f"Agency: {group.agency}",
        f"Contractor: {group.contractor}",
        f"Award count: {len(group.awards)}",
        f"Total obligated value: ${total_value:,.2f}",
        f"Date range: {min(start_dates).isoformat()} to {max(start_dates).isoformat()}",
        f"NAICS codes: {', '.join(naics_codes) if naics_codes else 'Unknown'}",
        "Source awards:",
    ]
    for award in group.awards:
        end_date = award.end_date.isoformat() if award.end_date else "open"
        description = " ".join(award.description.split())
        lines.extend(
            [
                f"- Award ID: {award.award_id}",
                f"  Period: {award.start_date.isoformat()} to {end_date}",
                f"  Obligation: ${award.award_amount:,.2f}",
                f"  NAICS: {award.naics_code or 'Unknown'}",
                f"  Type: {award.award_type or 'Unknown'}",
                f"  Place of performance: {award.place_of_performance_state or 'Unknown'}",
                f"  Description: {description or 'No description provided'}",
            ]
        )
    return "\n".join(lines)


async def fetch_contract_groups(
    engine: AsyncEngine,
    *,
    group_limit: int = 500,
    contracts_per_group: int = 5,
    min_awards_per_group: int = 1,
) -> list[ContractGroup]:
    """Fetch contract rows grouped by agency and contractor from the contracts table."""

    if group_limit <= 0 or contracts_per_group <= 0 or min_awards_per_group <= 0:
        return []

    statement = select(contracts_table).order_by(
        contracts_table.c.awarding_agency.asc(),
        contracts_table.c.recipient_name.asc(),
        contracts_table.c.start_date.desc(),
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(statement)).mappings().all()

    ordered_keys: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[ContractAward]] = {}
    for row in rows:
        key = (str(row["awarding_agency"]), str(row["recipient_name"]))
        if key not in grouped:
            if len(ordered_keys) >= group_limit:
                continue
            ordered_keys.append(key)
            grouped[key] = []
        if len(grouped[key]) < contracts_per_group:
            grouped[key].append(_award_from_row(row))

    return [
        ContractGroup(
            agency=agency,
            contractor=contractor,
            awards=tuple(grouped[(agency, contractor)]),
        )
        for agency, contractor in ordered_keys
        if len(grouped[(agency, contractor)]) >= min_awards_per_group
    ]


async def generate_training_examples(
    groups: Sequence[ContractGroup],
    generator: SnippetGenerator,
    *,
    limit: int | None = None,
) -> list[TrainingExample]:
    """Generate grounded synthetic examples from grouped contract contexts."""

    examples: list[TrainingExample] = []
    for group in groups:
        if limit is not None and len(examples) >= limit:
            break
        context = format_contract_context(group)
        user_message = (
            f"{DEFAULT_INSTRUCTION}\n\n"
            f"Contract context:\n{context}\n\n"
            "Return a concise report snippet with an executive insight, competitive "
            "signal, and at least one cited award ID."
        )
        response = (await generator.generate(SYNTHETIC_SYSTEM_PROMPT, user_message)).strip()
        if not _is_grounded_response(response, group):
            continue
        examples.append(
            TrainingExample(
                instruction=DEFAULT_INSTRUCTION,
                context=context,
                response=response,
            )
        )
    return examples


def write_jsonl(examples: Sequence[TrainingExample], output_path: str | Path) -> int:
    """Write examples as newline-delimited JSON and return the row count."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json_record(), ensure_ascii=False))
            handle.write("\n")
    return len(examples)


def validate_curated_jsonl(path: str | Path, *, minimum_examples: int = 50) -> int:
    """Validate a hand-curated training file before QLoRA fine-tuning."""

    records = _read_jsonl_records(Path(path))
    if len(records) < minimum_examples:
        raise ValueError(f"curated dataset must contain at least {minimum_examples} examples")
    for index, record in enumerate(records, start=1):
        _validate_training_record(record, row_number=index)
    return len(records)


async def generate_synthetic_dataset(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    group_limit: int = 500,
    contracts_per_group: int = 5,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> int:
    """Generate and write a synthetic JSONL dataset from live DB rows or samples."""

    if dry_run:
        groups = sample_contract_groups(group_limit)
        generator: SnippetGenerator = DryRunSnippetGenerator()
        examples = await generate_training_examples(groups, generator, limit=group_limit)
        return write_jsonl(examples, output_path)

    resolved_settings = settings or get_settings()
    engine = create_async_engine(resolved_settings.database_url, echo=False)
    try:
        groups = await fetch_contract_groups(
            engine,
            group_limit=group_limit,
            contracts_per_group=contracts_per_group,
        )
        examples = await generate_training_examples(
            groups,
            GeminiClient(settings=resolved_settings),
            limit=group_limit,
        )
        return write_jsonl(examples, output_path)
    finally:
        await engine.dispose()


class DryRunSnippetGenerator:
    """Deterministic local generator for CLI smoke tests."""

    async def generate(self, system: str, user_message: str) -> str:
        """Return a grounded synthetic response without external services."""

        del system
        award_ids = _award_ids_from_context(user_message)
        award_id = award_ids[0] if award_ids else "UNKNOWN-AWARD"
        return (
            f"The grouped awards show a measurable GovCon opportunity grounded in {award_id}. "
            "The contractor has repeatable agency demand signals and should be tracked for "
            "follow-on recompetes."
        )


def sample_contract_groups(limit: int) -> list[ContractGroup]:
    """Return deterministic sample groups for dry-run generation."""

    if limit <= 0:
        return []
    groups: list[ContractGroup] = []
    for index in range(1, limit + 1):
        agency = "Department of Homeland Security" if index % 2 else "Department of Defense"
        contractor = f"Sample Contractor {index:03d}"
        award_id = f"SAMPLE-{index:03d}"
        groups.append(
            ContractGroup(
                agency=agency,
                contractor=contractor,
                awards=(
                    ContractAward(
                        award_id=award_id,
                        recipient_name=contractor,
                        awarding_agency=agency,
                        award_amount=1_000_000.0 + index,
                        start_date=date(2024, 1, 1),
                        end_date=None,
                        naics_code="541512",
                        description="Sample cybersecurity modernization services",
                        place_of_performance_state="VA",
                        award_type="Definitive Contract",
                    ),
                ),
            )
        )
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `python -m govintel.training.synthetic`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--contracts-per-group", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic local sample data and skip PostgreSQL/Gemini.",
    )
    args = parser.parse_args(argv)

    written = asyncio.run(
        generate_synthetic_dataset(
            output_path=args.output,
            group_limit=args.limit,
            contracts_per_group=args.contracts_per_group,
            dry_run=args.dry_run,
        )
    )
    print(f"Wrote {written} training examples to {args.output}")
    return 0


def _award_from_row(row: RowMapping) -> ContractAward:
    """Convert a SQLAlchemy mapping row into a ContractAward."""

    return ContractAward(
        award_id=str(row["award_id"]),
        recipient_name=str(row["recipient_name"]),
        awarding_agency=str(row["awarding_agency"]),
        award_amount=float(row["award_amount"] or 0.0),
        start_date=_as_date(row["start_date"]),
        end_date=_as_optional_date(row["end_date"]),
        naics_code=str(row["naics_code"] or ""),
        description=str(row["description"] or ""),
        place_of_performance_state=str(row["place_of_performance_state"] or ""),
        award_type=str(row["award_type"] or ""),
    )


def _as_date(value: object) -> date:
    """Normalize SQL date values for ContractAward."""

    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"expected date-compatible value, got {value!r}")


def _as_optional_date(value: object) -> date | None:
    """Normalize nullable SQL date values."""

    if value is None:
        return None
    return _as_date(value)


def _is_grounded_response(response: str, group: ContractGroup) -> bool:
    """Return whether a generated response is non-empty and cites source evidence."""

    if not response.strip():
        return False
    return any(award.award_id in response for award in group.awards)


def _read_jsonl_records(path: Path) -> list[dict[str, str]]:
    """Read JSONL rows and ensure each row is an object of string values."""

    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
            ):
                raise ValueError(f"row {row_number} must be an object of string fields")
            records.append(raw)
    return records


def _validate_training_record(record: Mapping[str, str], *, row_number: int) -> None:
    """Validate one curated row for schema, completeness, and source grounding."""

    if set(record) != _REQUIRED_JSONL_FIELDS:
        raise ValueError(f"row {row_number} must contain instruction, context, and response")
    if not all(record[field].strip() for field in _REQUIRED_JSONL_FIELDS):
        raise ValueError(f"row {row_number} contains an empty training field")
    award_ids = _award_ids_from_context(record["context"])
    if award_ids and not any(award_id in record["response"] for award_id in award_ids):
        raise ValueError(f"row {row_number} response is not grounded in source award IDs")


def _award_ids_from_context(context: str) -> list[str]:
    """Extract source award IDs from rendered training context."""

    return _AWARD_ID_LINE_PATTERN.findall(context)


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke checks
    raise SystemExit(main())
