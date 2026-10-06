"""The labelled synthetic set: reports plus the facts a correct extraction should return.

Each expected field lists every acceptable answer; scoring accepts any of them.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from triage.distances import load_distances
from triage.intake import create_report
from triage.models import Report, SourceTag
from triage.tiers import TIER_TABLE

LABELLED_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "labelled.jsonl"


class ExpectedFault(BaseModel):
    """One fault's acceptable answers. No defaults: every scored field is stated for every fault."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Each entry is one acceptable taxonomy_match, compared as a set; () means "no match".
    taxonomy_match: tuple[tuple[str, ...], ...] = Field(min_length=1)
    alternative_mentioned: tuple[bool, ...] = Field(min_length=1)
    coping_mentioned: tuple[bool, ...] = Field(min_length=1)
    hazard_status: tuple[Literal["none", "described", "unclear"], ...] = Field(min_length=1)
    mechanism_type: tuple[Literal["active", "conditional"] | None, ...] = Field(min_length=1)
    harm_claimed: tuple[bool, ...] = Field(min_length=1)
    fault_or_sign: tuple[Literal["fault", "sign"], ...] = Field(min_length=1)
    claim_mismatch: tuple[Literal["over", "under"] | None, ...] = Field(min_length=1)

    @field_validator("taxonomy_match")
    @classmethod
    def _names_on_the_list(cls, answers: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        unknown = sorted({n for answer in answers for n in answer if n not in TIER_TABLE})
        if unknown:
            raise ValueError(f"taxonomy names not in TIER_TABLE: {unknown}")
        return answers


class LabelledReport(BaseModel):
    """One line of labelled.jsonl."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    # Who wrote the text, so results can be split by author.
    origin: Literal["gemini", "team", "claude-code"]
    # Officer/tenant pair this report belongs to (the register-gap test), or None.
    pair: str | None
    source_tag: SourceTag
    community: str
    reported_at: AwareDatetime
    text: str = Field(min_length=1)
    expected: tuple[ExpectedFault, ...]
    notes: str  # judgment calls behind the labels, for review


def load_labelled(path: Path = LABELLED_PATH) -> tuple[list[Report], dict[str, tuple[ExpectedFault, ...]]]:
    """Reports (record id kept as source_file) and, separately, their expected faults by request_id."""
    distances = load_distances()
    reports: list[Report] = []
    expected: dict[str, tuple[ExpectedFault, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = LabelledReport.model_validate_json(line)
        # Logistics needs a distance for every community; an unknown one is a typo, not a new place.
        if record.community not in distances:
            raise ValueError(f"{record.id}: community {record.community!r} is not in the distance table")
        report = create_report(
            tenant_id=f"S-{record.id}", raw_text=record.text, source_tag=record.source_tag,
            community=record.community, original_report_timestamp=record.reported_at, source_file=record.id,
        )
        reports.append(report)
        expected[report.request_id] = record.expected
    ids = [r.source_file for r in reports]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {path.name}")
    return reports, expected
