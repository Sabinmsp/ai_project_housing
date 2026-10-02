"""Shared data contract for the six-stage triage pipeline.

Stages 1, 2 and 6 are implemented in this package. Stages 3, 4 and 5 are
owned by teammates; the models they produce (VerifiedSpan, EnrichedJob) are
defined here so every stage agrees on field names and types.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Stage 1: Intake
# ---------------------------------------------------------------------------

class SourceTag(str, Enum):
    OFFICER = "officer"              # transcribed phone call
    TENANT_DIRECT = "tenant_direct"  # self-filled web form / email


class Report(BaseModel):
    """Raw report plus exactly four stamped fields. No interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    source_tag: SourceTag
    community: str = Field(min_length=1)  # needed by Stage 5 distance lookup
    original_report_timestamp: datetime

    @field_validator("original_report_timestamp")
    @classmethod
    def _must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("original_report_timestamp must be timezone-aware")
        return v


# ---------------------------------------------------------------------------
# Stage 2: Extraction (the only model call)
# ---------------------------------------------------------------------------

SpanField = Literal[
    "fault_description",
    "taxonomy_match",
    "alternative_mentioned",
    "coping_mentioned",
    "impact_status",
    "hazard_mechanism",
]


class QuotedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: SpanField
    text: str = Field(min_length=1)


class ExtractedFacts(BaseModel):
    """Response schema for the LLM and the validation boundary.

    Presence checks only. There is deliberately no numeric field anywhere in
    this model: nothing that leaves Stage 2 can be a score.
    """

    model_config = ConfigDict(extra="forbid")

    fault_description: Optional[str] = None
    taxonomy_match: list[str] = Field(default_factory=list)
    alternative_mentioned: bool = False
    coping_mentioned: bool = False
    impact_status: Literal["ongoing", "intermittent"] = "ongoing"
    hazard_mechanism: Optional[str] = None
    mechanism_type: Optional[Literal["active", "conditional"]] = None
    quoted_spans: list[QuotedSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _claims_need_spans(self) -> "ExtractedFacts":
        cited = {s.field for s in self.quoted_spans}
        required: list[str] = []
        if self.fault_description:
            required.append("fault_description")
        if self.alternative_mentioned:
            required.append("alternative_mentioned")
        if self.coping_mentioned:
            required.append("coping_mentioned")
        if self.hazard_mechanism:
            required.append("hazard_mechanism")
        missing = [f for f in required if f not in cited]
        if missing:
            raise ValueError(f"claimed facts without a quoted span: {missing}")

        if self.hazard_mechanism and self.mechanism_type is None:
            raise ValueError("hazard_mechanism given without mechanism_type")
        if not self.hazard_mechanism and self.mechanism_type is not None:
            raise ValueError("mechanism_type given without hazard_mechanism")
        return self


class ExtractionStatus(str, Enum):
    OK = "ok"
    FLAGGED_FOR_HUMAN = "flagged_for_human"  # failed validation twice
    NO_FAULT_NAMED = "no_fault_named"        # out of scope, coordinator contacts tenant


class ExtractionResult(BaseModel):
    request_id: str
    status: ExtractionStatus
    facts: Optional[ExtractedFacts] = None
    attempts: int
    errors: list[str] = Field(default_factory=list)
    extractor: str  # "llm:<model>" or "offline"


# ---------------------------------------------------------------------------
# Stages 3 to 5 output: what Stage 6 consumes (owned by teammates)
# ---------------------------------------------------------------------------

class VerifiedSpan(BaseModel):
    field: str
    text: str
    verified: bool


class EnrichedJob(BaseModel):
    """Output of Stage 5. Stage 6 reads only three of these fields for order."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    community: str
    original_report_timestamp: datetime
    fault_description: Optional[str] = None

    # Stage 4 evaluation. urgency_tally None means REVIEW BAND.
    tier: Optional[Literal["dangerous", "standard"]] = None
    base_points: Optional[int] = None
    no_redundancy: int = 0  # 0 or 1
    urgency_tally: Optional[int] = Field(default=None, ge=2, le=4)
    safety_flag: bool = False
    safety_level: Literal["active", "conditional", "none"] = "none"
    flags: list[str] = Field(default_factory=list)
    spans: list[VerifiedSpan] = Field(default_factory=list)

    # Stage 5 logistics: display only, never read by the sort
    distance_cost_km: Optional[float] = None
    capacity_block_flag: bool = False
    next_actionable: Optional[str] = None
    shared_route_opportunities: list[str] = Field(default_factory=list)
    starvation_line: Optional[str] = None

    @model_validator(mode="after")
    def _tally_consistent(self) -> "EnrichedJob":
        if self.urgency_tally is None:
            return self
        if self.tier is None or self.base_points is None:
            raise ValueError("a scored job must carry its tier and base_points")
        if self.base_points + self.no_redundancy != self.urgency_tally:
            raise ValueError("urgency_tally must equal base_points + no_redundancy")
        return self

    @property
    def in_review_band(self) -> bool:
        return self.urgency_tally is None


# ---------------------------------------------------------------------------
# Stage 6: Ranking + why-trace
# ---------------------------------------------------------------------------

class ReasoningTrace(BaseModel):
    """Structured explanation. Both renderers read this and nothing else."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    fault_description: Optional[str]
    tier: Literal["dangerous", "standard"]
    tier_source: str
    base_points: int
    no_redundancy: int
    no_redundancy_reason: str
    urgency_tally: int
    safety_flag: bool
    safety_reason: str
    evidence_spans: list[VerifiedSpan]
    flags: list[str]
    sort_key: tuple[bool, int, int]
    position: int
    queue_length: int
    distance_cost_km: Optional[float]
    logistics_notes: list[str]


class ReviewBandEntry(BaseModel):
    request_id: str
    community: str
    fault_description: Optional[str]
    original_report_timestamp: datetime
    reason: str = "Fault not on the government list: awaiting coordinator tier call"


class RankingResult(BaseModel):
    review_band: list[ReviewBandEntry]
    ranked: list[EnrichedJob]
    traces: list[ReasoningTrace]
