"""Pydantic models passed between triage stages.

Ranking input/output (RankInput, RankedJob, RankResult), intake (SourceTag,
Report), extraction (QuotedSpan, ExtractedFacts, ExtractionStatus,
ExtractionResult) and the enriched job ranking is fed from (VerifiedSpan,
EnrichedJob).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from triage.tiers import TIER_TABLE

# Invariant 7: pattern is a regex search, so r"\S" requires at least one non-space character.
Reason = Annotated[str, Field(pattern=r"\S")]


class RankInput(BaseModel):
    # frozen: assignment after creation raises ValidationError.
    # extra="forbid": unknown fields (e.g. an override) are rejected, not silently dropped.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # §3.4, NFR5: IDs never encode region. str so Stage 1's request_id passes through unchanged;
    # strict rejects non-str values (e.g. a UUID object) instead of coercing them.
    job_id: str = Field(strict=True, min_length=1)
    # Safety G3: 0 none, 1 conditional, 2 active. strict blocks coercion, so True or "2" fail.
    safety_level: int = Field(strict=True, ge=0, le=2)
    # Urgency G1/G2: None = no tier. No default, so callers must state it. strict: "4" and 4.0 fail.
    tally: int | None = Field(strict=True, ge=2, le=4)
    # FIFO tie-break: AwareDatetime rejects naive times, which can't be compared safely.
    original_timestamp: AwareDatetime
    # Logistics G1: carried for display; rank() must never read it. None = unknown, so a
    # display never shows a missing distance as 0 km.
    # allow_inf_nan=False: ge=0 alone lets inf through.
    distance_km: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    # mode="before" runs on the raw input, before Pydantic would turn a number into a
    # Unix-epoch datetime. A defaulted 0 would become 1970 and jump the FIFO queue.
    # ISO strings still parse, so JSON and database rows round-trip.
    @field_validator("original_timestamp", mode="before")
    @classmethod
    def _reject_numeric_timestamp(cls, value: object) -> object:
        if isinstance(value, (int, float)):
            raise ValueError("original_timestamp must be a datetime or ISO string with timezone, not a number")
        return value


class RankedJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int = Field(ge=1)
    job_id: str = Field(strict=True, min_length=1)
    flags: tuple[Reason, ...] = ()
    # description is schema metadata only; Pydantic doesn't enforce it.
    decided_by: Reason = Field(description="coordinator-only; tenant renderers must never emit")


class RankResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # No tier and no safety trigger: awaits a tier call instead of being ranked.
    review_band: tuple[str, ...]
    ranked: tuple[RankedJob, ...]


# ---------------------------------------------------------------------------
# Stage 1: Intake
# ---------------------------------------------------------------------------

class SourceTag(str, Enum):
    OFFICER = "officer"              # transcribed phone call
    TENANT_DIRECT = "tenant_direct"  # self-filled web form / email


class Report(BaseModel):
    """Raw report plus the stamped intake fields. No interpretation.

    The last four fields record where the report came from (provenance). They
    are never sent to the model and never read by ranking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    source_tag: SourceTag
    community: str = Field(min_length=1)  # needed by Stage 5 distance lookup
    original_report_timestamp: datetime

    region: Optional[str] = None
    source_file: Optional[str] = None
    source_item: Optional[int] = None  # item # within a multi-issue form
    timestamp_source: Optional[str] = None  # how original_report_timestamp was set

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
    "hazard",
    "harm_claimed",
    "fault_or_sign",
    "mismatch_claim",
    "mismatch_detail",
    "worsening_mentioned",
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
    # §3.2.2 2026-10-04 fields: no defaults, so a response that omits one fails validation
    # instead of silently reading as "no hazard" / "no harm".
    hazard_status: Literal["none", "described", "unclear"]
    mechanism_type: Optional[Literal["active", "conditional"]] = None
    harm_claimed: bool
    fault_or_sign: Literal["fault", "sign"]
    # §3.2.2 Field 8 (severity_mismatch in the doc). Renamed: model-facing names must not
    # invite judging how the tenant writes (CLAUDE.md invariant 3).
    claim_mismatch: Optional[Literal["over", "under"]]
    worsening_mentioned: bool
    quoted_spans: list[QuotedSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _claims_need_spans(self) -> "ExtractedFacts":
        cited = {s.field for s in self.quoted_spans}
        required: list[str] = []
        if self.fault_description:
            required.append("fault_description")
        if self.taxonomy_match:
            required.append("taxonomy_match")
        if self.impact_status == "intermittent":
            required.append("impact_status")
        if self.alternative_mentioned:
            required.append("alternative_mentioned")
        if self.coping_mentioned:
            required.append("coping_mentioned")
        if self.hazard_status in ("described", "unclear"):
            required.append("hazard")
        if self.harm_claimed:
            required.append("harm_claimed")
        if self.fault_or_sign == "sign":
            required.append("fault_or_sign")
        if self.claim_mismatch is not None:
            required += ["mismatch_claim", "mismatch_detail"]
        if self.worsening_mentioned:
            required.append("worsening_mentioned")
        missing = [f for f in required if f not in cited]
        if missing:
            raise ValueError(f"claimed facts without a quoted span: {missing}")

        # A list match with no fault named is inconsistent extraction: reject so extract() retries.
        if self.taxonomy_match and not self.fault_description:
            raise ValueError("fault_description: required when taxonomy_match is non-empty")
        # Unclear means no pathway was described, so evaluation picks the safety level, not the model.
        if self.hazard_status == "described" and self.mechanism_type is None:
            raise ValueError("mechanism_type: required when hazard_status is 'described'")
        if self.hazard_status != "described" and self.mechanism_type is not None:
            raise ValueError(f"mechanism_type: must be null when hazard_status is {self.hazard_status!r}")
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: SpanField
    text: str
    verified: bool


class EnrichedJob(BaseModel):
    """Output of Stage 5. Stage 6 reads only three of these fields for order."""

    # frozen: assignment would skip _consistent, so a scored job could lose its tally unchecked.
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    community: str
    original_report_timestamp: datetime
    fault_description: Optional[str] = None
    # Fault-list names matched in Stage 2, for the why-trace. Optional so
    # Stage 5 output without it still validates; never read by the sort.
    taxonomy_match: list[str] = Field(default_factory=list)

    # Evaluation output. No defaults: an omission must raise, not read as "no tier" or
    # "no hazard" (same reason as ExtractedFacts).
    tier: Optional[Literal["dangerous", "standard"]]
    # TIER_TABLE name whose sources back the tier (the candidate that produced the tally).
    tier_entry: Optional[str]
    base_points: Optional[int] = Field(ge=2, le=3)
    # The +1: removed by an alternative, a sign-only report, or a degraded-by-definition fault.
    severity_bump: Optional[int] = Field(ge=0, le=1)
    urgency_tally: Optional[int] = Field(ge=2, le=4)
    tally_reasons: tuple[Reason, ...]
    safety_flag: bool = False
    safety_level: Literal["active", "conditional", "none"]
    safety_reason: Reason
    flags: tuple[Reason, ...]
    spans: list[VerifiedSpan] = Field(default_factory=list)

    # Stage 5 logistics: display only, never read by the sort
    distance_cost_km: Optional[float] = None
    capacity_block_flag: bool = False
    next_actionable: Optional[str] = None
    shared_route_opportunities: list[str] = Field(default_factory=list)
    starvation_line: Optional[str] = None

    @model_validator(mode="after")
    def _consistent(self) -> "EnrichedJob":
        if self.safety_flag != (self.safety_level == "active"):
            raise ValueError(f"safety_flag={self.safety_flag} disagrees with safety_level={self.safety_level!r}")
        if self.urgency_tally is None:
            # A tier with no tally would be held in the review band or crash the coordinator view.
            if any(v is not None for v in (self.tier, self.tier_entry, self.base_points, self.severity_bump)):
                raise ValueError("tier, tier_entry, base_points and severity_bump must be None when urgency_tally is None")
            return self
        # The coordinator's tier line cites this entry's sources, so it must back the stated tier.
        if self.tier_entry not in TIER_TABLE or TIER_TABLE[self.tier_entry].tier != self.tier:
            raise ValueError(f"tier_entry {self.tier_entry!r} is not a {self.tier} entry in TIER_TABLE")
        if self.tier is None or self.base_points is None or self.severity_bump is None:
            raise ValueError("a scored job must carry its tier, base_points and severity_bump")
        if self.base_points + self.severity_bump != self.urgency_tally:
            raise ValueError("urgency_tally must equal base_points + severity_bump")
        return self

    @property
    def in_review_band(self) -> bool:
        # §3.4, §4.4: a no-tier safety job is ranked, not held.
        return self.urgency_tally is None and self.safety_level == "none"
