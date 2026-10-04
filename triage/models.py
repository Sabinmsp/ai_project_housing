from typing import Annotated
from uuid import UUID

from pydantic import UUID4, AwareDatetime, BaseModel, ConfigDict, Field, field_validator

# Invariant 7: pattern is a regex search, so r"\S" requires at least one non-space character.
Reason = Annotated[str, Field(pattern=r"\S")]


class RankInput(BaseModel):
    # frozen: assignment after creation raises ValidationError.
    # extra="forbid": unknown fields (e.g. an override) are rejected, not silently dropped.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # §3.4, NFR5: IDs never encode region. UUID4 accepts only random UUIDs, not name- or number-derived ones.
    job_id: UUID4
    # Safety G3: 0 none, 1 conditional, 2 active. strict blocks coercion, so True or "2" fail.
    safety_level: int = Field(strict=True, ge=0, le=2)
    # Urgency G1/G2: None = no tier. No default, so callers must state it. strict: "4" and 4.0 fail.
    tally: int | None = Field(strict=True, ge=2, le=4)
    # FIFO tie-break: AwareDatetime rejects naive times, which can't be compared safely.
    original_timestamp: AwareDatetime
    # Logistics G1: carried for display; rank() must never read it.
    # allow_inf_nan=False: ge=0 alone lets inf through.
    distance_km: float = Field(default=0.0, ge=0, allow_inf_nan=False)

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
    job_id: UUID
    flags: tuple[Reason, ...] = ()
    # description is schema metadata only; Pydantic doesn't enforce it.
    decided_by: Reason = Field(description="coordinator-only; tenant renderers must never emit")


class RankResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # No tier and no safety trigger: awaits a tier call instead of being ranked.
    review_band: tuple[UUID, ...]
    ranked: tuple[RankedJob, ...]
