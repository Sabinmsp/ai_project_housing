from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class RankInput(BaseModel):
    # frozen: assignment after creation raises ValidationError.
    # extra="forbid": unknown fields (e.g. an override) are rejected, not silently dropped.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # §3.4, NFR5: IDs never encode region.
    job_id: UUID
    # Safety G3: 0 none, 1 conditional, 2 active. strict blocks coercion, so True or "2" fail.
    safety_level: int = Field(strict=True, ge=0, le=2)
    # Urgency G1/G2: None = no tier. No default, so callers must state it.
    tally: int | None = Field(ge=2, le=4)
    # FIFO tie-break: AwareDatetime rejects naive times, which can't be compared safely.
    original_timestamp: AwareDatetime
    # Logistics G1: carried for display; rank() must never read it.
    # allow_inf_nan=False: ge=0 alone lets inf through.
    distance_km: float = Field(default=0.0, ge=0, allow_inf_nan=False)
