"""The fault tier table: a fault's tier is looked up here, never inferred (master §4.4, §4.8).

Used by evaluation (tiers and sources) and extraction (FAULT_NAMES only, never tiers).
Two authorities:
- nt.gov.au repairs guidance: "dangerous things are repaired first".
- NT Residential Tenancies Act 1999 s63(2) emergency repairs, as in force 1 Aug 2025.
"""

from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# s63(2)(e), (k), (m) and (n) are excluded: "dangerous", "unsafe", "likely to injure" and
# "serious" are judgments, not lookups (Part 0, master E2). "Serious" is dropped from roof
# leak and storm damage for the same reason; the accepted cost is a false-high on minor cases.


class FaultEntry(BaseModel):
    """One tier-table row: fault name, tier, whether it is degraded by definition, sources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tier: Literal["dangerous", "standard"]
    degraded: bool
    sources: tuple[str, ...] = Field(min_length=1)
    # True only for a coordinator's own call on one job: never in FAULT_NAMES, so the model can't match it.
    coordinator_only: bool


def _entry(name: str, tier: Literal["dangerous", "standard"], *sources: str, degraded: bool = False) -> FaultEntry:
    return FaultEntry(name=name, tier=tier, degraded=degraded, sources=sources, coordinator_only=False)


COORDINATOR_SOURCE = "coordinator's call"
NO_FIT_EMERGENCY = "no listed fault fits — emergency"
NO_FIT_GENERAL = "no listed fault fits — general"


_ENTRIES = (
    _entry("blocked or broken toilet", "dangerous", "nt.gov.au", "RTA s63(2)(b)"),
    _entry("blocked drain", "dangerous", "nt.gov.au"),
    _entry("sewage leak", "dangerous", "nt.gov.au"),
    _entry("leaking or burst water main or pipe", "dangerous", "nt.gov.au", "RTA s63(2)(a)"),
    _entry("exposed electrical wires", "dangerous", "nt.gov.au"),
    _entry("gas leak", "dangerous", "nt.gov.au", "RTA s63(2)(d)"),
    _entry("roof leak", "dangerous", "RTA s63(2)(c)"),
    _entry("flooding or flood damage", "dangerous", "RTA s63(2)(f)"),
    _entry("storm, fire or impact damage", "dangerous", "RTA s63(2)(g)"),
    _entry("no gas, electricity or water supply", "dangerous", "RTA s63(2)(h)"),
    _entry("hot water system not working", "dangerous", "RTA s63(2)(j)"),
    _entry("stove or oven not working", "dangerous", "RTA s63(2)(j)"),
    _entry("dripping tap or tap tight to turn", "standard", "nt.gov.au", degraded=True),
    _entry("stove element not working", "standard", "nt.gov.au"),
    _entry("fan not working properly", "standard", "nt.gov.au"),
    _entry("power point not working", "standard", "nt.gov.au"),
    # Master §4.4/§4.8/FR2z as revised: a named coordinator's recorded call on a single job is the
    # authority here, never the model and never reused automatically.
    FaultEntry(name=NO_FIT_EMERGENCY, tier="dangerous", degraded=False, sources=(COORDINATOR_SOURCE,), coordinator_only=True),
    FaultEntry(name=NO_FIT_GENERAL, tier="standard", degraded=False, sources=(COORDINATOR_SOURCE,), coordinator_only=True),
)

# MappingProxyType is a read-only view; _ENTRIES is a tuple, so nothing else holds the dict.
TIER_TABLE = MappingProxyType({e.name: e for e in _ENTRIES})

# What the extraction model may see: names only, never tiers or sources, and never a
# coordinator-only entry. Also the recording key (recording.prompt_hash), so it must not change.
FAULT_NAMES = tuple(name for name, e in TIER_TABLE.items() if not e.coordinator_only)
