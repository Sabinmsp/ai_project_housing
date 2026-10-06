"""Fault tier table: a fault's tier is looked up here, never inferred (master §4.4).

Two authorities:
- nt.gov.au repairs guidance: "dangerous things are repaired first" (master §4.8).
- NT Residential Tenancies Act 1999 s63(2) emergency repairs, as in force 1 Aug 2025.
"""

from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# s63(2)(e), (k), (m) and (n) are excluded: "dangerous", "unsafe", "likely to injure" and
# "serious" are judgments, not lookups (Part 0). "Serious" is dropped from roof leak and
# storm damage for the same reason; the accepted cost is a false-high on minor cases.


class FaultEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tier: Literal["dangerous", "standard"]
    degraded: bool
    sources: tuple[str, ...] = Field(min_length=1)


def _entry(name: str, tier: Literal["dangerous", "standard"], *sources: str, degraded: bool = False) -> FaultEntry:
    return FaultEntry(name=name, tier=tier, degraded=degraded, sources=sources)


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
)

# MappingProxyType is a read-only view; _ENTRIES is a tuple, so nothing else holds the dict.
TIER_TABLE = MappingProxyType({e.name: e for e in _ENTRIES})

# What the extraction model may see: names only, never tiers or sources.
FAULT_NAMES = tuple(TIER_TABLE)
