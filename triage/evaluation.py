"""Stage "evaluation": pure functions from extracted facts to tier, tally and safety (master §4).

No I/O, no model.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from triage.models import Reason
from triage.tiers import TIER_TABLE, FaultEntry

_TIER_ORDER = {"standard": 0, "dangerous": 1}


class TierResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[FaultEntry, ...]
    tier: Literal["dangerous", "standard"] | None
    flag: Reason | None


def lookup_tier(taxonomy_match: Sequence[str]) -> TierResult:
    """Look up each matched fault in TIER_TABLE; several matches take the highest tier, flagged."""
    unknown = [n for n in dict.fromkeys(taxonomy_match) if n not in TIER_TABLE]
    if unknown:
        raise ValueError(f"fault not in TIER_TABLE: {', '.join(map(repr, unknown))}")
    # Table order, not input order, so the same set of matches always gives the same result.
    matched = set(taxonomy_match)
    entries = tuple(e for name, e in TIER_TABLE.items() if name in matched)
    if not entries:
        return TierResult(entries=(), tier=None, flag=None)
    # Invariant 8: ambiguity errs high, so the highest tier among the candidates wins.
    tier = max((e.tier for e in entries), key=_TIER_ORDER.__getitem__)
    if len(entries) == 1:
        return TierResult(entries=entries, tier=tier, flag=None)
    # §3.2.4, FR2p; invariant 9: the flag names every candidate, not a bare marker.
    # Quoted and joined with "or": table names themselves contain commas.
    candidates = " or ".join(f'"{e.name}"' for e in entries)
    flag = f"Fault type unclear — could be {candidates}. Scored at the highest tier among them ({tier})."
    return TierResult(entries=entries, tier=tier, flag=flag)
