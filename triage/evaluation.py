"""Stage "evaluation": pure functions from extracted facts to tier, tally and safety (master §4).

No I/O, no model.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from triage.models import ExtractedFacts, Reason
from triage.tiers import TIER_TABLE, FaultEntry

_TIER_ORDER = {"standard": 0, "dangerous": 1}
# §4.2 tier mapping.
_BASE_POINTS = {"dangerous": 3, "standard": 2}

NO_ALTERNATIVE = "No alternative named — no-redundancy default: +1"
ALTERNATIVE = "Alternative named in report: +0"
SIGN = "Report describes a sign of the fault, not the fault itself: +0"
DEGRADED = "This fault leaves the function working by definition: +0"
UNVERIFIED_ALTERNATIVE = "Alternative mentioned, but the quote isn't in the report — ignored (+1 kept)"
UNVERIFIED_SIGN = "Sign-only reading, but the quote isn't in the report — scored as the fault"


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


class TallyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tally: int | None = Field(ge=2, le=4)
    reasons: tuple[Reason, ...]
    flags: tuple[Reason, ...]


def compute_tally(tier: TierResult, facts: ExtractedFacts, unverified: frozenset[str]) -> TallyResult:
    """Base points for the tier plus the +1 bump, taking the highest-scoring candidate fault."""
    if tier.tier is None:
        return TallyResult(tally=None, reasons=(), flags=())

    # D1 (invariant 8): an unverified span never lowers a score, so its claim is ignored.
    alternative = facts.alternative_mentioned and "alternative_mentioned" not in unverified
    sign = facts.fault_or_sign == "sign" and "fault_or_sign" not in unverified

    def score(entry: FaultEntry) -> tuple[int, tuple[str, ...]]:
        zero_reasons = []
        if entry.degraded:  # FR2t: dripping or stiff taps leave the function working (§4.3).
            zero_reasons.append(DEGRADED)
        # §4.3 no-redundancy default: only an alternative removes the +1. G2a: coping never
        # counts, so coping_mentioned is deliberately not read.
        if alternative:
            zero_reasons.append(ALTERNATIVE)
        if sign:  # §4.2 sign vs fault: a sign gets the tier of its fault but no +1.
            zero_reasons.append(SIGN)
        if zero_reasons:
            return _BASE_POINTS[entry.tier], tuple(zero_reasons)
        return _BASE_POINTS[entry.tier] + 1, (NO_ALTERNATIVE,)

    # D2 (invariant 8): ambiguity errs high, so the best-scoring candidate wins.
    # max() keeps the first of equal scores, i.e. table order.
    tally, reasons = max((score(e) for e in tier.entries), key=lambda s: s[0])

    # Only flag an ignored claim when it actually kept the +1; otherwise the flag text is false.
    flags: list[str] = []
    if reasons == (NO_ALTERNATIVE,):
        if facts.alternative_mentioned and not alternative:
            flags.append(UNVERIFIED_ALTERNATIVE)
        if facts.fault_or_sign == "sign" and not sign:
            flags.append(UNVERIFIED_SIGN)
    return TallyResult(tally=tally, reasons=reasons, flags=tuple(flags))
