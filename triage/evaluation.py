"""Stage "evaluation": pure functions from extracted facts to tier, tally and safety (master §4).

No I/O, no model.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from triage.models import ExtractedFacts, Reason
from triage.tiers import TIER_TABLE, FaultEntry

_TIER_ORDER = {"standard": 0, "dangerous": 1}
# (field, quote text) pairs whose quote was not found in the report.
Unverified = frozenset[tuple[str, str]]
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


def compute_tally(tier: TierResult, facts: ExtractedFacts, unverified: Unverified) -> TallyResult:
    """Base points for the tier plus the +1 bump, taking the highest-scoring candidate fault."""
    if tier.tier is None:
        return TallyResult(tally=None, reasons=(), flags=())

    # D1 (invariant 8): an unverified span never lowers a score, so its claim is ignored.
    failed = {field for field, _ in unverified}
    alternative = facts.alternative_mentioned and "alternative_mentioned" not in failed
    sign = facts.fault_or_sign == "sign" and "fault_or_sign" not in failed

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


UNCLEAR_HAZARD = "Possible hazard — needs a direct look: '{quote}'"
UNVERIFIED_HAZARD = "Hazard quote isn't in the report — level kept, check the reading"
NO_HAZARD = "No hazard mechanism described"


class SafetyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int = Field(ge=0, le=2)
    reason: Reason
    flags: tuple[Reason, ...]


def _quote(facts: ExtractedFacts, field: str) -> str:
    # The validator guarantees a span for every claimed field, so next() always finds one.
    return next(s.text for s in facts.quoted_spans if s.field == field)


def compute_safety(facts: ExtractedFacts, unverified: Unverified) -> SafetyResult:
    """Safety level 0-2 from the hazard reading, plus flags for unclear or claimed-only harm."""
    # §4.4, Safety G1: safety is independent of tier, so taxonomy_match is never read.
    flags: list[str] = []
    if facts.hazard_status == "described":
        quote = _quote(facts, "hazard")
        # Safety G3: active is a full override; conditional is elevated only.
        if facts.mechanism_type == "active":
            level, reason = 2, f"Active hazard described: '{quote}' — full override"
        else:
            level, reason = 1, f"Conditional hazard described: '{quote}' — elevated, does not bypass active hazards"
        # D3: a described pathway always wins, so harm_claimed adds no G5 flag (§4.5).
    elif facts.hazard_status == "unclear":
        quote = _quote(facts, "hazard")
        # §4.5 / invariant 8: unclear errs high to conditional. Self-mitigation is not a field,
        # so avoidance ("we avoid that spot") can't lower it.
        level, reason = 1, f"Unclear hazard: '{quote}' — treated as conditional"
        # D3: the unclear flag already sends it to a human, so harm_claimed adds no G5 flag.
        flags.append(UNCLEAR_HAZARD.format(quote=quote))
    else:
        level, reason = 0, NO_HAZARD
        # Safety G5: a harm claim with no pathway is checked, never trusted or dismissed. D1: the
        # flag fires even if the harm quote is unverified, since dropping it would lower the outcome.
        if facts.harm_claimed:
            flags.append(
                f"Safety claim — unconfirmed: '{_quote(facts, 'harm_claimed')}' — fast human check, no override"
            )
    # D1 (invariant 8): an unverified hazard quote never lowers the level; a human checks it.
    if level > 0 and any(field == "hazard" for field, _ in unverified):
        flags.append(UNVERIFIED_HAZARD)
    return SafetyResult(level=level, reason=reason, flags=tuple(flags))


# FR16c: these two only flag. They never return or change a tally or level.


def mismatch_flag(facts: ExtractedFacts, tier: TierResult) -> Reason | None:
    """Flag a claim stronger than the report's details, or a played-down repair-first fault."""
    if facts.claim_mismatch is None:
        return None
    claim, detail = _quote(facts, "mismatch_claim"), _quote(facts, "mismatch_detail")
    if facts.claim_mismatch == "over":
        return f"Claim stronger than the report's own details: '{claim}' vs '{detail}' — check before acting"
    # §3.2.2 Field 8: the model never sees tiers, so code keeps "under" for dangerous-tier faults only.
    if tier.tier == "dangerous":
        return f"Report plays down a fault on the repair-first list: '{claim}' vs '{detail}' — check before scheduling"
    return None


def unverified_flag(facts: ExtractedFacts, unverified: Unverified) -> Reason | None:
    """Flag every quote whose field failed verification, even when no score moved."""
    # §3.2.5: a fabricated quote means the extraction itself is unreliable.
    # Exact pairs, so a verified quote in the same field is never listed. Dedupe, keep span order.
    quotes = dict.fromkeys(s.text for s in facts.quoted_spans if (s.field, s.text) in unverified)
    if not quotes:
        return None
    listed = ", ".join(f"'{q}'" for q in quotes)
    return f"Quoted words not found in the report: {listed} — check the reading"


class Evaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: TierResult
    tally: TallyResult
    safety: SafetyResult
    # Deduped, in order: safety, ambiguity, tally, mismatch, unverified.
    flags: tuple[Reason, ...]


def evaluate(facts: ExtractedFacts, unverified: Unverified) -> Evaluation:
    """Tier, tally, safety and every flag for one report's extracted facts."""
    # D4: evaluation must never score a report that names no fault.
    if not facts.fault_description:
        raise ValueError("no fault named — out of scope, route to coordinator contact (master §3.2.4)")
    spans = {(s.field, s.text) for s in facts.quoted_spans}
    if not unverified <= spans:
        raise ValueError(f"unverified pairs are not quoted spans of this report: {sorted(unverified - spans)}")

    tier = lookup_tier(facts.taxonomy_match)
    tally = compute_tally(tier, facts, unverified)
    safety = compute_safety(facts, unverified)
    candidates = (*safety.flags, tier.flag, *tally.flags, mismatch_flag(facts, tier), unverified_flag(facts, unverified))
    flags = tuple(dict.fromkeys(f for f in candidates if f is not None))
    return Evaluation(tier=tier, tally=tally, safety=safety, flags=flags)
