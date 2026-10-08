"""Evaluation: the policy applied to one fault's facts (master §3.5, §4).

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
Input: ExtractedFacts plus verification's unverified (field, quote) pairs. Output: an
Evaluation with tier, urgency tally, safety level and reasoned flags. The model reports
facts; this code applies policy. Pure functions, no I/O, no model.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """The tier-table entries matched, the tier taken, and an ambiguity flag if several."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[FaultEntry, ...]
    tier: Literal["dangerous", "standard"] | None
    flag: Reason | None


def lookup_tier(taxonomy_match: Sequence[str], *, coordinator_call: bool) -> TierResult:
    """Look up each matched fault in TIER_TABLE; several matches take the highest tier, flagged.

    coordinator_call is True only for a coordinator's tier call; the model's matches pass False.
    It has no default, so a caller must say which it is.

    Raises:
        ValueError: a name is not in TIER_TABLE, or a coordinator-only name came from extraction.
    """
    unknown = [n for n in dict.fromkeys(taxonomy_match) if n not in TIER_TABLE]
    if unknown:
        raise ValueError(f"fault not in TIER_TABLE: {', '.join(map(repr, unknown))}")
    # Invariant 2: the model never decides a tier, so it can't name a coordinator-only entry.
    reserved = [n for n in dict.fromkeys(taxonomy_match) if TIER_TABLE[n].coordinator_only]
    if reserved and not coordinator_call:
        raise ValueError(f"coordinator-only repair type from extraction: {', '.join(map(repr, reserved))}")
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
    """Urgency tally = base + bump, the entry that produced it, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tally: int | None = Field(ge=2, le=4)
    base: int | None = Field(ge=2, le=3)
    bump: int | None = Field(ge=0, le=1)
    # TIER_TABLE name of the candidate that produced the tally; its sources back the tier.
    winner: str | None
    reasons: tuple[Reason, ...]
    flags: tuple[Reason, ...]

    @model_validator(mode="after")
    def _parts_add_up(self) -> "TallyResult":
        if self.tally is None:
            if self.base is not None or self.bump is not None or self.winner is not None:
                raise ValueError("base, bump and winner must be None when tally is None")
        elif self.winner is None:
            raise ValueError("a tally must name the fault that produced it")
        elif self.base is None or self.bump is None or self.base + self.bump != self.tally:
            raise ValueError(f"tally {self.tally} must equal base {self.base} + bump {self.bump}")
        return self


def compute_tally(tier: TierResult, facts: ExtractedFacts, unverified: Unverified) -> TallyResult:
    """Base points for the tier plus the +1 bump, taking the highest-scoring candidate fault."""
    if tier.tier is None:
        return TallyResult(tally=None, base=None, bump=None, winner=None, reasons=(), flags=())

    # D1 (invariant 8): an unverified span never lowers a score, so its claim is ignored.
    failed = {field for field, _ in unverified}
    alternative = facts.alternative_mentioned and "alternative_mentioned" not in failed
    sign = facts.fault_or_sign == "sign" and "fault_or_sign" not in failed

    def score(entry: FaultEntry) -> tuple[int, int, str, tuple[str, ...]]:
        zero_reasons = []
        if entry.degraded:  # FR2t, §4.3: dripping or stiff taps leave the function working.
            zero_reasons.append(DEGRADED)
        # §4.3 no-redundancy default: only an alternative removes the +1. Urgency G2a: coping
        # never counts, so coping_mentioned is deliberately not read.
        if alternative:
            zero_reasons.append(ALTERNATIVE)
        if sign:  # §4.2 sign vs fault: a sign gets the tier of its fault but no +1.
            zero_reasons.append(SIGN)
        if zero_reasons:
            return _BASE_POINTS[entry.tier], 0, entry.name, tuple(zero_reasons)
        return _BASE_POINTS[entry.tier], 1, entry.name, (NO_ALTERNATIVE,)

    # D2 (invariant 8): ambiguity errs high, so the best-scoring candidate wins.
    # max() keeps the first of equal scores, i.e. table order.
    base, bump, winner, reasons = max((score(e) for e in tier.entries), key=lambda s: s[0] + s[1])

    # Only flag an ignored claim when it actually kept the +1; otherwise the flag text is false.
    flags: list[str] = []
    if bump == 1:
        if facts.alternative_mentioned and not alternative:
            flags.append(UNVERIFIED_ALTERNATIVE)
        if facts.fault_or_sign == "sign" and not sign:
            flags.append(UNVERIFIED_SIGN)
    return TallyResult(
        tally=base + bump, base=base, bump=bump, winner=winner, reasons=reasons, flags=tuple(flags)
    )


UNCLEAR_HAZARD = "Possible hazard — needs a direct look: '{quote}'"
UNVERIFIED_HAZARD = "Hazard quote isn't in the report — level kept, check the reading"
NO_HAZARD = "No hazard mechanism described"


class SafetyResult(BaseModel):
    """Safety level (0 none, 1 conditional, 2 active), its reason, and any flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int = Field(ge=0, le=2)
    reason: Reason
    flags: tuple[Reason, ...]


def _quote(facts: ExtractedFacts, field: str, unverified: Unverified) -> str:
    # Cite a verified quote when there is one, so a reason never repeats words that aren't the
    # tenant's. The validator guarantees a span for every claimed field, so texts is never empty.
    texts = [s.text for s in facts.quoted_spans if s.field == field]
    return next((t for t in texts if (field, t) not in unverified), texts[0])


def _none_verified(facts: ExtractedFacts, field: str, unverified: Unverified) -> bool:
    return all((field, s.text) in unverified for s in facts.quoted_spans if s.field == field)


def compute_safety(facts: ExtractedFacts, unverified: Unverified) -> SafetyResult:
    """Safety level 0-2 from the hazard reading, plus flags for unclear or claimed-only harm."""
    # §3.5, §4.5: safety is independent of tier, so taxonomy_match is never read.
    flags: list[str] = []
    if facts.hazard_status == "described":
        quote = _quote(facts, "hazard", unverified)
        # Safety G3: active is a full override; conditional is elevated only.
        if facts.mechanism_type == "active":
            level, reason = 2, f"Active hazard described: '{quote}' — full override"
        else:
            level, reason = 1, f"Conditional hazard described: '{quote}' — elevated, does not bypass active hazards"
        # §4.5: a described pathway always wins over G5, so harm_claimed adds no flag.
    elif facts.hazard_status == "unclear":
        quote = _quote(facts, "hazard", unverified)
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
                f"Safety claim — unconfirmed: '{_quote(facts, 'harm_claimed', unverified)}' — fast human check, no override"
            )
    # D1 (invariant 8): an unverified hazard quote never lowers the level; a human checks it.
    # One verified quote backs the level, so only flag when none is verified.
    if level > 0 and _none_verified(facts, "hazard", unverified):
        flags.append(UNVERIFIED_HAZARD)
    return SafetyResult(level=level, reason=reason, flags=tuple(flags))


# FR16c: these two only flag. They never change a tally or level.


def mismatch_flag(facts: ExtractedFacts, tier: TierResult, unverified: Unverified) -> Reason | None:
    """Flag a claim stronger than the report's details, or a played-down repair-first fault."""
    if facts.claim_mismatch is None:
        return None
    claim, detail = _quote(facts, "mismatch_claim", unverified), _quote(facts, "mismatch_detail", unverified)
    if facts.claim_mismatch == "over":
        return f"Claim stronger than the report's own details: '{claim}' vs '{detail}' — check before acting"
    # §3.2.2: the model never sees tiers, so code keeps "under" for dangerous-tier faults only.
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
    """Everything evaluation decided about one fault."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: TierResult
    tally: TallyResult
    safety: SafetyResult
    # Deduped, in order: safety, ambiguity, tally, mismatch, unverified.
    flags: tuple[Reason, ...]


def evaluate(facts: ExtractedFacts, unverified: Unverified) -> Evaluation:
    """Tier, tally, safety and every flag for one fault's extracted facts.

    Raises:
        ValueError: no fault is named, or unverified holds a pair that isn't one of the
            facts' quoted spans.
    """
    # D4: evaluation must never score a report that names no fault.
    if not facts.fault_description:
        raise ValueError("no fault named — out of scope, route to coordinator contact (master §3.2.4)")
    spans = {(s.field, s.text) for s in facts.quoted_spans}
    if not unverified <= spans:
        raise ValueError(f"unverified pairs are not quoted spans of this report: {sorted(unverified - spans)}")

    tier = lookup_tier(facts.taxonomy_match, coordinator_call=False)
    tally = compute_tally(tier, facts, unverified)
    safety = compute_safety(facts, unverified)
    candidates = (*safety.flags, tier.flag, *tally.flags, mismatch_flag(facts, tier, unverified), unverified_flag(facts, unverified))
    flags = tuple(dict.fromkeys(f for f in candidates if f is not None))
    return Evaluation(tier=tier, tally=tally, safety=safety, flags=flags)
