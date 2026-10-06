"""Stage "verification": check that every quoted span is the report's own words.

Pure functions, no I/O, no model.
"""

from triage.evaluation import Unverified
from triage.models import ExtractedFacts, QuotedSpan


def _supports_claim(span: QuotedSpan, facts: ExtractedFacts) -> bool:
    # A span backs a claim only when its field's value needs a quote; the rest are noise the
    # model added (e.g. impact_status quoting "ongoing") and must not count as fabrications.
    match span.field:
        case "fault_description" | "taxonomy_match":
            return True
        case "impact_status":
            return facts.impact_status == "intermittent"
        case "fault_or_sign":
            return facts.fault_or_sign == "sign"
        case "hazard":
            return facts.hazard_status != "none"
        case "mismatch_claim" | "mismatch_detail":
            return facts.claim_mismatch is not None
        case "alternative_mentioned":
            return facts.alternative_mentioned
        case "coping_mentioned":
            return facts.coping_mentioned
        case "harm_claimed":
            return facts.harm_claimed
        case "worsening_mentioned":
            return facts.worsening_mentioned
    raise ValueError(f"span field {span.field!r} has no claim rule")


def claim_spans(facts: ExtractedFacts) -> tuple[QuotedSpan, ...]:
    """The spans that back a claim, in their original order."""
    return tuple(s for s in facts.quoted_spans if _supports_claim(s, facts))


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def verify_spans(raw_text: str, facts: ExtractedFacts) -> Unverified:
    """(field, quote) pairs whose quote is not in the report text."""
    report = _normalise(raw_text)
    # Exact substring after case and whitespace folding, no fuzzy matching: a near-match is not
    # the tenant's words. An all-whitespace quote normalises to "" and would match anything.
    return frozenset(
        (s.field, s.text) for s in claim_spans(facts) if not _normalise(s.text) or _normalise(s.text) not in report
    )
