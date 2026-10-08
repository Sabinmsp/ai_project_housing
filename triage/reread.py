"""Re-read safety net: a second extraction of a report whose hazard the second reader flagged.

Pipeline: extraction -> second reader -> [re-read] -> verification -> evaluation -> ranking.
Input: the first reading of each fault and, when triggered, a second ExtractionResult for the
same report. Output: per fault, the facts to evaluate (the safer hazard reading, everything
else from read 1), a ReRead record for the coordinator, and flags. It can only raise safety.
The second reader only triggers it; no value comes from Jev.
"""

from __future__ import annotations

from triage.models import (
    EnrichedJob,
    ExtractedFacts,
    ExtractionResult,
    ExtractionStatus,
    HazardReading,
    ReRead,
    SecondReading,
)
from triage.second_reader import flagged_fields

# Invariant 8: uncertainty errs high, so the higher reading wins. Evaluation maps these to
# safety levels 0, 1, 1, 2, so a higher reading never gives a lower level.
READING_ORDER: tuple[HazardReading, ...] = ("none", "unclear", "conditional", "active")
TRIGGER_FIELDS = frozenset({"hazard", "mechanism"})


def should_reread(client_name: str, second: SecondReading | None) -> bool:
    """Live extraction only (never offline or recorded), and only after a hazard or mechanism flag."""
    live = client_name.startswith("llm:")
    return live and second is not None and bool(flagged_fields(second) & TRIGGER_FIELDS)


def reading(facts: ExtractedFacts) -> HazardReading:
    """The hazard reading of one fault: active > conditional > unclear > none."""
    if facts.hazard_status == "described":
        return "active" if facts.mechanism_type == "active" else "conditional"
    return facts.hazard_status


def _match(read_1: ExtractedFacts, rereads: tuple[ExtractedFacts, ...], single: bool) -> ExtractedFacts | None:
    """The re-read fault matching read 1: directly when both reads name one fault, else by
    taxonomy (never position), the safest if several."""
    if single and len(rereads) == 1:
        # One fault each: they describe the same thing, so no taxonomy is needed (e.g. an
        # unlisted fault, whose taxonomy is empty).
        return rereads[0]
    names = set(read_1.taxonomy_match)
    if not names:
        return None
    same = [f for f in rereads if set(f.taxonomy_match) == names] or [f for f in rereads if names & set(f.taxonomy_match)]
    return max(same, key=lambda f: READING_ORDER.index(reading(f)), default=None)


def _with_hazard_of(read_1: ExtractedFacts, read_2: ExtractedFacts) -> ExtractedFacts:
    """read 1 with read 2's hazard reading and hazard quotes; nothing else changes."""
    spans = [s.model_dump() for s in read_1.quoted_spans if s.field != "hazard"]
    spans += [s.model_dump() for s in read_2.quoted_spans if s.field == "hazard"]
    return ExtractedFacts.model_validate({
        **read_1.model_dump(), "hazard_status": read_2.hazard_status,
        "mechanism_type": read_2.mechanism_type, "quoted_spans": spans,
    })


def _unavailable(read_1: ExtractedFacts, why: str) -> tuple[ExtractedFacts, ReRead, tuple[str, ...]]:
    record = ReRead(read_1=reading(read_1), read_2=None, unavailable_reason=why, used="read 1")
    return read_1, record, (f"Re-read unavailable — {why}; read 1 kept. Check.",)


def combine(read_1: ExtractedFacts, reread: ExtractionResult,
            single: bool) -> tuple[ExtractedFacts, ReRead, tuple[str, ...]]:
    """The facts to evaluate, the record of both readings, and the coordinator flags.

    single: read 1 of this report named exactly one fault.
    """
    first = reading(read_1)
    if reread.status is not ExtractionStatus.OK or reread.extraction is None:
        return _unavailable(read_1, "; ".join(reread.errors)[:200] or reread.status.value)
    match = _match(read_1, reread.extraction.faults, single)
    if match is None:
        return _unavailable(read_1, f"no re-read fault matches taxonomy {read_1.taxonomy_match}")
    second = reading(match)
    flags = () if second == first else (
        f"Readings inconsistent — read 1: {first}, read 2: {second}; the safer reading is used. Check.",)
    if READING_ORDER.index(second) > READING_ORDER.index(first):
        return _with_hazard_of(read_1, match), ReRead(read_1=first, read_2=second, unavailable_reason=None,
                                                      used="read 2"), flags
    return read_1, ReRead(read_1=first, read_2=second, unavailable_reason=None, used="read 1"), flags


def attach(job: EnrichedJob, record: ReRead, flags: tuple[str, ...]) -> EnrichedJob:
    """The job with its re-read record and flags added for the coordinator."""
    return EnrichedJob.model_validate({**job.model_dump(), "reread": record, "flags": (*job.flags, *flags)})
