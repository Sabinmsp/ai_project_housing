"""Extraction: the only stage where a model reads anything.

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
Input: one Report. Output: an ExtractionResult holding a ReportExtraction (facts with quoted
spans, one entry per fault) or a flag for a human. The model sees the report text and the
fault names only, never tiers, points, scoring rules or other jobs.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol

from pydantic import ValidationError

from .models import (
    ExtractedFacts,
    ExtractionResult,
    ExtractionStatus,
    QuotedSpan,
    Report,
    ReportExtraction,
)
# Invariant 1: import names only. TIER_TABLE carries tiers and must never be imported here.
from .tiers import FAULT_NAMES

try:
    from openai import OpenAIError as _ProviderError  # optional dependency
except ImportError:
    _ProviderError = OSError  # type: ignore[misc, assignment]

# The only examples the model sees. None may come from reports/ (a test checks), so the
# prompt can't teach the answer to a report it will later be scored on.
PROMPT_EXAMPLES = {
    "sign": "the laundry stinks of sewage",
    "fault": "the toilet won't flush",
    "alternative": "using the shower in the other bathroom",
    "coping": "washing at my sister's place",
    "unclear_hazard": "the switchboard is buzzing and smells like burning",
    "active_hazard": "water is touching the wire",
}

# Invariant 1: no tiers, sources, points or other jobs. Invariant 3: no word that invites
# judging how bad something is or how the tenant writes (banned-words test).
SYSTEM_PROMPT = f"""You read housing maintenance reports from tenants in remote Northern \
Territory communities and record what each report says. You never judge how bad, pressing \
or alarming anything is.

General rules:
1. If the report names no fault, return an empty faults list. Never return an entry that \
names no fault.
2. Return one entry per distinct fault. Never combine two faults into one entry, and never \
split one fault into two entries.
3. A fault that is not on the fault list is normal: return it with an empty taxonomy_match. \
No match is a correct answer; never force the nearest list entry.
4. Match informal, regional or misspelt wording by meaning, not by its exact words. An \
unfamiliar word is not evidence that a fault is off the list. This applies whoever wrote the \
report.
5. Report presence, never absence: say the report does not mention something, never that \
the thing does not exist. A short report is complete as written; record only what is there.
6. Every span's text is words copied exactly from the REPORT TEXT — never a fault list name, \
never a field value such as 'ongoing', 'fault' or 'sign'. Add a span only where a field below \
says one is required.

### faults
One entry per distinct fault the report names, in the order they appear. Empty when the \
report names no fault.

### fault_description
The fault in the report's own words, lightly trimmed. Required in every entry. Span field: \
fault_description.

### taxonomy_match
Names from the fault list that this fault matches by meaning. If the words could fit more \
than one entry, list every entry they fit; do not choose between them. Empty if none fits. Span field: taxonomy_match, quoting the \
words that describe the fault. When the list is not empty it needs its own span with field \
taxonomy_match, even if those are the same words as fault_description.

### alternative_mentioned
true only if the report names another working instance of the same thing in the home \
(e.g. "{PROMPT_EXAMPLES['alternative']}"). Otherwise false. Span required when true.

### coping_mentioned
true if the report describes any other way of getting by: a bucket, the servo, a neighbour, \
takeaway (e.g. "{PROMPT_EXAMPLES['coping']}"). Coping is never an alternative: the same \
words never set both fields. Span required when true.

### impact_status
"ongoing" unless the report says the fault comes and goes; then "intermittent", with a span.

### hazard_status
"described": the report states a one-step physical pathway to harm. Never a chain of \
events, never invented, never inferred from how alarmed the writer sounds.
"unclear": a possible harm source (electrical, gas, structural, fire) plus an active \
abnormality (buzzing, sparking, burning smell, sagging, cracking), with no pathway stated \
(e.g. "{PROMPT_EXAMPLES['unclear_hazard']}").
"none": anything else. A dead appliance on its own is none.
A gas leak reported as present is "described" with mechanism_type "active": the pathway is \
part of the fault.
Avoidance or self-mitigation ("we keep the kids out") never changes hazard_status.
Span field: hazard, required for "described" and "unclear".

### mechanism_type
Set only when hazard_status is "described": "active" if the pathway is happening now \
(e.g. "{PROMPT_EXAMPLES['active_hazard']}"), "conditional" if it could happen only under a \
condition the report states. Null otherwise.

### harm_claimed
true if the report names a harm to a person. A health condition counts only when the report \
links it directly to this fault; a vulnerable person simply living there does not count. \
Span required when true.

### fault_or_sign
"sign" only when the report gives a sensed cue alone: a smell, sound or stain \
(e.g. "{PROMPT_EXAMPLES['sign']}"). Any described effect is the fault \
(e.g. "{PROMPT_EXAMPLES['fault']}"). Unsure: "fault". Gas smells and burning smells are \
never signs. A smell of gas is the fault itself (a gas leak), so fault_or_sign is 'fault'. \
Span required for "sign".

### claim_mismatch
"over" when the wording is more dramatic than the report's own details; "under" when the \
wording explicitly plays the fault down (e.g. "nothing too bad"). Compare only against \
details in the same report. Plain or brief wording is never a mismatch. Null otherwise. When \
set, quote both: mismatch_claim (the wording) and mismatch_detail (the detail it is compared \
with).

### worsening_mentioned
true if the report says the fault is getting worse. Span required when true.

### quoted_spans
Each span names a field and copies the report's exact words. Allowed field values: \
fault_description, taxonomy_match, alternative_mentioned, coping_mentioned, impact_status, \
hazard, harm_claimed, fault_or_sign, mismatch_claim, mismatch_detail, worsening_mentioned.
"""


def build_user_prompt(raw_text: str, fault_names: tuple[str, ...] = FAULT_NAMES) -> str:
    """The per-report message: the fault-name list and the report text verbatim."""
    faults = "\n".join(f"- {name}" for name in fault_names)
    # report_text_from_prompt parses this block back out; keep the two in step.
    return f"FAULT LIST:\n{faults}\n\nREPORT TEXT:\n<<<\n{raw_text}\n>>>"


def report_text_from_prompt(user: str) -> str:
    """The report text inside a message built by build_user_prompt.

    Raises:
        ValueError: the message has no REPORT TEXT block.
    """
    m = re.search(r"REPORT TEXT:\n<<<\n(.*)\n>>>", user, re.S)
    if m is None:
        raise ValueError("user prompt has no REPORT TEXT block")
    return m.group(1)


class LLMClient(Protocol):
    """Anything extract() can call: live, recorded, recording or offline."""

    name: str

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Return a JSON string conforming to schema (structured-output mode)."""
        ...


# One report's facts and quotes fit easily. Without a cap the provider reserves the model's
# maximum (65k tokens) against the account balance on every call.
MAX_OUTPUT_TOKENS = 1024

# Probe 2026-10-06 gave 0/21 validation failures vs 17/21 for gpt-4o-mini.
DEFAULT_MODEL = "gpt-4o"


def configured_model() -> str:
    """TRIAGE_MODEL, or the default when it is unset or empty."""
    # `or`, not a get() default: a blank `TRIAGE_MODEL=` line in .env loads as "".
    return os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL


class OpenAICompatibleClient:
    """Any OpenAI-compatible endpoint that supports json_schema response format."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None) -> None:
        from openai import OpenAI  # optional dependency

        self.model = model or configured_model()
        self.name = f"llm:{self.model}"
        self._client = OpenAI(
            # An empty TRIAGE_BASE_URL means unset: None lets the SDK use its default endpoint.
            base_url=base_url or os.environ.get("TRIAGE_BASE_URL") or None,
            api_key=api_key or os.environ.get("TRIAGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """One structured-output call at temperature 0; returns the raw JSON text."""
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ReportExtraction", "strict": True, "schema": schema},
            },
            # Opt-in, OpenRouter only: a reasoning model's hidden thinking can use up
            # MAX_OUTPUT_TOKENS and return empty content. OpenAI's API rejects this field.
            extra_body={"reasoning": {"enabled": False}} if os.environ.get("TRIAGE_DISABLE_REASONING") else None,
        )
        return resp.choices[0].message.content or ""


# Keywords strict mode rejects, plus "description": Pydantic fills it from developer
# docstrings, which mention scores, and the model must never see scoring language.
_DROP_FROM_SCHEMA = ("default", "minLength", "maxLength", "description")


def response_schema() -> dict:
    """ReportExtraction as a strict structured-output schema.

    Strict mode makes the provider constrain decoding to the schema instead of
    treating it as a hint. It requires every property to be listed as required
    (optional ones are nullable instead) and no extra properties. Pydantic
    still validates the result, including the rules strict mode cannot express.
    """
    def tighten(node):
        if isinstance(node, dict):
            for key in _DROP_FROM_SCHEMA:
                node.pop(key, None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for value in node:
                tighten(value)
        return node

    return tighten(ReportExtraction.model_json_schema())


def _parse(raw_json: str, fault_names: tuple[str, ...]) -> ReportExtraction:
    """Validate the model's JSON; a fault name not on the list raises ValueError."""
    extraction = ReportExtraction.model_validate_json(raw_json)
    unknown = [m for f in extraction.faults for m in f.taxonomy_match if m not in fault_names]
    if unknown:
        raise ValueError(f"taxonomy_match not on the fault list: {unknown}")
    return extraction


def _redact(text: str) -> str:
    for key in (os.environ.get("TRIAGE_API_KEY"), os.environ.get("OPENAI_API_KEY")):
        if key:
            text = text.replace(key, "<redacted>")
    return text


def extract(report: Report, client: LLMClient,
            fault_names: tuple[str, ...] = FAULT_NAMES) -> ExtractionResult:
    """One model call, validated at the boundary. Retry once, then flag for a human.

    Returns:
        status OK with the extraction, NO_FAULT_NAMED for an empty faults list, or
        FLAGGED_FOR_HUMAN with both attempts' errors. Never raises for a bad answer or a
        provider error, so one report can't stop the run.
    """
    schema = response_schema()
    user = build_user_prompt(report.raw_text, fault_names)
    errors: list[str] = []

    for attempt in (1, 2):
        try:
            raw = client.complete_json(SYSTEM_PROMPT, user, schema)
            extraction = _parse(raw, fault_names)
        # Provider and network errors too: one failed call flags its report, never ends the run.
        except (ValidationError, ValueError, json.JSONDecodeError, _ProviderError, OSError) as e:
            # Redact before truncating, or the cut could leave part of a key behind.
            errors.append(f"attempt {attempt}: {e.__class__.__name__}: {_redact(str(e))[:300]}")
            continue
        status = ExtractionStatus.OK if extraction.faults else ExtractionStatus.NO_FAULT_NAMED
        return ExtractionResult(request_id=report.request_id, status=status,
                                extraction=extraction, attempts=attempt, errors=errors,
                                extractor=client.name)

    return ExtractionResult(request_id=report.request_id,
                            status=ExtractionStatus.FLAGGED_FOR_HUMAN,
                            extraction=None, attempts=2, errors=errors,
                            extractor=client.name)


# ---------------------------------------------------------------------------
# Offline test double (demo.py --offline, CI): a regex reader, not the real extractor.
# It returns the same ReportExtraction contract so later stages run unchanged. It is
# deliberately conservative: unknown text gives no match.
# ---------------------------------------------------------------------------

_FAULT_PATTERNS: dict[str, str] = {
    "blocked or broken toilet": r"\b(toilet|dunny|loo)\b[^.\n]*?\b(blocked|broken|not flushing|overflow\w*)\b|\b(blocked|broken)\s+(toilet|dunny)\b",
    "blocked drain": r"\bdrain\b[^.\n]*\bblocked\b|\bblocked drain\b",
    "sewage leak": r"\bsewage\b|\bsewer\b",
    "leaking or burst water main or pipe": r"\b(burst|busted)\s+pipe\b|\bpipe\b[^.\n]*\bburst\b|\bwater (is )?leaking\b",
    "exposed electrical wires": r"\b(sparking|sparks|exposed wires?|electric shock|zapped)\b",
    "gas leak": r"\b(smell (of )?gas|gas leak)\b",
    "roof leak": r"\broof\b[^.\n]*\bleak\w*\b|\bleak\w*\b[^.\n]*\broof\b",
    "flooding or flood damage": r"\bflood\w*\b",
    "storm, fire or impact damage": r"\b(storm|cyclone|fire)\b[^.\n]*\bdamage\w*\b",
    "no gas, electricity or water supply": r"\bno (power|electricity|water|gas)\b|\bpower('s| is)? (off|out)\b",
    "hot water system not working": r"\bhot water\b[^.\n]*\b(not working|broken|gone|cold|no)\b|\bno hot water\b",
    # Lookahead: "stove element" is its own standard entry (nt.gov.au), not the stove itself.
    "stove or oven not working": r"\b(stove|cooktop|oven)\b(?![^.\n]*\belements?\b)[^.\n]*\b(not working|broken|dead)\b",
    "dripping tap or tap tight to turn": r"\btaps?\b[^.\n]*\b(dripping|drips|tight|stiff)\b|\bdripping taps?\b",
    "stove element not working": r"\belements?\b[^.\n]*\b(not working|broken|dead)\b",
    "fan not working properly": r"\bfan\b[^.\n]*\b(not working|broken|dead)\b",
    "power point not working": r"\bpower ?points?\b[^.\n]*\b(not working|broken|dead)\b",
}
_ALTERNATIVE = r"\busing the other (one|toilet|shower|bathroom)\b|\bother (toilet|shower|bathroom) (works|is working|still works)\b"
_COPING = r"\b(bucket|neighbou?r'?s|the servo|servo|the shop|the clinic|family'?s place)\b"
# A fault that is described but not on the list must reach the review band
# (fault_description set, taxonomy_match empty), not "no fault named".
_GENERIC_FAULT = r"[^.,\n]*\b(broken|busted|not working|wobbl\w*|leak\w*|cracked|smashed|stuck|falling|blocked|dead|faulty|damaged|won'?t (open|close|work|turn))\b[^.,\n]*"
_INTERMITTENT =r"\b(comes and goes|on and off|sometimes|now and then)\b"
_ACTIVE_HAZARD = r"water (is )?coming (through|out of) the (light|power point|switch)[^.\n]*|\bsparking\b[^.\n]*|\bsmell (of )?gas\b[^.\n]*|\bexposed wires?\b[^.\n]*"
_CONDITIONAL_HAZARD = r"\bif it rains\b[^.\n]*|\bwhen it rains\b[^.\n]*"


class OfflineExtractor:
    """Regex test double behind the LLMClient interface. Never emits unclear, sign,
    mismatch, harm or worsening."""

    name = "offline"

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Read the report out of the user message; system and schema are ignored."""
        return self.read(report_text_from_prompt(user)).model_dump_json()

    @staticmethod
    def read(text: str) -> ReportExtraction:
        """One entry for the fault the patterns find, or [] when none is named."""
        facts = OfflineExtractor._read_facts(text)
        return ReportExtraction(faults=[facts] if facts.fault_description else [])

    @staticmethod
    def _read_facts(text: str) -> ExtractedFacts:
        spans: list[QuotedSpan] = []
        matches: list[str] = []
        first_fault_span: Optional[str] = None
        for name, pat in _FAULT_PATTERNS.items():
            hit = re.search(pat, text, re.I)
            if hit:
                matches.append(name)
                spans.append(QuotedSpan(field="taxonomy_match", text=hit.group(0)))
                first_fault_span = first_fault_span or hit.group(0)

        fault_description = first_fault_span
        if fault_description is None:
            generic = re.search(_GENERIC_FAULT, text, re.I)
            if generic:
                fault_description = generic.group(0).strip()
        if fault_description:
            spans.append(QuotedSpan(field="fault_description", text=fault_description))

        alt = re.search(_ALTERNATIVE, text, re.I)
        if alt:
            spans.append(QuotedSpan(field="alternative_mentioned", text=alt.group(0)))
        cope = re.search(_COPING, text, re.I)
        if cope:
            spans.append(QuotedSpan(field="coping_mentioned", text=cope.group(0)))
        inter = re.search(_INTERMITTENT, text, re.I)
        if inter:
            spans.append(QuotedSpan(field="impact_status", text=inter.group(0)))

        hazard, mtype = None, None
        act = re.search(_ACTIVE_HAZARD, text, re.I)
        cond = re.search(_CONDITIONAL_HAZARD, text, re.I)
        if act:
            hazard, mtype = act.group(0).strip(), "active"
        elif cond:
            hazard, mtype = cond.group(0).strip(), "conditional"
        if hazard:
            spans.append(QuotedSpan(field="hazard", text=hazard))

        return ExtractedFacts(
            fault_description=fault_description,
            taxonomy_match=matches,
            alternative_mentioned=bool(alt),
            coping_mentioned=bool(cope),
            impact_status="intermittent" if inter else "ongoing",
            hazard_status="described" if hazard else "none",
            mechanism_type=mtype,
            # Stand-in only: offline never emits "unclear", "sign", a mismatch, harm or worsening.
            harm_claimed=False,
            fault_or_sign="fault",
            claim_mismatch=None,
            worsening_mentioned=False,
            quoted_spans=spans,
        )
