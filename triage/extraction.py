"""Stage 2: Extraction. The only model call in the system.

The model answers reading questions only. It receives the report text and a
flat list of fault NAMES. It never sees tiers, points, scoring rules, other
jobs or rank positions, so it structurally cannot influence a score.
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

SYSTEM_PROMPT = """You read housing maintenance reports from tenants in remote \
Northern Territory communities. You answer reading questions about the text. \
You do not judge how serious anything is.

Rules:
1. Only report what the text says. Every fact you claim must include a quoted \
span copied EXACTLY, character for character, from the report text. Each \
span's field names the fact it supports. That includes taxonomy_match (cite the \
words that describe the fault) and impact_status when it is "intermittent".
2. taxonomy_match: pick names from the fault list that the report describes. \
Returning an empty list is correct and expected when nothing on the list fits. \
List several only if the text genuinely fits several.
3. alternative_mentioned: true only if the tenant names another working \
instance of the same function in their own house (e.g. "using the other toilet").
4. coping_mentioned: true if the tenant describes a workaround that is not a \
working instance in the house (bucket, neighbour's, the servo, the shop). \
Coping is NOT an alternative.
5. impact_status: "intermittent" only if the text says the fault comes and \
goes; otherwise "ongoing".
6. hazard_mechanism: a described physical pathway to harm, in the tenant's \
words, or null. It must say HOW a person could be hurt (e.g. water reaching \
electrics, sparking, a gas smell, exposed wires, a ceiling about to fall). A \
fault or damage on its own (a leak, something not heating, mould, a stain) is \
not a hazard_mechanism unless the text also says how it could hurt someone. \
mechanism_type: "active" if the harm pathway is happening now \
(e.g. water coming through a light fitting), "conditional" if it could happen \
under some condition (e.g. "if it rains"). null if no hazard_mechanism.
7. If the report names no fault at all, return fault_description null and an \
empty taxonomy_match.
"""


def build_user_prompt(raw_text: str, fault_names: tuple[str, ...] = FAULT_NAMES) -> str:
    faults = "\n".join(f"- {name}" for name in fault_names)
    return f"FAULT LIST:\n{faults}\n\nREPORT TEXT:\n<<<\n{raw_text}\n>>>"


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Return a JSON string conforming to schema (structured-output mode)."""
        ...


# One report's facts and quotes fit comfortably in this. Without a cap the
# provider reserves the model's maximum (65k tokens) against the account
# balance for every call.
MAX_OUTPUT_TOKENS = 1024


class OpenAICompatibleClient:
    """Any OpenAI-compatible endpoint that supports json_schema response format."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None) -> None:
        from openai import OpenAI  # optional dependency

        self.model = model or os.environ.get("TRIAGE_MODEL", "gpt-4o-mini")
        self.name = f"llm:{self.model}"
        self._client = OpenAI(
            base_url=base_url or os.environ.get("TRIAGE_BASE_URL"),
            api_key=api_key or os.environ.get("TRIAGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )

    def complete_json(self, system: str, user: str, schema: dict) -> str:
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
        )
        return resp.choices[0].message.content or ""


# Keywords strict mode does not accept, plus "description": Pydantic fills it
# from docstrings written for developers
# (which mention scores), and the model must never see scoring language.
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
    extraction = ReportExtraction.model_validate_json(raw_json)
    unknown = [m for f in extraction.faults for m in f.taxonomy_match if m not in fault_names]
    if unknown:
        raise ValueError(f"taxonomy_match not on the fault list: {unknown}")
    return extraction


def extract(report: Report, client: LLMClient,
            fault_names: tuple[str, ...] = FAULT_NAMES) -> ExtractionResult:
    """One model call, validated at the boundary. Retry once, then flag for a human."""
    schema = response_schema()
    user = build_user_prompt(report.raw_text, fault_names)
    errors: list[str] = []

    for attempt in (1, 2):
        try:
            raw = client.complete_json(SYSTEM_PROMPT, user, schema)
            extraction = _parse(raw, fault_names)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            errors.append(f"attempt {attempt}: {e.__class__.__name__}: {str(e)[:300]}")
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
# Offline fallback: deterministic keyword reader for demos without an API key.
# It produces the same ExtractedFacts contract, so Stages 3 to 6 cannot tell
# the difference. It is deliberately conservative: unknown text -> no match.
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
# A fault is described but not one on the list: this must reach the REVIEW
# BAND (fault_description set, taxonomy_match empty), not "no fault named".
_GENERIC_FAULT = r"[^.,\n]*\b(broken|busted|not working|wobbl\w*|leak\w*|cracked|smashed|stuck|falling|blocked|dead|faulty|damaged|won'?t (open|close|work|turn))\b[^.,\n]*"
_INTERMITTENT =r"\b(comes and goes|on and off|sometimes|now and then)\b"
_ACTIVE_HAZARD = r"water (is )?coming (through|out of) the (light|power point|switch)[^.\n]*|\bsparking\b[^.\n]*|\bsmell (of )?gas\b[^.\n]*|\bexposed wires?\b[^.\n]*"
_CONDITIONAL_HAZARD = r"\bif it rains\b[^.\n]*|\bwhen it rains\b[^.\n]*"


class OfflineExtractor:
    name = "offline"

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        m = re.search(r"REPORT TEXT:\n<<<\n(.*)\n>>>", user, re.S)
        text = m.group(1) if m else ""
        return self.read(text).model_dump_json()

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


def default_client() -> LLMClient:
    """LLM if a key is configured, otherwise the offline reader."""
    if os.environ.get("TRIAGE_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAICompatibleClient()
        except ImportError:
            pass
    return OfflineExtractor()
