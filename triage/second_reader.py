"""Second reader: Jev (TypeSafe AI) reads the same report independently, for coordinator flags.

Pipeline: extraction -> verification -> evaluation -> [second reader] -> ranking -> explain.
Input: the report text (plus the fault's own words for a compound report) and the extraction
model's facts for that fault. Output: a SecondReading: agrees, disagreement or low-confidence
flags, or why it did not run. Flags only: it never changes safety level, tally, tier or rank.
API: https://docs.typesafe.ai/api (POST /v1/systemone, typed "choice" questions).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from triage.models import EnrichedJob, ExtractedFacts, SecondReading

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
TIMEOUT_S = 30
# Uncalibrated: an assumed cut-off, not measured against labelled reports. Below it the
# coordinator is asked to check, whatever Jev chose.
LOW_CONFIDENCE = 0.7

_IF_FAULT_NAMED = (
    " If the input names one fault to look at, answer only about that fault, using the whole report as context."
)

# Invariant 1: Jev sees the report and these questions only, never tiers, points, scoring
# rules or other jobs. Invariant 3: no words that judge how bad something is or how the tenant
# writes (same banned-words test as the extraction prompt). All four are "choice" questions:
# only choice answers carry a confidence in the API.
QUESTIONS: dict[str, dict[str, object]] = {
    "hazard": {
        "type": "choice",
        "instructions": "Does the report state a way the fault could physically hurt someone?" + _IF_FAULT_NAMED,
        "criteria": {
            "described": "The report states a one-step physical pathway to harm, such as water touching a wire, "
            "or a gas leak reported as present.",
            "unclear": "The report names a possible source of harm (electrical, gas, structure, fire) with something "
            "abnormal happening (buzzing, sparking, burning smell, sagging, cracking), but states no pathway.",
            "none": "Anything else, including an appliance that has simply stopped working.",
        },
    },
    "mechanism": {
        "type": "choice",
        "instructions": "If the report states a way the fault could physically hurt someone, is that happening "
        "now, or could it happen only under a condition the report states?" + _IF_FAULT_NAMED,
        "criteria": {
            "happening now": "The pathway to harm is happening now.",
            "could happen": "The pathway could happen only under a condition the report states.",
            "no hazard": "The report states no pathway to harm.",
        },
    },
    "alternative": {
        "type": "choice",
        "instructions": "Does the report name another working instance of the same thing in the home, such as "
        "a second toilet or another shower? Other ways of getting by (a bucket, a neighbour) do not count."
        + _IF_FAULT_NAMED,
        "criteria": {
            "yes": "The report names another working one of the same thing in the home.",
            "no": "The report names no other working one of the same thing in the home.",
        },
    },
    "fault_or_sign": {
        "type": "choice",
        "instructions": "Does the report describe the fault itself, or only something sensed (a smell, sound or "
        "stain)? A smell of gas or of burning counts as the fault itself." + _IF_FAULT_NAMED,
        "criteria": {
            "fault": "The report describes the fault or its effect.",
            "sensed cue only": "The report gives only a smell, sound or stain.",
        },
    },
}


# extra="ignore", unlike the pipeline's own models: this is a third-party response, and a field
# the API adds (id, usage, created...) must not make the second reader "unavailable". Only the
# fields compare() uses are validated; a missing one still raises.
class JevAnswer(BaseModel):
    """One choice answer: the fields compare() reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["choice"]
    choice: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float] | None = None  # optional: never read


class JevResponse(BaseModel):
    """The /v1/systemone response body: the answers only."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    answers: dict[str, JevAnswer]


class JevClient(Protocol):
    """Anything that answers the questions for one input: the real API or a test fake."""

    def ask(self, state: str | dict[str, str], questions: dict[str, dict[str, object]]) -> dict[str, object]:
        """The raw JSON response body."""
        ...


class TypeSafeClient:
    """The real Jev API over HTTPS."""

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def ask(self, state: str | dict[str, str], questions: dict[str, dict[str, object]]) -> dict[str, object]:
        """POST the questions; raises on an HTTP error, a timeout or a non-JSON body."""
        body = json.dumps({"model": JEV_MODEL, "state": state, "questions": questions}).encode("utf-8")
        request = urllib.request.Request(
            JEV_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))

    def redact(self, text: str) -> str:
        return text.replace(self._key, "<redacted>")


def client_from_env() -> TypeSafeClient | None:
    """The real client if TYPESAFE_API_KEY is set (an empty value counts as no key)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    return TypeSafeClient(key) if key else None


def not_run(status: Literal["not run (no key)", "not run (offline)", "not run (recorded mode)"]) -> SecondReading:
    return SecondReading(status=status, flags=(), detail=None)


def state_for(raw_text: str, fault_description: str | None, compound: bool) -> str | dict[str, str]:
    """Jev's input: the report alone, or for a compound report the report plus this fault's words."""
    if compound and fault_description:
        return {"report": raw_text, "fault": fault_description}
    return raw_text


def _expected(facts: ExtractedFacts) -> dict[str, str | None]:
    """The extraction model's facts, as the option Jev would pick if it read the same."""
    mechanism = {"active": "happening now", "conditional": "could happen"}
    return {
        "hazard": facts.hazard_status,
        # An unclear hazard records no mechanism, so there is nothing to compare it with.
        "mechanism": (
            mechanism[facts.mechanism_type] if facts.mechanism_type
            else "no hazard" if facts.hazard_status == "none" else None
        ),
        "alternative": "yes" if facts.alternative_mentioned else "no",
        "fault_or_sign": "fault" if facts.fault_or_sign == "fault" else "sensed cue only",
    }


def compare(facts: ExtractedFacts, answers: dict[str, JevAnswer]) -> tuple[str, ...]:
    """Coordinator flags, at most one per field: a disagreement (which states the confidence),
    else an answer below LOW_CONFIDENCE."""
    flags: list[str] = []
    for field, expected in _expected(facts).items():
        # Jev saw no stated pathway, so its mechanism answer has nothing to describe.
        if expected is None or (field == "mechanism" and answers["hazard"].choice in ("none", "unclear")):
            continue
        answer = answers[field]
        if answer.choice != expected:
            flags.append(
                f"disagrees on {field} — check (extraction: {expected}; second reader: {answer.choice}, "
                f"confidence {answer.confidence:.2f})"
            )
        elif answer.confidence < LOW_CONFIDENCE:
            flags.append(f"low confidence on {field} — check (second reader: {answer.choice}, "
                         f"confidence {answer.confidence:.2f})")
    return tuple(flags)


def flagged_fields(reading: SecondReading) -> frozenset[str]:
    """The question ids compare() flagged ("disagrees on hazard — ..." -> "hazard")."""
    return frozenset(flag.split(" — ")[0].rsplit(" on ", 1)[1] for flag in reading.flags)


def _validate(raw: dict[str, object]) -> JevResponse:
    response = JevResponse.model_validate(raw)
    for field, question in QUESTIONS.items():
        answer = response.answers.get(field)
        if answer is None:
            raise ValueError(f"no answer for {field}")
        if answer.choice not in question["criteria"]:  # type: ignore[operator]
            raise ValueError(f"{field}: {answer.choice!r} is not one of the options")
    return response


def read(client: JevClient, state: str | dict[str, str], facts: ExtractedFacts) -> SecondReading:
    """Ask Jev about one fault and compare. Any failure is "unavailable", never an exception."""
    try:
        response = _validate(client.ask(state, QUESTIONS))
    except Exception as e:  # network, HTTP 4xx/5xx, timeout, malformed body: none may block the job
        detail = f"{e.__class__.__name__}: {e}"
        redact = getattr(client, "redact", None)
        return SecondReading(status="unavailable", flags=(), detail=(redact(detail) if redact else detail)[:200])
    return SecondReading(status="ran", flags=compare(facts, response.answers), detail=None)


def attach(job: EnrichedJob, reading: SecondReading) -> EnrichedJob:
    """The job with its second reading added for the coordinator; nothing else changes."""
    return EnrichedJob.model_validate({**job.model_dump(), "second_reader": reading})
