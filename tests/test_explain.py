from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from triage.adapter import to_rank_input
from triage.evaluation import NO_ALTERNATIVE, NO_HAZARD
from triage.explain import (
    ReasoningTrace,
    build_traces,
    render_coordinator,
    render_review_entry,
    render_tenant_sms,
)
from triage.models import EnrichedJob, VerifiedSpan
from triage.ranking import NO_TIER_FLAG, REVIEW_BAND_REASON, rank
from triage.tiers import TIER_TABLE

MON = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
ACTIVE_REASON = "Active hazard described: 'water coming through the light fitting' — full override"


def panel_d_job(**overrides: Any) -> EnrichedJob:
    fields: dict[str, Any] = {
        "request_id": "R-2291",
        "parent_report_id": "R-2291",
        "community": "Wadeye",
        "original_report_timestamp": datetime.fromtimestamp(1726041600, tz=timezone.utc),
        "fault_description": "roof leaking",
        "taxonomy_match": ["roof leak"],
        "tier": "dangerous",
        "tier_entry": "roof leak",
        "base_points": 3,
        "severity_bump": 1,
        "urgency_tally": 4,
        "tally_reasons": (NO_ALTERNATIVE,),
        "safety_flag": True,
        "safety_level": "active",
        "safety_reason": ACTIVE_REASON,
        "flags": (),
        "spans": [
            VerifiedSpan(field="hazard", text="water coming through the light fitting", verified=True),
            VerifiedSpan(field="coping_mentioned", text="made up", verified=False),
        ],
        "distance_cost_km": 412,
    }
    return EnrichedJob(**{**fields, **overrides})


def standard_job(request_id: str, **overrides: Any) -> EnrichedJob:
    fields: dict[str, Any] = {
        "request_id": request_id,
        "parent_report_id": request_id,
        "community": "Darwin",
        "original_report_timestamp": MON,
        "tier": "standard",
        "tier_entry": "power point not working",
        "base_points": 2,
        "severity_bump": 1,
        "urgency_tally": 3,
        "tally_reasons": (NO_ALTERNATIVE,),
        "safety_level": "none",
        "safety_reason": NO_HAZARD,
        "flags": (),
    }
    return EnrichedJob(**{**fields, **overrides})


def traces(*jobs: EnrichedJob) -> list[ReasoningTrace]:
    result = rank([to_rank_input(j) for j in jobs])
    return build_traces(result, {j.request_id: j for j in jobs})


def test_panel_d_trace() -> None:
    (tr,) = traces(panel_d_job())
    assert tr.original_timestamp == datetime.fromtimestamp(1726041600, tz=timezone.utc)
    assert tr.taxonomy_match == ("roof leak",)
    assert tr.tally_reasons == (NO_ALTERNATIVE,)
    assert tr.safety_level == 2 and tr.safety_reason == ACTIVE_REASON
    assert [s.text for s in tr.evidence_spans] == ["water coming through the light fitting"]
    view = render_coordinator(tr)
    assert "412 km" in view and "not in sort_key" in view
    assert "roof leak" in view and "2024-09-11" in view
    assert "made up" not in view  # unverified spans never reach an audience


def test_trace_text_is_tenant_words_never_model_wording() -> None:
    job = panel_d_job(
        fault_description="roof leaking. model stitched this sentence together",
        spans=[
            VerifiedSpan(field="fault_description", text="invented by model", verified=False),
            VerifiedSpan(field="fault_description", text="roof leaking", verified=True),
        ],
    )
    (tr,) = traces(job)
    assert tr.fault_description == "roof leaking"
    sms = render_tenant_sms(tr)
    assert "stitched" not in sms and "invented" not in sms
    (no_span,) = traces(panel_d_job(spans=[]))
    assert no_span.fault_description is None
    assert '"your repair"' in render_tenant_sms(no_span)


def test_tenant_sms_uses_only_trace_fields_and_hides_distance() -> None:
    (tr,) = traces(panel_d_job())
    sms = render_tenant_sms(tr)
    assert "R-2291" in sms
    assert "km" not in sms and "412" not in sms
    # Our no-redundancy default, not something the tenant said.
    assert "no other working one" not in sms


def test_tenant_sms_invariant_to_position_queue_and_decided_by() -> None:
    jobs = [
        standard_job("R-3F9A1C2B"),
        standard_job("R-7D04E8A1", original_report_timestamp=MON + timedelta(days=1)),
        standard_job("R-0B62F5C9", original_report_timestamp=MON + timedelta(days=2)),
    ]
    middle = next(t for t in traces(*jobs) if t.position == 2)
    # model_validate rather than model_copy, so the copy is a validated trace.
    moved = ReasoningTrace.model_validate(
        {**middle.model_dump(), "position": 41, "queue_length": 97, "decided_by": "below #40: higher urgency score above"}
    )
    # §5.2: nothing comparative, so the SMS can't depend on where the job sits.
    assert render_tenant_sms(moved) == render_tenant_sms(middle)


def test_other_job_id_reaches_coordinator_never_tenant() -> None:
    (tr,) = traces(standard_job("R-3F9A1C2B", shared_route_opportunities=["R-7D04E8A1"]))
    assert "R-7D04E8A1" not in render_tenant_sms(tr)
    assert "R-7D04E8A1" in render_coordinator(tr)


def test_safety_sms_says_safety_job_without_comparison() -> None:
    (tr,) = traces(panel_d_job())
    sms = render_tenant_sms(tr)
    assert "marked as a safety job" in sms
    assert "sits above" not in sms and "non-safety" not in sms


def test_safety_reason_comes_from_the_job() -> None:
    reason = "Conditional hazard described: 'if it rains' — elevated, does not bypass active hazards"
    (tr,) = traces(standard_job("R-AAAA", safety_level="conditional", safety_reason=reason))
    assert tr.safety_level == 1 and tr.safety_reason == reason


def test_coordinator_view_shows_tally_and_safety_reasons() -> None:
    (tr,) = traces(panel_d_job())
    view = render_coordinator(tr)
    assert NO_ALTERNATIVE in view and ACTIVE_REASON in view
    assert "+1" in view and "(3 + 1)" in view


def test_distance_none_renders_as_unknown() -> None:
    (tr,) = traces(panel_d_job(distance_cost_km=None))
    view = render_coordinator(tr)
    assert "distance unknown" in view and "0 km" not in view


def test_trace_carries_enriched_job_flags() -> None:
    flag = "could be blocked drain or sewage leak; scored as sewage leak"
    (tr,) = traces(panel_d_job(flags=[flag]))
    assert tr.flags == (flag,)
    assert flag in render_coordinator(tr)


def test_untiered_safety_job_carries_no_tier_flag_and_renders() -> None:
    job = panel_d_job(tier=None, tier_entry=None, base_points=None, severity_bump=None, urgency_tally=None, tally_reasons=())
    (tr,) = traces(job)
    assert NO_TIER_FLAG in tr.flags
    assert "untiered" in render_coordinator(tr)
    assert "confirming its priority" in render_tenant_sms(tr)


def test_coordinator_view_shows_position_and_decided_by() -> None:
    all_traces = traces(standard_job("R-AAAA"), standard_job("R-BBBB", original_report_timestamp=MON + timedelta(days=1)))
    view = render_coordinator(all_traces[1])
    assert "2 of 2" in view
    assert all_traces[1].decided_by in view


def test_review_band_entry_uses_review_band_reason() -> None:
    job = standard_job("R-AAAA", tier=None, tier_entry=None, base_points=None, severity_bump=None, urgency_tally=None, tally_reasons=())
    result = rank([to_rank_input(job)])
    assert result.review_band == ("R-AAAA",)
    view = render_review_entry(job)
    assert REVIEW_BAND_REASON in view and "R-AAAA" in view


# --- tier source attribution (Step 4.6c) ---------------------------------------

ALL_SOURCES = sorted({src for e in TIER_TABLE.values() for src in e.sources})


def entry_job(name: str) -> EnrichedJob:
    tier = TIER_TABLE[name].tier
    base = 3 if tier == "dangerous" else 2
    return standard_job(
        "R-3F9A1C2B", taxonomy_match=[name], tier=tier, tier_entry=name, base_points=base, urgency_tally=base + 1
    )


def shown(source: str, view: str) -> bool:
    # RTA sources render by section number; nt.gov.au renders as itself.
    return (source.removeprefix("RTA ") if source.startswith("RTA ") else source) in view


@pytest.mark.parametrize("name", list(TIER_TABLE))
def test_coordinator_shows_exactly_the_entry_sources(name: str) -> None:
    (tr,) = traces(entry_job(name))
    view = render_coordinator(tr)
    for source in ALL_SOURCES:
        assert shown(source, view) == (source in TIER_TABLE[name].sources), source


def test_rta_only_entry_never_shows_nt_gov() -> None:
    (tr,) = traces(entry_job("hot water system not working"))
    view = render_coordinator(tr)
    assert "nt.gov.au" not in view
    assert "NT Residential Tenancies Act s63(2)(j) — emergency repair" in view


def test_nt_gov_only_entry_never_shows_rta() -> None:
    (tr,) = traces(entry_job("blocked drain"))
    view = render_coordinator(tr)
    assert "RTA" not in view and "Residential Tenancies" not in view
    assert "nt.gov.au — on the repaired-first list" in view


def test_several_sources_joined() -> None:
    (tr,) = traces(entry_job("gas leak"))
    assert "nt.gov.au — on the repaired-first list; NT Residential Tenancies Act s63(2)(d) — emergency repair" in render_coordinator(tr)


@pytest.mark.parametrize("name", list(TIER_TABLE))
def test_tenant_sms_names_no_source_or_tier_label(name: str) -> None:
    (tr,) = traces(entry_job(name))
    sms = render_tenant_sms(tr)
    for word in ("dangerous", "standard", "nt.gov", "RTA", "Act"):
        assert word not in sms, word
    expected = "urgent repair" if TIER_TABLE[name].tier == "dangerous" else "general repair"
    assert f"It's being treated as an {expected}." in sms or f"It's being treated as a {expected}." in sms


@pytest.mark.parametrize("name", list(TIER_TABLE))
def test_coordinator_never_says_nt_fault_list_name(name: str) -> None:
    (tr,) = traces(entry_job(name))
    view = render_coordinator(tr)
    assert "NT fault list name" not in view
    assert any(name in line and line.endswith("(reference list name)") for line in view.splitlines())


# --- invariant 10: tenant SMS never carries flags or reasons -------------------------

TENANT_UNSAFE_FLAGS = (
    "Report plays down a fault on the repair-first list: 'nothing too bad' vs 'sewage up the shower' — check before scheduling",
    "Safety claim — unconfirmed: 'gonna electrocute the kids' — fast human check, no override",
    "Quoted words not found in the report: 'sparks shooting out' — check the reading",
)


def test_tenant_sms_ignores_flags_and_reasons() -> None:
    (plain,) = traces(panel_d_job())
    flagged = ReasoningTrace.model_validate({
        **plain.model_dump(),
        "flags": TENANT_UNSAFE_FLAGS,
        "safety_reason": "Unclear hazard: 'sparks shooting out' — treated as conditional",
        "tally_reasons": ("Alternative named in report: +0",),
    })
    sms = render_tenant_sms(flagged)
    assert sms == render_tenant_sms(plain)
    for text in (*TENANT_UNSAFE_FLAGS, flagged.safety_reason, *flagged.tally_reasons):
        assert text not in sms
    for fragment in ("plays down", "unconfirmed", "isn't in the report", "not found", "nothing too bad"):
        assert fragment not in sms
