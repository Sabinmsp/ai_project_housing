from datetime import datetime, timedelta, timezone
from typing import Any

from triage.adapter import to_rank_input
from triage.explain import (
    ReasoningTrace,
    build_traces,
    render_coordinator,
    render_review_entry,
    render_tenant_sms,
)
from triage.models import EnrichedJob, VerifiedSpan
from triage.ranking import NO_TIER_FLAG, REVIEW_BAND_REASON, rank

MON = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)


def panel_d_job(**overrides: Any) -> EnrichedJob:
    fields: dict[str, Any] = {
        "request_id": "R-2291",
        "community": "Wadeye",
        "original_report_timestamp": datetime.fromtimestamp(1726041600, tz=timezone.utc),
        "fault_description": "roof leaking",
        "taxonomy_match": ["serious roof leak"],
        "tier": "dangerous",
        "base_points": 3,
        "no_redundancy": 1,
        "urgency_tally": 4,
        "safety_flag": True,
        "safety_level": "active",
        "spans": [
            VerifiedSpan(field="hazard_mechanism", text="water coming through the light fitting", verified=True),
            VerifiedSpan(field="coping_mentioned", text="made up", verified=False),
        ],
        "distance_cost_km": 412,
    }
    return EnrichedJob(**{**fields, **overrides})


def standard_job(request_id: str, **overrides: Any) -> EnrichedJob:
    fields: dict[str, Any] = {
        "request_id": request_id,
        "community": "Darwin",
        "original_report_timestamp": MON,
        "tier": "standard",
        "base_points": 2,
        "no_redundancy": 1,
        "urgency_tally": 3,
    }
    return EnrichedJob(**{**fields, **overrides})


def traces(*jobs: EnrichedJob) -> list[ReasoningTrace]:
    result = rank([to_rank_input(j) for j in jobs])
    return build_traces(result, {j.request_id: j for j in jobs})


def test_panel_d_trace() -> None:
    (tr,) = traces(panel_d_job())
    assert tr.original_timestamp == datetime.fromtimestamp(1726041600, tz=timezone.utc)
    assert tr.taxonomy_match == ("serious roof leak",)
    assert tr.defaults_applied == ("no-redundancy default applied: +1 (no alternative named)",)
    assert tr.safety_level == 2 and "active hazard" in tr.safety_reason
    assert [s.text for s in tr.evidence_spans] == ["water coming through the light fitting"]
    view = render_coordinator(tr)
    assert "412 km" in view and "not in sort_key" in view
    assert "serious roof leak" in view and "2024-09-11" in view
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


def test_safety_sms_says_safety_job_without_comparison() -> None:
    (tr,) = traces(panel_d_job())
    sms = render_tenant_sms(tr)
    assert "marked as a safety job" in sms
    assert "sits above" not in sms and "non-safety" not in sms


def test_conditional_safety_reason_from_level() -> None:
    (tr,) = traces(standard_job("R-AAAA", safety_level="conditional"))
    assert tr.safety_level == 1 and "conditional hazard" in tr.safety_reason


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
    job = panel_d_job(tier=None, base_points=None, no_redundancy=0, urgency_tally=None)
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
    job = standard_job("R-AAAA", tier=None, base_points=None, no_redundancy=0, urgency_tally=None)
    result = rank([to_rank_input(job)])
    assert result.review_band == ("R-AAAA",)
    view = render_review_entry(job)
    assert REVIEW_BAND_REASON in view and "R-AAAA" in view
