import inspect
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from triage.adapter import to_rank_input
from triage.evaluation import NO_ALTERNATIVE, NO_HAZARD
from triage.explain import (
    ELECTRICAL_ADVICE,
    GAS_ADVICE,
    SMS_PATH_SENTENCES,
    ReasoningTrace,
    build_traces,
    render_coordinator,
    render_review_entry,
    render_tenant_sms,
    UNKNOWN_REF_REPLY,
    tenant_sms,
    tenant_sms_report,
    tenant_why,
)
from triage.models import EnrichedJob, ExtractionResult, ExtractionStatus, VerifiedSpan
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
        "distance_cost_km": 250,
        "nearest_office": "Palmerston office",
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
    assert "~250 km to nearest NT Housing office (Palmerston office) — straight-line; actual dispatch point not known" in view and "not in sort_key" in view
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
    assert "Housing repair R-2291: we have your message." in render_tenant_sms(no_span)
    assert '"your repair"' not in render_tenant_sms(no_span)


def test_tenant_sms_uses_only_trace_fields_and_hides_distance() -> None:
    (tr,) = traces(panel_d_job())
    sms = render_tenant_sms(tr)
    assert "R-2291" in sms
    assert "km" not in sms and "250" not in sms and "office" not in sms
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
    (tr,) = traces(panel_d_job(distance_cost_km=None, nearest_office=None))
    view = render_coordinator(tr)
    assert "unknown (community not in location table)" in view and " km" not in view


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
    assert "marked as a safety job and is being handled as a priority" in render_tenant_sms(tr)


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
    expected = "urgent repair under NT rules" if TIER_TABLE[name].tier == "dangerous" else "routine repair"
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


# --- tenant SMS per path (master §5.2) ------------------------------------------------

UNTIERED: dict[str, Any] = {"tier": None, "tier_entry": None, "base_points": None, "severity_bump": None,
                            "urgency_tally": None, "tally_reasons": ()}


def fault_span(text: str) -> list[VerifiedSpan]:
    return [VerifiedSpan(field="fault_description", text=text, verified=True)]


def path_job(path: str, request_id: str = "R-3F9A1C2B", fault: str = "kitchen tap dripping") -> EnrichedJob:
    """A job that takes the given SMS path, with its fault words as a verified span."""
    by_path: dict[str, dict[str, Any]] = {
        "safety_active": {"safety_level": "active", "safety_flag": True, "safety_reason": ACTIVE_REASON},
        "safety_conditional": {"safety_level": "conditional", "safety_reason": "Conditional hazard: 'if it rains'"},
        "urgent": {"tier": "dangerous", "tier_entry": "roof leak", "base_points": 3, "urgency_tally": 4},
        "routine": {},
        "review": UNTIERED,
    }
    return standard_job(request_id, spans=fault_span(fault), **by_path[path])


def not_extracted(status: ExtractionStatus) -> ExtractionResult:
    return ExtractionResult(request_id="R-3F9A1C2B", status=status, attempts=2, extractor="offline")


def every_path_sms(fault: str = "kitchen tap dripping") -> dict[str, str]:
    sms = {p: tenant_sms(path_job(p, fault=fault)) for p in ("safety_active", "safety_conditional", "urgent", "routine", "review")}
    sms["flagged"] = tenant_sms(not_extracted(ExtractionStatus.FLAGGED_FOR_HUMAN))
    sms["out_of_scope"] = tenant_sms(not_extracted(ExtractionStatus.NO_FAULT_NAMED))
    return sms


REPLY = "Reply HELP with R-3F9A1C2B if anything changes or gets worse."
RANKED_PATHS = ("safety_active", "safety_conditional", "urgent", "routine")


@pytest.mark.parametrize("path", ["safety_active", "safety_conditional", "urgent", "routine", "review"])
def test_each_job_path_shows_its_fault_words_path_sentence_and_reply(path: str) -> None:
    sms = tenant_sms(path_job(path, fault="kitchen tap dripping"))
    assert sms == (f'Housing repair R-3F9A1C2B: we have your report about "kitchen tap dripping". '
                   f"{SMS_PATH_SENTENCES[path]} {REPLY}")  # type: ignore[index]
    assert ("We'll keep you updated." in sms) == (path in RANKED_PATHS)


@pytest.mark.parametrize(("status", "path"), [(ExtractionStatus.FLAGGED_FOR_HUMAN, "flagged"),
                                              (ExtractionStatus.NO_FAULT_NAMED, "out_of_scope")])
def test_unextracted_paths_have_own_sentence_and_reply(status: ExtractionStatus, path: str) -> None:
    sms = tenant_sms(not_extracted(status))
    assert sms == f"Housing repair R-3F9A1C2B: we have your message. {SMS_PATH_SENTENCES[path]} {REPLY}"  # type: ignore[index]


def test_every_path_has_a_distinct_sentence() -> None:
    assert len(set(every_path_sms().values())) == len(SMS_PATH_SENTENCES) == 7


def test_ok_extraction_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="EnrichedJob"):
        tenant_sms(not_extracted(ExtractionStatus.OK))


def test_fault_words_are_trimmed_and_capped() -> None:
    sms = tenant_sms(path_job("routine", fault="  kitchen \n tap   dripping "))
    assert '"kitchen tap dripping"' in sms
    long_sms = tenant_sms(path_job("routine", fault="water " * 30))
    quoted = long_sms.split('"')[1]
    assert len(quoted) == 60 and quoted.endswith("…")


def test_fault_text_is_the_verified_span_never_model_wording() -> None:
    job = path_job("routine", fault="tap dripping").model_dump()
    job["fault_description"] = "model paraphrase of the tap"
    job["spans"] = [{"field": "fault_description", "text": "invented", "verified": False}, *job["spans"]]
    sms = tenant_sms(EnrichedJob.model_validate(job))
    assert '"tap dripping"' in sms and "paraphrase" not in sms and "invented" not in sms


BANNED = ("unclear", "not confirmed", "unconfirmed", "exaggerat", "overstat", "position", "queue", "ahead",
          "behind", "other job", "other tenant", "jobs", "tier", "point", "score", "s63", "dangerous", "standard",
          "Residential Tenancies", "RTA", "nt.gov", "wait time", "staff", "trades available", "#", "logged",
          "for your area", "available trade", '"your repair"', "general", "completed by", "fixed by",
          "guarantee")


def test_no_template_uses_banned_wording() -> None:
    texts = [*every_path_sms().values(), *SMS_PATH_SENTENCES.values(), GAS_ADVICE, ELECTRICAL_ADVICE,
             tenant_sms_report(compound(("urgent", "R-A1", "roof leaking"), ("routine", "R-B2", "door sticks")))]
    for text in texts:
        for word in BANNED:
            assert word.lower() not in text.lower(), (word, text)
        # The only tier-like wording allowed (master §5.2, E7).
        for phrase in ("urgent", "routine"):
            if phrase in text:
                assert f"{phrase} repair" in text, text


def test_sms_is_the_same_in_every_community() -> None:
    # No timeframe is stated, so the community can't change tenant text.
    for path in (*RANKED_PATHS, "review"):
        job = path_job(path)
        assert len({tenant_sms(EnrichedJob.model_validate({**job.model_dump(), "community": c}))
                    for c in ("Darwin", "Wadeye", "Atlantis")}) == 1, path


def test_reply_line_on_every_message() -> None:
    for path, sms in every_path_sms().items():
        assert sms.endswith(REPLY), path


# --- invariant 10 / I7: other queue contents never change the SMS ---------------------

def other_job(i: int, days: int, path: str) -> EnrichedJob:
    job = path_job(path, f"R-OTHER{i:04d}", fault=f"other fault {i}")
    return EnrichedJob.model_validate({**job.model_dump(), "original_report_timestamp": MON + timedelta(days=days)})


others_strategy = st.lists(
    st.builds(other_job, st.integers(0, 9999), st.integers(-30, 30),
        st.sampled_from(["safety_active", "safety_conditional", "urgent", "routine", "review"])),
    max_size=8, unique_by=lambda j: j.request_id,
)


@given(path=st.sampled_from(["safety_active", "safety_conditional", "urgent", "routine"]),
       queue_a=others_strategy, queue_b=others_strategy)
def test_same_own_facts_different_queue_give_identical_sms(path: str, queue_a: list[EnrichedJob],
                                                           queue_b: list[EnrichedJob]) -> None:
    own = path_job(path)
    sms_a = render_tenant_sms(next(t for t in traces(own, *queue_a) if t.job_id == own.request_id))
    sms_b = render_tenant_sms(next(t for t in traces(own, *queue_b) if t.job_id == own.request_id))
    alone = render_tenant_sms(traces(own)[0])
    assert sms_a == sms_b == alone == tenant_sms(own)
    for other in (*queue_a, *queue_b):
        assert other.request_id not in sms_a and "other fault" not in sms_a


def test_logistics_fields_naming_other_jobs_never_change_the_sms() -> None:
    own = path_job("routine")
    busy = EnrichedJob.model_validate({**own.model_dump(), "shared_route_opportunities": ["R-7D04E8A1"],
                                       "starvation_line": "3 jobs waiting over 14 days", "distance_cost_km": 412, "nearest_office": "Palmerston office",
                                       "capacity_block_flag": True, "next_actionable": "Thursday"})
    assert tenant_sms(busy) == tenant_sms(own)
    assert "R-7D04E8A1" not in tenant_sms(busy)


# --- compound reports (master §5.2 C5) -------------------------------------------------

def compound(*paths_faults: tuple[str, str, str]) -> list[EnrichedJob]:
    return [EnrichedJob.model_validate({**path_job(p, rid, fault=f).model_dump(), "parent_report_id": "R-PARENT1"})
            for p, rid, f in paths_faults]


CHILDREN = (("safety_active", "R-C1A", "wires sparking in the kitchen"), ("routine", "R-C2B", "back door sticks"),
            ("review", "R-C3C", "fly screen torn"))


def test_compound_message_lists_every_child_fault_and_ref() -> None:
    jobs = compound(*CHILDREN)
    sms = tenant_sms_report(jobs)
    assert sms.startswith("Housing repairs: we've received 3 repairs from your report: ")
    for _, rid, fault in CHILDREN:
        assert f'"{fault}" (ref {rid})' in sms
    assert sms.endswith("Reply HELP with the ref of any repair that changes or gets worse: R-C1A, R-C2B, R-C3C.")
    assert "R-PARENT1" not in sms


@given(st.lists(st.sampled_from(["safety_active", "safety_conditional", "urgent", "routine", "review"]),
                min_size=2, max_size=5))
def test_compound_message_names_exactly_its_children(paths: list[str]) -> None:
    jobs = compound(*[(p, f"R-K{i}X", f"fault number {i}") for i, p in enumerate(paths)])
    sms = tenant_sms_report(jobs)
    assert f"we've received {len(jobs)} repairs" in sms
    assert [j.request_id for j in jobs] == [rid for rid in (f"R-K{i}X" for i in range(9)) if f"(ref {rid})" in sms]


def test_child_single_message_never_names_a_sibling() -> None:
    plain = compound(*CHILDREN)
    ids = [j.request_id for j in plain]
    # Duplicate flags name siblings (demo._duplicate_flags); the coordinator sees them, the tenant never does.
    jobs = [EnrichedJob.model_validate({**j.model_dump(), "flags": (f"Possible duplicate (also {', '.join(set(ids) - {j.request_id})})",)})
            for j in plain]
    for job, before in zip(jobs, plain):
        sms = tenant_sms(job)
        assert sms == tenant_sms(before)
        assert job.request_id in sms
        for sibling in jobs:
            if sibling is not job:
                assert sibling.request_id not in sms and sibling.spans[0].text not in sms


def test_single_job_report_gets_its_single_message() -> None:
    (job,) = compound(CHILDREN[1])
    assert tenant_sms_report([job]) == tenant_sms(job)


def test_compound_rejects_jobs_from_two_reports() -> None:
    with pytest.raises(ValueError, match="exactly one report"):
        tenant_sms_report([path_job("routine", "R-A1"), path_job("routine", "R-B2")])
    with pytest.raises(ValueError, match="exactly one report"):
        tenant_sms_report([])


# --- safety advice: fixed text, gas and exposed wires only ------------------------------

# Verbatim from Prabin; a second copy so a paraphrase in explain.py fails here.
GAS_TEXT = ("If you smell gas: leave the building or area and call Fire and Emergency Services on 000. "
            "If it is safe to do so, turn off the gas at the cylinder or meter. Do not enter the gas affected area.")
ELECTRICAL_TEXT = "Stay away from the exposed wiring. In an emergency call 000."


def hazard_job(entry: str, request_id: str = "R-3F9A1C2B", fault: str = "smell gas in the kitchen",
               safety: str = "active") -> EnrichedJob:
    tier = TIER_TABLE[entry].tier
    base = 3 if tier == "dangerous" else 2
    hazard = {"active": {"safety_level": "active", "safety_flag": True, "safety_reason": ACTIVE_REASON},
              "conditional": {"safety_level": "conditional", "safety_reason": "Conditional hazard: 'if it rains'"},
              "none": {}}[safety]
    return standard_job(request_id, taxonomy_match=[entry], tier=tier, tier_entry=entry, base_points=base,
                        urgency_tally=base + 1, spans=fault_span(fault), **hazard)


@pytest.mark.parametrize("safety", ["active", "conditional", "none"])
def test_gas_taxonomy_gets_gas_advice_first_at_any_safety_level(safety: str) -> None:
    # Keyed on taxonomy only: a casual gas report that came back with no hazard still gets it.
    job = hazard_job("gas leak", safety=safety)
    path = "urgent" if safety == "none" else f"safety_{safety}"
    sms = tenant_sms(job)
    assert sms == (f'{GAS_TEXT} Housing repair R-3F9A1C2B: we have your report about "smell gas in the kitchen". '
                   f"{SMS_PATH_SENTENCES[path]} {REPLY}")  # type: ignore[index]
    if safety != "none":
        (tr,) = traces(job)
        assert render_tenant_sms(tr) == sms


@pytest.mark.parametrize("safety", ["active", "conditional", "none"])
def test_wires_taxonomy_gets_electrical_advice_first_at_any_safety_level(safety: str) -> None:
    sms = tenant_sms(hazard_job("exposed electrical wires", fault="wires hanging out of the wall", safety=safety))
    assert sms.startswith(f"{ELECTRICAL_TEXT} Housing repair R-3F9A1C2B: ")
    assert sms.endswith(REPLY) and GAS_TEXT not in sms


def test_advice_is_the_first_sentence_and_never_follows_the_reply() -> None:
    sms = tenant_sms(hazard_job("gas leak", safety="none"))
    assert sms.index(GAS_TEXT) == 0 < sms.index("Housing repair") < sms.index(REPLY)


def test_both_matches_get_both_lines_in_fixed_order() -> None:
    job = EnrichedJob.model_validate({**hazard_job("gas leak").model_dump(),
                                      "taxonomy_match": ["exposed electrical wires", "gas leak"]})
    assert tenant_sms(job).startswith(f"{GAS_TEXT} {ELECTRICAL_TEXT} Housing repair ")


@pytest.mark.parametrize("entry", [e for e in TIER_TABLE if e not in ("gas leak", "exposed electrical wires")])
def test_no_advice_for_any_other_fault(entry: str) -> None:
    sms = tenant_sms(hazard_job(entry, fault="roof leaking"))
    assert GAS_TEXT not in sms and ELECTRICAL_TEXT not in sms and "000" not in sms
    assert sms.startswith("Housing repair R-3F9A1C2B: ")


def test_compound_gas_and_tap_gets_gas_first_against_the_gas_fault_only() -> None:
    gas = hazard_job("gas leak", "R-GAS1", fault="smell gas in the kitchen", safety="none")
    tap = path_job("routine", "R-TAP2", fault="kitchen tap dripping")
    jobs = [EnrichedJob.model_validate({**j.model_dump(), "parent_report_id": "R-PARENT1"}) for j in (gas, tap)]
    sms = tenant_sms_report(jobs)
    assert sms == (f'For "smell gas in the kitchen" (ref R-GAS1): {GAS_TEXT} '
                   "Housing repairs: we've received 2 repairs from your report: "
                   '"smell gas in the kitchen" (ref R-GAS1); "kitchen tap dripping" (ref R-TAP2). '
                   "We'll update you on each one separately. "
                   "Reply HELP with the ref of any repair that changes or gets worse: R-GAS1, R-TAP2.")
    assert sms.count(GAS_TEXT) == 1 and ELECTRICAL_TEXT not in sms
    assert GAS_TEXT in tenant_sms(jobs[0]) and "000" not in tenant_sms(jobs[1])


def test_compound_child_without_verified_words_has_no_quoted_placeholder() -> None:
    jobs = [EnrichedJob.model_validate({**path_job("routine", rid).model_dump(), "parent_report_id": "R-P", "spans": []})
            for rid in ("R-A1", "R-B2")]
    sms = tenant_sms_report(jobs)
    assert "a repair (ref R-A1); a repair (ref R-B2)." in sms and '"' not in sms


# --- tenant WHY answer --------------------------------------------------------------------

HELP = ("If anyone in the house is unwell, elderly or very young, or this is affecting anyone's health or safety, "
        "reply HELP R-3F9A1C2B and a coordinator will look at it again.")


def in_community(job: EnrichedJob, community: str) -> EnrichedJob:
    return EnrichedJob.model_validate({**job.model_dump(), "community": community})


WHY_IMPACT = "We know this is hard to live with. A coordinator can see how long it has been waiting."


@pytest.mark.parametrize(("path", "status"), [
    ("safety_active", "is booked as a safety repair."),
    ("safety_conditional", "is booked as a safety repair."),
    ("urgent", "is booked as an urgent repair."),
    ("routine", "is booked as a routine repair."),
    ("review", "is with a coordinator, who is deciding what kind of repair it is."),
])
@pytest.mark.parametrize("community", ["Darwin", "Wadeye", "Atlantis"])
def test_why_per_category(path: str, status: str, community: str) -> None:
    why = tenant_why(in_community(path_job(path, fault="kitchen tap dripping"), community))
    assert why == (f'Your "kitchen tap dripping" repair (R-3F9A1C2B) {status} '
                   f"Yours was received on 28 September 2026. {WHY_IMPACT} {HELP}")


def test_why_date_is_nt_local() -> None:
    # 20:00 UTC on the 28th is 05:30 on the 29th in Darwin.
    job = EnrichedJob.model_validate({**path_job("routine").model_dump(),
                                      "original_report_timestamp": datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)})
    assert "Yours was received on 29 September 2026." in tenant_why(job)


@pytest.mark.parametrize("status", [ExtractionStatus.FLAGGED_FOR_HUMAN, ExtractionStatus.NO_FAULT_NAMED])
def test_flagged_and_out_of_scope_why(status: ExtractionStatus) -> None:
    why = tenant_why(not_extracted(status))
    assert why == f"Your repair (R-3F9A1C2B) is with a coordinator, who is deciding what kind of repair it is. {WHY_IMPACT} {HELP}"


def test_ok_extraction_result_has_no_why() -> None:
    with pytest.raises(ValueError, match="EnrichedJob"):
        tenant_why(not_extracted(ExtractionStatus.OK))


WHY_BANNED = ("position", "queue", "number", "ahead", "behind", "other job", "other tenant", "tier", "point",
              "score", "s63", "dangerous", "standard", "general", "Residential Tenancies", "RTA", "nt.gov",
              "completed by", "fixed by", "guarantee", "unclear", "not confirmed", "unconfirmed", "exaggerat",
              "#", "decided")


def every_why() -> list[str]:
    whys = [tenant_why(in_community(path_job(p), c)) for p in ("safety_active", "safety_conditional", "urgent",
                                                             "routine", "review") for c in ("Darwin", "Wadeye", "Atlantis")]
    return [*whys, tenant_why(not_extracted(ExtractionStatus.FLAGGED_FOR_HUMAN)),
            tenant_why(not_extracted(ExtractionStatus.NO_FAULT_NAMED))]


def test_no_why_uses_banned_wording() -> None:
    for why in every_why():
        for word in WHY_BANNED:
            assert word.lower() not in why.lower(), (word, why)
        assert "R-3F9A1C2B" in why and why.endswith(HELP)


def test_why_reads_only_the_tenants_own_job() -> None:
    # No queue, trace or position can reach it: its only input is this job's own record.
    assert list(inspect.signature(tenant_why).parameters) == ["job"]


@given(path=st.sampled_from(["safety_active", "safety_conditional", "urgent", "routine", "review"]),
       queue_a=others_strategy, queue_b=others_strategy)
def test_same_own_facts_different_queue_give_identical_why(path: str, queue_a: list[EnrichedJob],
                                                           queue_b: list[EnrichedJob]) -> None:
    own = path_job(path)
    whys = []
    for queue in (queue_a, queue_b, []):
        jobs = {j.request_id: j for j in (own, *queue)}
        rank([to_rank_input(j) for j in jobs.values()])  # the queue exists; WHY must not depend on it
        whys.append(tenant_why(jobs[own.request_id]))
    assert whys[0] == whys[1] == whys[2]
    for other in (*queue_a, *queue_b):
        assert other.request_id not in whys[0]


def test_logistics_fields_naming_other_jobs_never_change_the_why() -> None:
    own = path_job("routine")
    busy = EnrichedJob.model_validate({**own.model_dump(), "shared_route_opportunities": ["R-7D04E8A1"],
                                       "starvation_line": "3 jobs waiting over 14 days", "distance_cost_km": 412, "nearest_office": "Palmerston office",
                                       "capacity_block_flag": True, "next_actionable": "Thursday",
                                       "flags": ("Possible duplicate (also R-7D04E8A1)",)})
    assert tenant_why(busy) == tenant_why(own)


def test_coordinator_view_shows_tenant_asked_why_and_tenant_text_does_not_change() -> None:
    a, b = standard_job("R-AAAA"), standard_job("R-BBBB", original_report_timestamp=MON + timedelta(days=1))
    result = rank([to_rank_input(a), to_rank_input(b)])
    asked = build_traces(result, {"R-AAAA": a, "R-BBBB": b}, frozenset({"R-BBBB"}))
    plain = build_traces(result, {"R-AAAA": a, "R-BBBB": b})
    assert ["tenant asked why" in render_coordinator(t) for t in asked] == [False, True]
    assert not any("tenant asked why" in render_coordinator(t) for t in plain)
    assert [render_tenant_sms(t) for t in asked] == [render_tenant_sms(t) for t in plain]


def test_review_entry_shows_tenant_asked_why() -> None:
    job = path_job("review")
    assert "tenant asked why" in render_review_entry(job, asked_why=True)
    assert "tenant asked why" not in render_review_entry(job)


def test_unknown_ref_reply_text() -> None:
    assert UNKNOWN_REF_REPLY == ("We couldn't find that reference. Please check the number or call the maintenance "
                                 "call centre on 1800 104 076.")


RANKING_WORDS = ("order", "ordered", "alternative", "another working", "score", "tally", "priority list",
                 "ahead", "behind", "business days", "hours", "within", "days", "fs17", "response time",
                 "target")


def test_no_tenant_text_explains_ranking_or_states_a_timeframe() -> None:
    texts = [
        *every_path_sms().values(), *SMS_PATH_SENTENCES.values(), GAS_ADVICE, ELECTRICAL_ADVICE,
        tenant_sms_report(compound(("urgent", "R-A1", "roof leaking"), ("routine", "R-B2", "door sticks"))),
        *every_why(), UNKNOWN_REF_REPLY,
        *[tenant_sms(in_community(path_job(p), c)) for p in RANKED_PATHS for c in ("Darwin", "Wadeye", "Atlantis")],
    ]
    for text in texts:
        for word in RANKING_WORDS:
            assert word not in text.lower(), (word, text)



@pytest.mark.parametrize("entry", ["exposed electrical wires", "sewage leak", "blocked drain"])
def test_every_dangerous_entry_reads_urgent(entry: str) -> None:
    # Category word follows the tier: these cite no RTA s63, and must still read urgent.
    job = hazard_job(entry, fault="drain overflowing", safety="none")
    assert "It's being treated as an urgent repair under NT rules." in tenant_sms(job)
    assert "is booked as an urgent repair." in tenant_why(job)



def test_review_entry_shows_the_same_distance_row() -> None:
    job = standard_job("R-AAAA", community="Wadeye", distance_cost_km=250, nearest_office="Palmerston office",
                       **UNTIERED)
    assert "~250 km to nearest NT Housing office (Palmerston office) — straight-line; actual dispatch point not known" in render_review_entry(job)
    unknown = standard_job("R-BBBB", community="Atlantis", **UNTIERED)
    assert "unknown (community not in location table)" in render_review_entry(unknown)
