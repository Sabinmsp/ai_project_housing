"""Property tests: the equity guarantee as executable assertions."""
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from triage.models import EnrichedJob, VerifiedSpan
from triage.ranking import rank, render_coordinator, render_tenant_sms, sort_key

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
COMMUNITIES = ["Wadeye", "Maningrida", "Darwin", "Katherine", "Galiwinku"]


@st.composite
def scored_job(draw, i=None):
    tier = draw(st.sampled_from(["dangerous", "standard"]))
    base = 3 if tier == "dangerous" else 2
    nr = draw(st.integers(0, 1))
    safety = draw(st.sampled_from(["active", "conditional", "none"]))
    return EnrichedJob(
        request_id=f"R-{draw(st.uuids()).hex[:8]}",
        community=draw(st.sampled_from(COMMUNITIES)),
        original_report_timestamp=EPOCH + timedelta(minutes=draw(st.integers(0, 60 * 24 * 90))),
        fault_description="fault",
        tier=tier, base_points=base, no_redundancy=nr, urgency_tally=base + nr,
        safety_flag=(safety == "active"), safety_level=safety,
        distance_cost_km=draw(st.floats(0, 2000, allow_nan=False)),
        capacity_block_flag=draw(st.booleans()),
        shared_route_opportunities=draw(st.lists(st.just("R-OTHER"), max_size=2)),
    )


@st.composite
def review_job(draw):
    return EnrichedJob(
        request_id=f"R-{draw(st.uuids()).hex[:8]}",
        community=draw(st.sampled_from(COMMUNITIES)),
        original_report_timestamp=EPOCH + timedelta(minutes=draw(st.integers(0, 10000))),
        fault_description="something not on the list",
    )


queues = st.lists(scored_job(), min_size=0, max_size=40)


@given(queues)
def test_no_unflagged_job_outranks_a_flagged_one(jobs):
    ranked = rank(jobs).ranked
    seen_unflagged = False
    for j in ranked:
        if not j.safety_flag:
            seen_unflagged = True
        assert not (seen_unflagged and j.safety_flag)


@given(queues)
def test_equal_flag_and_tally_order_oldest_first(jobs):
    ranked = rank(jobs).ranked
    for a, b in zip(ranked, ranked[1:]):
        if (a.safety_flag, a.urgency_tally) == (b.safety_flag, b.urgency_tally):
            assert a.original_report_timestamp <= b.original_report_timestamp


@given(queues, st.data())
def test_changing_any_logistics_value_never_changes_any_position(jobs, data):
    before = [j.request_id for j in rank(jobs).ranked]
    mutated = [
        j.model_copy(update={
            "distance_cost_km": data.draw(st.floats(0, 5000, allow_nan=False)),
            "community": data.draw(st.sampled_from(COMMUNITIES)),
            "capacity_block_flag": data.draw(st.booleans()),
            "shared_route_opportunities": [],
            "starvation_line": "3 jobs, oldest 40 days, no shared route",
        })
        for j in jobs
    ]
    after = [j.request_id for j in rank(mutated).ranked]
    assert before == after


@given(queues)
def test_higher_tally_never_below_lower_within_same_flag(jobs):
    ranked = rank(jobs).ranked
    for a, b in zip(ranked, ranked[1:]):
        if a.safety_flag == b.safety_flag:
            assert a.urgency_tally >= b.urgency_tally


@given(queues, st.lists(review_job(), max_size=10))
def test_review_band_is_held_outside_the_sort(jobs, review):
    res = rank(jobs + review)
    ranked_ids = {j.request_id for j in res.ranked}
    band_ids = [e.request_id for e in res.review_band]
    assert not ranked_ids & set(band_ids)
    assert len(res.ranked) == len(jobs) and len(res.review_band) == len(review)
    ts = [e.original_report_timestamp for e in res.review_band]
    assert ts == sorted(ts)


@given(queues)
def test_input_order_does_not_matter(jobs):
    a = [sort_key(j) for j in rank(jobs).ranked]
    b = [sort_key(j) for j in rank(list(reversed(jobs))).ranked]
    assert a == b


@settings(max_examples=50)
@given(queues)
def test_every_trace_is_arithmetically_consistent(jobs):
    res = rank(jobs)
    for pos, (job, tr) in enumerate(zip(res.ranked, res.traces), start=1):
        assert tr.request_id == job.request_id
        assert tr.base_points + tr.no_redundancy == tr.urgency_tally
        assert tr.position == pos and tr.queue_length == len(res.ranked)
        assert tr.sort_key == sort_key(job)


# --- the model cannot influence order ------------------------------------
# Everything Stage 2's model produces (fault text, taxonomy matches, quoted
# spans, coping/alternative wording) reaches Stage 6 only through fields that
# are not in the sort key. Vary all of them at once: the order must not move.

_SPLITS = {4: [("dangerous", 3, 1)], 3: [("dangerous", 3, 0), ("standard", 2, 1)],
           2: [("standard", 2, 0)]}
_WORDS = st.text(alphabet="abcdefghij klmnop", min_size=1, max_size=30)


@given(queues, st.data())
def test_only_safety_tally_and_timestamp_decide_order(jobs, data):
    before = [j.request_id for j in rank(jobs).ranked]
    mutated = []
    for j in jobs:
        tier, base, nr = data.draw(st.sampled_from(_SPLITS[j.urgency_tally]))
        level = "active" if j.safety_flag else data.draw(st.sampled_from(["conditional", "none"]))
        mutated.append(j.model_copy(update={
            "fault_description": data.draw(st.one_of(st.none(), _WORDS)),
            "taxonomy_match": data.draw(st.lists(_WORDS, max_size=3)),
            "spans": [VerifiedSpan(field=f, text=t, verified=v) for f, t, v in data.draw(
                st.lists(st.tuples(st.sampled_from(["fault_description", "taxonomy_match",
                                                    "coping_mentioned", "hazard_mechanism"]),
                                   _WORDS, st.booleans()), max_size=4))],
            "flags": data.draw(st.lists(st.sampled_from(["ambiguity_flag", "unverified_span"]),
                                        max_size=2)),
            "tier": tier, "base_points": base, "no_redundancy": nr,
            "safety_level": level,
            "distance_cost_km": data.draw(st.one_of(st.none(), st.floats(0, 5000, allow_nan=False))),
            "community": data.draw(st.sampled_from(COMMUNITIES)),
        }))
    after = [j.request_id for j in rank(mutated).ranked]
    assert before == after


def test_ranking_module_has_no_model_access():
    """Stage 6 is pure: it imports only the shared models, so no code path
    from ranking can reach an LLM client."""
    src = (Path(__file__).resolve().parents[1] / "triage" / "ranking.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert imported <= {"__future__", "typing", ".models"}, imported
    for word in ("openai", "complete_json", "LLMClient", "extraction"):
        assert word not in src, word


# --- Panel D worked example ----------------------------------------------

def panel_d_job():
    return EnrichedJob(
        request_id="R-2291", community="Wadeye",
        original_report_timestamp=datetime.fromtimestamp(1726041600, tz=timezone.utc),
        fault_description="roof leaking", taxonomy_match=["serious roof leak"],
        tier="dangerous", base_points=3, no_redundancy=1, urgency_tally=4,
        safety_flag=True, safety_level="active",
        spans=[VerifiedSpan(field="hazard_mechanism",
                            text="water coming through the light fitting", verified=True),
               VerifiedSpan(field="coping_mentioned", text="made up", verified=False)],
        distance_cost_km=412,
    )


def test_panel_d_trace():
    tr = rank([panel_d_job()]).traces[0]
    assert tr.sort_key == (True, 4, -1726041600)
    assert tr.original_report_timestamp == datetime.fromtimestamp(1726041600, tz=timezone.utc)
    assert tr.taxonomy_match == ["serious roof leak"]
    assert tr.defaults_applied == ["no-redundancy default applied: +1 (no alternative named)"]
    assert "mechanism_type=active" in tr.safety_reason
    assert [s.text for s in tr.evidence_spans] == ["water coming through the light fitting"]
    view = render_coordinator(tr)
    assert "412 km" in view and "not in sort_key" in view
    assert "serious roof leak" in view and "2024-09-11" in view
    assert "made up" not in view  # unverified spans never reach an audience


def test_trace_text_is_tenant_words_never_model_wording():
    """Panel D: no value in the trace originates inside the model. The model's
    own fault_description is replaced by the Stage 3 verified span."""
    job = panel_d_job().model_copy(update={
        "fault_description": "roof leaking. model stitched this sentence together",
        "spans": [VerifiedSpan(field="fault_description", text="invented by model", verified=False),
                  VerifiedSpan(field="fault_description", text="roof leaking", verified=True)],
    })
    tr = rank([job]).traces[0]
    assert tr.fault_description == "roof leaking"
    sms = render_tenant_sms(tr)
    assert "stitched" not in sms and "invented" not in sms
    no_span = rank([job.model_copy(update={"spans": []})]).traces[0]
    assert no_span.fault_description is None
    assert '"your repair"' in render_tenant_sms(no_span)


def test_tenant_sms_uses_only_trace_fields_and_hides_distance():
    tr = rank([panel_d_job()]).traces[0]
    sms = render_tenant_sms(tr)
    assert "R-2291" in sms and "1 of 1" in sms
    assert "km" not in sms and "412" not in sms
