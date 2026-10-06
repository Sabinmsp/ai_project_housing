from datetime import datetime, timedelta, timezone

import pytest

from hypothesis import given
from hypothesis import strategies as st

from triage.escalation import UNMATCHED, escalate, rematch_child
from triage.extraction import OfflineExtractor, extract
from triage.intake import SQLiteReportRepository, UnknownRequestError, create_report
from triage.models import ChildJob, ExtractedFacts, ExtractionResult, ExtractionStatus, ReportExtraction

T0 = datetime(2026, 9, 10, tzinfo=timezone(timedelta(hours=9, minutes=30)))


def saved_report(repo, text):
    r = create_report(tenant_id="T-1", raw_text=text, source_tag="tenant_direct",
                      community="Wurrumiyanga", original_report_timestamp=T0)
    repo.save(r)
    return r


def test_escalation_re_enters_stage_2_and_keeps_timestamp():
    repo, client = SQLiteReportRepository(), OfflineExtractor()
    r = saved_report(repo, "roof leaking in the bedroom")
    assert extract(r, client).extraction.faults[0].mechanism_type is None

    later = T0 + timedelta(days=20)
    updated, res = escalate(repo, r.request_id,
                            "now water coming through the light fitting", later, client)

    assert res.extraction.faults[0].mechanism_type == "active"  # re-read over the whole history
    assert updated.request_id == r.request_id
    assert updated.original_report_timestamp == T0  # fairness never resets


def test_escalation_never_fuzzy_matches():
    repo = SQLiteReportRepository()
    r = saved_report(repo, "toilet blocked")
    with pytest.raises(UnknownRequestError):
        escalate(repo, r.request_id.lower(), "worse", T0, OfflineExtractor())



# --- compound reports: escalation by a child job id -----------------------------------

COMPOUND = "wire sparking near the sink and I can smell gas"
WIRE_NAMES, GAS_NAMES = ["exposed electrical wires"], ["gas leak"]


def facts(description: str, names: list[str], worse: bool = False) -> ExtractedFacts:
    spans = [{"field": "fault_description", "text": description}]
    spans += [{"field": "taxonomy_match", "text": description}] if names else []
    spans += [{"field": "worsening_mentioned", "text": "worse"}] if worse else []
    return ExtractedFacts.model_validate({
        "fault_description": description, "taxonomy_match": names, "alternative_mentioned": False,
        "coping_mentioned": False, "impact_status": "ongoing", "hazard_status": "none",
        "mechanism_type": None, "harm_claimed": False, "fault_or_sign": "fault", "claim_mismatch": None,
        "worsening_mentioned": worse, "quoted_spans": spans,
    })


WIRE, GAS = facts("wire sparking", WIRE_NAMES), facts("smell gas", GAS_NAMES)
WIRE_WORSE, GAS_WORSE = facts("wire sparking", WIRE_NAMES, worse=True), facts("smell gas", GAS_NAMES, worse=True)


class Reads:
    """Fake model client: every call returns the same re-extraction."""
    name = "fake"

    def __init__(self, *faults: ExtractedFacts) -> None:
        self.raw = ReportExtraction(faults=faults).model_dump_json()

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        return self.raw


def compound(repo: SQLiteReportRepository, first: ExtractedFacts = WIRE,
             second: ExtractedFacts = GAS) -> tuple[ChildJob, ChildJob]:
    r = saved_report(repo, COMPOUND)
    children = tuple(ChildJob.model_validate({"job_id": job_id, "parent_report_id": r.request_id,
                                              "facts": f, "flags": ()})
                     for job_id, f in (("R-0000000A", first), ("R-0000000B", second)))
    for c in children:
        repo.save_child(c)
    return children


def test_escalating_a_child_id_updates_that_child_only() -> None:
    repo = SQLiteReportRepository()
    child1, child2 = compound(repo)
    later = T0 + timedelta(days=3)
    report, res = escalate(repo, child2.job_id, "gas worse now", later, Reads(WIRE_WORSE, GAS_WORSE))
    assert res.status is ExtractionStatus.OK
    assert report.request_id == child2.parent_report_id and report.raw_text.endswith("gas worse now")
    assert report.original_report_timestamp == T0  # fairness never resets
    assert repo.get_child(child2.job_id).facts == GAS_WORSE
    assert repo.get_child(child2.job_id).flags == ()
    assert repo.get_child(child1.job_id) == child1  # WIRE_WORSE was re-read too, but child 1 was not escalated


def test_reordered_faults_update_the_matching_child_not_the_same_position() -> None:
    repo = SQLiteReportRepository()
    child1, child2 = compound(repo)
    escalate(repo, child2.job_id, "gas worse now", T0, Reads(GAS_WORSE, WIRE_WORSE))
    assert repo.get_child(child2.job_id).facts == GAS_WORSE
    assert repo.get_child(child1.job_id) == child1


def expect_flagged_unchanged(repo: SQLiteReportRepository, child: ChildJob, *flags: str) -> None:
    after = repo.get_child(child.job_id)
    assert after.facts == child.facts
    assert after.flags == flags


def test_two_faults_sharing_the_childs_entry_flag_and_do_not_update() -> None:
    repo = SQLiteReportRepository()
    _, child2 = compound(repo)
    escalate(repo, child2.job_id, "gas", T0, Reads(GAS_WORSE, facts("gas", GAS_NAMES)))
    expect_flagged_unchanged(repo, child2, UNMATCHED)


def test_new_fault_in_escalation_flags_the_escalated_child() -> None:
    repo = SQLiteReportRepository()
    child1, child2 = compound(repo)
    flood = facts("near the sink", ["flooding or flood damage"])
    escalate(repo, child2.job_id, "now flooding", T0, Reads(WIRE, GAS_WORSE, flood))
    new = 'New fault in escalation: "near the sink" matches no existing job from this report — review the re-extracted report.'
    expect_flagged_unchanged(repo, child2, new, UNMATCHED)  # count changed 2 -> 3
    assert repo.get_child(child1.job_id) == child1


def test_fault_count_change_flags_even_with_one_match() -> None:
    repo = SQLiteReportRepository()
    _, child2 = compound(repo)
    escalate(repo, child2.job_id, "gas worse", T0, Reads(GAS_WORSE))
    expect_flagged_unchanged(repo, child2, UNMATCHED)


def test_child_with_no_taxonomy_entry_is_never_matched() -> None:
    repo = SQLiteReportRepository()
    untiered = facts("near the sink", [])
    _, child2 = compound(repo, second=untiered)
    escalate(repo, child2.job_id, "worse", T0, Reads(WIRE, facts("near the sink", [], worse=True)))
    new = 'New fault in escalation: "near the sink" matches no existing job from this report — review the re-extracted report.'
    expect_flagged_unchanged(repo, child2, new, UNMATCHED)


def test_failed_re_extraction_flags_and_keeps_facts() -> None:
    class Broken:
        name = "broken"

        def complete_json(self, system: str, user: str, schema: dict) -> str:
            return "not json"
    repo = SQLiteReportRepository()
    _, child2 = compound(repo)
    _, res = escalate(repo, child2.job_id, "worse", T0, Broken())
    assert res.status is ExtractionStatus.FLAGGED_FOR_HUMAN
    expect_flagged_unchanged(repo, child2, UNMATCHED)


def test_flags_carry_forward_across_escalations() -> None:
    repo = SQLiteReportRepository()
    _, child2 = compound(repo)
    escalate(repo, child2.job_id, "gas", T0, Reads(GAS_WORSE))  # count changed: flagged
    escalate(repo, child2.job_id, "gas", T0, Reads(WIRE, GAS_WORSE))  # matched: updated
    after = repo.get_child(child2.job_id)
    assert (after.facts, after.flags) == (GAS_WORSE, (UNMATCHED,))


NAMES = ["exposed electrical wires", "gas leak", "blocked drain"]
fault_facts = st.builds(lambda names, worse: facts("wire sparking", names, worse),
                        st.lists(st.sampled_from(NAMES), unique=True, max_size=2), st.booleans())


@given(st.lists(fault_facts, min_size=1, max_size=3), st.integers(min_value=0, max_value=2),
       st.lists(fault_facts, max_size=4), st.randoms(use_true_random=False))
def test_rematch_never_reads_fault_order_and_never_drops_or_guesses(
        stored: list[ExtractedFacts], pick: int, reread: list[ExtractedFacts], rnd) -> None:
    siblings = [ChildJob.model_validate({"job_id": f"R-{i:08X}", "parent_report_id": "R-PARENT00",
                                         "facts": f, "flags": ("earlier flag",)}) for i, f in enumerate(stored)]
    child = siblings[pick % len(siblings)]

    def run(faults: list[ExtractedFacts]) -> ChildJob:
        return rematch_child(child, siblings, ExtractionResult(
            request_id="R-PARENT00", status=ExtractionStatus.OK, extraction=ReportExtraction(faults=faults),
            attempts=1, errors=[], extractor="fake"))

    out = run(reread)
    shuffled = list(reread)
    rnd.shuffle(shuffled)
    again = run(shuffled)
    # Order of the re-read faults never changes the outcome.
    assert (again.facts, sorted(again.flags)) == (out.facts, sorted(out.flags))
    assert (out.job_id, out.parent_report_id) == (child.job_id, child.parent_report_id)
    assert out.flags[:1] == ("earlier flag",)  # never dropped
    hits = [f for f in reread if set(child.facts.taxonomy_match) & set(f.taxonomy_match)]
    updated = bool(child.facts.taxonomy_match) and len(reread) == len(siblings) and len(hits) == 1
    assert out.facts == (hits[0] if updated else child.facts)
    assert (UNMATCHED in out.flags) == (not updated)
