import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from triage.extraction import PROMPT_EXAMPLES
from triage.labelled import LABELLED_PATH, ExpectedFault, LabelledReport, load_labelled
from triage.report_files import parse_report_text, read_text

ROOT = Path(__file__).resolve().parent.parent
NT = timezone(timedelta(hours=9, minutes=30))
REMOTE = {"Wadeye", "Maningrida", "Galiwinku"}


def records() -> list[LabelledReport]:
    return [LabelledReport.model_validate_json(line) for line in LABELLED_PATH.read_text(encoding="utf-8").splitlines()]


def by_id() -> dict[str, LabelledReport]:
    return {r.id: r for r in records()}


def normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


# --- shape and provenance ----------------------------------------------------------------

def test_set_size_and_origins() -> None:
    rows = records()
    assert len(rows) == 31
    assert len({r.id for r in rows}) == 31
    origins = {r.id: r.origin for r in rows}
    assert {i for i, o in origins.items() if o == "team"} == {"D1", "D2", "D3"}
    assert {i for i, o in origins.items() if o == "gemini"} == {f"P{n}{v}" for n in range(1, 9) for v in "ot"} | {
        "B1", "B2", "B3", "B4"}
    assert {i for i, o in origins.items() if o == "claude-code"} == {"P9o", "P9t", "P10o", "P10t", "X1", "S1", "A1", "A2"}


def test_officer_tenant_pairs() -> None:
    rows = records()
    pairs = sorted({r.pair for r in rows if r.pair}, key=lambda p: int(p[1:]))
    assert len(pairs) >= 10
    ids = by_id()
    for n, pair in enumerate(pairs, start=1):
        officer, tenant = ids[f"{pair}o"], ids[f"{pair}t"]
        assert (officer.pair, tenant.pair) == (pair, pair)
        assert (officer.source_tag.value, officer.community) == ("officer", "Darwin")
        assert tenant.source_tag.value == "tenant_direct" and tenant.community in REMOTE
        # Odd pairs: tenant reported first; even pairs: officer first.
        first, second = (tenant, officer) if n % 2 else (officer, tenant)
        assert first.reported_at < second.reported_at, pair


def test_singles_are_remote_tenant_reports() -> None:
    for r in records():
        if r.pair is None:
            assert r.source_tag.value == "tenant_direct" and r.community in REMOTE, r.id


def test_timestamps_distinct_nt_time_in_range() -> None:
    stamps = [r.reported_at for r in records()]
    assert len(set(stamps)) == len(stamps)
    assert all(ts.utcoffset() == timedelta(hours=9, minutes=30) for ts in stamps)
    assert all(datetime(2026, 9, 22, tzinfo=NT) <= ts < datetime(2026, 10, 6, tzinfo=NT) for ts in stamps)


# --- loader ---------------------------------------------------------------------------------

def test_loader_keeps_id_as_source_file_and_returns_expected_separately() -> None:
    reports, expected = load_labelled()
    ids = by_id()
    assert len(reports) == 31 and set(expected) == {r.request_id for r in reports}
    for report in reports:
        record = ids[report.source_file]
        assert report.raw_text == record.text
        assert report.original_report_timestamp == record.reported_at
        assert (report.community, report.source_tag) == (record.community, record.source_tag)
        assert expected[report.request_id] == record.expected


def test_unknown_community_is_rejected(tmp_path: Path) -> None:
    line = json.loads(LABELLED_PATH.read_text(encoding="utf-8").splitlines()[0])
    line["community"] = "Katherine"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Katherine"):
        load_labelled(path)


# --- no overlap with anything a model or prompt has already seen ------------------------------

def _probe_samples() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("probe_llm", ROOT / "scripts" / "probe_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAMPLES


def test_no_report_matches_a_probe_sample_prompt_example_or_demo_report() -> None:
    seen = {normalise(s): "probe sample" for s in _probe_samples()}
    seen |= {normalise(s): "PROMPT_EXAMPLES" for s in PROMPT_EXAMPLES.values()}
    seen |= {normalise(parse_report_text(read_text(p)).raw_text): f"reports/{p.name}"
             for p in sorted((ROOT / "reports").glob("*.pdf"))}
    clashes = {r.id: seen[normalise(r.text)] for r in records() if normalise(r.text) in seen}
    assert clashes == {}


# --- coverage required by BUILD.md #2 --------------------------------------------------------

def test_required_case_types_are_present() -> None:
    rows = records()
    texts = " ".join(normalise(r.text) for r in rows)
    assert all(term in texts for term in ("dunny", "chocked", "won't go down"))
    faults = [f for r in rows for f in r.expected]
    assert any(len(r.expected) >= 2 for r in rows)  # compound
    assert any(f.fault_or_sign == ("sign",) for f in faults)  # sign-only
    assert any(f.alternative_mentioned == (True,) for f in faults)
    assert any(f.coping_mentioned == (True,) for f in faults)
    assert any("under" in f.claim_mismatch for f in faults)
    assert any("over" in f.claim_mismatch for f in faults)
    assert any(f.taxonomy_match == ((),) for f in faults)  # a fault not on the list


# --- ExpectedFault contract ------------------------------------------------------------------

def expected_fault() -> dict:
    return by_id()["P1t"].expected[0].model_dump(mode="json")


@pytest.mark.parametrize("field", list(ExpectedFault.model_fields))
def test_expected_fault_field_must_be_given(field: str) -> None:
    data = expected_fault()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        ExpectedFault.model_validate(data)


def test_expected_fault_rejects_unknown_name_extra_field_and_assignment() -> None:
    with pytest.raises(ValidationError, match="not in TIER_TABLE"):
        ExpectedFault.model_validate(expected_fault() | {"taxonomy_match": [["serious roof leak"]]})
    with pytest.raises(ValidationError):
        ExpectedFault.model_validate(expected_fault() | {"severity": ["high"]})
    fault = ExpectedFault.model_validate(expected_fault())
    with pytest.raises(ValidationError):
        fault.harm_claimed = (True,)  # type: ignore[misc]
