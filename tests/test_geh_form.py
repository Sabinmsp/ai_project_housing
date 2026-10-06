from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from triage.geh_form import (
    FormParseError,
    community_from_address,
    parse_geh_form,
    parse_items,
    raw_text_for,
    tenant_id_for,
)
from triage.report_files import load_reports

ACST = timezone(timedelta(hours=9, minutes=30))
PDF_DIR = Path(__file__).resolve().parents[1] / "pdf"
needs_pdfs = pytest.mark.skipif(not PDF_DIR.is_dir(), reason="pdf/ sample forms not present")

ROWS = [
    "1", "Split system air conditioner blowing warm air only, indoor unit dripping", "water",
    "Urgent", "Main bedroom", "28/09/2026", "Phone call",
    "No contractor booked yet. Daytime temps", "above 38 C, infant in household.",
    "2", "Outside sensor light not working", "Routine", "Outside, east side", "NIL", "NIL", "NIL",
]


def test_rows_become_separate_items_with_wrapped_text_joined():
    a, b = parse_items(ROWS)
    assert a.issue == "Split system air conditioner blowing warm air only, indoor unit dripping water"
    assert a.location == "Main bedroom" and a.previously_reported == "28/09/2026"
    assert a.method == "Phone call"
    assert a.comments == "No contractor booked yet. Daytime temps above 38 C, infant in household."
    assert b.location == "Outside, east side"
    assert (b.previously_reported, b.method, b.comments) == (None, None, None)


def test_tenant_priority_never_reaches_raw_text():
    for item in parse_items(ROWS):
        text = raw_text_for(item).lower()
        assert not any(w in text for w in ("urgent", "routine", "immediate"))


def test_broken_table_is_rejected():
    with pytest.raises(FormParseError):
        parse_items(["1", "Leaking tap", "Routine"])
    with pytest.raises(FormParseError):
        parse_items(["2", "starts at the wrong number"])


@pytest.mark.parametrize("address, community", [
    ("Lot 42, 18 Wattlebird Court, Humpty Doo NT 0836", "Humpty Doo"),
    ("Lot 317, Yuendumu NT 0872", "Yuendumu"),
    ("Lot 204, Wurrumiyanga, Bathurst Island NT 0822", "Wurrumiyanga"),
    ("Unit 3, 27 Banksia Street, Darwin City NT 0800", "Darwin City"),
])
def test_community_from_address(address, community):
    assert community_from_address(address) == community


def test_tenant_id_is_stable_and_not_the_email():
    tid = tenant_id_for("Priya.Raman@example.com")
    assert tid == tenant_id_for("priya.raman@example.com")
    assert "priya" not in tid.lower() and tid.startswith("T-")


# --- the five sample forms -------------------------------------------------

@needs_pdfs
def test_every_form_gives_one_report_per_issue():
    reports, skipped = load_reports(PDF_DIR)
    assert skipped == []
    per_file = {}
    for r in reports:
        per_file.setdefault(r.source_file, []).append(r.source_item)
    known = {"01_Greater-Darwin_Humpty-Doo_burst-pipe.pdf": 3,
             "02_Central-Australia_Yuendumu_air-conditioner.pdf": 3,
             "03_Big-Rivers_Ngukurr_break-in-damage.pdf": 3,
             "04_Barkly-Remote_Ali-Curung_electrical-fault.pdf": 4,
             "05_Top-End_Wurrumiyanga_roof-leak.pdf": 4}
    for name, count in known.items():
        assert len(per_file[name]) == count, name
    assert all(items == list(range(1, len(items) + 1)) for items in per_file.values())
    assert len({r.request_id for r in reports}) == len(reports)


@needs_pdfs
def test_form_fields_land_in_the_right_place():
    reports = parse_geh_form(PDF_DIR / "02_Central-Australia_Yuendumu_air-conditioner.pdf")
    first, second, third = reports
    assert {r.community for r in reports} == {"Yuendumu"}
    assert {r.region for r in reports} == {"Central Australia"}
    assert len({r.tenant_id for r in reports}) == 1
    assert "Second unit, so no working cooling in the house" in second.raw_text
    # previously reported: the tenant keeps their original place in the queue
    assert first.original_report_timestamp == datetime(2026, 9, 28, tzinfo=ACST)
    assert "previously reported" in first.timestamp_source
    assert third.timestamp_source == "form received (file time)"


@needs_pdfs
def test_no_personal_details_or_tenant_priority_in_any_raw_text():
    reports, _ = load_reports(PDF_DIR)
    for r in reports:
        text = r.raw_text.lower()
        assert "@" not in text and "0491" not in text and "lot " not in text
        assert not any(w in text for w in ("urgent", "routine", "immediate"))
