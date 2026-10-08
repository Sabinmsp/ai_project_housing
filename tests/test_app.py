"""The web workspace (app/server.py): it shows the pipeline's results and records decisions,
but never orders or scores anything itself."""
import json
import re
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.server as server
from triage.evaluation import ALTERNATIVE
from triage.explain import tenant_sms, tenant_why
from triage.extraction import OfflineExtractor, report_text_from_prompt
from triage.recording import RECORDED_DIR, RecordedClient

REPORTS = {
    "a.txt": ("T-01", "Darwin", "2026-09-20 09:00", "toilet blocked"),
    "b.txt": ("T-02", "Wadeye", "2026-09-22 09:00", "power point sparking in the kitchen"),
    "c.txt": ("T-03", "Maningrida", "2026-09-21 09:00", "the ceiling fan wobbles and makes a noise"),
    "d.txt": ("T-04", "Galiwinku", "2026-09-19 09:00", "toilet blocked"),
}


class _Offline(OfflineExtractor):
    def describe(self) -> dict:
        return {"recorded": "offline", "recorded_on": None, "new_text": "offline"}


def _report_file(tenant: str, community: str, when: str, message: str) -> str:
    return f"Tenant ID: {tenant}\nCommunity: {community}\nSource: officer\nReported: {when}\nMessage:\n{message}\n"


@pytest.fixture
def client():
    # Starts empty, like the real app; the offline reader keeps tests independent of recordings and keys.
    server._workspace = server.Workspace(db_path=":memory:", client=_Offline())
    c = TestClient(server.app)
    assert c.post("/api/login", json={"username": "admin", "password": "Admin1!"}).status_code == 200
    for name, fields in REPORTS.items():
        r = c.post("/api/reports/upload", files={"file": (name, _report_file(*fields).encode(), "text/plain")})
        assert r.status_code == 200, r.text
    yield c
    server._workspace = None


@pytest.fixture
def officer(client):
    c = TestClient(server.app)
    assert c.post("/api/login", json={"username": "officer", "password": "Officer1!"}).status_code == 200
    return c


def _job(queue: dict, text: str, community: str) -> dict:
    return next(j for j in queue["ranked"] + queue["review_band"] if text in j["fault"] and j["community"] == community)


def test_queue_puts_safety_first_and_holds_unlisted_faults(client):
    q = client.get("/api/queue").json()
    assert [j["position"] for j in q["ranked"]] == list(range(1, len(q["ranked"]) + 1))
    assert q["ranked"][0]["community"] == "Wadeye" and q["ranked"][0]["safety_level"] == 2
    assert [j["community"] for j in q["review_band"]] == ["Maningrida"]


def test_equal_jobs_are_oldest_first_whatever_the_distance(client):
    toilets = [j for j in client.get("/api/queue").json()["ranked"] if j["fault"] == "toilet blocked"]
    # Galiwinku (560 km, reported 19 Sep) before Darwin (0 km, reported 20 Sep).
    assert [j["community"] for j in toilets] == ["Galiwinku", "Darwin"]


def test_tier_call_moves_a_review_band_job_into_the_queue_and_is_audited(client):
    fan = _job(client.get("/api/queue").json(), "ceiling fan", "Maningrida")
    r = client.post(f"/api/jobs/{fan['job_id']}/tier", json={"fault_name": "fan not working properly", "reason": "Fan fault"})
    assert r.status_code == 200
    q = client.get("/api/queue").json()
    assert q["review_band"] == []
    moved = _job(q, "ceiling fan", "Maningrida")
    assert moved["tier"] == "standard" and moved["urgency_tally"] == 3
    detail = client.get(f"/api/jobs/{fan['job_id']}").json()
    assert any(a["action"] == "Tier call" for a in detail["audit"])


def test_followup_re_ranks_but_keeps_the_original_report_time(client):
    before = _job(client.get("/api/queue").json(), "toilet blocked", "Darwin")
    r = client.post(f"/api/jobs/{before['job_id']}/followup", json={"text": "we are using the other one now"})
    assert r.json()["status"] == "ok"
    after = client.get(f"/api/jobs/{before['job_id']}").json()
    assert after["original_timestamp"] == before["original_timestamp"]
    assert after["urgency_tally"] == 3  # another working toilet named: the +1 is removed


def test_tenant_answer_never_mentions_distance_or_other_jobs(client):
    q = client.get("/api/queue").json()
    ids = {j["job_id"] for j in q["ranked"] + q["review_band"]}
    for job_id in ids:
        detail = client.get(f"/api/jobs/{job_id}").json()
        for text in (detail["why"], detail["sms"]):
            assert " km" not in text
            assert not (ids - {job_id}) & set(text.replace('"', " ").replace(":", " ").replace(".", " ").split())


def test_decision_needs_a_note_and_completed_jobs_leave_the_queue(client):
    job = client.get("/api/queue").json()["ranked"][0]
    assert client.post(f"/api/jobs/{job['job_id']}/decision", json={"status": "Completed"}).status_code == 422
    ok = client.post(f"/api/jobs/{job['job_id']}/decision", json={"status": "Completed", "note": "Fixed today"})
    assert ok.status_code == 200
    q = client.get("/api/queue").json()
    assert job["job_id"] not in {j["job_id"] for j in q["ranked"]}
    assert job["job_id"] in {j["job_id"] for j in q["completed"]}


def test_fairness_view_compares_but_does_not_reorder(client):
    ranked_before = [j["job_id"] for j in client.get("/api/queue").json()["ranked"]]
    f = client.get("/api/fairness").json()
    assert [r["job_id"] for r in f["what_if"]] == ranked_before
    assert sorted(r["nearest_first_position"] for r in f["what_if"]) == list(range(1, len(ranked_before) + 1))
    assert [j["job_id"] for j in client.get("/api/queue").json()["ranked"]] == ranked_before


def test_signed_out_users_get_nothing(client):
    anon = TestClient(server.app)
    assert anon.get("/api/queue").status_code == 401
    assert anon.post("/api/login", json={"username": "admin", "password": "wrong"}).status_code == 401


def test_officer_records_reports_but_cannot_see_the_queue_or_assign(client, officer):
    assert officer.get("/api/queue").status_code == 403
    assert officer.get("/api/tradies").status_code == 403
    r = officer.post("/api/reports/text", json={"community": "Wadeye", "message": "toilet blocked, using the other one"})
    assert r.status_code == 200
    job_id = r.json()["reports"][0]["jobs"][0]["job_id"]
    assert officer.post(f"/api/jobs/{job_id}/assign", json={"tradie_id": 1, "note": "trying"}).status_code == 403
    mine = officer.get("/api/my-reports").json()
    assert [x["job_id"] for x in mine[0]["jobs"]] == [job_id]
    assert "urgency_tally" not in mine[0]["jobs"][0] and "position" not in mine[0]["jobs"][0]
    # The coordinator sees the officer's report in the queue.
    assert job_id in {j["job_id"] for j in client.get("/api/queue").json()["ranked"]}


def test_recommendation_is_a_qualified_available_tradie_and_never_reorders(client):
    q = client.get("/api/queue").json()
    order = [j["job_id"] for j in q["ranked"]]
    toilet = _job(q, "toilet blocked", "Darwin")
    recs = client.get(f"/api/jobs/{toilet['job_id']}").json()["recommendations"]
    top = [r for r in recs if r["recommended"]]
    assert len(top) == 1 and "Plumber" in top[0]["trades"] and top[0]["available"]
    assert recs[0] is not None and recs[0]["recommended"]
    assert [j["job_id"] for j in client.get("/api/queue").json()["ranked"]] == order


def test_assigning_a_tradie_is_recorded_and_unavailable_tradies_are_refused(client):
    job = _job(client.get("/api/queue").json(), "toilet blocked", "Darwin")
    plumber = next(t for t in client.get("/api/tradies").json() if "Plumber" in t["trades"])
    r = client.post(f"/api/jobs/{job['job_id']}/assign", json={"tradie_id": plumber["id"], "note": "Thursday run"})
    assert r.status_code == 200
    detail = client.get(f"/api/jobs/{job['job_id']}").json()
    assert detail["status"] == "Assigned" and detail["tradie"] == plumber["name"]
    assert any(a["action"] == "Tradie assigned" for a in detail["audit"])
    assert {"job_id": job["job_id"], "community": "Darwin"} in next(
        t for t in client.get("/api/tradies").json() if t["id"] == plumber["id"])["jobs"]
    other = next(t for t in client.get("/api/tradies").json() if t["id"] != plumber["id"])
    client.post(f"/api/tradies/{other['id']}/availability")
    assert client.post(f"/api/jobs/{job['job_id']}/assign", json={"tradie_id": other["id"], "note": "x y z"}).status_code == 400


def test_a_tradie_already_going_there_is_recommended_first(client):
    q = client.get("/api/queue").json()
    darwin, galiwinku = _job(q, "toilet blocked", "Darwin"), _job(q, "toilet blocked", "Galiwinku")
    far_plumber = next(t for t in client.get("/api/tradies").json() if t["base"] != "Darwin" and "Plumber" in t["trades"])
    client.post(f"/api/jobs/{galiwinku['job_id']}/assign", json={"tradie_id": far_plumber["id"], "note": "on Gove run"})
    client.post("/api/reports/text", json={"community": "Galiwinku", "message": "toilet blocked"})
    q = client.get("/api/queue").json()
    second = next(j for j in q["ranked"] if j["community"] == "Galiwinku" and j["job_id"] != galiwinku["job_id"])
    top = next(r for r in client.get(f"/api/jobs/{second['job_id']}").json()["recommendations"] if r["recommended"])
    assert top["id"] == far_plumber["id"] and top["same_trip"] == [galiwinku["job_id"]]


FORM = Path(__file__).parent / "fixtures" / "geh_form_burst_pipe.pdf"


@pytest.fixture
def empty():
    server._workspace = server.Workspace(db_path=":memory:", client=_Offline())
    c = TestClient(server.app)
    assert c.post("/api/login", json={"username": "admin", "password": "Admin1!"}).status_code == 200
    yield c
    server._workspace = None


def test_app_starts_empty_and_loads_no_sample_files(empty):
    q = empty.get("/api/queue").json()
    assert q["ranked"] == q["review_band"] == q["needs_human"] == q["completed"] == []
    source = Path(server.__file__).read_text()
    assert "DEFAULT_FOLDERS" not in source and 'ROOT / "pdf"' not in source and 'ROOT / "reports"' not in source


def test_an_uploaded_form_runs_all_six_stages_and_returns_its_stage6_result(empty):
    r = empty.post("/api/reports/upload", files={"file": (FORM.name, FORM.read_bytes(), "application/pdf")})
    assert r.status_code == 200, r.text
    reports = r.json()["reports"]
    assert len(reports) == 3  # one report per issue row on the form
    queue = empty.get("/api/queue").json()
    positions = {j["job_id"]: j["position"] for j in queue["ranked"]}
    for rep in reports:
        s1 = rep["stage1"]
        assert s1["request_id"] == rep["report_id"] and s1["request_id"].startswith("R-")
        assert s1["original_report_timestamp"].endswith("+09:30")
        # Privacy: nothing personal and no tenant urgency rating reaches the model.
        for private in ("Marcus", "Ellery", "example.com", "0491", "Wattlebird", "Immediate", "Routine"):
            assert private not in s1["raw_text"]
        if rep["stage2"]["status"] != "ok":
            # Never dropped: a row Stage 2 can't use waits for a human read, with no jobs.
            assert rep["jobs"] == [] and rep["report_id"] in {r["report_id"] for r in queue["needs_human"]}
            continue
        for job in rep["jobs"]:
            assert all(span["verified"] for span in job["stage3"])
            assert job["stage4"]["urgency_tally"] == job["urgency_tally"]
            assert job["stage6"]["sms"] and job["stage6"]["why"]
            if job["stage6"]["in_review_band"]:
                assert job["stage6"]["position"] is None
            else:
                # The Stage 6 result shown after upload is the same one the queue shows.
                assert job["stage6"]["position"] == positions[job["job_id"]]
    burst = next(j for rep in reports for j in rep["jobs"] if "burst" in (j["fault"] or ""))
    assert burst["stage5"]["required_trades"] == ["Plumber"]
    assert burst["stage5"]["recommended_tradie"] is not None


def test_uploads_and_decisions_survive_a_restart_without_calling_the_model(tmp_path):
    db = str(tmp_path / "fairfix.db")
    server._workspace = server.Workspace(db_path=db, client=_Offline())
    c = TestClient(server.app)
    c.post("/api/login", json={"username": "admin", "password": "Admin1!"})
    c.post("/api/reports/upload", files={"file": (FORM.name, FORM.read_bytes(), "application/pdf")})
    before = c.get("/api/queue").json()
    job = before["ranked"][0]
    c.post(f"/api/jobs/{job['job_id']}/assign", json={"tradie_id": 1, "note": "Thursday run"})

    class NoCalls(_Offline):
        def complete_json(self, *a, **k):
            raise AssertionError("restart must rebuild from saved Stage 2 results, not call the model")

    server._workspace = server.Workspace(db_path=db, client=NoCalls())
    after = c.get("/api/queue").json()
    assert [j["job_id"] for j in after["ranked"]] == [j["job_id"] for j in before["ranked"]]
    restored = next(j for j in after["ranked"] if j["job_id"] == job["job_id"])
    assert restored["status"] == "Assigned" and restored["original_timestamp"] == job["original_timestamp"]
    server._workspace = None


def test_the_same_file_uploaded_twice_is_refused_and_makes_no_second_job(empty):
    first = empty.post("/api/reports/upload", files={"file": (FORM.name, FORM.read_bytes(), "application/pdf")})
    jobs = empty.get("/api/queue").json()
    again = empty.post("/api/reports/upload", files={"file": ("renamed.pdf", FORM.read_bytes(), "application/pdf")})
    assert again.status_code == 409
    for rep in first.json()["reports"]:
        assert rep["report_id"] in again.json()["detail"]
    assert empty.get("/api/queue").json() == jobs


def test_a_different_file_with_the_same_fault_is_still_accepted(empty):
    # The two files differ only in their last characters, so only a whole-file comparison passes.
    for message in ("toilet blocked", "toilet blocked, second toilet"):
        body = _report_file("T-01", "Darwin", "2026-09-20 09:00", message).encode()
        assert empty.post("/api/reports/upload", files={"file": ("a.txt", body, "text/plain")}).status_code == 200


def test_a_repeat_upload_is_still_refused_after_a_restart_but_not_after_reset(tmp_path):
    db = str(tmp_path / "fairfix.db")
    server._workspace = server.Workspace(db_path=db, client=_Offline())
    c = TestClient(server.app)
    c.post("/api/login", json={"username": "admin", "password": "Admin1!"})
    upload = lambda: c.post("/api/reports/upload", files={"file": (FORM.name, FORM.read_bytes(), "application/pdf")})
    assert upload().status_code == 200
    server._workspace = server.Workspace(db_path=db, client=_Offline())
    assert upload().status_code == 409
    c.post("/api/reset")
    assert upload().status_code == 200
    server._workspace = None


def test_reset_clears_everything(empty):
    empty.post("/api/reports/upload", files={"file": (FORM.name, FORM.read_bytes(), "application/pdf")})
    assert empty.post("/api/reset").status_code == 200
    q = empty.get("/api/queue").json()
    assert q["ranked"] == q["review_band"] == q["needs_human"] == []


def test_no_tradie_is_recommended_until_the_trade_is_known(client):
    fan = _job(client.get("/api/queue").json(), "ceiling fan", "Maningrida")
    assert fan["needed_trades"] == []
    assert not any(r["recommended"] for r in client.get(f"/api/jobs/{fan['job_id']}").json()["recommendations"])
    client.post(f"/api/jobs/{fan['job_id']}/tier", json={"fault_name": "fan not working properly", "reason": "Fan fault"})
    recs = client.get(f"/api/jobs/{fan['job_id']}").json()["recommendations"]
    top = next(r for r in recs if r["recommended"])
    assert "Electrician" in top["trades"]


def test_fault_names_are_tidied_only_to_an_exact_list_name():
    import json
    raw = json.dumps({"faults": [{"taxonomy_match": [": blocked or broken toilet", ">roof leak", "roof leaking"]}]})
    tidied = json.loads(server._tidy_fault_names(raw))["faults"][0]["taxonomy_match"]
    # Noise around a real name is removed; a name that isn't on the list stays as written, so it is still rejected.
    assert tidied == ["blocked or broken toilet", "roof leak", "roof leaking"]
    assert server._tidy_fault_names("not json") == "not json"


def test_a_retry_asks_the_model_again_instead_of_replaying_the_rejected_answer():
    from triage.recording import RecordingMissing

    class Saved:
        name = "recorded:x"
        def __init__(self): self.answer = None
        def complete_json(self, *a):
            if self.answer is None:
                raise RecordingMissing("none")
            return self.answer

    class Live:
        name = "llm:x"
        def __init__(self, saved): self.saved, self.calls = saved, 0
        def complete_json(self, *a):
            self.calls += 1
            self.saved.answer = '{"faults": []}'  # like RecordingClient, it saves what it returns
            return self.saved.answer

    c = server.WorkspaceClient.__new__(server.WorkspaceClient)
    saved = Saved()
    c.saved, c.offline, c._last_live_user, c.name = [saved], None, None, "x"
    c.live = Live(saved)
    c.complete_json("sys", "report text", {})
    c.complete_json("sys", "report text", {})  # extract()'s retry for the same text
    assert c.live.calls == 2


def test_coordinator_chooses_the_trade_for_an_unlisted_fault_and_it_is_recorded(client):
    q = client.get("/api/queue").json()
    order = [j["job_id"] for j in q["ranked"]]
    fan = _job(q, "ceiling fan", "Maningrida")  # not on the fault list: no trade from Stage 5
    detail = client.get(f"/api/jobs/{fan['job_id']}").json()
    assert detail["logistics"]["trade_source"] is None and not any(r["recommended"] for r in detail["recommendations"])
    preview = client.get(f"/api/jobs/{fan['job_id']}?trade=Electrician").json()
    top = next(r for r in preview["recommendations"] if r["recommended"])
    assert "Electrician" in top["trades"] and preview["logistics"]["trade_source"] == "coordinator"
    assert client.get(f"/api/jobs/{fan['job_id']}?trade=Wizard").status_code == 400
    r = client.post(f"/api/jobs/{fan['job_id']}/assign", json={"tradie_id": top["id"], "note": "Mia on Thursday", "trade": "Electrician"})
    assert r.status_code == 200
    after = client.get(f"/api/jobs/{fan['job_id']}").json()
    assert after["status"] == "Assigned" and after["logistics"]["needed_trades"] == ["Electrician"]
    assert any("coordinator's trade call" in a["note"] for a in after["audit"])
    # A trade choice for display never moves a job: still in the review band, queue order unchanged.
    q2 = client.get("/api/queue").json()
    assert fan["job_id"] in {j["job_id"] for j in q2["review_band"]}
    assert [j["job_id"] for j in q2["ranked"]] == order


def test_a_listed_fault_keeps_its_stage5_trade_whatever_the_coordinator_sends(client):
    toilet = _job(client.get("/api/queue").json(), "toilet blocked", "Darwin")
    d = client.get(f"/api/jobs/{toilet['job_id']}?trade=Roofer").json()
    assert d["logistics"]["needed_trades"] == ["Plumber"] and d["logistics"]["trade_source"] == "pipeline"


# ---- Coordinator override (master §5.1) ------------------------------------------------------
# The client fixture ranks: Wadeye sparking power point (safety) #1, Galiwinku toilet #2, Darwin toilet #3.

def _ranked(c) -> list[tuple[str, str]]:
    return [(j["community"], j["fault"]) for j in c.get("/api/queue").json()["ranked"]]


def _pin(c, job: dict, target: int, tag: str = "local knowledge"):
    return c.post(f"/api/jobs/{job['job_id']}/pin", json={"target_position": target, "reason_tag": tag})


def test_a_pin_shows_both_positions_and_never_changes_the_ranking(client):
    before = server._workspace.ranking()
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    assert darwin["position"] == 3
    assert _pin(client, darwin, 2).status_code == 200
    q = client.get("/api/queue").json()
    assert [j["community"] for j in q["ranked"]] == ["Wadeye", "Darwin", "Galiwinku"]
    row = _job(q, "toilet", "Darwin")
    assert (row["position"], row["system_position"], row["pinned"], row["decided_by"]) == (2, 3, True, "coordinator pin")
    assert row["pin"]["reason_tag"] == "local knowledge" and row["pin"]["age_days"] == 0
    assert q["pin_count"] == 1
    other = _job(q, "toilet", "Galiwinku")
    assert (other["position"], other["system_position"], other["pinned"]) == (3, 2, False)
    # Never: ranking with a pin in the workspace is identical to ranking without one.
    assert server._workspace.ranking() == before


def test_pin_refusals(client, officer):
    q = client.get("/api/queue").json()
    darwin, wadeye = _job(q, "toilet", "Darwin"), _job(q, "sparking", "Wadeye")
    fan = q["review_band"][0]
    assert _pin(client, fan, 1).json()["detail"] == "Choose repair type first"
    assert _pin(client, darwin, 2, "already made safe").status_code == 400
    assert _pin(client, darwin, 0).status_code == 400
    assert client.post(f"/api/jobs/{darwin['job_id']}/pin",
                       json={"target_position": 2, "reason_tag": "other", "note": "free text"}).status_code == 422
    for job, target in ((darwin, 1), (wadeye, 2), (wadeye, 3)):
        r = _pin(client, job, target)
        assert r.status_code == 400 and r.json()["detail"] == server.SAFETY_BLOCK
    assert client.post("/api/jobs/R-NOPE/pin", json={"target_position": 1, "reason_tag": "other"}).status_code == 404
    assert _pin(officer, darwin, 2).status_code == 403
    assert officer.delete(f"/api/jobs/{darwin['job_id']}/pin").status_code == 403
    assert _ranked(client) == [(j["community"], j["fault"]) for j in q["ranked"]]


def test_safety_jobs_can_be_reordered_among_themselves(client):
    body = _report_file("T-09", "Yuendumu", "2026-09-25 09:00", "strong smell of gas in the kitchen").encode()
    assert client.post("/api/reports/upload", files={"file": ("e.txt", body, "text/plain")}).status_code == 200
    assert [c for c, _ in _ranked(client)][:2] == ["Wadeye", "Yuendumu"]  # both safety, oldest first
    gas = _job(client.get("/api/queue").json(), "gas", "Yuendumu")
    assert _pin(client, gas, 1, "access / road").status_code == 200
    assert [c for c, _ in _ranked(client)][:2] == ["Yuendumu", "Wadeye"]
    assert _pin(client, gas, 3).json()["detail"] == server.SAFETY_BLOCK  # still not below a non-safety job


def test_unpin_restores_the_system_position_and_both_are_audited(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    system = _ranked(client)
    _pin(client, darwin, 2, "tenant contact")
    assert client.delete(f"/api/jobs/{darwin['job_id']}/pin").status_code == 200
    assert _ranked(client) == system
    assert client.delete(f"/api/jobs/{darwin['job_id']}/pin").status_code == 404
    audit = [(a["action"], a["note"]) for a in client.get(f"/api/jobs/{darwin['job_id']}").json()["audit"]][::-1]
    assert ("Pinned #2 (system #3)", "tenant contact") in audit and ("Unpinned", "") in audit


@pytest.mark.parametrize("close", ["assign", "complete"])
def test_dispatching_or_closing_a_job_clears_its_pin(client, close):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    _pin(client, darwin, 2)
    if close == "assign":
        client.post(f"/api/jobs/{darwin['job_id']}/assign", json={"tradie_id": 1, "note": "Thursday run"})
    else:
        client.post(f"/api/jobs/{darwin['job_id']}/decision", json={"status": "Completed", "note": "Fixed it"})
    assert darwin["job_id"] not in server._workspace.pins
    assert client.get("/api/queue").json()["pin_count"] == 0
    event = "cleared-by-assign" if close == "assign" else "cleared-by-completion"
    assert [h["event"] for h in server._workspace.pin_history] == ["pinned", event]


def test_pins_survive_a_restart_and_a_follow_up_but_not_a_reset(tmp_path):
    db = str(tmp_path / "fairfix.db")
    server._workspace = server.Workspace(db_path=db, client=_Offline())
    c = TestClient(server.app)
    c.post("/api/login", json={"username": "admin", "password": "Admin1!"})
    for name, fields in REPORTS.items():
        c.post("/api/reports/upload", files={"file": (name, _report_file(*fields).encode(), "text/plain")})
    darwin = _job(c.get("/api/queue").json(), "toilet", "Darwin")
    _pin(c, darwin, 2)
    server._workspace = server.Workspace(db_path=db, client=_Offline())
    assert _job(c.get("/api/queue").json(), "toilet", "Darwin")["position"] == 2
    c.post(f"/api/jobs/{darwin['job_id']}/followup", json={"text": "still blocked"})
    assert _job(c.get("/api/queue").json(), "toilet", "Darwin")["pinned"]  # same job id, so the pin stays
    c.post("/api/reset")
    assert server._workspace.pins == {} and server._workspace.pin_history == []
    server._workspace = None


def test_a_pin_leaves_the_tenant_sms_identical_and_adds_one_plain_line_to_why(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    before = client.get(f"/api/jobs/{darwin['job_id']}").json()
    _pin(client, darwin, 2, "bundling with nearby job")
    after = client.get(f"/api/jobs/{darwin['job_id']}").json()
    assert after["sms"] == before["sms"]
    assert after["why"] != before["why"] and after["why"].replace(server.tenant_why.__globals__["WHY_PINNED_LINE"] + " ", "") == before["why"]
    # No tag, position or other job; WHY_BANNED in test_explain covers the wording in general.
    for leak in ("bundling", "nearby", "#", "position", "system", "pin", "Galiwinku", "Wadeye"):
        assert leak not in after["why"].lower(), leak


def test_a_new_arrival_above_a_pin_is_marked(client):
    galiwinku = _job(client.get("/api/queue").json(), "toilet", "Galiwinku")
    _pin(client, galiwinku, 3)
    body = _report_file("T-10", "Wadeye", "2026-09-10 09:00", "toilet blocked").encode()
    client.post("/api/reports/upload", files={"file": ("late.txt", body, "text/plain")})
    q = client.get("/api/queue").json()
    late = next(j for j in q["ranked"] if j["community"] == "Wadeye" and j["fault"] == "toilet blocked")
    assert late["arrived_above_pin"] and late["system_position"] == 2
    assert not _job(q, "toilet", "Darwin")["arrived_above_pin"]


def test_a_job_with_no_arrival_time_is_reported_not_guessed(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    _pin(client, darwin, 2)
    server._workspace.submitted.pop(_job(client.get("/api/queue").json(), "toilet", "Galiwinku")["report_id"])
    with pytest.raises(ValueError, match="no arrival time"):
        client.get("/api/queue")


def test_fairness_counts_every_override_by_community_tag_and_direction(client):
    q = client.get("/api/queue").json()
    darwin, galiwinku = _job(q, "toilet", "Darwin"), _job(q, "toilet", "Galiwinku")
    _pin(client, darwin, 2, "access / road")
    client.delete(f"/api/jobs/{darwin['job_id']}/pin")
    _pin(client, galiwinku, 3, "other")
    rows = {o["community"]: o for o in client.get("/api/fairness").json()["overrides"]}
    assert (rows["Darwin"]["jobs"], rows["Darwin"]["repins"]) == (1, 0) and (rows["Darwin"]["up"], rows["Darwin"]["down"]) == (1, 0)
    assert rows["Darwin"]["by_tag"]["access / road"] == 1 and sum(rows["Darwin"]["by_tag"].values()) == 1
    assert (rows["Galiwinku"]["up"], rows["Galiwinku"]["down"], rows["Galiwinku"]["by_tag"]["other"]) == (0, 1, 1)
    assert set(rows) == {"Darwin", "Galiwinku"}


class _SplitsOnFollowUp(_Offline):
    """The offline reader never splits a report, so this one returns two faults once the
    follow-up mentions sparking: one per pattern, each quoting the report's own words."""

    def complete_json(self, system, user, schema):
        text = report_text_from_prompt(user)
        if "sparking" not in text:
            return super().complete_json(system, user, schema)
        toilet, sparks = self.read("toilet blocked").model_dump(), self.read("sparking").model_dump()
        return json.dumps({**toilet, "faults": toilet["faults"] + sparks["faults"]})


def test_a_re_read_that_splits_the_job_drops_its_pin_and_says_so(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    _pin(client, darwin, 2)
    server._workspace.client = _SplitsOnFollowUp()
    client.post(f"/api/jobs/{darwin['job_id']}/followup", json={"text": "and now the power point is sparking too"})
    assert darwin["job_id"] not in server._workspace.jobs  # two faults now, so two new job ids
    assert server._workspace.pins == {}
    assert [h["event"] for h in server._workspace.pin_history] == ["pinned", "replaced-by-reread"]
    assert any(a["action"] == "Unpinned" and a["note"] == "replaced by reread" for a in server._workspace.audit)


def test_one_job_pinned_three_times_counts_as_one_job_and_two_re_pins(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    _pin(client, darwin, 2, "access / road")
    _pin(client, darwin, 3, "other")
    _pin(client, darwin, 2, "tenant contact")  # the latest pin decides tag and direction
    rows = {o["community"]: o for o in client.get("/api/fairness").json()["overrides"]}
    assert set(rows) == {"Darwin"}
    assert (rows["Darwin"]["jobs"], rows["Darwin"]["repins"], rows["Darwin"]["up"], rows["Darwin"]["down"]) == (1, 2, 1, 0)
    assert {t: n for t, n in rows["Darwin"]["by_tag"].items() if n} == {"tenant contact": 1}


def test_the_move_buttons_stay_inside_the_jobs_safety_group(client):
    q = client.get("/api/queue").json()
    move = {k: client.get(f"/api/jobs/{_job(q, *k)['job_id']}").json()["move_range"]
            for k in (("sparking", "Wadeye"), ("toilet", "Darwin"))}
    assert move == {("sparking", "Wadeye"): {"top": 1, "last": 1}, ("toilet", "Darwin"): {"top": 2, "last": 3}}
    assert client.get(f"/api/jobs/{q['review_band'][0]['job_id']}").json()["move_range"] is None


class _CannotReadFollowUp(_Offline):
    def complete_json(self, system, user, schema):
        if "still blocked" in report_text_from_prompt(user):
            raise ConnectionError("model unreachable")
        return super().complete_json(system, user, schema)


def test_an_unreadable_follow_up_keeps_the_job_and_flags_it_for_a_human(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    server._workspace.client = _CannotReadFollowUp()
    r = client.post(f"/api/jobs/{darwin['job_id']}/followup", json={"text": "still blocked"})
    assert r.status_code == 200 and r.json()["status"] != "ok"
    after = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    assert after["job_id"] == darwin["job_id"] and after["urgency_tally"] == darwin["urgency_tally"]
    assert 'Follow-up could not be read automatically, review it: "still blocked"' in after["flags"]


def test_stepping_three_places_then_saving_is_one_pin_and_no_re_pins(client):
    for i, community in enumerate(("Katherine", "Tennant Creek", "Nhulunbuy")):
        body = _report_file(f"T-2{i}", community, f"2026-09-2{3 + i} 09:00", "toilet blocked").encode()
        client.post("/api/reports/upload", files={"file": (f"t{i}.txt", body, "text/plain")})
    last = _job(client.get("/api/queue").json(), "toilet", "Nhulunbuy")
    assert last["position"] == 6
    # The UI's preview: three "Move up" clicks change nothing on the server, then one save.
    assert _pin(client, last, last["position"] - 3, "access / road").status_code == 200
    assert _job(client.get("/api/queue").json(), "toilet", "Nhulunbuy")["position"] == 3
    assert [h["event"] for h in server._workspace.pin_history] == ["pinned"]
    row = next(o for o in client.get("/api/fairness").json()["overrides"] if o["community"] == "Nhulunbuy")
    assert (row["jobs"], row["repins"], row["up"]) == (1, 0, 1)


def test_the_override_moves_are_a_preview_and_only_save_sends_a_pin():
    # No JS runtime in CI, so this checks the page source: the move buttons and Cancel are plain
    # buttons whose handlers never call the API, and the one pin request is the Save submit.
    js = (Path(server.__file__).parent / "static" / "app.js").read_text()
    for button in ("mvUp", "mvDown", "mvTop", "mvCancel"):
        assert f'id="{button}" class="btn btn-outline btn-sm" type="button"' in js or \
               f'id="{button}" class="btn btn-outline flex-1" type="button"' in js, button
    assert 'id="mvSave" class="btn btn-primary flex-1" type="submit"' in js
    handlers = js[js.index("$('#mvUp').onclick"):js.index("submitForm('#pinForm'")]
    assert "api(" not in handlers and "fetch(" not in handlers
    assert js.count("/pin`, { method: 'POST'") == 1
    assert "submitForm('#pinForm', f => api(`/api/jobs/${id}/pin`, { method: 'POST', body: { target_position: preview," in js


# ---- Plain language, "Why it's here" and "Choose repair type" ---------------------------------

APP_JS = (Path(server.__file__).parent / "static" / "app.js").read_text()
# What can reach the screen: the source without its // comments.
APP_JS_SHOWN = re.sub(r"^\s*//.*$|\s//\s.*$", "", APP_JS, flags=re.M)


class _Recorded(RecordedClient):
    def describe(self) -> dict:
        return {"recorded": self.name, "recorded_on": self.recorded_on, "new_text": "flagged"}


@pytest.fixture
def recorded(monkeypatch):
    """The recorded reports/ and pdf/ sets, replayed from saved model answers (no API calls), with
    one coordinator repair-type call and one pin so those lines are covered too."""
    monkeypatch.setenv("TRIAGE_MODEL", "anthropic/claude-sonnet-5.5")  # the model the answers were saved for
    server._workspace = server.Workspace(db_path=":memory:", client=_Recorded(RECORDED_DIR))
    for folder in ("reports", "pdf"):
        server._workspace.load_folder(server.ROOT / folder, by="test")
    c = TestClient(server.app)
    assert c.post("/api/login", json={"username": "admin", "password": "Admin1!"}).status_code == 200
    q = c.get("/api/queue").json()
    assert len(q["ranked"]) >= 10 and q["review_band"]
    aircon = next(j for j in q["review_band"] if "air conditioner" in j["fault"].lower())
    assert c.post(f"/api/jobs/{aircon['job_id']}/tier",
                  json={"fault_name": "fan not working properly", "reason": "Cooling fault, treat like a fan"}).status_code == 200
    general = [j for j in c.get("/api/queue").json()["ranked"] if j["safety_level"] == 0]
    assert _pin(c, general[-1], general[0]["position"], "access / road").status_code == 200
    yield c
    server._workspace = None


def _allowed_lines(job, trace, row, call, date_only) -> set[str]:
    """Every sentence the box may say for this job, each tied to one value in its trace."""
    if trace is None:
        return {server.NOT_LISTED, server.SAFETY_SENTENCES[0], server.waiting_sentence(job, date_only)}
    allowed = {f"#{row.display_position} in the queue.", server.SAFETY_SENTENCES[trace.safety_level],
               server.reported_sentence(trace.original_timestamp, date_only)}
    allowed |= {server.LOSS_SENTENCES[r] for r in trace.tally_reasons}
    if row.pin:
        word = "up" if row.display_position < trace.position else "down"
        allowed.add(f"Moved {word} from #{trace.position} by a coordinator (reason: {row.pin.reason_tag}).")
    if call:
        allowed.add(f"Treated like: {call.fault_name} ({server.REPAIR_TYPE[trace.tier]}) — coordinator's call: {call.reason}.")
    elif trace.tier is None:
        allowed.add(server.NOT_LISTED)
    else:
        allowed |= {server.source_sentence(trace.tier, s) for s in server.TIER_TABLE[trace.tier_entry].sources}
    return allowed


def test_why_here_says_only_what_each_jobs_trace_holds(recorded):
    w = server._workspace
    band, traces, rows = w.display()
    by_id = {t.job_id: t for t in traces}
    kinds = set()
    for job_id, job in w.jobs.items():
        trace, row, call = by_id.get(job_id), rows.get(job_id), w.tier_calls.get(job_id)
        date_only = server.is_date_only(w.repo.get(job.parent_report_id))
        lines = recorded.get(f"/api/jobs/{job_id}").json()["why_here"]
        allowed = _allowed_lines(job, trace, row, call, date_only)
        assert set(lines) <= allowed, (job_id, set(lines) - allowed)  # nothing the trace doesn't hold
        assert len(lines) == len(set(lines))
        if trace is None:
            kinds.add("not listed")
            assert lines[-1] == server.waiting_sentence(job, date_only)
            continue
        # ...and everything it does hold is said: position, every loss reason, safety, the date.
        assert lines[0] == f"#{row.display_position} in the queue."
        assert {server.LOSS_SENTENCES[r] for r in trace.tally_reasons} <= set(lines)
        assert server.SAFETY_SENTENCES[trace.safety_level] in lines
        assert lines[-1] == server.reported_sentence(trace.original_timestamp, date_only)
        if ALTERNATIVE in trace.tally_reasons:
            assert not any("doesn't mention a working alternative" in l for l in lines)
        # Only the Act is called law; nt.gov.au is guidance, and a coordinator's call cites neither.
        sources = server.TIER_TABLE[trace.tier_entry].sources if trace.tier_entry and not call else ()
        assert any("under NT law" in l for l in lines) == any(s.startswith("RTA ") for s in sources)
        kinds |= {"pinned"} if row.pin else set()
        kinds |= {"called"} if call else set()
        kinds |= {"safety"} if trace.safety_level else set()
        kinds |= {"emergency"} if trace.tier == "dangerous" and not call else set()
    assert kinds == {"not listed", "pinned", "called", "safety", "emergency"}


def test_why_here_wording_says_what_the_report_mentions_never_what_exists(recorded):
    texts = [l for job_id in server._workspace.jobs for l in recorded.get(f"/api/jobs/{job_id}").json()["why_here"]]
    texts += [*server.LOSS_SENTENCES.values(), *server.SAFETY_SENTENCES.values(), server.NOT_LISTED]
    for line in texts:
        low = line.lower()
        assert "no alternative" not in low and not re.search(r"\bnone\b", low), line
        assert not re.search(r"\bno other\b", low) or "mentioned" in low, line
        for word in ("score", "ranked by", "the ai", "the model"):  # never implies the AI scored anything
            assert word not in low, line


def test_every_score_reason_has_a_fixed_sentence_and_an_unknown_source_fails():
    from triage import evaluation
    assert set(server.LOSS_SENTENCES) == {evaluation.NO_ALTERNATIVE, evaluation.ALTERNATIVE, evaluation.SIGN, evaluation.DEGRADED}
    with pytest.raises(ValueError, match="no wording"):
        server.source_sentence("dangerous", "some new authority")


def test_tenant_sms_and_why_pass_through_unchanged(recorded):
    w = server._workspace
    rows = w.display()[2]
    for job_id, job in w.jobs.items():
        detail = recorded.get(f"/api/jobs/{job_id}").json()
        assert detail["sms"] == tenant_sms(job)
        assert detail["why"] == tenant_why(job, pinned=bool(rows.get(job_id) and rows[job_id].pinned))


def test_previous_calls_lists_every_call_newest_first_and_changes_nothing(client, officer):
    fan = client.get("/api/queue").json()["review_band"][0]
    ids = [fan["job_id"]]
    for n in (1, 2):
        body = _report_file(f"T-3{n}", "Ngukurr", f"2026-09-1{n} 09:00", "the ceiling fan wobbles and makes a noise").encode()
        r = client.post("/api/reports/upload", files={"file": (f"f{n}.txt", body, "text/plain")})
        ids.append(r.json()["reports"][0]["jobs"][0]["job_id"])
    for i, job_id in enumerate(ids):
        assert client.post(f"/api/jobs/{job_id}/tier",
                           json={"fault_name": "fan not working properly", "reason": f"call {i}"}).status_code == 200
    before = (dict(server._workspace.tier_calls), client.get("/api/queue").json())
    calls = client.get("/api/tier-calls").json()
    assert [c["reason"] for c in calls] == ["call 2", "call 1", "call 0"]
    assert [c["job_id"] for c in calls] == ids[::-1]
    assert {(c["treated_like"], c["repair_type"], c["job_open"]) for c in calls} == {("fan not working properly", "General", True)}
    assert all("wobbles" in c["fault"] and c["by"] for c in calls)
    assert (dict(server._workspace.tier_calls), client.get("/api/queue").json()) == before  # read-only
    assert officer.get("/api/tier-calls").status_code == 403


def _js_function(name: str) -> str:
    start = APP_JS.index(f"function {name}(")
    return APP_JS[start:APP_JS.index("\n}\n", start)]


def test_the_repair_type_form_has_no_default_and_past_calls_never_fill_it():
    start = APP_JS.index('<form id="tierForm"')
    form = APP_JS[start:APP_JS.index("</form>", start)]
    assert "selected" not in form and "selected" not in _js_function("repairOptions")
    assert '<option value="">Choose a listed repair…</option>' in form and "Treat like" in form
    assert "repairOptions(refData.repair_list)" in form  # built from the lists only, never from past calls
    assert "#treatLike').value" not in APP_JS and "selected" not in _js_function("previousCalls")
    assert ("Safety risks go first. Then jobs are ordered by the type of repair and whether the household has lost the "
            "use of it completely. When jobs are level, the earlier report goes first.") in APP_JS


def test_the_repair_list_is_grouped_and_names_each_source(client):
    rows = client.get("/api/reference").json()["repair_list"]
    assert {r["repair_type"] for r in rows} == {"Emergency", "General"}
    roof = next(r for r in rows if r["name"] == "roof leak")
    assert roof == {"name": "roof leak", "repair_type": "Emergency", "sources": ["NT Residential Tenancies Act s63(2)(c)"]}
    assert "group('Emergency repairs', 'Emergency') + group('General repairs', 'General')" in _js_function("repairOptions")


def test_a_called_job_shows_treated_like_in_the_working_and_the_box(recorded):
    job_id = next(iter(server._workspace.tier_calls))
    detail = recorded.get(f"/api/jobs/{job_id}").json()
    row = next(r for r in detail["trace"] if r["label"] == "Treated like")
    assert row["value"] == "fan not working properly (General) — coordinator's call: Cooling fault, treat like a fan"
    assert "Treated like: fan not working properly (General) — coordinator's call: Cooling fault, treat like a fan." in detail["why_here"]


def test_no_score_column_and_no_old_vocabulary_on_screen():
    assert ">Score<" not in APP_JS and not re.search(r"\bLow\b", APP_JS)
    for old in ("review band", "needs a read", "human read", "nt category", "tier call", "untiered", "second reader",
                "stage 5", "first-come", "display only."):
        assert old not in APP_JS_SHOWN.lower(), old


def test_the_working_uses_plain_labels_and_keeps_every_row(recorded):
    for trace in server._workspace.display()[1]:
        detail = recorded.get(f"/api/jobs/{trace.job_id}").json()
        labels = [r["label"] for r in detail["trace"]]
        for must in ("Repair type", "Safety", "Required trade", "Distance", "Reported", "Queue place"):
            assert must in labels
        if trace.tier:
            assert {"Repair list match", "Base points", "Severity bump", "Urgency score"} <= set(labels)
        shown = " ".join(f"{r['label']} {r['value']} {r['note']}" for r in detail["trace"])
        shown = (shown + " " + " ".join(detail["flags"] + detail["why_here"])).lower()
        for word in ("tier", "review band", "stage 5", "first-come", "second reader"):
            assert word not in shown, (word, trace.job_id)


def test_every_header_and_badge_has_a_tooltip():
    tips = APP_JS[APP_JS.index("const TH_TIPS"):APP_JS.index("function applyTips")]
    start = APP_JS.index("const PRIORITY_TIP = {")
    tips += APP_JS[start:APP_JS.index("};", start)]
    keys = set(re.findall(r"""(?:^|[\s{,])(?:'([^']+)'|"([^"]+)"|([A-Za-z]+)):""", tips))
    keys = {a or b or c for a, b, c in keys}
    for label in set(re.findall(r"<th[^>]*>([^<$]+)</th>", APP_JS)):
        assert label in keys, label
    for text in set(re.findall(r'class="badge[^"]*"[^>]*>([^<$]+)<', APP_JS)):
        assert text in keys, text
    # A badge whose text changes carries its own title, unless every value it can show has a tip.
    covered = ("${priorityOf(j)}", "${REPAIR_TYPE[j.tier]}", "${k}", "${icon(")
    for tag, content in re.findall(r'(<span class="badge[^>]*>)(\$\{[^<]*)', APP_JS):
        assert "title=" in tag or content.startswith(covered), (tag, content[:40])


def test_the_fixed_sentences_are_exactly_the_agreed_wording():
    from triage import evaluation
    assert server.LOSS_SENTENCES[evaluation.NO_ALTERNATIVE] == \
        "The report doesn't mention a working alternative, so it's treated as a full loss of use."
    assert server.SAFETY_SENTENCES == {2: "The report describes a safety risk now.",
                                       1: "The report describes a possible safety risk.", 0: "No safety risk described."}
    assert server.NOT_LISTED == "This fault isn't on the NT repair lists. A coordinator needs to choose how to treat it."
    assert server.source_sentence("dangerous", "RTA s63(2)(c)") == \
        "Emergency repair under NT law (Residential Tenancies Act s63(2)(c))."
    assert server.source_sentence("dangerous", "nt.gov.au") == "Emergency repair in NT Government repairs guidance (nt.gov.au)."
    assert server.source_sentence("standard", "nt.gov.au") == "General repair in NT Government repairs guidance (nt.gov.au)."


def test_a_date_only_form_report_shows_the_date_alone_never_a_midnight_time(empty):
    # This form's first two rows carry a "previously reported to DIPL" date with no time.
    form = server.ROOT / "pdf" / "05_Top-End_Wurrumiyanga_roof-leak.pdf"
    r = empty.post("/api/reports/upload", files={"file": (form.name, form.read_bytes(), "application/pdf")})
    dated = [rep for rep in r.json()["reports"] if rep["stage1"]["date_only"]]
    assert dated, "the sample form should have a date-only row"
    checked = 0
    for rep in dated:
        assert rep["stage1"]["timestamp_source"].startswith("date previously reported to DIPL")
        for job in rep["jobs"]:
            detail = empty.get(f"/api/jobs/{job['job_id']}").json()
            assert detail["date_only"] is True
            shown = " ".join(detail["why_here"])
            shown += " " + " ".join(row["value"] for row in detail["trace"] or [] if row["label"] == "Reported")
            assert not re.search(r"\b0?0:00\b", shown), shown
            assert re.search(r"Reported \d{1,2} [A-Z][a-z]{2} —", shown), shown
            checked += 1
    assert checked, "no date-only row became a job, so nothing was checked"


def test_a_timed_report_keeps_its_time(client):
    darwin = _job(client.get("/api/queue").json(), "toilet", "Darwin")
    detail = client.get(f"/api/jobs/{darwin['job_id']}").json()
    assert detail["date_only"] is False
    assert detail["why_here"][-1].startswith("Reported 20 Sep, 9:00 — ")
    assert next(r["value"] for r in detail["trace"] if r["label"] == "Reported") == "20 Sep 2026, 09:00"


def test_the_needs_a_decision_box_says_when_it_was_reported_and_how_long_it_has_waited(client):
    fan = client.get("/api/queue").json()["review_band"][0]
    lines = client.get(f"/api/jobs/{fan['job_id']}").json()["why_here"]
    days = server.days_waiting(server._workspace.jobs[fan["job_id"]])
    assert lines == [server.NOT_LISTED, "No safety risk described.",
                     f"Reported 21 Sep, 9:00 — waiting {days} day{'' if days == 1 else 's'}"]
    assert days == fan["days_waiting"] and days > 0


def test_the_screen_shows_a_date_only_report_without_a_time():
    assert "const reportedAt = (iso, dateOnly) => dateOnly ? new Date(iso).toLocaleDateString() : new Date(iso).toLocaleString();" in APP_JS
    assert "reportedAt(j.original_timestamp, j.date_only)" in APP_JS
    assert "reportedAt(s1.original_report_timestamp, s1.date_only)" in APP_JS
    assert "new Date(j.original_timestamp).toLocaleString()" not in APP_JS


JARGON = ("tier", "score", "urgency", "untiered", "review band", "points", "bump")


def _backed_by_box(reason: str, box: list[str]) -> bool:
    """True if the row reason is a short form of a line in the job's own box."""
    if reason == "Needs a decision":
        return server.NOT_LISTED in box
    if reason in ("Safety risk now", "Possible safety risk"):
        level = 2 if reason == "Safety risk now" else 1
        return server.SAFETY_SENTENCES[level] in box
    if reason.startswith("Treated like: "):
        return any(line.startswith(reason + " — coordinator's call: ") for line in box)
    if reason == "Emergency repair under NT law":
        return any(line.startswith("Emergency repair under NT law (Residential Tenancies Act ") for line in box)
    m = re.fullmatch(r"(Emergency|General) repair \(NT Government guidance\)", reason)
    return bool(m) and f"{m.group(1)} repair in NT Government repairs guidance (nt.gov.au)." in box


def test_every_queue_row_shows_a_plain_reason_taken_from_its_own_box(recorded):
    q = recorded.get("/api/queue").json()
    rows = q["ranked"] + q["review_band"]
    seen = set()
    for j in rows:
        box = recorded.get(f"/api/jobs/{j['job_id']}").json()["why_here"]
        assert j["reason"], j["job_id"]
        assert _backed_by_box(j["reason"], box), (j["job_id"], j["reason"], box)
        assert not any(word in j["reason"].lower() for word in JARGON), j["reason"]
        # Safety outranks everything, so it is the reason whenever the box describes a risk.
        if j["safety_level"]:
            assert j["reason"] in ("Safety risk now", "Possible safety risk")
        seen.add(j["reason"].split(":")[0].split(" (")[0])
    assert {"Needs a decision", "Safety risk now", "Possible safety risk", "Treated like",
            "Emergency repair under NT law", "General repair"} <= seen


def test_the_queue_page_renders_the_row_reason_for_ranked_and_undecided_rows():
    queue = _js_function("renderQueue")
    assert queue.count("${rowReason(j)}") == 2
    assert "${esc(j.reason)}" in _js_function("rowReason")


def test_an_nt_gov_au_only_emergency_row_is_called_guidance_not_law(empty):
    # Blocked drain: an emergency repair on nt.gov.au only, not in the Act, with no safety risk.
    drain = Path(__file__).parent / "fixtures" / "blocked_drain.txt"
    assert server.TIER_TABLE["blocked drain"].sources == ("nt.gov.au",)
    r = empty.post("/api/reports/upload", files={"file": (drain.name, drain.read_bytes(), "text/plain")})
    job = r.json()["reports"][0]["jobs"][0]
    row = next(j for j in empty.get("/api/queue").json()["ranked"] if j["job_id"] == job["job_id"])
    assert (row["tier"], row["safety_level"]) == ("dangerous", 0)
    assert row["reason"] == "Emergency repair (NT Government guidance)"
    box = empty.get(f"/api/jobs/{job['job_id']}").json()["why_here"]
    assert "Emergency repair in NT Government repairs guidance (nt.gov.au)." in box
    assert not any("NT law" in line for line in box)
