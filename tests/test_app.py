"""The web workspace (app/server.py): it shows the pipeline's results and records decisions,
but never orders or scores anything itself."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.server as server
from triage.extraction import OfflineExtractor

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
