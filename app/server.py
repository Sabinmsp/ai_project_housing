"""FairFix NT web app: a coordinator workspace over the triage pipeline.

The server owns no scoring. A report enters only when someone uploads a document or records
a call; it then runs through triage/pipeline.py (Stages 1 to 6), the same code the CLI uses.
Nothing is loaded at start-up except what earlier uploads saved in SQLite.

Run from the repo root:  python -m app.server   (or: uvicorn app.server:app)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from triage.adapter import to_rank_input
from triage import second_reader
from triage.distances import community_names, km_between
from triage.escalation import escalate
from triage.evaluation import ALTERNATIVE, DEGRADED, NO_ALTERNATIVE, SIGN, compute_tally, lookup_tier
from triage.explain import ReasoningTrace, tenant_sms, tenant_why
from triage.extraction import OfflineExtractor, OpenAICompatibleClient, configured_model
from triage.intake import DuplicateRequestError, SQLiteReportRepository, UnknownRequestError, create_report
from triage.models import EnrichedJob, ExtractedFacts, ExtractionResult, ExtractionStatus, Report, SourceTag
from triage.overrides import REASON_TAGS, SAFETY_BLOCK, DisplayRow, Pin, apply_pins, pin_breaks_safety
from triage.pipeline import build_jobs, build_report_jobs, enrich_fault, save_children, stage2_extract, stage6_rank
from triage.ranking import NO_TIER_FLAG
from triage.recording import RECORDED_DIR, RecordedClient, RecordingClient, RecordingMissing
from triage.report_files import load_reports
from triage.tiers import COORDINATOR_SOURCE, FAULT_NAMES, NO_FIT_EMERGENCY, NO_FIT_GENERAL, TIER_TABLE
from triage.trades import ALL_TRADES, required_trades
from triage.verification import verify_spans

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
NT_TIME = timezone(timedelta(hours=9, minutes=30))
DB_PATH = os.environ.get("FAIRFIX_DB", str(ROOT / "data" / "fairfix.db"))

SOURCE_LABELS = {"nt.gov.au": "NT Government repairs guidance (nt.gov.au)",
                 COORDINATOR_SOURCE: "coordinator's call — not from the NT repair lists"}

# Prototype accounts only, not production authentication. The officer records reports;
# the admin (maintenance coordinator) works the queue and assigns tradies.
USERS = {
    "officer": {"password": "Officer1!", "role": "officer", "name": "Housing Officer"},
    "admin": {"password": "Admin1!", "role": "admin", "name": "Maintenance Coordinator"},
}

# Stage 5 static data from the design ("demo trade roster"): five tradies with different trades
# and home bases. Names are fictional; the phone numbers are from Australia's range for fiction.
SEED_TRADIES = (
    {"name": "Jack Walker", "trades": ["Plumber", "Gas fitter"], "base": "Darwin", "phone": "0491 570 006"},
    {"name": "Mia Nguyen", "trades": ["Electrician"], "base": "Katherine", "phone": "0491 570 313"},
    {"name": "Sam O'Brien", "trades": ["Roofer", "Builder"], "base": "Darwin", "phone": "0491 570 737"},
    {"name": "Tara Riley", "trades": ["Electrician", "Plumber"], "base": "Tennant Creek", "phone": "0491 571 266"},
    {"name": "Alex Kelly", "trades": ["Builder", "General maintenance"], "base": "Alice Springs", "phone": "0491 571 491"},
)


def _source_label(source: str) -> str:
    if source.startswith("RTA "):
        return f"NT Residential Tenancies Act {source.removeprefix('RTA ')}"
    return SOURCE_LABELS.get(source, source)


def _now() -> datetime:
    return datetime.now(NT_TIME)


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env (git-ignored). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Stage 2 client
# ---------------------------------------------------------------------------

_NAME_NOISE = " \t\n:;>-*•\"'`.,"


def _tidy_fault_names(raw: str) -> str:
    """Strip stray punctuation some models put around fault names (": blocked or broken toilet").
    A name is kept only if what remains is exactly a list name; anything else is left as the
    model wrote it, so Stage 2 validation still rejects it. Nothing is guessed or matched loosely."""
    try:
        data = json.loads(raw)
        for fault in data.get("faults", []):
            fault["taxonomy_match"] = [m.strip(_NAME_NOISE) if isinstance(m, str) and m.strip(_NAME_NOISE) in FAULT_NAMES else m
                                       for m in fault.get("taxonomy_match", [])]
        return json.dumps(data)
    except (ValueError, AttributeError, TypeError):
        return raw  # not the expected shape: let Stage 2 validation report it

class WorkspaceClient:
    """Reads a report with the model. A saved answer for the exact same text is reused (it is the
    model's real answer, recorded earlier); new text goes to the live model when TRIAGE_LIVE=1
    and a key is set, and the answer is saved. TRIAGE_OFFLINE_FALLBACK=1 uses the regex stand-in,
    labelled as such. Otherwise the report is flagged for a human, never guessed."""

    def __init__(self) -> None:
        _load_dotenv()
        # Each model's answers live in their own subfolder, so switching models never overwrites
        # another model's saved answers (the file name is the report text's hash, not the model's).
        own = RECORDED_DIR / "by_model" / "".join(c if c.isalnum() else "_" for c in configured_model())
        self.saved = [RecordedClient(own), RecordedClient(RECORDED_DIR)]
        self.recorded = next((c for c in self.saved if c.recorded_on), self.saved[1])
        has_key = bool(os.environ.get("TRIAGE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        self.live = RecordingClient(OpenAICompatibleClient(), own) if os.environ.get("TRIAGE_LIVE") and has_key else None
        self.offline = OfflineExtractor() if os.environ.get("TRIAGE_OFFLINE_FALLBACK") else None
        self.name = self.recorded.name
        self._last_live_user: Optional[str] = None

    def describe(self) -> dict:
        new_text = ("read live by " + self.live.model if self.live
                    else "read by the offline keyword stand-in (not the real reader)" if self.offline
                    else "flagged for a human unless the same text was read before (set TRIAGE_LIVE=1 with a funded key)")
        return {"recorded": self.recorded.name, "recorded_on": self.recorded.recorded_on, "new_text": new_text,
                "live": bool(self.live)}

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        # extract() reads .name after this returns, so each result names the reader actually used.
        missing: Optional[RecordingMissing] = None
        # A second call for the same text is extract()'s retry after a rejected answer:
        # ask the model again rather than replaying the answer it just rejected.
        if user != self._last_live_user:
            for saved in self.saved:
                try:
                    answer = saved.complete_json(system, user, schema)
                    self.name = saved.name
                    return _tidy_fault_names(answer)
                except RecordingMissing as e:
                    missing = e
        fallback = self.live or self.offline
        if fallback is None:
            raise missing or RecordingMissing("no saved answer")  # flagged for a human by extract(), never guessed
        self.name = fallback.name
        self._last_live_user = user
        return _tidy_fault_names(fallback.complete_json(system, user, schema))


# ---------------------------------------------------------------------------
# Persistence: the pipeline's repository plus two tables for the app
# ---------------------------------------------------------------------------

class WorkspaceStore(SQLiteReportRepository):
    """Reports, follow-ups and child jobs (the pipeline's tables), plus each report's Stage 2
    result and the coordinator's decisions. Usable from any server thread because every access
    goes through _lock."""

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        connect = sqlite3.connect
        sqlite3.connect = lambda p: connect(p, check_same_thread=False)
        try:
            super().__init__(path)
        finally:
            sqlite3.connect = connect
        self._conn.execute("CREATE TABLE IF NOT EXISTS stage2_results (report_id TEXT PRIMARY KEY, result TEXT NOT NULL)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()

    def save_result(self, result: ExtractionResult) -> None:
        self._conn.execute("INSERT INTO stage2_results VALUES (?, ?) ON CONFLICT(report_id) DO UPDATE SET result = excluded.result",
                           (result.request_id, result.model_dump_json()))
        self._conn.commit()

    def results(self) -> dict[str, ExtractionResult]:
        rows = self._conn.execute("SELECT report_id, result FROM stage2_results ORDER BY rowid").fetchall()
        return {rid: ExtractionResult.model_validate_json(raw) for rid, raw in rows}

    def save_state(self, values: dict) -> None:
        self._conn.executemany("INSERT INTO app_state VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                               [(k, json.dumps(v)) for k, v in values.items()])
        self._conn.commit()

    def load_state(self) -> dict:
        return {k: json.loads(v) for k, v in self._conn.execute("SELECT key, value FROM app_state").fetchall()}

    def clear(self) -> None:
        for table in ("followups", "child_jobs", "stage2_results", "app_state", "reports"):
            self._conn.execute(f"DELETE FROM {table}")
        self._conn.commit()


class TierCall(BaseModel):
    """A coordinator's repair-type call on one job. community is stored at call time, so the
    Communities counts keep the call even if a re-read later replaces the job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fault_name: str
    reason: str
    by: str
    at: datetime
    community: str


# ---------------------------------------------------------------------------
# Workspace: the queue as the pipeline sees it, plus the coordinator's decisions
# ---------------------------------------------------------------------------

class Workspace:
    def __init__(self, db_path: str = DB_PATH, client=None) -> None:
        self.client = client or WorkspaceClient()
        self.db_path = db_path
        self.repo = WorkspaceStore(db_path)
        self.results: dict[str, ExtractionResult] = {}     # Stage 2 output per report
        self.jobs: dict[str, EnrichedJob] = {}              # Stage 5 output per job
        self.facts: dict[str, ExtractedFacts] = {}          # the fault each job was built from
        state = self.repo.load_state()
        # Calls saved before community was stored take it from the job's report, which never changes.
        self.tier_calls = {k: TierCall.model_validate(v if "community" in v else {**v, "community": self._report_community(k)})
                           for k, v in state.get("tier_calls", {}).items()}
        self.status: dict[str, dict] = state.get("status", {})
        self.extra_flags: dict[str, list[str]] = state.get("extra_flags", {})
        self.tradies = {int(k): v for k, v in state.get("tradies", {}).items()} or \
            {i: {"id": i, "available": True, **t} for i, t in enumerate(SEED_TRADIES, start=1)}
        self.submitted: dict[str, dict] = state.get("submitted", {})
        # sha256 of each uploaded file -> the reports it made, so a file uploaded twice by mistake
        # is refused instead of making a second copy of every job.
        self.uploads: dict[str, list[str]] = state.get("uploads", {})
        self.audit: list[dict] = state.get("audit", [])
        # Master §5.1: pins sit on top of the ranking and never feed back into it. The history is
        # append-only, for reporting (the fairness view), since a pin is removed on dispatch.
        self.pins = {k: Pin.model_validate(v) for k, v in state.get("pins", {}).items()}
        self.pin_history: list[dict] = state.get("pin_history", [])
        self.skipped: list[dict] = []
        self._restore()

    def _report_community(self, job_id: str) -> str:
        """The community of the report a job came from (a compound report's child resolves to its parent)."""
        child = self.repo.get_child(job_id)
        report = self.repo.get(child.parent_report_id if child else job_id)
        if report is None:
            raise ValueError(f"saved repair-type call for {job_id}, but no report to take its community from")
        return report.community

    # ---- persistence -------------------------------------------------------
    def save(self) -> None:
        self.repo.save_state({
            "tier_calls": {k: v.model_dump(mode="json") for k, v in self.tier_calls.items()},
            "status": self.status, "extra_flags": self.extra_flags, "tradies": self.tradies,
            "submitted": self.submitted, "uploads": self.uploads, "audit": self.audit,
            "pins": {k: v.model_dump(mode="json") for k, v in self.pins.items()}, "pin_history": self.pin_history,
        })

    def _restore(self) -> None:
        """Rebuild jobs from saved Stage 2 results: Stages 3 to 5 rerun, the model is not called."""
        saved = self.repo.results()
        for report in self.repo.all():
            result = saved.get(report.request_id)
            if result is None:  # stored but never read (e.g. the server stopped mid-upload)
                result = stage2_extract(report, self.client)
                self.repo.save_result(result)
            self._apply(report, result, existing=True)

    # ---- Stage 1 -----------------------------------------------------------
    def load_folder(self, folder: Path, by: str) -> list[str]:
        reports, skipped = load_reports(folder)
        self.skipped += [{"file": p.name, "reason": why} for p, why in skipped]
        return [rid for r in reports for rid in [self.add_report(r, by)] if rid]

    def add_report(self, report: Report, by: str) -> Optional[str]:
        try:
            self.repo.save(report)
        except DuplicateRequestError:
            self.skipped.append({"file": report.source_file or report.request_id, "reason": "duplicate request_id"})
            return None
        self.submitted[report.request_id] = {"by": by, "at": _now().isoformat()}
        self.log(report.request_id, by, "Report received",
                 f"{report.source_tag.value}, {report.community}" + (f", {report.source_file}" if report.source_file else ""))
        self.process(report)
        return report.request_id

    # ---- Stages 2 to 5 -------------------------------------------------------
    def process(self, report: Report) -> None:
        result = stage2_extract(report, self.client)
        self.repo.save_result(result)
        self._apply(report, result)
        self.save()

    def _apply(self, report: Report, result: ExtractionResult, existing: bool = False) -> None:
        """Stage 2 output straight into Stages 3 to 5, replacing the report's previous jobs."""
        self.results[report.request_id] = result
        for job_id in [j for j, job in self.jobs.items() if job.parent_report_id == report.request_id]:
            self.jobs.pop(job_id)
            self.facts.pop(job_id, None)
        if result.status is not ExtractionStatus.OK or result.extraction is None:
            return
        children = self.repo.children(report.request_id) if existing else []
        if children:  # a compound report: keep its job ids and any escalation flags
            faults = [c.facts for c in children]
            jobs = build_jobs(report, faults, [c.job_id for c in children], [c.flags for c in children])
        elif existing:  # restart: rebuild from the saved reading, never call the second reader again
            faults = list(result.extraction.faults)
            jobs = build_jobs(report, faults)
        else:
            jev, jev_not_run = self.jev_reader()
            extraction, jobs = build_report_jobs(report, result.extraction, self.client, jev, jev_not_run)
            faults = list(extraction.faults)
            save_children(self.repo, faults, jobs)
        for job, facts in zip(jobs, faults, strict=True):
            self.store_job(job, facts)

    def jev_reader(self) -> tuple[Optional[second_reader.JevClient], second_reader.SecondReading]:
        """The second reader (Jev) runs only with live extraction and its own key, as in demo.py."""
        if isinstance(self.client, OfflineExtractor):
            return None, second_reader.not_run("not run (offline)")
        if getattr(self.client, "live", None) is None:
            return None, second_reader.not_run("not run (recorded mode)")
        return second_reader.client_from_env(), second_reader.not_run("not run (no key)")

    def store_job(self, job: EnrichedJob, facts: ExtractedFacts) -> None:
        self.facts[job.request_id] = facts
        if job.request_id in self.tier_calls and job.urgency_tally is None:
            job = self._apply_tier_call(job, facts, self.tier_calls[job.request_id])
        if self.extra_flags.get(job.request_id):
            job = EnrichedJob.model_validate({**job.model_dump(), "flags": (*job.flags, *self.extra_flags[job.request_id])})
        self.jobs[job.request_id] = job
        self.status.setdefault(job.request_id, {"status": "Open", "tradie_id": None, "note": None})

    def _apply_tier_call(self, job: EnrichedJob, facts: ExtractedFacts, call: TierCall) -> EnrichedJob:
        """The coordinator's tier call goes through the same Stage 4 rules as a list match."""
        report = self.repo.get(job.parent_report_id)
        unverified = verify_spans(report.raw_text, facts) if report else frozenset()
        tier = lookup_tier([call.fault_name], coordinator_call=True)
        tally = compute_tally(tier, facts, unverified)
        return EnrichedJob.model_validate({
            **job.model_dump(), "tier": tier.tier, "tier_entry": tally.winner, "base_points": tally.base,
            "severity_bump": tally.bump, "urgency_tally": tally.tally, "tally_reasons": tally.reasons,
            "required_trades": required_trades(tally.winner),
            "flags": (*job.flags, f"Repair type chosen by a coordinator: {call_text(call)}"),
        })

    def log(self, job_id: Optional[str], actor: str, action: str, note: str = "") -> None:
        self.audit.append({"job_id": job_id, "actor": actor, "action": action, "note": note, "at": _now().isoformat()})

    # ---- Stage 6 ---------------------------------------------------------
    def open_jobs(self) -> dict[str, EnrichedJob]:
        return {k: j for k, j in self.jobs.items() if self.status[k]["status"] != "Completed"}

    def ranking(self) -> tuple[list[str], list[ReasoningTrace]]:
        out = stage6_rank(list(self.open_jobs().values()))
        return list(out.result.review_band), list(out.traces)

    def display(self) -> tuple[list[str], list[ReasoningTrace], dict[str, DisplayRow]]:
        """The ranking, then the coordinator's pins laid over it for display."""
        band, traces = self.ranking()
        jobs = self.open_jobs()
        safety = {t.job_id: t.safety_level for t in traces}
        # When the report entered the app, not the tenant's reported time: a form uploaded today can
        # carry last week's date and still be a new arrival. No fallback: apply_pins raises if absent.
        arrived = {t.job_id: datetime.fromisoformat(self.submitted[jobs[t.job_id].parent_report_id]["at"])
                   for t in traces if jobs[t.job_id].parent_report_id in self.submitted}
        rows = apply_pins([t.job_id for t in traces], self.pins, safety, arrived)
        return band, traces, {r.job_id: r for r in rows}

    def unpin(self, job_id: str, actor: str, event: str) -> None:
        """Remove a pin, if there is one, and record why in the pin history and the audit trail."""
        if self.pins.pop(job_id, None) is None:
            return
        community = next(h["community"] for h in reversed(self.pin_history) if h["job_id"] == job_id)
        self.pin_history.append({"event": event, "job_id": job_id, "community": community, "by": actor, "at": _now().isoformat()})
        self.log(job_id, actor, "Unpinned", "" if event == "unpinned" else event.replace("-", " "))


_lock = threading.Lock()
_workspace: Optional[Workspace] = None


def ws() -> Workspace:
    global _workspace
    if _workspace is None:
        _workspace = Workspace()
    return _workspace


# ---------------------------------------------------------------------------
# JSON shaping (display only)
# ---------------------------------------------------------------------------

def distance_of(job: EnrichedJob) -> dict:
    """Stage 5's straight-line km to the nearest housing office; both None if the community is unlisted."""
    return {"km": job.distance_cost_km, "office": job.nearest_office}


def days_waiting(job: EnrichedJob) -> int:
    return max(0, (_now() - job.original_report_timestamp).days)


SAFETY_NAMES = {2: "active", 1: "conditional", 0: "none"}


def trades_needed(w: Workspace, job: EnrichedJob, choice: Optional[str] = None) -> tuple[list[str], Optional[str]]:
    """The trade(s) a job needs and who decided: Stage 5 for a listed fault, otherwise the
    coordinator's choice (an unlisted fault such as an air conditioner has no trade on the list)."""
    if job.required_trades:
        return list(job.required_trades), "pipeline"
    chosen = choice or w.status[job.request_id].get("trade")
    return ([chosen], "coordinator") if chosen else ([], None)


def recommend_tradies(w: Workspace, job: EnrichedJob, choice: Optional[str] = None) -> list[dict]:
    """Who the coordinator might send to THIS job. Display only: it ranks tradies for one job,
    never jobs against each other, and the coordinator makes the choice.

    Order: qualified for Stage 5's required trade and available first, then anyone already going
    to this community (one trip, two jobs), then nearest home base, then lightest workload.
    """
    needed, _ = trades_needed(w, job, choice)
    rows = []
    for t in w.tradies.values():
        assigned = [k for k, s in w.status.items()
                    if s.get("tradie_id") == t["id"] and s["status"] == "Assigned" and k in w.jobs]
        same_trip = [k for k in assigned if k != job.request_id and w.jobs[k].community == job.community]
        qualified = not needed or bool(set(needed) & set(t["trades"]))
        km = km_between(t["base"], job.community)
        reasons = [("Qualified: " + ", ".join(sorted(set(needed) & set(t["trades"])))) if needed and qualified
                   else ("Trade not confirmed yet" if not needed else "Not listed for " + " or ".join(needed)),
                   "Available" if t["available"] else "Unavailable",
                   f"≈{km:g} km from {t['base']}" if km is not None else f"Based in {t['base']}",
                   f"{len(assigned)} assigned job(s)"]
        if same_trip:
            reasons.append("Already going to " + job.community + " (" + ", ".join(same_trip) + ")")
        rows.append({**t, "qualified": qualified, "km": km, "workload": len(assigned), "same_trip": same_trip,
                     "reasons": reasons, "assigned_here": w.status[job.request_id].get("tradie_id") == t["id"]})
    rows.sort(key=lambda r: (not r["qualified"], not r["available"], not r["same_trip"],
                             r["km"] if r["km"] is not None else 10**6, r["workload"]))
    for r in rows:
        r["recommended"] = False
    # No recommendation until the trade is known (from Stage 5, or the coordinator's choice).
    top = next((r for r in rows if r["qualified"] and r["available"]), None) if needed else None
    if top:
        top["recommended"] = True
    return rows


# ---- Plain wording (display only: each sentence restates one value the trace already holds) ----

REPAIR_TYPE = {"dangerous": "Emergency", "standard": "General"}
# Keyed by evaluation's own reason constants, so a new reason raises instead of showing words
# nobody wrote. Each says what the report mentions, never what exists.
LOSS_SENTENCES = {
    NO_ALTERNATIVE: "The report doesn't mention a working alternative, so it's treated as a full loss of use.",
    ALTERNATIVE: "The report mentions a working alternative, so it isn't treated as a full loss of use.",
    SIGN: "The report describes a sign of the fault rather than the fault itself, so it isn't treated as a full loss of use.",
    DEGRADED: "This kind of fault leaves it still working, so it isn't treated as a full loss of use.",
}
SAFETY_SENTENCES = {2: "The report describes a safety risk now.", 1: "The report describes a possible safety risk.",
                    0: "No safety risk described."}
SAFETY_PLAIN = {2: "Safety risk now", 1: "Possible safety risk", 0: "No safety risk described"}
NOT_LISTED = "This fault isn't on the NT repair lists. A coordinator needs to choose how to treat it."
FLAG_PLAIN = {NO_TIER_FLAG: "Not on the NT repair lists: a coordinator needs to choose its repair type"}


# evaluation.lookup_tier's several-matches flag; same fact, the repair type named in plain words.
_HIGHEST_TIER = re.compile(r"Scored at the highest tier among them \((dangerous|standard)\)\.")


NO_FIT = {NO_FIT_EMERGENCY: "emergency", NO_FIT_GENERAL: "general"}


def call_text(call: TierCall) -> str:
    """A coordinator's repair-type call in words. A "no listed fault fits" call names no list
    entry and cites neither the Act nor nt.gov.au: the coordinator is the authority."""
    repair = REPAIR_TYPE[TIER_TABLE[call.fault_name].tier]
    if call.fault_name in NO_FIT:
        return f"Coordinator's call: {repair} repair — no listed fault fits. Reason: {call.reason.rstrip('.')}."
    return f"Treated like: {call.fault_name} ({repair}) — coordinator's call: {call.reason.rstrip('.')}."


def plain_flag(flag: str) -> str:
    """A pipeline flag in the screen's vocabulary. The fact it states is unchanged."""
    flag = FLAG_PLAIN.get(flag, flag)
    return _HIGHEST_TIER.sub(lambda m: f"Treated as the higher repair type among them ({REPAIR_TYPE[m.group(1)]}).", flag)


def source_sentence(tier: str, source: str) -> str:
    """One authority behind a listed fault, in words. Only the Act is called law (nt.gov.au is guidance)."""
    if source.startswith("RTA "):
        return f"Emergency repair under NT law (Residential Tenancies Act {source.removeprefix('RTA ')})."
    if source == "nt.gov.au":
        return f"{REPAIR_TYPE[tier]} repair in {SOURCE_LABELS[source]}."
    raise ValueError(f"no wording for source {source!r}")


def is_date_only(report: Optional[Report]) -> bool:
    """True when the report gave a date but no time (a GEHSF03 "previously reported to DIPL" date)."""
    # geh_form's own timestamp_source wording; midnight there is a placeholder, not a reported time.
    return bool(report and report.timestamp_source and report.timestamp_source.startswith("date previously reported to DIPL"))


def reported_when(when: datetime, date_only: bool) -> str:
    """"22 Sep, 9:15", or "22 Sep" alone when no time was reported: never a time nobody gave."""
    t = when.astimezone(NT_TIME)
    return f"{t.day} {t:%b}" if date_only else f"{t.day} {t:%b}, {t.hour}:{t:%M}"


def reported_sentence(when: datetime, date_only: bool) -> str:
    return f"Reported {reported_when(when, date_only)} — when everything above is equal, the earlier report goes first."


def waiting_sentence(job: EnrichedJob, date_only: bool) -> str:
    # Master §4.6: waiting time is what shows a remote job stuck awaiting a decision.
    days = days_waiting(job)
    return f"Reported {reported_when(job.original_report_timestamp, date_only)} — waiting {days} day{'' if days == 1 else 's'}"


def why_here(job: EnrichedJob, trace: Optional[ReasoningTrace], row: Optional[DisplayRow],
             call: Optional[TierCall], date_only: bool) -> list[str]:
    """The "Why it's here" box: a headline, then one fixed sentence per value in the trace.
    A job that needs a decision has no trace, so it gets only what its own record says."""
    if trace is None:
        return [NOT_LISTED, SAFETY_SENTENCES[to_rank_input(job).safety_level], waiting_sentence(job, date_only)]
    lines = [f"#{row.display_position if row else trace.position} in the queue."]
    if row and row.pin:
        moved = ("Moved up from" if row.display_position < trace.position
                 else "Moved down from" if row.display_position > trace.position else "Held at")
        lines.append(f"{moved} #{trace.position} by a coordinator (reason: {row.pin.reason_tag}).")
    if call:
        # The coordinator's call, not the lists: an air conditioner treated like a stove is not
        # an emergency repair under the Act, so no source sentence is shown for it.
        lines.append(call_text(call))
    elif trace.tier is None:
        lines.append(NOT_LISTED)
    else:
        lines += [source_sentence(trace.tier, src) for src in TIER_TABLE[trace.tier_entry].sources]
    lines += [LOSS_SENTENCES[reason] for reason in trace.tally_reasons]
    lines.append(SAFETY_SENTENCES[trace.safety_level])
    lines.append(reported_sentence(trace.original_timestamp, date_only))
    return lines


def row_reason(job: EnrichedJob, trace: Optional[ReasoningTrace], call: Optional[TierCall]) -> Optional[str]:
    """The one-line reason under a queue row (master §5.1: explanation visible by default).
    Each is a short form of a line in the job's own "Why it's here" box, never a new claim."""
    if job.in_review_band:
        return "Needs a decision"
    if trace is None:  # completed: no longer in the queue
        return None
    if trace.safety_level:  # safety outranks everything else, so it is the reason when present
        return SAFETY_PLAIN[trace.safety_level]
    if call:
        return call_text(call) if call.fault_name in NO_FIT else f"Treated like: {call.fault_name} ({REPAIR_TYPE[trace.tier]})"
    if any(src.startswith("RTA ") for src in TIER_TABLE[trace.tier_entry].sources):
        return "Emergency repair under NT law"
    return f"{REPAIR_TYPE[trace.tier]} repair (NT Government guidance)"


def plain_decided_by(text: str) -> str:
    """ranking's own reason, with its two tier words in plain language (the rule is unchanged)."""
    return (text.replace("untiered safety job above, needs a human tier call first",
                         "safety job above that is not on the repair lists and needs a coordinator decision first")
                .replace("both untiered", "both not on the repair lists"))


def pin_view(pin: Pin) -> dict:
    return {**pin.model_dump(mode="json"), "age_days": max(0, (_now() - pin.at).days)}


def job_summary(w: Workspace, job: EnrichedJob, trace: Optional[ReasoningTrace] = None,
                row: Optional[DisplayRow] = None) -> dict:
    report = w.repo.get(job.parent_report_id)
    level = to_rank_input(job).safety_level
    verified_fault = next((s.text for s in job.spans if s.verified and s.field == "fault_description"), None)
    status = w.status[job.request_id]
    return {
        "job_id": job.request_id,
        "report_id": job.parent_report_id,
        "community": job.community,
        "region": report.region if report else None,
        "source_tag": report.source_tag.value if report else None,
        "source_file": report.source_file if report else None,
        "source_item": report.source_item if report else None,
        "fault": verified_fault or job.fault_description,
        "fault_name": job.tier_entry or (job.taxonomy_match[0] if job.taxonomy_match else None),
        "tier": job.tier,
        "base_points": job.base_points,
        "severity_bump": job.severity_bump,
        "urgency_tally": job.urgency_tally,
        "safety_level": level,
        "safety_name": SAFETY_NAMES[level],
        "flags": [plain_flag(f) for f in list(job.flags) + (list(trace.flags[len(job.flags):]) if trace else [])],
        "original_timestamp": job.original_report_timestamp.isoformat(),
        "date_only": is_date_only(report),
        "days_waiting": days_waiting(job),
        "distance": distance_of(job),
        # row: the coordinator's view with pins; without one, the system position.
        "position": row.display_position if row else trace.position if trace else None,
        "system_position": trace.position if trace else None,
        "pinned": bool(row and row.pinned),
        "pin": pin_view(row.pin) if row and row.pin else None,
        "arrived_above_pin": bool(row and row.arrived_above_pin),
        "decided_by": "coordinator pin" if row and row.pinned else trace.decided_by if trace else None,
        "reason": row_reason(job, trace, w.tier_calls.get(job.request_id)),
        "in_review_band": job.in_review_band,
        "needed_trades": list(job.required_trades),
        "tier_call": w.tier_calls[job.request_id].model_dump(mode="json") if job.request_id in w.tier_calls else None,
        **status,
        "tradie": w.tradies[status["tradie_id"]]["name"] if status["tradie_id"] else None,
    }


def move_range(traces: list[ReasoningTrace], job_id: str) -> Optional[dict]:
    """The first and last slot a pin may use for this job: its own safety group (master §3.4)."""
    if job_id not in {t.job_id for t in traces}:
        return None
    k = sum(t.safety_level >= 1 for t in traces)
    safety = next(t.safety_level for t in traces if t.job_id == job_id) >= 1
    return {"top": 1, "last": k} if safety else {"top": k + 1, "last": len(traces)}


def trace_rows(trace: ReasoningTrace, row: Optional[DisplayRow], call: Optional[TierCall], date_only: bool) -> list[dict]:
    rows: list[dict] = []
    if call:
        rows.append({"label": "Coordinator's call" if call.fault_name in NO_FIT else "Treated like",
                     "value": call_text(call) if call.fault_name in NO_FIT
                     else f"{call.fault_name} ({REPAIR_TYPE[trace.tier]}) — coordinator's call: {call.reason}",
                     "note": f"{call.by}, {call.at.astimezone(NT_TIME):%d %b %Y}"})
    if trace.tier is None:
        rows.append({"label": "Repair type", "value": "Not on the repair lists", "note": "Choose repair type"})
    else:
        sources = "; ".join(_source_label(s) for s in TIER_TABLE[trace.tier_entry].sources)
        rows += [
            {"label": "Repair list match",
             "value": "none — coordinator's call" if TIER_TABLE[trace.tier_entry].coordinator_only else trace.tier_entry,
             "note": "no listed fault fits" if TIER_TABLE[trace.tier_entry].coordinator_only else "name on the NT repair lists"},
            {"label": "Repair type", "value": REPAIR_TYPE[trace.tier], "note": sources},
            {"label": "Base points", "value": str(trace.base_points), "note": "from the repair type, not the text"},
            {"label": "Severity bump", "value": f"+{trace.severity_bump}", "note": "; ".join(trace.tally_reasons)},
            {"label": "Urgency score", "value": str(trace.urgency_tally), "note": f"{trace.base_points} + {trace.severity_bump}"},
        ]
    rows += [
        {"label": "Safety", "value": f"{SAFETY_PLAIN[trace.safety_level]} (level {trace.safety_level})", "note": trace.safety_reason},
        {"label": "Required trade", "value": ", ".join(trace.required_trades) or "not confirmed", "note": "Shown for planning — never changes the order"},
        {"label": "Distance", "value": f"{trace.distance_km:g} km" if trace.distance_km is not None else "unknown",
         "note": f"straight line to {trace.nearest_office}, not in sort key" if trace.nearest_office else "community not in the table, not in sort key"},
        *([{"label": "Second check (independent reader)", "value": trace.second_reader.status,
            "note": "; ".join(trace.second_reader.flags) or "no disagreement"}] if trace.second_reader else []),
        *([{"label": "Second check (safety re-read)", "value": "yes", "note": "read again after the independent reader disagreed on safety"}] if trace.reread else []),
        {"label": "Reported", "value": f"{trace.original_timestamp.astimezone(NT_TIME):%d %b %Y}" + ("" if date_only else f", {trace.original_timestamp.astimezone(NT_TIME):%H:%M}"), "note": "When everything above is equal, the earlier report goes first. Never overwritten."},
        {"label": "Queue place", "value": f"{trace.position} of {trace.queue_length}", "note": plain_decided_by(trace.decided_by)},
        *([{"label": "Coordinator pin", "value": f"{row.display_position} of {trace.queue_length}",
            "note": f"coordinator pin ({row.pin.reason_tag}); system position {trace.position}"}] if row and row.pin else []),
    ]
    return rows


def facts_view(facts: ExtractedFacts) -> list[dict]:
    return [
        {"field": "Fault list match", "value": ", ".join(facts.taxonomy_match) or "none (not on the list)"},
        {"field": "Another working one named", "value": "yes" if facts.alternative_mentioned else "no"},
        {"field": "Coping described", "value": "yes" if facts.coping_mentioned else "no"},
        {"field": "Hazard", "value": facts.hazard_status + (f" ({facts.mechanism_type})" if facts.mechanism_type else "")},
        {"field": "Harm claimed", "value": "yes" if facts.harm_claimed else "no"},
        {"field": "Fault or sign", "value": facts.fault_or_sign},
        {"field": "Getting worse", "value": "yes" if facts.worsening_mentioned else "no"},
        {"field": "Comes and goes", "value": "yes" if facts.impact_status == "intermittent" else "no"},
    ]


def _plain_failure(errors: list[str]) -> str:
    """Why a reading was not used, in words a coordinator can act on."""
    last = errors[-1] if errors else ""
    if "RecordingMissing" in last:
        return "The model has not read this text: set TRIAGE_LIVE=1 with a funded key, or a person reads it."
    if "not on the fault list" in last:
        return "The model named a fault that isn't on the list (twice), so its answer was not used."
    if "ValidationError" in last or "JSON" in last:
        return "The model's answer was incomplete or invalid (twice), so it was not used."
    return "The model could not be reached (twice), so a person needs to read this report."


PRIVACY_NOTE = {
    "form": "Name, phone, email, address and the tenant's own Immediate/Urgent/Routine rating were kept out of the text sent to the model.",
    "text": "Only the message text is sent to the model. A tenant reference is stored as a one-way code, never as entered.",
}


def stage_view(w: Workspace, report_id: str) -> dict:
    """What each stage produced for one report, ending with its Stage 6 result."""
    report, result = w.repo.get(report_id), w.results[report_id]
    band, traces, rows = w.display()
    by_id = {t.job_id: t for t in traces}
    jobs = [j for j in w.jobs.values() if j.parent_report_id == report_id]
    out_jobs = []
    for job in jobs:
        trace, row = by_id.get(job.request_id), rows.get(job.request_id)
        sms, why = tenant_sms(job), tenant_why(job, pinned=bool(row and row.pinned),
                                               classified_by_coordinator=job.request_id in w.tier_calls)
        tier_sources = "; ".join(_source_label(s) for s in TIER_TABLE[job.tier_entry].sources) if job.tier_entry else None
        top = next((r for r in recommend_tradies(w, job) if r["recommended"]), None)
        out_jobs.append({
            **job_summary(w, job, trace, row),
            "stage3": [s.model_dump() for s in job.spans],
            "stage4": {"tier": job.tier, "tier_entry": job.tier_entry, "tier_sources": tier_sources,
                       "base_points": job.base_points, "severity_bump": job.severity_bump,
                       "urgency_tally": job.urgency_tally, "tally_reasons": list(job.tally_reasons),
                       "safety_level": to_rank_input(job).safety_level, "safety_reason": job.safety_reason},
            "stage5": {"required_trades": job.required_trades, "distance": distance_of(job),
                       "recommended_tradie": top["name"] if top else None},  # None until the trade is known
            "stage6": {"in_review_band": job.request_id in band, "position": row.display_position if row else None,
                       "queue_length": trace.queue_length if trace else None,
                       "decided_by": plain_decided_by(trace.decided_by) if trace else "Needs a decision: choose repair type",
                       "trace": trace_rows(trace, row, w.tier_calls.get(job.request_id), is_date_only(report)) if trace else None,
                       "why_here": why_here(job, trace, row, w.tier_calls.get(job.request_id), is_date_only(report)),
                       "sms": sms, "why": why},
        })
    return {
        "report_id": report_id,
        "stage1": {"request_id": report.request_id, "tenant_id": report.tenant_id, "source_tag": report.source_tag.value,
                   "community": report.community, "region": report.region,
                   "original_report_timestamp": report.original_report_timestamp.isoformat(),
                   "date_only": is_date_only(report), "timestamp_source": report.timestamp_source, "source_file": report.source_file,
                   "source_item": report.source_item, "raw_text": report.raw_text,
                   "privacy": PRIVACY_NOTE["form" if report.source_item is not None else "text"]},
        "stage2": {"status": result.status.value, "extractor": result.extractor, "attempts": result.attempts,
                   "problem": None if result.status is ExtractionStatus.OK else
                   ("No fault named: contact the tenant" if result.status is ExtractionStatus.NO_FAULT_NAMED else _plain_failure(result.errors)),
                   "sms": None if result.status is ExtractionStatus.OK else tenant_sms(result),
                   "faults": [{"fault_description": f.fault_description, "facts": facts_view(f)}
                              for f in (result.extraction.faults if result.extraction else [])]},
        "jobs": out_jobs,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="FairFix NT")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_sessions: dict[str, str] = {}  # session token -> username, held on the server so a role can't be forged


def require(request: Request, *roles: str) -> dict:
    """The signed-in user, or 401. With roles given, 403 for anyone else."""
    username = _sessions.get(request.cookies.get("ff_session", ""))
    if username is None:
        raise HTTPException(401, "Please sign in")
    user = {"username": username, "role": USERS[username]["role"], "name": USERS[username]["name"]}
    if roles and user["role"] not in roles:
        raise HTTPException(403, "Your role can't do this")
    return user


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(data: LoginIn) -> JSONResponse:
    username = data.username.strip().lower()
    account = USERS.get(username)
    if account is None or not secrets.compare_digest(account["password"], data.password):
        raise HTTPException(401, "Wrong username or password")
    token = secrets.token_urlsafe(24)
    _sessions[token] = username
    resp = JSONResponse({"username": username, "role": account["role"], "name": account["name"]})
    resp.set_cookie("ff_session", token, httponly=True, samesite="lax")
    return resp


@app.post("/api/logout")
async def logout(request: Request) -> JSONResponse:
    _sessions.pop(request.cookies.get("ff_session", ""), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ff_session")
    return resp


@app.get("/api/me")
async def me(request: Request) -> dict:
    return require(request)


@app.get("/api/reference")
async def reference(request: Request) -> dict:
    require(request)
    with _lock:
        w = ws()
        communities = sorted(set(community_names()) | {j.community for j in w.jobs.values()})
        return {"communities": communities, "fault_names": list(FAULT_NAMES), "mode": w.client.describe(),
                "trades": list(ALL_TRADES), "reason_tags": list(REASON_TAGS),
                "repair_list": [{"name": e.name, "repair_type": REPAIR_TYPE[e.tier], "coordinator_only": e.coordinator_only,
                                 "sources": [_source_label(s) for s in e.sources]} for e in TIER_TABLE.values()]}


@app.get("/api/queue")
async def queue(request: Request) -> dict:
    require(request, "admin")
    with _lock:
        w = ws()
        band, traces, rows = w.display()
        by_id = {t.job_id: t for t in traces}
        needs_human = []
        for rid, res in w.results.items():
            if res.status is ExtractionStatus.OK:
                continue
            report = w.repo.get(rid)
            reason = ("No fault named: contact the tenant" if res.status is ExtractionStatus.NO_FAULT_NAMED
                      else _plain_failure(res.errors))
            needs_human.append({"report_id": rid, "community": report.community, "raw_text": report.raw_text,
                                "reason": reason, "source_file": report.source_file, "source_item": report.source_item,
                                "days_waiting": max(0, (_now() - report.original_report_timestamp).days)})
        jobs = w.open_jobs()
        return {
            "mode": w.client.describe(),
            # Display order: pins over the system order; each row carries both positions.
            "ranked": [job_summary(w, jobs[r.job_id], by_id[r.job_id], r)
                       for r in sorted(rows.values(), key=lambda r: r.display_position)],
            "pin_count": sum(r.pinned for r in rows.values()),
            "review_band": [job_summary(w, jobs[j]) for j in band],
            "needs_human": needs_human,
            "completed": [job_summary(w, j) for k, j in w.jobs.items() if w.status[k]["status"] == "Completed"],
            "skipped": w.skipped,
        }


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str, request: Request, trade: Optional[str] = None) -> dict:
    require(request, "admin")
    if trade is not None and trade not in ALL_TRADES:
        raise HTTPException(400, f"Unknown trade: {trade}")
    with _lock:
        w = ws()
        job = w.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        needed, trade_source = trades_needed(w, job, trade)
        band, traces, rows = w.display()
        trace, row = next((t for t in traces if t.job_id == job_id), None), rows.get(job_id)
        report = w.repo.get(job.parent_report_id)
        result = w.results[job.parent_report_id]
        same_community = [j for k, j in w.open_jobs().items() if j.community == job.community]
        others = [j.request_id for j in same_community if j.request_id != job_id]
        oldest = max((days_waiting(j) for j in same_community), default=0)
        sms, why = tenant_sms(job), tenant_why(job, pinned=bool(row and row.pinned),
                                               classified_by_coordinator=job_id in w.tier_calls)
        return {
            **job_summary(w, job, trace, row),
            "raw_text": report.raw_text,
            "timestamp_source": report.timestamp_source,
            "extractor": result.extractor,
            "facts": facts_view(w.facts[job_id]),
            "spans": [s.model_dump() for s in job.spans],
            "trace": trace_rows(trace, row, w.tier_calls.get(job_id), is_date_only(report)) if trace else None,
            "why_here": why_here(job, trace, row, w.tier_calls.get(job_id), is_date_only(report)),
            "review_reason": NOT_LISTED if job.in_review_band else None,
            "sms": sms,
            "why": why,
            "logistics": {
                "distance": distance_of(job),
                "needed_trades": needed,
                "trade_source": trade_source,
                "shared_trip": others,
                "community_line": f"{job.community}: {len(same_community)} open job(s), oldest waiting {oldest} day(s)",
            },
            "recommendations": recommend_tradies(w, job, trade),
            "move_range": move_range(traces, job_id),
            "audit": [a for a in w.audit if a["job_id"] in (job_id, job.parent_report_id)][::-1],
        }


class TextReportIn(BaseModel):
    community: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_tag: Literal["officer", "tenant_direct"] = "officer"
    reported_at: Optional[datetime] = None
    tenant_ref: str = ""


@app.post("/api/reports/text")
async def add_text_report(data: TextReportIn, request: Request) -> dict:
    """A phone call or message typed in by an officer: Stage 1 builds the report from it."""
    user = require(request, "officer", "admin")
    with _lock:
        w = ws()
        when = data.reported_at or _now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=NT_TIME)  # a time with no zone is NT time
        ref = data.tenant_ref.strip()
        # Pseudonymous: the same tenancy reference always gives the same id; the reference itself is not stored.
        tenant = "T-" + (hashlib.sha256(ref.lower().encode()).hexdigest()[:8].upper() if ref else secrets.token_hex(4).upper())
        report = create_report(tenant_id=tenant, raw_text=data.message.strip(), source_tag=SourceTag(data.source_tag),
                               community=data.community.strip(), original_report_timestamp=when,
                               timestamp_source="reported time entered at intake")
        rid = w.add_report(report, by=user["name"])
        return {"reports": [stage_view(w, rid)] if rid else []}


@app.post("/api/reports/upload")
async def upload_report(request: Request, file: UploadFile = File(...)) -> dict:
    """An uploaded document: a GEHSF03 form (one report per issue row) or a .txt report."""
    user = require(request, "officer", "admin")
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in (".pdf", ".txt"):
        raise HTTPException(400, "Upload a GEHSF03 PDF form or a .txt report")
    content = await file.read()
    if not content:
        raise HTTPException(400, "The file is empty")
    digest = hashlib.sha256(content).hexdigest()
    with _lock, tempfile.TemporaryDirectory() as tmp:
        w = ws()
        # Exact bytes only, whatever the file name: a different file about the same fault is a
        # genuine report and goes through (duplicates are flagged, never merged — master C4).
        if digest in w.uploads:
            raise HTTPException(409, "This file was already uploaded (reports "
                                     + ", ".join(w.uploads[digest]) + "). Use Follow-up on those jobs instead.")
        (Path(tmp) / name).write_bytes(content)
        before = len(w.skipped)
        ids = w.load_folder(Path(tmp), by=user["name"])
        problems = w.skipped[before:]
        if not ids:
            raise HTTPException(400, problems[0]["reason"] if problems else "No report found in that file")
        w.uploads[digest] = ids
        w.save()
        return {"reports": [stage_view(w, rid) for rid in ids]}


@app.get("/api/my-reports")
async def my_reports(request: Request) -> list[dict]:
    """What the officer recorded: references and progress only, never the priority."""
    user = require(request, "officer", "admin")
    with _lock:
        w = ws()
        out = []
        for rid, sub in w.submitted.items():
            if sub["by"] != user["name"] or w.repo.get(rid) is None:
                continue
            report, res = w.repo.get(rid), w.results[rid]
            jobs = [{"job_id": k, "status": w.status[k]["status"],
                     "tradie": w.tradies[w.status[k]["tradie_id"]]["name"] if w.status[k]["tradie_id"] else None}
                    for k, j in w.jobs.items() if j.parent_report_id == rid]
            out.append({"report_id": rid, "community": report.community, "submitted_at": sub["at"],
                        "raw_text": report.raw_text, "read": res.status is ExtractionStatus.OK, "jobs": jobs})
        return out[::-1]


class FollowupIn(BaseModel):
    text: str = Field(min_length=1)


@app.post("/api/jobs/{job_id}/followup")
async def followup(job_id: str, data: FollowupIn, request: Request) -> dict:
    """Escalation: matched by exact id, the report is re-read, the original timestamp is kept."""
    user = require(request, "admin")
    with _lock:
        w = ws()
        job = w.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        try:
            report, result = escalate(w.repo, job_id, data.text.strip(), _now(), w.client)
        except UnknownRequestError:
            raise HTTPException(404, "No report with that id")
        read_ok = result.status is ExtractionStatus.OK and result.extraction is not None
        child = w.repo.get_child(job_id)
        if child is not None:
            # escalate() already re-matched this child (or flagged it); rebuild only this job.
            if read_ok:
                w.results[report.request_id] = result
                w.repo.save_result(result)
            w.store_job(enrich_fault(report, child.facts, child.job_id, child.flags), child.facts)
        elif read_ok:
            w.repo.save_result(result)
            w._apply(report, result)
        else:
            # Never drop a job because its follow-up could not be read: keep its last reading and ask a human.
            w.extra_flags.setdefault(job_id, []).append(
                f'Follow-up could not be read automatically, review it: "{data.text.strip()}"')
            w.store_job(enrich_fault(report, w.facts[job_id], job_id), w.facts[job_id])
        # A re-read that splits one fault into several makes new job ids; the old job's pin goes with it.
        for pinned_id in [k for k in w.pins if k not in w.jobs]:
            w.unpin(pinned_id, user["name"], "replaced-by-reread")
        w.log(job_id, "tenant", "Follow-up received", f'"{data.text.strip()}" (read by {result.extractor}, {result.status.value})')
        w.save()
        return {"ok": True, "status": result.status.value, "extractor": result.extractor}


@app.get("/api/tier-calls")
async def previous_tier_calls(request: Request) -> list[dict]:
    """Every coordinator repair-type call so far, newest first. Read-only reference: nothing
    here is ever pre-selected, suggested or applied to another job."""
    require(request, "admin")
    with _lock:
        w = ws()
        rows = []
        for job_id, call in sorted(w.tier_calls.items(), key=lambda kv: kv[1].at, reverse=True):
            job = w.jobs.get(job_id)
            fault = (next((s.text for s in job.spans if s.verified and s.field == "fault_description"), None)
                     or job.fault_description) if job else None
            rows.append({"job_id": job_id, "fault": fault, "job_open": job is not None, "treated_like": call.fault_name,
                         "no_fit": call.fault_name in NO_FIT,
                         "repair_type": REPAIR_TYPE[TIER_TABLE[call.fault_name].tier], "by": call.by,
                         "reason": call.reason, "at": call.at.isoformat()})
        return rows


class TierCallIn(BaseModel):
    fault_name: str
    reason: str = Field(min_length=3)


@app.post("/api/jobs/{job_id}/tier")
async def tier_call(job_id: str, data: TierCallIn, request: Request) -> dict:
    """The coordinator's tier call for an untiered job: it then enters the sort normally."""
    user = require(request, "admin")
    with _lock:
        w = ws()
        job = w.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        if job.urgency_tally is not None:
            raise HTTPException(400, "This job already has a repair type from the NT repair lists")
        if data.fault_name not in TIER_TABLE:
            raise HTTPException(400, "Choose a fault from the list")
        w.tier_calls[job_id] = TierCall.model_validate({"fault_name": data.fault_name, "reason": data.reason.strip(),
                                                        "by": user["name"], "at": _now(), "community": job.community})
        w.store_job(job, w.facts[job_id])
        w.log(job_id, user["name"], "Tier call", f'Counted as "{data.fault_name}" ({TIER_TABLE[data.fault_name].tier}). Reason: {data.reason.strip()}')
        w.save()
        return {"ok": True}


class PinIn(BaseModel):
    # Plain int and str, checked below, so a bad value gets a 400 with a reason, not a 422.
    model_config = ConfigDict(extra="forbid")  # a note is refused: tags only (master §5.1)
    target_position: int
    reason_tag: str


@app.post("/api/jobs/{job_id}/pin")
async def pin_job(job_id: str, data: PinIn, request: Request) -> dict:
    """The coordinator's override: show this job at a chosen position. Display only, never ranking."""
    user = require(request, "admin")
    with _lock:
        w = ws()
        job = w.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        if data.target_position < 1:
            raise HTTPException(400, "Position must be 1 or more")
        if data.reason_tag not in REASON_TAGS:
            raise HTTPException(400, "Choose a reason: " + ", ".join(REASON_TAGS))
        band, traces = w.ranking()
        if job_id in band:
            raise HTTPException(400, "Choose repair type first")
        if w.status[job_id]["status"] != "Open":
            raise HTTPException(400, "Only an open job that has not been dispatched can be pinned")
        system = [t.job_id for t in traces]
        if pin_breaks_safety(system, {t.job_id: t.safety_level for t in traces}, job_id, data.target_position):
            raise HTTPException(400, SAFETY_BLOCK)
        at, system_position = _now(), system.index(job_id) + 1
        w.pins[job_id] = Pin.model_validate({"job_id": job_id, "target_position": data.target_position,
                                             "reason_tag": data.reason_tag, "by": user["name"], "at": at,
                                             "system_position_at_pin": system_position})
        direction = "up" if data.target_position < system_position else "down" if data.target_position > system_position else "same"
        w.pin_history.append({"event": "pinned", "job_id": job_id, "community": job.community, "by": user["name"],
                              "at": at.isoformat(), "tag": data.reason_tag, "target": data.target_position,
                              "system_position": system_position, "direction": direction})
        w.log(job_id, user["name"], f"Pinned #{data.target_position} (system #{system_position})", data.reason_tag)
        w.save()
        return {"ok": True}


@app.delete("/api/jobs/{job_id}/pin")
async def unpin_job(job_id: str, request: Request) -> dict:
    """Remove the coordinator's pin; the job goes back to its system position."""
    user = require(request, "admin")
    with _lock:
        w = ws()
        if job_id not in w.pins:
            raise HTTPException(404, "This job is not pinned")
        w.unpin(job_id, user["name"], "unpinned")
        w.save()
        return {"ok": True}


class AssignIn(BaseModel):
    tradie_id: int
    note: str = Field(min_length=3)
    trade: Optional[str] = None  # the coordinator's call, used only when Stage 5 has no trade


@app.post("/api/jobs/{job_id}/assign")
async def assign(job_id: str, data: AssignIn, request: Request) -> dict:
    """The coordinator sends a tradie. The recommendation is advice; this is the decision."""
    user = require(request, "admin")
    with _lock:
        w = ws()
        job = w.jobs.get(job_id)
        tradie = w.tradies.get(data.tradie_id)
        if job is None or tradie is None:
            raise HTTPException(404, "No such job or tradie")
        if not tradie["available"]:
            raise HTTPException(400, f"{tradie['name']} is marked unavailable")
        if data.trade is not None and data.trade not in ALL_TRADES:
            raise HTTPException(400, f"Unknown trade: {data.trade}")
        needed, source = trades_needed(w, job, data.trade)
        mismatch = needed and not set(needed) & set(tradie["trades"])
        chosen = needed[0] if source == "coordinator" else None
        w.status[job_id] = {"status": "Assigned", "tradie_id": tradie["id"], "note": data.note.strip(), "trade": chosen}
        w.unpin(job_id, user["name"], "cleared-by-assign")  # master §5.1: a pin lasts until dispatch
        w.log(job_id, user["name"], "Tradie assigned",
              f"{tradie['name']} ({', '.join(tradie['trades'])}, {tradie['base']})"
              + (f" as {chosen} (coordinator's trade call: not on the fault list)" if chosen else "")
              + (f" [not listed for {' or '.join(needed)}]" if mismatch else "") + f". {data.note.strip()}")
        w.save()
        return {"ok": True}


class DecisionIn(BaseModel):
    status: Literal["Open", "Completed"]
    note: str = Field(min_length=3)


@app.post("/api/jobs/{job_id}/decision")
async def decision(job_id: str, data: DecisionIn, request: Request) -> dict:
    """Close a job, or reopen it (which also clears the assigned tradie). Recorded with a note."""
    user = require(request, "admin")
    with _lock:
        w = ws()
        if job_id not in w.jobs:
            raise HTTPException(404, "No job with that id")
        tradie_id = w.status[job_id]["tradie_id"] if data.status == "Completed" else None
        w.status[job_id] = {"status": data.status, "tradie_id": tradie_id, "note": data.note.strip(),
                            "trade": w.status[job_id].get("trade")}
        if data.status == "Completed":
            w.unpin(job_id, user["name"], "cleared-by-completion")
        w.log(job_id, user["name"], "Job completed" if data.status == "Completed" else "Job reopened", data.note.strip())
        w.save()
        return {"ok": True}


@app.get("/api/tradies")
async def list_tradies(request: Request) -> list[dict]:
    require(request, "admin")
    with _lock:
        w = ws()
        return [{**t, "jobs": [{"job_id": k, "community": w.jobs[k].community}
                               for k, s in w.status.items()
                               if s.get("tradie_id") == t["id"] and s["status"] == "Assigned" and k in w.jobs]}
                for t in w.tradies.values()]


class TradieIn(BaseModel):
    name: str = Field(min_length=2)
    trades: list[str] = Field(min_length=1)
    base: str = Field(min_length=2)
    phone: str = ""


@app.post("/api/tradies")
async def add_tradie(data: TradieIn, request: Request) -> dict:
    user = require(request, "admin")
    unknown = [t for t in data.trades if t not in ALL_TRADES]
    if unknown:
        raise HTTPException(400, f"Unknown trade: {', '.join(unknown)}")
    with _lock:
        w = ws()
        tid = max(w.tradies, default=0) + 1
        w.tradies[tid] = {"id": tid, "name": data.name.strip(), "trades": data.trades, "base": data.base.strip(),
                          "phone": data.phone.strip(), "available": True}
        w.log(None, user["name"], "Tradie added", f"{data.name.strip()} ({', '.join(data.trades)}, {data.base.strip()})")
        w.save()
        return {"ok": True, "id": tid}


@app.post("/api/tradies/{tradie_id}/availability")
async def toggle_tradie(tradie_id: int, request: Request) -> dict:
    user = require(request, "admin")
    with _lock:
        w = ws()
        tradie = w.tradies.get(tradie_id)
        if tradie is None:
            raise HTTPException(404, "No such tradie")
        tradie["available"] = not tradie["available"]
        w.log(None, user["name"], "Tradie availability", f"{tradie['name']}: {'available' if tradie['available'] else 'unavailable'}")
        w.save()
        return {"ok": True, "available": tradie["available"]}


@app.get("/api/communities")
async def communities(request: Request) -> list[dict]:
    """Who is waiting where, and what coordinators did by hand in each community. Display only.

    Ranking can't favour town (distance isn't in the sort key; FIFO; property tests), so no
    nearest-first comparison is shown. Pins and repair-type calls are human choices, which is
    where town-first bias could re-enter, so they are counted here for the coordinator to see.
    """
    require(request, "admin")
    with _lock:
        w = ws()
        rows: dict[str, dict] = {}

        def row(community: str) -> dict:
            return rows.setdefault(community, {
                "community": community, "open": 0, "safety": 0, "review_band": 0, "oldest_days": 0,
                "distance": next((distance_of(j) for j in w.jobs.values() if j.community == community),
                                 {"km": None, "office": None}),
                "moved": {"jobs": 0, "repins": 0, "up": 0, "down": 0, "by_tag": dict.fromkeys(REASON_TAGS, 0)},
                "no_fit_calls": {"emergency": 0, "general": 0}})

        for job in w.open_jobs().values():
            c = row(job.community)
            c["open"] += 1
            c["safety"] += to_rank_input(job).safety_level > 0
            c["review_band"] += job.in_review_band
            c["oldest_days"] = max(c["oldest_days"], days_waiting(job))
        # From the append-only history, so pins already cleared still count. One count per pinned
        # job, read from its latest pin; moving the same job again is a re-pin.
        latest: dict[str, dict] = {}
        for h in (h for h in w.pin_history if h["event"] == "pinned"):
            if h["job_id"] in latest:
                row(h["community"])["moved"]["repins"] += 1
            latest[h["job_id"]] = h
        for h in latest.values():
            moved = row(h["community"])["moved"]
            moved["jobs"] += 1
            moved["by_tag"][h["tag"]] += 1
            if h["direction"] != "same":
                moved[h["direction"]] += 1
        # "No listed fault fits" calls, emergency vs general: labelling remote faults general and town
        # faults emergency would show here. Read from the call itself, so a re-read never drops one.
        for call in w.tier_calls.values():
            if call.fault_name in NO_FIT:
                row(call.community)["no_fit_calls"][NO_FIT[call.fault_name]] += 1
        return sorted(rows.values(), key=lambda c: (-c["oldest_days"], c["community"]))


@app.post("/api/reset")
async def reset(request: Request) -> dict:
    """Delete every report, job and decision (the tradie roster returns to its starting five)."""
    user = require(request, "admin")
    global _workspace
    with _lock:
        w = ws()
        w.repo.clear()
        _workspace = Workspace(db_path=w.db_path, client=w.client)
        _workspace.log(None, user["name"], "All data cleared")
        _workspace.save()
        return {"ok": True}


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8040"))
    print(f"FairFix NT: http://127.0.0.1:{port}   (Ctrl+C to stop)  data: {DB_PATH}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
