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
from pydantic import BaseModel, Field

from triage.adapter import to_rank_input
from triage import second_reader
from triage.distances import community_names, km_between
from triage.escalation import escalate
from triage.evaluation import compute_tally, lookup_tier
from triage.explain import ReasoningTrace, tenant_sms, tenant_why
from triage.extraction import OfflineExtractor, OpenAICompatibleClient, configured_model
from triage.intake import DuplicateRequestError, SQLiteReportRepository, UnknownRequestError, create_report
from triage.models import EnrichedJob, ExtractedFacts, ExtractionResult, ExtractionStatus, Report, SourceTag
from triage.pipeline import build_jobs, build_report_jobs, enrich_fault, save_children, stage2_extract, stage6_rank
from triage.recording import RECORDED_DIR, RecordedClient, RecordingClient, RecordingMissing
from triage.report_files import load_reports
from triage.tiers import FAULT_NAMES, TIER_TABLE
from triage.trades import ALL_TRADES, required_trades
from triage.verification import verify_spans

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
NT_TIME = timezone(timedelta(hours=9, minutes=30))
DB_PATH = os.environ.get("FAIRFIX_DB", str(ROOT / "data" / "fairfix.db"))

SOURCE_LABELS = {"nt.gov.au": "NT Government repairs guidance (nt.gov.au)"}

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
    fault_name: str
    reason: str
    by: str
    at: datetime


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
        self.tier_calls = {k: TierCall.model_validate(v) for k, v in state.get("tier_calls", {}).items()}
        self.status: dict[str, dict] = state.get("status", {})
        self.extra_flags: dict[str, list[str]] = state.get("extra_flags", {})
        self.tradies = {int(k): v for k, v in state.get("tradies", {}).items()} or \
            {i: {"id": i, "available": True, **t} for i, t in enumerate(SEED_TRADIES, start=1)}
        self.submitted: dict[str, dict] = state.get("submitted", {})
        self.audit: list[dict] = state.get("audit", [])
        self.skipped: list[dict] = []
        self._restore()

    # ---- persistence -------------------------------------------------------
    def save(self) -> None:
        self.repo.save_state({
            "tier_calls": {k: v.model_dump(mode="json") for k, v in self.tier_calls.items()},
            "status": self.status, "extra_flags": self.extra_flags, "tradies": self.tradies,
            "submitted": self.submitted, "audit": self.audit,
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
        tier = lookup_tier([call.fault_name])
        tally = compute_tally(tier, facts, unverified)
        return EnrichedJob.model_validate({
            **job.model_dump(), "tier": tier.tier, "tier_entry": tally.winner, "base_points": tally.base,
            "severity_bump": tally.bump, "urgency_tally": tally.tally, "tally_reasons": tally.reasons,
            "required_trades": required_trades(tally.winner),
            "flags": (*job.flags, f'Tier set by coordinator as "{call.fault_name}": {call.reason}'),
        })

    def log(self, job_id: Optional[str], actor: str, action: str, note: str = "") -> None:
        self.audit.append({"job_id": job_id, "actor": actor, "action": action, "note": note, "at": _now().isoformat()})

    # ---- Stage 6 ---------------------------------------------------------
    def open_jobs(self) -> dict[str, EnrichedJob]:
        return {k: j for k, j in self.jobs.items() if self.status[k]["status"] != "Completed"}

    def ranking(self) -> tuple[list[str], list[ReasoningTrace]]:
        out = stage6_rank(list(self.open_jobs().values()))
        return list(out.result.review_band), list(out.traces)


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


def job_summary(w: Workspace, job: EnrichedJob, trace: Optional[ReasoningTrace] = None) -> dict:
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
        "flags": list(job.flags) + (list(trace.flags[len(job.flags):]) if trace else []),
        "original_timestamp": job.original_report_timestamp.isoformat(),
        "days_waiting": days_waiting(job),
        "distance": distance_of(job),
        "position": trace.position if trace else None,
        "decided_by": trace.decided_by if trace else None,
        "in_review_band": job.in_review_band,
        "needed_trades": list(job.required_trades),
        "tier_call": w.tier_calls[job.request_id].model_dump(mode="json") if job.request_id in w.tier_calls else None,
        **status,
        "tradie": w.tradies[status["tradie_id"]]["name"] if status["tradie_id"] else None,
    }


def trace_rows(trace: ReasoningTrace) -> list[dict]:
    rows: list[dict] = []
    if trace.tier is None:
        rows.append({"label": "Tier", "value": "untiered", "note": "needs a coordinator tier call"})
    else:
        sources = "; ".join(_source_label(s) for s in TIER_TABLE[trace.tier_entry].sources)
        rows += [
            {"label": "Fault list match", "value": trace.tier_entry, "note": "reference list name"},
            {"label": "Tier", "value": trace.tier, "note": sources},
            {"label": "Base points", "value": str(trace.base_points), "note": "from the tier, not the text"},
            {"label": "Severity bump", "value": f"+{trace.severity_bump}", "note": "; ".join(trace.tally_reasons)},
            {"label": "Urgency score", "value": str(trace.urgency_tally), "note": f"{trace.base_points} + {trace.severity_bump}"},
        ]
    rows += [
        {"label": "Safety level", "value": f"{trace.safety_level} ({SAFETY_NAMES[trace.safety_level]})", "note": trace.safety_reason},
        {"label": "Required trade", "value": ", ".join(trace.required_trades) or "not confirmed", "note": "Stage 5, display only"},
        {"label": "Distance", "value": f"{trace.distance_km:g} km" if trace.distance_km is not None else "unknown",
         "note": f"straight line to {trace.nearest_office}, not in sort key" if trace.nearest_office else "community not in the table, not in sort key"},
        *([{"label": "Second reader", "value": trace.second_reader.status,
            "note": "; ".join(trace.second_reader.flags) or "no disagreement"}] if trace.second_reader else []),
        *([{"label": "Re-read", "value": "yes", "note": "safety re-read after a second-reader flag"}] if trace.reread else []),
        {"label": "Reported", "value": f"{trace.original_timestamp.astimezone(NT_TIME):%d %b %Y, %H:%M}", "note": "first-come order input, never overwritten"},
        {"label": "Position", "value": f"{trace.position} of {trace.queue_length}", "note": trace.decided_by},
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
    band, traces = w.ranking()
    by_id = {t.job_id: t for t in traces}
    jobs = [j for j in w.jobs.values() if j.parent_report_id == report_id]
    out_jobs = []
    for job in jobs:
        trace = by_id.get(job.request_id)
        sms, why = tenant_sms(job), tenant_why(job)
        tier_sources = "; ".join(_source_label(s) for s in TIER_TABLE[job.tier_entry].sources) if job.tier_entry else None
        top = next((r for r in recommend_tradies(w, job) if r["recommended"]), None)
        out_jobs.append({
            **job_summary(w, job, trace),
            "stage3": [s.model_dump() for s in job.spans],
            "stage4": {"tier": job.tier, "tier_entry": job.tier_entry, "tier_sources": tier_sources,
                       "base_points": job.base_points, "severity_bump": job.severity_bump,
                       "urgency_tally": job.urgency_tally, "tally_reasons": list(job.tally_reasons),
                       "safety_level": to_rank_input(job).safety_level, "safety_reason": job.safety_reason},
            "stage5": {"required_trades": job.required_trades, "distance": distance_of(job),
                       "recommended_tradie": top["name"] if top else None},  # None until the trade is known
            "stage6": {"in_review_band": job.request_id in band, "position": trace.position if trace else None,
                       "queue_length": trace.queue_length if trace else None,
                       "decided_by": trace.decided_by if trace else "Held in the review band until a coordinator tier call",
                       "trace": trace_rows(trace) if trace else None, "sms": sms, "why": why},
        })
    return {
        "report_id": report_id,
        "stage1": {"request_id": report.request_id, "tenant_id": report.tenant_id, "source_tag": report.source_tag.value,
                   "community": report.community, "region": report.region,
                   "original_report_timestamp": report.original_report_timestamp.isoformat(),
                   "timestamp_source": report.timestamp_source, "source_file": report.source_file,
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
                "trades": list(ALL_TRADES)}


@app.get("/api/queue")
async def queue(request: Request) -> dict:
    require(request, "admin")
    with _lock:
        w = ws()
        band, traces = w.ranking()
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
            "ranked": [job_summary(w, jobs[t.job_id], t) for t in traces],
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
        band, traces = w.ranking()
        trace = next((t for t in traces if t.job_id == job_id), None)
        report = w.repo.get(job.parent_report_id)
        result = w.results[job.parent_report_id]
        same_community = [j for k, j in w.open_jobs().items() if j.community == job.community]
        others = [j.request_id for j in same_community if j.request_id != job_id]
        oldest = max((days_waiting(j) for j in same_community), default=0)
        sms, why = tenant_sms(job), tenant_why(job)
        return {
            **job_summary(w, job, trace),
            "raw_text": report.raw_text,
            "timestamp_source": report.timestamp_source,
            "extractor": result.extractor,
            "facts": facts_view(w.facts[job_id]),
            "spans": [s.model_dump() for s in job.spans],
            "trace": trace_rows(trace) if trace else None,
            "review_reason": "Not on the fault list and no safety risk described: waiting for a coordinator tier call." if job.in_review_band else None,
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
    with _lock, tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / name).write_bytes(content)
        w = ws()
        before = len(w.skipped)
        ids = w.load_folder(Path(tmp), by=user["name"])
        problems = w.skipped[before:]
        if not ids:
            raise HTTPException(400, problems[0]["reason"] if problems else "No report found in that file")
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
    require(request, "admin")
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
        w.log(job_id, "tenant", "Follow-up received", f'"{data.text.strip()}" (read by {result.extractor}, {result.status.value})')
        w.save()
        return {"ok": True, "status": result.status.value, "extractor": result.extractor}


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
            raise HTTPException(400, "This job already has a tier from the fault list")
        if data.fault_name not in TIER_TABLE:
            raise HTTPException(400, "Choose a fault from the list")
        w.tier_calls[job_id] = TierCall(fault_name=data.fault_name, reason=data.reason.strip(), by=user["name"], at=_now())
        w.store_job(job, w.facts[job_id])
        w.log(job_id, user["name"], "Tier call", f'Counted as "{data.fault_name}" ({TIER_TABLE[data.fault_name].tier}). Reason: {data.reason.strip()}')
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


@app.get("/api/fairness")
async def fairness(request: Request) -> dict:
    """The equity view: who is waiting where, and what a nearest-first queue would do instead."""
    require(request, "admin")
    with _lock:
        w = ws()
        band, traces = w.ranking()
        jobs = w.open_jobs()
        communities: dict[str, dict] = {}
        for job in jobs.values():
            c = communities.setdefault(job.community, {"community": job.community, "open": 0, "safety": 0, "review_band": 0,
                                                       "oldest_days": 0, "distance": distance_of(job)})
            c["open"] += 1
            c["safety"] += to_rank_input(job).safety_level > 0
            c["review_band"] += job.in_review_band
            c["oldest_days"] = max(c["oldest_days"], days_waiting(job))

        # Counterfactual only: the same jobs sorted nearest-first. Our queue never does this.
        def km(t: ReasoningTrace) -> float:
            d = distance_of(jobs[t.job_id])["km"]
            return d if d is not None else float("inf")
        what_if = {t.job_id: i for i, t in enumerate(sorted(traces, key=lambda t: (km(t), t.position)), start=1)}
        rows = [{**job_summary(w, jobs[t.job_id], t), "nearest_first_position": what_if[t.job_id],
                 "change": what_if[t.job_id] - t.position} for t in traces]
        losers = [r for r in rows if r["change"] > 0]
        return {
            "communities": sorted(communities.values(), key=lambda c: -c["oldest_days"]),
            "what_if": rows,
            "summary": {
                "jobs_pushed_back": len(losers),
                "places_lost": sum(r["change"] for r in losers),
                "safety_jobs_pushed_back": sum(r["safety_level"] > 0 for r in losers),
                "worst": max(losers, key=lambda r: r["change"], default=None),
            },
        }


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
