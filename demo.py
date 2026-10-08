"""Command-line run of all six stages (triage/pipeline.py), the same code the web app uses.

Reports are read from the reports/ folder (one .pdf or .txt per report; see
triage/report_files.py for the layout). Run: python demo.py [folder]
"""
import argparse
import os
import sys
from pathlib import Path

from triage.extraction import LLMClient, OfflineExtractor, OpenAICompatibleClient, extract
from triage.intake import DuplicateRequestError, SQLiteReportRepository
from triage.models import EnrichedJob, ExtractionStatus, Report, ReportExtraction
from triage.adapter import to_rank_input
from triage.recording import RECORDED_DIR, RecordedClient, RecordingClient
from triage.verification import claim_spans
from triage.pipeline import build_jobs, duplicate_flags, enrich_fault, save_children
from triage.explain import build_traces, render_coordinator, render_review_entry, render_tenant_sms
from triage.ranking import rank
from triage.report_files import load_reports

REPORTS_DIR = Path(__file__).with_name("reports")

# ---- Stages 3 to 5 live in triage/pipeline.py, shared with the web app (app/server.py) ----
# These names are kept so existing callers and tests keep working.
_standin_stages_3_to_5 = enrich_fault
_duplicate_flags = duplicate_flags


def _standin_jobs(report: Report, extraction: ReportExtraction) -> list[EnrichedJob]:
    """One job per fault, none dropped or merged. [] never gets here: extract() routes it out of scope."""
    return build_jobs(report, extraction.faults)


def _save_children(repo: SQLiteReportRepository, extraction: ReportExtraction, jobs: list[EnrichedJob]) -> None:
    """Store a compound report's child jobs, so a tenant replying with a child's id can escalate it."""
    save_children(repo, extraction.faults, jobs)


def _choose_client(offline: bool, record: bool) -> LLMClient:
    """Offline double, live model, recorded replay, or live-and-record; prints the mode first."""
    if offline:
        print("MODE: offline. OFFLINE STAND-IN: regex test double, not the real extractor. "
              "Dialect handling requires the LLM path (master §3.2.3).")
        return OfflineExtractor()
    _load_dotenv()
    has_key = bool(os.environ.get("TRIAGE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if record:
        if not has_key:
            sys.exit("--record needs TRIAGE_API_KEY or OPENAI_API_KEY (environment or .env).")
        live = OpenAICompatibleClient()
        print(f"MODE: live {live.model}, recording every response to {RECORDED_DIR} — paid API calls.")
        return RecordingClient(live, RECORDED_DIR)
    if has_key:
        live = OpenAICompatibleClient()
        print(f"MODE: live {live.model} — paid API calls. Use --offline for the regex test double.")
        return live
    # No key: replay real recorded answers. A miss is flagged for a human, never sent to the
    # regex double, which would pass a test double's guess off as the model's reading.
    recorded = RecordedClient(RECORDED_DIR)
    when = recorded.recorded_on or "none match the current prompt"
    print(f"MODE: recorded {recorded.model} responses (recorded {when}) — no API calls. "
          "Set OPENAI_API_KEY to read new reports live.")
    return recorded


def _load_dotenv(path: Path = Path(__file__).with_name(".env")) -> None:
    """Read KEY=value lines from .env (git-ignored). Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _one_line(text: str) -> str:
    return text.replace("\n", " | ")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the triage pipeline on a folder of reports.")
    parser.add_argument("folder", nargs="?", type=Path, default=REPORTS_DIR)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true",
                       help="regex test double; never builds the API client (CI, local gate)")
    modes.add_argument("--record", action="store_true",
                       help=f"live run that also saves every response to {RECORDED_DIR.name}/ (needs a key)")
    args = parser.parse_args(argv)
    folder = args.folder
    if not folder.is_dir():
        sys.exit(f"Report folder not found: {folder}")
    repo = SQLiteReportRepository()
    client = _choose_client(args.offline, args.record)

    # ---- STAGE 1: intake ---------------------------------------------------
    reports, skipped = load_reports(folder)
    for report in reports:
        try:
            repo.save(report)
        except DuplicateRequestError:
            skipped.append((Path(report.request_id), "duplicate request_id"))
    _heading(f"STAGE 1 - INTAKE (code)   {len(reports)} reports from {folder}/")
    for path, why in skipped:
        print(f"  skipped {path.name}: {why}")
    if not reports:
        print("Nothing to rank. Add .pdf or .txt reports to the folder.")
        return

    def origin(request_id: str) -> str:
        r = repo.get(request_id)
        if r is None or not r.source_file:
            return ""
        item = f" item {r.source_item}" if r.source_item else ""
        return f"  [{r.source_file}{item}]"

    for r in repo.all():
        print(f"\n{r.request_id}{origin(r.request_id)}")
        print(f"  tenant_id={r.tenant_id}  source_tag={r.source_tag.value}  "
              f"community={r.community}" + (f"  region={r.region}" if r.region else ""))
        print(f"  original_report_timestamp={r.original_report_timestamp.isoformat()}"
              + (f"  ({r.timestamp_source})" if r.timestamp_source else ""))
        print(f"  raw_text: {_one_line(r.raw_text)!r}")

    # ---- STAGE 2: extraction (the only model call) -------------------------
    _heading(f"STAGE 2 - EXTRACTION (model reads only)   extractor: {client.name}")
    extracted = []
    for report in repo.all():
        res = extract(report, client)
        print(f"\n{report.request_id}  status={res.status.value}  attempts={res.attempts}")
        if res.extraction is None:
            print(f"  -> coordinator follow-up: {res.errors}")
            continue
        for f in res.extraction.faults:
            print(f"  fault_description: {f.fault_description!r}")
            print(f"  taxonomy_match: {f.taxonomy_match}")
            print(f"  alternative_mentioned={f.alternative_mentioned}  "
                  f"coping_mentioned={f.coping_mentioned}  impact_status={f.impact_status}")
            print(f"  hazard_status: {f.hazard_status}  mechanism_type={f.mechanism_type}")
            claims = {(s.field, s.text) for s in claim_spans(f)}
            for span in f.quoted_spans:
                shown = repr(span.text) if (span.field, span.text) in claims else "(ignored — backs no claim)"
                print(f"  quote [{span.field}]: {shown}")
        if res.status is ExtractionStatus.OK:
            extracted.append((report, res.extraction))
        else:
            print("  -> no fault named: out of scope, coordinator contacts tenant")

    # ---- STAGES 3-5: verification, evaluation, logistics (triage/pipeline.py) ----
    jobs = []
    for report, extraction in extracted:
        report_jobs = _standin_jobs(report, extraction)
        _save_children(repo, extraction, report_jobs)
        jobs.extend(report_jobs)
    _heading("STAGES 3-5 - verification, evaluation, logistics (triage/pipeline.py)")

    # ---- STAGE 6: ranking + why-trace -------------------------------------
    by_id = {job.request_id: job for job in jobs}
    result = rank([to_rank_input(job) for job in jobs])
    _heading("STAGE 6 - RANKING + WHY-TRACE (code)")
    print("\nREVIEW BAND (held above and outside the sort)")
    for job_id in result.review_band:
        print(f"\n{job_id}{origin(job_id)}")
        print(render_review_entry(by_id[job_id]))

    print("\nRANKED QUEUE  sort_key = (-safety_level, tally-less first, -tally, original_timestamp, job_id)")
    for tr in build_traces(result, by_id):
        print(f"\n#{tr.position}{origin(tr.job_id)}")
        print(render_coordinator(tr))
        print("SMS:", render_tenant_sms(tr))


if __name__ == "__main__":
    main()
