"""End-to-end demo of Stages 1, 2 and 6.

Stages 3 to 5 belong to teammates. The `_standin_*` functions below are
minimal placeholders so the demo runs; replace them with the real modules
once they land.

Reports are read from the reports/ folder (one .pdf or .txt per report; see
triage/report_files.py for the layout). Run: python demo.py [folder]
"""
import argparse
import os
import sys
from pathlib import Path

from triage.extraction import LLMClient, OfflineExtractor, OpenAICompatibleClient, extract
from triage.intake import DuplicateRequestError, SQLiteReportRepository, new_request_id
from triage.models import EnrichedJob, ExtractionStatus, ReportExtraction, VerifiedSpan
from triage.adapter import to_rank_input
from triage.evaluation import evaluate
from triage.recording import RECORDED_DIR, RecordedClient, RecordingClient
from triage.verification import claim_spans, verify_spans
from triage.explain import build_traces, render_coordinator, render_review_entry, render_tenant_sms
from triage.ranking import rank
from triage.report_files import load_reports

REPORTS_DIR = Path(__file__).with_name("reports")

# ---- stand-ins for teammates' stages (NOT part of Stages 1, 2, 6) --------
_STANDIN_DISTANCE = {"Wadeye": 412, "Maningrida": 510, "Darwin": 0, "Galiwinku": 560}


_SAFETY_LEVEL_NAMES = ("none", "conditional", "active")  # index = safety level


def _standin_stages_3_to_5(report, facts, job_id: str | None = None, extra_flags: tuple[str, ...] = ()) -> EnrichedJob:
    unverified = verify_spans(report.raw_text, facts)
    # Spans that back no claim (e.g. impact_status quoting "ongoing") are left out entirely.
    spans = [VerifiedSpan(field=s.field, text=s.text, verified=(s.field, s.text) not in unverified)
             for s in claim_spans(facts)]
    if not facts.fault_description:
        # D4: extract() routes these out of scope, so evaluation must never see one.
        raise ValueError(f"{report.request_id}: no fault named, should not reach evaluation")
    ev = evaluate(facts, unverified)
    level = _SAFETY_LEVEL_NAMES[ev.safety.level]
    return EnrichedJob(
        request_id=job_id or report.request_id, parent_report_id=report.request_id,
        community=report.community,
        # Invariant 6: every job from a report keeps the report's intake timestamp (FIFO).
        original_report_timestamp=report.original_report_timestamp,
        fault_description=facts.fault_description, taxonomy_match=facts.taxonomy_match,
        spans=spans, distance_cost_km=_STANDIN_DISTANCE.get(report.community),
        tier=ev.tier.tier, tier_entry=ev.tally.winner,
        base_points=ev.tally.base, severity_bump=ev.tally.bump,
        urgency_tally=ev.tally.tally, tally_reasons=ev.tally.reasons,
        safety_flag=(level == "active"), safety_level=level, safety_reason=ev.safety.reason,
        flags=(*ev.flags, *extra_flags),
    )


def _duplicate_flags(faults) -> list[tuple[str, ...]]:
    """Per fault, a flag for each taxonomy entry another fault in the same report also matched."""
    flags = []
    for i, facts in enumerate(faults):
        others = {name for j, f in enumerate(faults) if j != i for name in f.taxonomy_match}
        # Flag, never merge: two entries may really be two faults, and a merge could drop one.
        flags.append(tuple(
            f'Possible duplicate: this report has two entries for "{name}". Check before dispatching both.'
            for name in dict.fromkeys(facts.taxonomy_match) if name in others
        ))
    return flags


def _standin_jobs(report, extraction: ReportExtraction) -> list[EnrichedJob]:
    """One job per fault, none dropped or merged. [] never gets here: extract() routes it out of scope."""
    faults = extraction.faults
    # One fault keeps the report's id. In a compound report every job gets its own id, so none
    # is mistaken for the report itself; parent_report_id links them back.
    ids = [report.request_id] if len(faults) == 1 else [new_request_id() for _ in faults]
    return [_standin_stages_3_to_5(report, facts, job_id, dups)
            for facts, job_id, dups in zip(faults, ids, _duplicate_flags(faults))]
# --------------------------------------------------------------------------


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

    # ---- STAGES 3-5: teammates (placeholders in this demo) ----------------
    jobs = [job for report, extraction in extracted for job in _standin_jobs(report, extraction)]
    _heading("STAGES 3-5 - teammates' stages (placeholders here, output not shown)")

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
