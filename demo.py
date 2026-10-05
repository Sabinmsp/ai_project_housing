"""End-to-end demo of Stages 1, 2 and 6.

Stages 3 to 5 belong to teammates. The `_standin_*` functions below are
minimal placeholders so the demo runs; replace them with the real modules
once they land.

Reports are read from the reports/ folder (one .pdf or .txt per report; see
triage/report_files.py for the layout). Run: python demo.py [folder]
"""
import os
import sys
from pathlib import Path

from triage.extraction import default_client, extract
from triage.intake import DuplicateRequestError, SQLiteReportRepository
from triage.models import EnrichedJob, ExtractionStatus, VerifiedSpan
from triage.ranking import rank, render_coordinator, render_tenant_sms
from triage.report_files import load_reports

REPORTS_DIR = Path(__file__).with_name("reports")

# ---- stand-ins for teammates' stages (NOT part of Stages 1, 2, 6) --------
_STANDIN_TIERS = {
    "blocked or broken toilet": "dangerous", "gas leak": "dangerous",
    "electrical fault: sparking or exposed wires": "dangerous",
    "serious roof leak": "dangerous", "no water supply": "dangerous",
    "sewage overflow": "dangerous", "no power to the house": "dangerous",
    "hot water system not working": "standard", "stove or cooktop not working": "standard",
    "air conditioner not working": "standard", "broken window or glazing": "standard",
    "broken external door lock": "standard",
}
_STANDIN_DISTANCE = {"Wadeye": 412, "Maningrida": 510, "Darwin": 0, "Galiwinku": 560}


def _standin_stages_3_to_5(report, facts) -> EnrichedJob:
    spans = [VerifiedSpan(field=s.field, text=s.text, verified=s.text in report.raw_text)
             for s in facts.quoted_spans]
    tiers = [_STANDIN_TIERS[m] for m in facts.taxonomy_match if m in _STANDIN_TIERS]
    common = dict(request_id=report.request_id, community=report.community,
                  original_report_timestamp=report.original_report_timestamp,
                  fault_description=facts.fault_description, spans=spans,
                  distance_cost_km=_STANDIN_DISTANCE.get(report.community))
    if not tiers:
        return EnrichedJob(**common)
    tier = "dangerous" if "dangerous" in tiers else "standard"
    base = 3 if tier == "dangerous" else 2
    nr = 0 if facts.alternative_mentioned else 1
    level = facts.mechanism_type or "none"
    flags = ["ambiguity_flag"] if len(facts.taxonomy_match) > 1 else []
    return EnrichedJob(**common, tier=tier, base_points=base, no_redundancy=nr,
                       urgency_tally=base + nr, safety_flag=(level == "active"),
                       safety_level=level, flags=flags)
# --------------------------------------------------------------------------


def _load_dotenv(path: Path = Path(__file__).with_name(".env")) -> None:
    """Read KEY=value lines from .env (git-ignored). Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _load_dotenv()
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS_DIR
    if not folder.is_dir():
        sys.exit(f"Report folder not found: {folder}")
    repo = SQLiteReportRepository()
    client = default_client()
    print(f"extractor: {client.name}")

    reports, skipped = load_reports(folder)
    print(f"reports: {len(reports)} read from {folder}/")
    for path, why in skipped:
        print(f"  skipped {path.name}: {why}")
    for report in reports:
        try:
            repo.save(report)
        except DuplicateRequestError:
            print(f"  skipped duplicate request_id {report.request_id}")
    if not reports:
        print("Nothing to rank. Add .pdf or .txt reports to the folder.")
        return
    print()

    jobs = []
    for report in repo.all():
        res = extract(report, client)
        if res.status is not ExtractionStatus.OK:
            print(f"{report.request_id}: {res.status.value}, coordinator follow-up")
            continue
        jobs.append(_standin_stages_3_to_5(report, res.facts))

    result = rank(jobs)
    print("\nREVIEW BAND (held above the sort)")
    for e in result.review_band:
        print(f"  {e.request_id}  {e.community}  {e.fault_description!r}")

    print("\nRANKED QUEUE")
    for tr in result.traces:
        print(f"\n#{tr.position}")
        print(render_coordinator(tr))
        print("SMS:", render_tenant_sms(tr))


if __name__ == "__main__":
    main()
