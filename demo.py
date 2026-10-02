"""End-to-end demo of Stages 1, 2 and 6.

Stages 3 to 5 belong to teammates. The `_standin_*` functions below are
minimal placeholders so the demo runs; replace them with the real modules
once they land. Run: python demo.py
"""
from datetime import datetime, timedelta, timezone

from triage.extraction import default_client, extract
from triage.intake import SQLiteReportRepository, create_report
from triage.models import EnrichedJob, ExtractionStatus, VerifiedSpan
from triage.ranking import rank, render_coordinator, render_tenant_sms

ACST = timezone(timedelta(hours=9, minutes=30))
NOW = datetime(2026, 9, 30, 15, 0, tzinfo=ACST)

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


INTAKE = [
    ("T-01", "Darwin", "officer", 2, "toilet blocked, using the other one"),
    ("T-02", "Wadeye", "tenant_direct", 5, "toilet blocked, going down the servo"),
    ("T-03", "Maningrida", "tenant_direct", 1,
     "roof leaking in kids room, water coming through the light fitting"),
    ("T-04", "Galiwinku", "tenant_direct", 3, "no hot water since last week"),
    ("T-05", "Wadeye", "officer", 4, "the ceiling fan wobbles and makes a noise"),
    ("T-06", "Darwin", "tenant_direct", 6, "toilet blocked"),
]


def main() -> None:
    repo = SQLiteReportRepository()
    client = default_client()
    print(f"extractor: {client.name}\n")

    for tenant, community, source, days_ago, text in INTAKE:
        repo.save(create_report(tenant_id=tenant, raw_text=text, source_tag=source,
                                community=community,
                                original_report_timestamp=NOW - timedelta(days=days_ago)))

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
