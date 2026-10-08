"""Stage contract: each stage's output type is exactly what the next stage takes, and a report
flows through all six with no conversion data in between."""
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import get_args, get_type_hints

from triage import pipeline
from triage.evaluation import Evaluation
from triage.explain import ReasoningTrace
from triage.extraction import OfflineExtractor
from triage.intake import create_report
from triage.models import EnrichedJob, ExtractedFacts, ExtractionResult, ExtractionStatus, Report
from triage.tiers import TIER_TABLE
from triage.trades import TRADE_FOR_FAULT

NT = timezone(timedelta(hours=9, minutes=30))


def hints(fn):
    return get_type_hints(fn, vars(pipeline))


def test_each_stage_returns_what_the_next_stage_takes():
    assert hints(pipeline.stage2_extract)["report"] is Report
    assert hints(pipeline.stage2_extract)["return"] is ExtractionResult
    # Stage 2 carries ExtractedFacts; Stage 3 takes exactly that.
    assert hints(pipeline.stage3_verify)["facts"] is ExtractedFacts
    assert hints(pipeline.stage3_verify)["return"] is pipeline.VerifiedFacts
    assert hints(pipeline.stage4_evaluate)["verified"] is pipeline.VerifiedFacts
    assert hints(pipeline.stage4_evaluate)["return"] is Evaluation
    assert hints(pipeline.stage5_enrich)["verified"] is pipeline.VerifiedFacts
    assert hints(pipeline.stage5_enrich)["evaluation"] is Evaluation
    assert hints(pipeline.stage5_enrich)["return"] is EnrichedJob
    jobs_param = hints(pipeline.stage6_rank)["jobs"]
    assert jobs_param.__origin__ is Sequence and get_args(jobs_param) == (EnrichedJob,)
    assert hints(pipeline.stage6_rank)["return"] is pipeline.Stage6Output


def test_a_report_flows_through_all_six_stages():
    report = create_report(tenant_id="T-1", raw_text="toilet blocked, using a bucket", source_tag="officer",
                           community="Wadeye", original_report_timestamp=datetime(2026, 9, 20, 9, tzinfo=NT))
    s2 = pipeline.stage2_extract(report, OfflineExtractor())
    assert s2.status is ExtractionStatus.OK
    (facts,) = s2.extraction.faults
    s3 = pipeline.stage3_verify(report, facts)
    s4 = pipeline.stage4_evaluate(s3)
    s5 = pipeline.stage5_enrich(report, s3, s4, report.request_id)
    s6 = pipeline.stage6_rank([s5])
    (trace,) = s6.traces
    assert isinstance(trace, ReasoningTrace)
    # Every Stage 6 number traces back to the earlier stage that produced it.
    assert (trace.urgency_tally, trace.base_points, trace.severity_bump) == (s4.tally.tally, s4.tally.base, s4.tally.bump)
    assert trace.original_timestamp == report.original_report_timestamp
    assert trace.required_trades == tuple(s5.required_trades) == ("Plumber",)
    assert trace.distance_km == s5.distance_cost_km == 412.0 and s5.distance_estimate_km is None


def test_estimate_only_when_no_road_distance_and_never_in_the_road_field():
    report = create_report(tenant_id="T-2", raw_text="toilet blocked", source_tag="officer",
                           community="Yuendumu", original_report_timestamp=datetime(2026, 9, 20, 9, tzinfo=NT))
    (job,) = pipeline.build_jobs(report, OfflineExtractor.read(report.raw_text).faults)
    assert job.distance_cost_km is None and job.distance_estimate_km and job.distance_estimate_km > 1000


def test_every_listed_fault_has_a_required_trade():
    assert set(TRADE_FOR_FAULT) == set(TIER_TABLE)
