# Housing maintenance triage

Ranks repair reports from remote Northern Territory housing so that safety comes first, then urgency, then first-reported.
A model reads each report for facts and quotes the tenant's words; plain code applies every rule, so the model never sets a tier, score or position.

## Pipeline

```
reports/*.pdf|.txt
   │  report_files.load_reports()   (GEH forms: geh_form.parse_geh_form, one Report per issue)
   ▼
intake          intake.create_report()  ──► Report {request_id, tenant_id, raw_text, source_tag,
   │            repo.save()                       community, original_report_timestamp}
   ▼
extraction      extraction.extract()  ──► report text + fault names ──► model (structured output)
   │            ◄── ReportExtraction {faults: ExtractedFacts[]}, validated; retry once, then flag
   ▼
verification    verification.verify_spans()  ──► quotes not found in the report text
   ▼
evaluation      evaluation.evaluate()  ──► tier, urgency tally, safety level, flags with reasons
   ▼            (logistics not built: demo.py adds distance for display only)
ranking         adapter.to_rank_input() ──► ranking.rank() ──► RankResult {review_band, ranked}
   ▼            sort_key = (-safety_level, tally-less first, -tally, original_timestamp, job_id)
explain         explain.build_traces() ──► render_coordinator(), render_tenant_sms(),
                render_review_entry()

escalation      escalation.escalate(): follow-up appended by exact ID ──► re-enters extraction;
                original_report_timestamp is never changed
```

## Run

```bash
pip install -r requirements.txt
python demo.py              # recorded mode when no API key is set
python demo.py --offline    # regex test double, never calls the API (what CI runs)
python demo.py --record     # live, and saves every response to data/recorded/
python demo.py pdf          # any folder of reports, e.g. the GEH forms in pdf/
```

`demo.py` prints its mode first:

| Mode | When | What reads the reports |
|---|---|---|
| recorded | no API key | real gpt-4o answers saved in `data/recorded/`, replayed with no API calls. A report with no recording, or one recorded under a different prompt, is flagged "Needs the live model", never guessed. |
| live | a key is set (environment or `.env`) | the model, gpt-4o by default (`TRIAGE_MODEL` overrides). Paid API calls. |
| record | `--record` (needs a key) | live, and saves each response for future recorded runs. |
| offline | `--offline` | a regex test double, not the real extractor. |

Recorded answers go through the same validation as live ones. For a live run set `TRIAGE_API_KEY` (or `OPENAI_API_KEY`), and optionally `TRIAGE_MODEL` and `TRIAGE_BASE_URL` for any OpenAI-compatible endpoint, in the environment or a git-ignored `.env` file.

## Tests

```bash
pytest -q                                  # unit, example and Hypothesis property tests
pytest -q && python demo.py --offline      # the full check CI runs
```

No test calls the API: `tests/conftest.py` blocks the OpenAI client.

## Folders

- `.github/` CI workflow: install, `pytest -q`, `python demo.py --offline`.
- `app/` placeholder for the coordinator UI (not built).
- `data/` distance table, recorded model responses, and a placeholder for synthetic reports.
  - `communities.json`: NT community coordinates (not read by code yet).
  - `housing_offices.json`: NT regional housing offices (not read by code yet).
- `pdf/` sample GEH repair request forms (synthetic).
- `reports/` the six synthetic reports the demo reads by default.
- `scripts/` `probe_llm.py`, a manual, paid probe of the live extractor.
- `tests/` mirrors `triage/`.
- `triage/` the pipeline stages and their shared models.

## Adding a report

Drop a `.pdf` or `.txt` file into `reports/`, one report per file, in this layout:

```
Tenant ID: T-07
Community: Galiwinku
Source: officer
Reported: 2026-09-23 09:15
Message:
power point in the kitchen is sparking and smells like burning
```

- `Source`: `officer` (or `phone`) for a transcribed call, `tenant_direct` (or `tenant direct`, `web`, `form`, `email`) for a tenant's own report.
- `Reported`: when the tenant reported the fault, as `2026-09-23 09:15` or `23/09/2026 9:15 am`. With no timezone it is read as NT time (UTC+09:30).
- `Request ID:` is optional; one is generated if absent.
- A file with a missing or bad field is skipped and listed, never guessed.

## GEH repair request forms (PDF)

`triage/geh_form.py` turns each issue row of an NT Government Employee Housing form (GEHSF03) into its own Report:

- `raw_text` = issue + location + comments, in the tenant's words.
- Not in `raw_text`: the tenant's own Immediate/Urgent/Routine choice (tiers come from the published fault list in evaluation, and the model never sees priority labels), or their name, phone, email and address.
- `tenant_id` is a pseudonymous hash of the email; `community` comes from the address; `region`, `source_file`, `source_item` record provenance.
- `original_report_timestamp` = the "date previously reported to DIPL" when given, otherwise the file's modified time. `timestamp_source` records which.

## Guarantees tested

- A job is never ranked above one with a higher safety level (`test_p1_no_job_ranked_above_higher_safety`).
- Equal safety and tally: the earlier report goes first (`test_p2_equal_safety_and_tally_older_first`).
- Changing distance never changes the result (`test_p3_distance_never_changes_result`); input order never does either (`test_p4_shuffling_never_changes_result`).
- A job is in the review band exactly when it has no tier and no safety trigger (`test_p5_band_exactly_when_untiered_and_no_safety`).
- The model's output is rejected if it carries a scoring field (`test_llm_cannot_add_scoring_fields`), and the response schema has no numeric fields.
- The extraction prompt contains no tier or scoring words, and extraction imports only `FAULT_NAMES` from `tiers.py`.
- Every claimed fact needs a quoted span; quotes not found in the report never lower a score.
- The tenant SMS never shows distance, queue position, `decided_by`, another job's ID, a source or a tier label.
- A `Report` carries only intake fields.

## Limitations


- Logistics (distance, trade capacity, bundling) is not built; `demo.py` shows distance for display only.
- No real fault-report data exists; every report and form here is synthetic.
- Recorded mode only covers report texts that have a recording; any other report is flagged for the live model.
- The offline test double cannot read dialect or informal wording, and never reports an unclear hazard, a sign, a mismatch, harm or worsening.
- A fault with no schema field, or detail the tenant never gave, is invisible to the system.
- Faults not on either tier authority (e.g. air conditioning) go to the review band for a coordinator's tier call.
- Dialect bias in extraction is reduced, not removed.
