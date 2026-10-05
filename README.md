# Triage pipeline: Stages 1, 2 and 6

Sabin's part of the six-stage housing maintenance triage pipeline.

| Stage | File | What it does |
|---|---|---|
| 1 Intake | `triage/intake.py`, `triage/report_files.py` | Wraps raw text in a `Report`, stamps `request_id`, `source_tag`, `community`, `original_report_timestamp`. SQLite repo behind an interface. Escalation appends text by exact `request_id`; the timestamp never changes. Reports are read from files in `reports/`; a file with a missing or bad field is skipped and listed, never guessed. |
| 2 Extraction | `triage/extraction.py` | The only model call. The model sees the report text and fault NAMES only (no tiers, points or ranks). Structured output, validated with Pydantic; retry once, then `FLAGGED_FOR_HUMAN`. Offline keyword reader with the same contract for demos with no API key. |
| 6 Ranking | `triage/ranking.py` | Pure function. `sort_key = (safety_flag, urgency_tally, -original_timestamp)`. Review-band jobs are held outside the sort. Builds a `ReasoningTrace` per job; two renderers (tenant SMS template, coordinator view) read only the trace. |

`triage/models.py` is the shared contract, including `EnrichedJob`, the shape Stage 6 expects from Stages 3 to 5.

## Run

```bash
pip install -r requirements.txt
pytest -q                  # includes Hypothesis property tests
python demo.py             # ranks every report in reports/
python demo.py some/folder # or any other folder
```

To use a real model, set `TRIAGE_API_KEY` (or `OPENAI_API_KEY`), and optionally `TRIAGE_MODEL` and `TRIAGE_BASE_URL` for any OpenAI-compatible endpoint (OpenRouter and LiteLLM proxy work). `demo.py` also reads these from a git-ignored `.env` file. With no key, the offline keyword reader is used.

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

- `Source`: `officer` (or `phone`) for a transcribed call, `tenant_direct` (or `web`, `form`, `email`) for a tenant's own report.
- `Reported`: when the tenant reported the fault, as `2026-09-23 09:15` or `23/09/2026 9:15 am`. With no timezone it is read as NT time (UTC+09:30).
- `Request ID:` is optional; one is generated if absent.

## Guarantees tested

- No unflagged job ever ranks above a flagged one.
- Equal flag and tally: oldest report first.
- Changing any logistics value (distance, community, capacity, shared route) never changes any position.
- Review-band jobs never enter the sorted queue.
- The extraction prompt contains no tier, score or rank words and no digits; the response schema has no numeric fields.
- Every claimed fact needs a quoted span; unverified spans never reach the SMS or coordinator view.
- Tenant SMS never shows distance.

## Integration notes for Stages 3 to 5

`demo.py` has clearly marked `_standin_*` functions for Stages 3 to 5. Replace them with the real modules; they only need to return an `EnrichedJob`. Set `urgency_tally=None` for off-list faults to send them to the review band.
