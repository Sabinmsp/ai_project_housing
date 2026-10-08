# Triage pipeline: Stages 1, 2 and 6

Sabin's part of the six-stage housing maintenance triage pipeline.

| Stage | File | What it does |
|---|---|---|
| 1 Intake | `triage/intake.py`, `triage/report_files.py` | Wraps raw text in a `Report`, stamps `request_id`, `source_tag`, `community`, `original_report_timestamp`. SQLite repo behind an interface. Escalation appends text by exact `request_id`; the timestamp never changes. Reports are read from files in `reports/`; a file with a missing or bad field is skipped and listed, never guessed. |
| 2 Extraction | `triage/extraction.py` | The only model call. The model sees the report text and fault NAMES only (no tiers, points or ranks). Structured output, validated with Pydantic; retry once, then `FLAGGED_FOR_HUMAN`. Offline keyword reader with the same contract for demos with no API key. |
| 6 Ranking | `triage/ranking.py` | Pure function. `sort_key = (safety_flag, urgency_tally, -original_timestamp)`. Review-band jobs are held outside the sort. Builds a `ReasoningTrace` per job (taxonomy match + verified spans, tier + source, defaults applied, arithmetic, safety reason, original timestamp, position, logistics); two template renderers (tenant SMS, coordinator view) read only the trace. No LLM: only Stage 2 calls a model. |

`triage/models.py` is the shared contract, including `EnrichedJob`, the shape Stage 6 expects from Stages 3 to 5.

## Data flow

```
reports/*.pdf|.txt
   │  report_files.load_reports()
   ▼
[1] intake.create_report() ──► Report {request_id, tenant_id, raw_text, source_tag,
   │  repo.save()                      community, original_report_timestamp}
   ▼
[2] extraction.extract()  ──► raw_text + fault NAMES ──► LLM (structured output)
   │                          ◄── JSON ── Pydantic ExtractedFacts (retry once, then human)
   ▼  ExtractedFacts {fault_description, taxonomy_match[], alternative_mentioned,
   │                  coping_mentioned, impact_status, hazard_mechanism,
   │                  mechanism_type, quoted_spans[]}
   ▼
[3][4][5] teammates  ──► EnrichedJob {safety_flag, urgency_tally, timestamp, tier,
   │                                  spans, taxonomy_match, logistics...}
   ▼
[6] ranking.rank()  ──► sort_key = (safety_flag, urgency_tally, -timestamp)
                    ──► RankingResult {review_band, ranked, traces[ReasoningTrace]}
                    ──► render_coordinator(trace), render_tenant_sms(trace)
```

Model reads, code decides: the only model call is in Stage 2, and nothing it returns is a number, tier or rank.

## Run

```bash
pip install -r requirements.txt
pytest -q                  # includes Hypothesis property tests
python demo.py             # ranks every report in reports/ — no key needed (recorded mode)
python demo.py some/folder # or any other folder
```

`demo.py` prints its mode first. It picks one of four:

| Mode | When | What reads the reports |
|---|---|---|
| **recorded** | no API key (the default reproduction) | real gpt-4o answers saved in `data/recorded/`, replayed with no API calls. A report with no recording, or one recorded under an older prompt, is flagged "Needs the live model" — never guessed. |
| **live** | a key is set (environment or `.env`) | the real model, gpt-4o by default (`TRIAGE_MODEL` overrides). Paid API calls. |
| **record** | `--record` (needs a key) | live, and saves every response to `data/recorded/` for future recorded runs. |
| **offline** | `--offline` | a regex test double, never the API. Used by CI and the local gate; not the real extractor. |

Recorded answers go through the same validation as live ones. For a live run, set `TRIAGE_API_KEY` (or `OPENAI_API_KEY`), and optionally `TRIAGE_MODEL` and `TRIAGE_BASE_URL` for any OpenAI-compatible endpoint (OpenRouter and LiteLLM proxy work), in the environment or a git-ignored `.env` file.

## Web workspace (coordinator UI)

`app/` is a browser front end over the same pipeline: the server owns no scoring, it calls
`triage/` and shows the results. Needs Python 3.10+.

```bash
pip install -r requirements.txt
python -m app.server        # then open http://127.0.0.1:8040
```

It starts empty and never loads sample files. A report enters only when someone uploads a
document (a GEHSF03 PDF, one report per issue row, or a .txt report) or records a call; it then
runs through `triage/pipeline.py`, the same Stages 1 to 6 the CLI uses, and the page shows what
each stage produced. Reports, each report's Stage 2 result and every coordinator decision are
saved in SQLite (`data/fairfix.db`, git-ignored; `FAIRFIX_DB=...` to move it), so a restart
rebuilds the queue without calling the model again. Reports → Clear all data empties it.

Stage 2 reuses the model's saved answer when the exact same text was read before
(`data/recorded/`). New text is read live with `TRIAGE_LIVE=1` and a funded key (the answer is
saved for next time); without that it is flagged for a human, never guessed.
`TRIAGE_OFFLINE_FALLBACK=1` uses the labelled regex stand-in instead. `PORT=...` changes the port.

Sign in with a prototype account (not production sign-in):

| Account | Password | Can do |
|---|---|---|
| `officer` | `Officer1!` | Record a phone call or message, upload a GEHSF03 / .txt form, see their own submissions and progress (never the priority) |
| `admin` | `Admin1!` | Work the queue, make tier calls, assign tradies, close jobs, see fairness |

| Admin page | What it shows |
|---|---|
| Dashboard | Open requests, critical, high priority, tradie matches; the top of the priority queue; workload |
| Upload report | Upload a document or record a call; shows the Stage 1 to 6 result for each report |
| Repair Requests | Every report, filterable: needs a read, review band, open, assigned, completed |
| Priority Queue | The review band, then the ranked queue with each job's reason |
| Job (click a row) | The exact text the model read, its facts with each quote checked, the why-trace, tenant SMS and "why is my repair here?" answer, tier call, **assign a tradie** with recommendations, close or reopen, follow-up (escalation), audit trail |
| Communities | Open jobs, safety jobs and the oldest wait per community |
| Fairness Monitor | What a nearest-first queue would do instead (comparison only) |
| Tradies | The roster (5 demo tradies), availability, current assignments, add a tradie |
| Reports | Counts by priority, NT category, status and community; CSV export |

The UI uses Tailwind and Lucide icons, both vendored in `app/static/vendor/` so it works offline.
"Priority" labels (Critical, High, Medium, Low) are display names for the pipeline's safety level
and urgency score; the order always comes from Stage 6.

**Tradie recommendations** are advice for one job: qualified for the fault's trade, available,
already assigned in the same community (one trip, two jobs), then nearest home base, then
lightest workload. They pick *who* goes, never *which job* goes first, and the admin decides.
Every decision needs a note and is kept in the audit trail. None of it changes the ranking rules.

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

## GEH repair request forms (PDF)

`python demo.py pdf` runs the pipeline on NT Government Employee Housing request forms (GEHSF03). `triage/geh_form.py` turns **each issue row into its own Report**:

- `raw_text` = issue + location + comments, in the tenant's words.
- **Not** in `raw_text`: the tenant's own Immediate/Urgent/Routine choice (tiers come from the published list in Stage 4, and the model never sees priority labels), or their name, phone, email and address.
- `tenant_id` is a pseudonymous hash of the email; `community` comes from the address; `region`, `source_file`, `source_item` record provenance.
- `original_report_timestamp` = the "date previously reported to DIPL" when given (escalation never resets queue fairness), otherwise the time the PDF arrived (file modified time). `timestamp_source` records which.

## Guarantees tested

- No unflagged job ever ranks above a flagged one.
- Equal flag and tally: oldest report first.
- Changing any logistics value (distance, community, capacity, shared route) never changes any position.
- Changing every field the model's output can reach (fault text, taxonomy matches, spans, flags, tier/base split at the same tally) never changes any position.
- `ranking.py` imports only the shared models: no code path from Stage 6 to an LLM.
- The model's output is rejected if it carries any scoring field (`priority`, `rank`, `tier`, `urgency_tally`, `score`, ...).
- A `Report` carries only the intake fields; nothing interpretive can be attached at Stage 1.
- Review-band jobs never enter the sorted queue.
- The extraction prompt contains no tier, score or rank words and no digits; the response schema has no numeric fields.
- Every claimed fact needs a quoted span, including each taxonomy match and an `intermittent` impact; unverified spans never reach the SMS or coordinator view.
- Tenant SMS never shows distance.

## Integration notes for Stages 3 to 5

`demo.py` has clearly marked `_standin_*` functions for Stages 3 to 5. Replace them with the real modules; they only need to return an `EnrichedJob`. Set `urgency_tally=None` for off-list faults to send them to the review band.
