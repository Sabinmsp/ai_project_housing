# Housing maintenance triage

Reads free-text repair reports from NT public housing tenants and ranks them: safety first, then
urgency, then earliest report. Every position comes with an explanation. Distance to the nearest
housing office is shown to the coordinator but never used to rank.

Two ways to run it, over the same pipeline (`triage/pipeline.py`):

- **Web app** (`app/`): officers upload reports, the coordinator works the ranked queue and assigns tradies.
- **Command line** (`demo.py`): runs a folder of reports and prints every stage.

The only AI step is extraction. It reads the tenant's words and returns facts with quotes;
it never scores or ranks. The recorded answers in `data/recorded/` come from **Claude Sonnet 5.5**
(Anthropic), called through OpenRouter.

## Pipeline

```
intake ─► extraction (LLM) ─► second reader (Jev, flag only) ─► re-read safety net
       ─► span verification ─► evaluation (tier table) ─► ranking
       ─► explanation (coordinator trace, tenant SMS, WHY answer)
```

| Stage | Code | What it does |
|---|---|---|
| intake | `triage/intake.py`, `triage/report_files.py`, `triage/geh_form.py` | Reads report files, stamps an opaque id (`R-` + 8 hex) and the original report time. |
| extraction | `triage/extraction.py` | The model returns facts with quoted spans, per fault. It never returns a tier, score or position. |
| second reader | `triage/second_reader.py` | Jev (TypeSafe AI) answers four questions independently. Disagreement or confidence < 0.7 is a coordinator flag only. |
| re-read | `triage/reread.py` | After a hazard/mechanism flag (live mode only), one more extraction; the higher hazard reading is used. |
| verification | `triage/verification.py` | Checks every quote is in the tenant's text. |
| evaluation | `triage/evaluation.py`, `triage/tiers.py` | Tier from the tier table, urgency tally, safety level, reasoned flags. |
| ranking | `triage/adapter.py`, `triage/ranking.py` | `sort_key = (-safety_level, tally-less first, -tally, original_timestamp, job_id)`; jobs with no tier and no safety trigger go to the review band. |
| logistics | `triage/distances.py`, `triage/trades.py` | Straight-line km to the nearest NT housing office, and the trade the fault needs. Display only. |
| explanation | `triage/explain.py` | Coordinator trace, tenant SMS, answer to `WHY <ref>`. |

`triage/pipeline.py` wires the stages together; `demo.py` and `app/server.py` both call it, so the
command line and the web app always give the same result.

## Quick start

Needs **Python 3.10 or newer** (3.12 is what CI uses).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TRIAGE_MODEL=anthropic/claude-sonnet-5.5   # needed to replay the recorded answers (no key)
python -m app.server                              # web app: http://127.0.0.1:8040
python demo.py                                    # or the command line
```

**Why `TRIAGE_MODEL`:** recorded answers are only replayed for the model they were recorded with,
and the code's default model is `gpt-4o`. Without this line every report is flagged
"Needs the live model" (nothing is guessed). The recordings cover the 6 reports in `reports/` and
21 of the 22 issue rows in the GEHSF03 forms in `pdf/`.

Optional `.env` in the repo root (git-ignored), for live extraction:

```
TRIAGE_API_KEY=sk-or-...                      # OpenRouter key (OPENAI_API_KEY also works, with OpenAI)
TRIAGE_BASE_URL=https://openrouter.ai/api/v1  # any OpenAI-compatible endpoint
TRIAGE_MODEL=anthropic/claude-sonnet-5.5      # the model the recordings were made with
TYPESAFE_API_KEY=...                          # optional: second reader (Jev)
```

Run modes (`demo.py` prints its mode first):

```bash
python demo.py               # live if a key is set; otherwise recorded
python demo.py --offline     # regex test double, never calls any API (what CI runs)
python demo.py --record      # live, and saves every response to data/recorded/ (needs a key)
python demo.py pdf           # any folder of reports, e.g. the GEH forms in pdf/
```

| Mode | When | Reads reports with | Second reader |
|---|---|---|---|
| live | `OPENAI_API_KEY` or `TRIAGE_API_KEY` set | `TRIAGE_MODEL` (default gpt-4o) at `TRIAGE_BASE_URL` (default OpenAI); paid calls | Jev if `TYPESAFE_API_KEY` set, else "not run (no key)" |
| recorded | no key | real Claude Sonnet 5.5 answers saved in `data/recorded/` (set `TRIAGE_MODEL=anthropic/claude-sonnet-5.5`); a report with no recording is flagged for a human, never guessed | "not run (recorded mode)" |
| record | `--record` | live, saving each response | Jev if keyed; no re-read (it would overwrite the recording) |
| offline | `--offline` | a regex test double, not the real extractor (CI only) | "not run (offline)" |

To run recorded mode with a key in `.env`, blank the key for that run:
`OPENAI_API_KEY= TRIAGE_API_KEY= TRIAGE_MODEL=anthropic/claude-sonnet-5.5 python demo.py`.

Other commands:

```bash
python demo.py reports --offline --why R-0000ABCD   # print the tenant's WHY answer for that ref
python scripts/probe_llm.py                         # paid: 7 fixed samples through the live extractor
python scripts/probe_jev.py                         # paid: the same 7 samples through Jev; exits 1 without TYPESAFE_API_KEY
```

`--why` matches the ref exactly. Refs are generated fresh on every run unless the report file
has a `Request ID:` line, so `--why` only finds a ref from an earlier run for such files.
An unknown ref prints: "We couldn't find that reference. Please check the number or call the
maintenance call centre on 1800 104 076." The coordinator view then marks the job "tenant asked why".

## Web workspace (coordinator UI)

`app/` is a browser front end over the same pipeline: the server owns no scoring, it calls
`triage/` and shows the results. Needs Python 3.10+.

```bash
source .venv/bin/activate
TRIAGE_MODEL=anthropic/claude-sonnet-5.5 python -m app.server   # then open http://127.0.0.1:8040
```

Press Ctrl+C to stop it. `PORT=8050` uses another port if 8040 is busy.

It starts empty and never loads sample files. A report enters only when someone uploads a
document (a GEHSF03 PDF, one report per issue row, or a .txt report) or records a call; it then
runs through `triage/pipeline.py`, the same Stages 1 to 6 the CLI uses, and the page shows what
each stage produced. Reports, each report's Stage 2 result and every coordinator decision are
saved in SQLite (`data/fairfix.db`, git-ignored; `FAIRFIX_DB=...` to move it), so a restart
rebuilds the queue without calling the model again. Reports → Clear all data empties it.

Stage 2 reuses the model's saved answer when the exact same text was read before
(`data/recorded/`). New text is read live with `TRIAGE_LIVE=1` and a funded key (the answer is
saved for next time, in `data/recorded/by_model/<model>/` so one model never overwrites
another's); without that it is flagged for a human, never guessed. The second reader runs only in
live mode with `TYPESAFE_API_KEY`, as in `demo.py`.
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

Distance is Stage 5's straight-line km to the nearest NT housing office (`triage/distances.py`);
communities not in `data/communities.json` show "unknown". It is never used for order.

**Tradie recommendations** are advice for one job: qualified for the fault's trade, available,
already assigned in the same community (one trip, two jobs), then nearest home base, then
lightest workload. For a fault not on the list, the admin chooses the trade first. They pick
*who* goes, never *which job* goes first, and the admin decides.
Every decision needs a note and is kept in the audit trail. None of it changes the ranking rules.

## Input format

Reports are files in a folder (`reports/` by default), one report per file: `.pdf` or `.txt`
with this layout.

```
Tenant ID: T-07
Community: Galiwinku
Source: tenant_direct
Reported: 2026-09-23 09:15
Request ID: R-0000ABCD
Message:
toilet blocked
```

- `Source`: `officer` or `phone` (transcribed call); `tenant_direct`, `tenant direct`, `web`, `form` or `email` (tenant's own words).
- `Reported`: when the tenant reported it, as `2026-09-23 09:15` (ISO) or `23/09/2026 9:15 am`. No timezone means NT time (UTC+09:30).
- `Request ID:` is optional.
- A file with a missing or unreadable field is skipped and listed, never guessed.
- NT Government Employee Housing repair forms (GEHSF03 PDFs, see `pdf/`) are also read: one report per issue row.

**Not supported:** typing a report on the command line, and CSV input.

## What the coordinator sees

Excerpt from `OPENAI_API_KEY= TRIAGE_API_KEY= TRIAGE_MODEL=anthropic/claude-sonnet-5.5 python demo.py` (recorded mode, `reports/T-03_maningrida.pdf`):

```
#1
job_id                  R-9FA600EC
taxonomy_match          roof leak  (reference list name)
tier                    dangerous  (NT Residential Tenancies Act s63(2)(c) — emergency repair)
base_points             3  (from tier, not text)
severity_bump           +1
tally_reason            No alternative named — no-redundancy default: +1
urgency_tally           4  (3 + 1)
safety_level            2  (Active hazard described: 'water coming through the light fitting' — full override)
span:fault_description  "roof leaking in kids room, water coming through the light fitting"  verified
span:taxonomy_match     "roof leaking in kids room, water coming through the light fitting"  verified
span:hazard             "water coming through the light fitting"  verified
original_timestamp      2026-09-29T15:00:00+09:30  FIFO input, never overwritten
position                1 of 5
decided_by              top of list
distance                ~280 km to nearest NT Housing office (Nhulunbuy office) — straight-line; actual dispatch point not known  not in sort_key
second_reader           not run (recorded mode)  flag only
```

Ids differ on every run. In live mode with Jev, `second_reader` reads `agrees`, or e.g.
`disagrees on hazard — check (extraction: none; second reader: unclear, confidence 0.48)`; a
re-read adds `re_read  read 1: none; read 2: active; used: read 2  safer reading kept`.

## What the tenant sees

Initial SMS (`--offline` run of the example report above):

> Housing repair R-0000ABCD: we have your report about "toilet blocked". It's being treated as an urgent repair under NT rules. We'll keep you updated. Reply HELP with R-0000ABCD if anything changes or gets worse.

Reply to `WHY R-0000ABCD`:

> Your "toilet blocked" repair (R-0000ABCD) is booked as an urgent repair. Yours was received on 23 September 2026. We know this is hard to live with. A coordinator can see how long it has been waiting. If anyone in the house is unwell, elderly or very young, or this is affecting anyone's health or safety, reply HELP R-0000ABCD and a coordinator will look at it again.

Gas report ("I can smell gas in the kitchen"): the fixed safety advice comes first.

> If you smell gas: leave the building or area and call Fire and Emergency Services on 000. If it is safe to do so, turn off the gas at the cylinder or meter. Do not enter the gas affected area. Housing repair R-0000GA5: we have your report about "smell gas". It is marked as a safety job and is being handled as a priority. We'll keep you updated. Reply HELP with R-0000GA5 if anything changes or gets worse.

All tenant text is fixed templates; no model writes it. A compound report also gets one message
listing each repair and its ref.

## Design guarantees (each enforced by a test)

| Guarantee | Test |
|---|---|
| Distance never changes the ranking | `tests/test_ranking_properties.py::test_p3_distance_never_changes_result` |
| FIFO uses the original report time, kept through escalation | `test_ranking_properties.py::test_p2_equal_safety_and_tally_older_first`, `test_intake.py::test_escalation_preserves_original_timestamp_and_appends_text`, `test_ranking.py::test_escalated_job_keeps_its_place_by_original_timestamp` |
| Safety above urgency | `test_ranking_properties.py::test_p1_no_job_ranked_above_higher_safety` |
| Jev never changes a value (safety, tally, tier, rank) | `test_second_reader.py::test_second_reader_never_changes_safety_tally_or_rank` |
| The re-read can only raise safety | `test_reread.py::test_reread_never_lowers_safety_and_never_changes_the_tally` |
| Tenant text never mentions other jobs, ranking or timeframes | `test_explain.py::test_same_own_facts_different_queue_give_identical_sms`, `test_same_own_facts_different_queue_give_identical_why`, `test_other_job_id_reaches_coordinator_never_tenant`, `test_no_tenant_text_explains_ranking_or_states_a_timeframe` |
| Quotes must be the tenant's own words | `test_verification.py::test_fabricated_quote_on_claimed_field_is_unverified`, `test_near_match_is_not_the_tenants_words`; `test_extraction.py::test_claim_without_span_rejected`; `test_explain.py::test_fault_text_is_the_verified_span_never_model_wording` |

## Data sources

- **Tier table** (`triage/tiers.py`):
  - NT Residential Tenancies Act 1999, s63(2) emergency repairs: https://legislation.nt.gov.au/Legislation/RESIDENTIAL-TENANCIES-ACT-1999
  - nt.gov.au, "Repairs and maintenance of your public housing home" (dangerous things are repaired first): https://nt.gov.au/property/social-housing/looking-after-your-home/repairs-and-maintenance-of-your-home
- **Housing offices** (`data/housing_offices.json`): nt.gov.au, "Contact your local housing office": https://nt.gov.au/property/social-housing/contacts-and-support-services/contact-your-local-housing-office. Coordinates are town-level, from the `source_url` on each row (Wikipedia).
- **Communities** (`data/communities.json`): coordinates from the `source_url` on each row (Wikipedia or Wikidata). Aliases only where that row's source page confirms them.
- **Gas advice** (tenant SMS): NT WorkSafe, gas safety: https://worksafe.nt.gov.au/safety-and-prevention/gas-safety
- **Electrical advice** (tenant SMS): Power and Water, safety and emergencies: https://www.powerwater.com.au/customers/safety-and-emergencies
- **Call-centre number** (unknown-ref reply): NT Housing fact sheet FS17: https://dhlgcd.nt.gov.au/media/documents/fact-sheets/repairs-and-maintenance-fs17.pdf

## Limitations

- The demo reports are synthetic. No real report data exists.
- There is no labelled set, so extraction accuracy is not measured.
- Casual wording sometimes loses a hazard (seen in a live probe). The second reader and the re-read mitigate this; they do not solve it. The re-read uses the same prompt at temperature 0, so it mostly catches non-determinism.
- Jev's 0.7 low-confidence threshold is uncalibrated.
- Distance is straight-line to the nearest NT Housing office, used as an assumed reference point. We don't know where trades are dispatched from. A community not in `data/communities.json` (or misspelt) shows "unknown"; matching is exact.
- The offline test double cannot read dialect or informal wording, and never reports an unclear hazard, a sign, a mismatch, harm or worsening.
- Faults not on the tier table (e.g. air conditioning) go to the review band for a coordinator's tier call.
- Not built: automatic job bundling (the web app only shows same-community jobs), the multi-step SMS sequence, actually sending SMS.
- `demo.py`'s SQLite store is in memory. The web app saves to `data/fairfix.db` (git-ignored).
- The web app's sign-in uses two prototype accounts with passwords in the code. Not production authentication.
- Tradie recommendations use a five-tradie demo roster and straight-line distance from each tradie's home base.
- Recorded answers replay only for the model they were made with (see Quick start).

## Tests

```bash
OPENAI_API_KEY= pytest -q                   # 729 passed
pytest -q && python demo.py --offline       # the full check CI runs (.github/workflows/tests.yml)
```

Unit, example and Hypothesis property tests. No test calls a paid API: `tests/conftest.py` blanks
the keys, blocks the OpenAI client and blocks all network calls.

**Mutation testing** here means: for each rule, we deliberately break the code (e.g. let distance
into the sort key, let Jev lower a safety level), run the tests, confirm at least one fails, then
revert. A rule whose break no test catches is not considered enforced.

## Folders

- `.github/` CI workflow: install, `pytest -q`, `python demo.py --offline`.
- `app/` the web app: `server.py` (FastAPI, sign-in, API, SQLite) and `static/` (the UI; Tailwind and Lucide vendored for offline use).
- `data/` community and office locations, recorded model responses, synthetic data placeholder.
- `docs/` design document and build notes.
- `pdf/` sample GEH repair request forms (synthetic) to upload in the web app or run with `python demo.py pdf`.
- `reports/` the six synthetic reports the demo reads by default.
- `scripts/` manual, paid probes: `probe_llm.py` (extractor), `probe_jev.py` (second reader).
- `tests/` mirrors `triage/`.
- `triage/` the pipeline stages, their shared models, and `pipeline.py` which wires them.

## Team

### Running the UI

See [Web workspace](#web-workspace-coordinator-ui) above. In short:

```bash
source .venv/bin/activate
TRIAGE_MODEL=anthropic/claude-sonnet-5.5 python -m app.server
```

Open http://127.0.0.1:8040, sign in as `admin` / `Admin1!` (coordinator) or `officer` / `Officer1!`,
then **Upload report** with a form from `pdf/`.

### AI used

- Extraction: Claude Sonnet 5.5 (Anthropic) via OpenRouter, for the recorded answers. Any
  OpenAI-compatible model can be set with `TRIAGE_MODEL`.
- Second reader: Jev (TypeSafe AI), live mode only.
- Development: Claude Code was used to help write code, tests and documentation.
