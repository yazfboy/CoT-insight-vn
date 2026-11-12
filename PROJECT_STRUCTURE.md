# CoT-insight-vn — Project Structure & Code Walkthrough

> **Audience:** engineers joining the next sprint who have not seen this repo before.
> **Goal of this doc:** explain what every folder, file, class, and function does, how the
> pieces connect, and where the sharp edges are — so you can be productive on day one.
> Written from reading the source only (no code was executed to produce this).

---

## 1. What this project actually is

This is a **research harness** for measuring how different **Chain-of-Thought (CoT) prompting
techniques** behave on a fixed set of **150 reasoning questions** (Vietnamese + English), using
Google's Gemini / Gemma models over the API.

The research questions it is built to answer:

1. **Does CoT help?** Compare plain answering vs. step-by-step reasoning across difficulty levels.
2. **Overconfidence / faithfulness** — when the model is *wrong*, is it *more* confident? Does its
   reasoning actually lead to the answer it gives (faithful), or does it reason correctly and then
   state a wrong answer (unfaithful)?
3. **Prompt leakage** — if you leak part of the context/formula into the few-shot examples, how much
   does accuracy jump? (A proxy for "benchmark contamination.")
4. **Interventions** — can targeted techniques (premise extraction, strict CoT, PAL, an NLI
   guardrail, confidence calibration) *fix* the cases the baseline got wrong?

The work is organized into **"Gói" (work packages)**. Package 1–2 built the dataset and the main
benchmark; Package 3 did error analysis; the `analysis/` folder is the Package-3/4 follow-up
experiment. Package 4 (a Streamlit web app) and Package 5 (report/slides) are the *upcoming* sprints
— the `app/` folder is still an empty placeholder, and the current branch is `sys/web`, which
strongly implies the web app is what's next.

**Tech stack:** Python, `pandas` + `openpyxl` for data wrangling and Excel I/O, `google-genai` SDK
*and* raw `requests` for two different API client implementations (see §6 and §7), `tqdm` for
progress bars, and `streamlit` + `plotly` declared for the not-yet-built web app.

---

## 2. Top-level folder map

```
CoT-insight-vn/
├── README.md              # Vietnamese quickstart + data-dictionary for the main benchmark
├── requirements.txt       # Python deps
├── .gitignore             # Ignores .env, .venv/, __pycache__/, Excel lock files
├── .vscode/settings.json  # Tells VS Code to load the .env file into the terminal
│
├── data/                  # INPUT: the 150-question dataset
├── benchmark/             # Gói 1–2: the main 6-method benchmark runner + its Excel outputs
├── analysis/              # Gói 3–4: the 5-intervention follow-up experiment + reviewed results
├── docs/                  # Human-written analysis deliverables (Word doc)
└── app/                   # Gói 4 placeholder: the Streamlit web app (not built yet)
```

| Folder | Stands for | Current state |
|---|---|---|
| `data/` | The single source-of-truth question set everything reads from | Done |
| `benchmark/` | The **main experiment**: 6 prompting methods × 2 models × 2 languages | Done, has results |
| `analysis/` | The **follow-up experiment**: 5 "fix the errors" interventions on 20 hand-reviewed failure cases | Done, has results |
| `docs/` | Written-up findings (faithfulness matrix, overconfidence analysis) | In progress |
| `app/` | Web dashboard to explore the results interactively | Empty (`.gitkeep` only) |

---

## 3. Root-level files

### `README.md`
Vietnamese-language quickstart for the **main benchmark only**. Explains the `.env` key format,
how to create the venv, `TEST_MODE`, the checkpoint-resume behaviour, and documents the Excel output
sheets plus the "next package" instructions.

> ⚠️ **Heads-up for the sprint:** the README has drifted from the code. It says the two models are
> `gemini-3.5-flash-lite` (Small) and `gemini-3.5-flash` (Large) and describes **4** output sheets.
> The current [`benchmark/run_benchmark.py`](benchmark/run_benchmark.py) actually uses
> `gemma-4-26b-a4b-it` (Small) + `gemini-3.5-flash-lite` (Large) and writes **5** sheets
> (it added `5_Faithfulness_Matrix`). Trust the code, not the README, and ideally update the README.

### `requirements.txt`
Six dependencies: `google-genai`, `pandas`, `openpyxl`, `tqdm` (used by the benchmark today) plus
`streamlit`, `plotly` (reserved for the Gói 4 web app). Note: `analysis/` uses raw `requests`
instead of `google-genai`, but `requests` is **not** pinned here — add it before anyone re-runs the
analysis script on a clean machine.

### `.gitignore`
Keeps secrets and local junk out of git: `.env` (the API keys), `.venv/`, `__pycache__/`, and Excel
temp lock files (`~$*.xlsx`). **Consequence worth knowing:** the memory note that "Gói 2/3 outputs
live in untracked `.venv`" is a symptom of this — anything written under `.venv/` is invisible to
git. Keep generated outputs under `benchmark/results/` or `analysis/`, never inside `.venv/`.

### `.vscode/settings.json`
One setting: `python.terminal.useEnvFile: true`. It makes the integrated terminal auto-load `.env`,
which is how API keys reach the process when you run from inside VS Code.

---

## 4. `data/` — the dataset

### `data/dataset_150.xlsx`
The **one input file** both experiments read. From how the code consumes it, it contains these
sheets and columns:

- **Sheets:** `data_test` (the 150 evaluation questions; the benchmark prefers this sheet if present)
  and `data_train` (≥30 extra questions used *only* by the analysis script to calibrate temperature
  and confidence — never mixed into the test set).
- **Columns (Vietnamese headers — important, the code indexes by these exact strings):**
  - `ID` — e.g. `EASY_001`, `MEDIUM_069`, `HARD_144`
  - `Mức độ` — difficulty: `Dễ` / `Trung bình` / `Khó` (mapped to Easy/Medium/Hard)
  - `Dạng câu hỏi` — question type; `"số nguyên"` means integer-answer, otherwise it's A/B/C/D multiple choice
  - `Câu hỏi Tiếng Việt` / `Câu hỏi Tiếng Anh` — the question in each language
  - `Đáp án` — the ground-truth answer
  - `Nhóm_Màu_Leak` — which "leak group" a question belongs to (e.g. `Leak_09_ChuyenDongGapNhau`), or
    `Khong_Lo_Unleaked` for questions with no leak. Drives the method-6 "leaked" prompt.

Because the whole pipeline keys off these exact Vietnamese column names, **renaming a column header
breaks everything downstream** — treat them as an API.

---

## 5. `benchmark/` — the main experiment (Gói 1–2)

```
benchmark/
├── run_benchmark.py          # the runner + auto-grader (single script, ~680 lines)
└── results/
    ├── benchmark_results_20261102_231148.xlsx   # an earlier full run
    ├── benchmark_results_20261106_095454.xlsx   # the latest full run
    ├── checkpoint.csv          # v1 checkpoint (DEPRECATED — had a truncated-output bug)
    └── checkpoint_v2.csv       # v2 checkpoint (current; resume reads this)
```

### `benchmark/results/` files
- **`benchmark_results_<timestamp>.xlsx`** — a finished run. Five sheets: `1_Raw_Logs`, `2_Summary`,
  `3_Leak_Compare`, `4_Errors`, `5_Faithfulness_Matrix` (see §6 `main()` for exactly how each is built).
- **`checkpoint_v2.csv`** — the **resume file**. On every full run the script writes rows here every
  5 completed groups and on exit, so a killed run picks up where it left off. The grid is **3600
  tasks** = 150 questions × 2 models × 2 languages × 6 methods.
- **`checkpoint.csv`** (v1) — kept for history only; the code explicitly switched to `checkpoint_v2.csv`
  because the v1 file was contaminated by a bug that cut outputs short. Don't resume from it.

### `benchmark/run_benchmark.py` — detailed walkthrough

This single file both **runs** the 6 prompting methods and **auto-grades** them, then exports Excel.

#### Module-level configuration (what you'll tweak most)
- `TEST_MODE` — `False` runs all 150; `True` runs a 5-question smoke test (`EASY_001`, `EASY_011`,
  `MEDIUM_069`, `MEDIUM_085`, `HARD_144`).
- `MODELS` — the two models under test. Each entry has `id`, `in_price`/`out_price` (USD per **1M**
  tokens), and a `thinking` level. `"Small"` → `gemma-4-26b-a4b-it` with `thinking:"minimal"`
  (Gemma defaults to ~2000 hidden "thinking" tokens that would eat the output budget, so it's turned
  down); `"Large"` → `gemini-3.5-flash-lite` with no thinking param.
- `MAX_TOK_DIRECT = 256` / `MAX_TOK_COT = 2048` — output caps for direct vs. reasoning prompts.
  The comments record that earlier caps (64 / 512) truncated outputs — a recurring failure mode in
  this project.
- `METHODS` — the six prompting strategies (the heart of the experiment):
  1. `1_Zero_Shot` — answer directly, no examples, no reasoning
  2. `2_Few_Shot` — a couple of worked examples, still no reasoning shown
  3. `3_Zero_Shot_CoT` — "think step by step", no examples
  4. `4_Few_Shot_CoT` — examples *with* reasoning
  5. `5_Self_Consistency` — run CoT 5× at temperature 0.7 and majority-vote
  6. `6_Few_Shot_CoT_Leaked` — like method 4 but a leak hint (context/formula) is injected
- `LANGS = ["VI","EN"]`, `SC_RUNS = 5`, `SC_REUSE_ZCOT = False` (self-consistency runs 5 fresh
  samples at T=0.7 per Wang et al., rather than reusing the zero-shot-CoT sample).
- `EXAMPLES` — the few-shot example blocks (direct + CoT) for VI and EN.
- `LEAK_HINTS` — a dict of 15 leak groups, each with a VI and EN hint string injected by method 6.

#### `load_keys(path) -> list[str]`
Reads `.env` line by line, strips comments / `KEY=` prefixes / quotes, returns the list of Gemini
API keys. **Affects:** nothing but its return value. Pure read.

#### `class KeyPool` — the rate-limit-aware key rotator
Wraps N API keys so the threaded runner can share them without tripping Gemini's 429 limits.
- `__init__(keys, min_gap=2.2)` — builds one `genai.Client` per key; sets up per-(key, model)
  bookkeeping: `dead` (key exhausted for that model?), `next_free` (earliest time this key may fire
  again), plus per-key token/request counters. Holds a `threading.Lock`.
- `acquire(model_id) -> (key_index, client, wait_seconds)` — picks the next alive key for that model
  round-robin, computes how long the caller must sleep to respect `min_gap`, and *reserves* the slot
  by pushing that key's `next_free` forward. **Affects:** `self.idx`, `self.next_free`. Raises if all
  keys are dead for the model.
- `add_usage(i, tok)` — adds token + request counts for key `i`. **Affects:** `self.toks`, `self.reqs`.
- `on_error(i, model_id, err)` — interprets an API error string: a *daily* quota error marks that
  (key, model) **dead**; a transient 429 just pushes `next_free` out by 8s; an auth error kills the
  key for *all* models. (Comment notes a past bug where "invalid" matched 400 errors and killed keys
  wrongly — now it only kills on genuine auth failures.) **Affects:** `self.dead`, `self.next_free`.
- `summary()` — prints per-key request/token totals and quota status. **Affects:** stdout only.

#### `make_prompt(row, method, lang) -> str`
The prompt factory. Given a dataset row, a method name, and a language, returns the exact prompt
string. It reads `Dạng câu hỏi` to decide whether to demand "a single integer" vs. "a letter
A/B/C/D", appends the mandatory `[Đáp án cuối cùng]/[Final Answer]` + `[Độ tự tin]/[Confidence]`
footer, and for methods 2/4/6 splices in `EXAMPLES` (and for method 6, the matching `LEAK_HINTS`
entry by `Nhóm_Màu_Leak`). **Affects:** nothing; pure string construction.

#### Parsing / grading helpers
- `ANS_TAG_RE`, `CONF_TAG_RE`, `BARE_ANS_RE` — compiled regexes for the answer tag, the confidence
  tag, and a "line that is nothing but an answer" fallback.
- `_extract_answer(line, is_int) -> str` — pulls an integer (stripping thousands separators) or a
  single A–D letter out of one line.
- `parse_output(text, q_type, finish_reason) -> (answer, confidence, status)` — the core extractor.
  Returns the predicted answer, a confidence (or `None` if the model omitted it — deliberately **not**
  defaulting to 50), and a `status` of `OK` / `TRUNCATED` (hit max tokens) / `NO_ANSWER` / `API_ERR`.
  It refuses to grab a stray trailing number unless that line is *only* an answer, to avoid scoring an
  intermediate step. **Affects:** nothing; pure.
- `is_match(pred, gt) -> 0|1` — normalizes case / commas / trailing `.0` and returns 1 on exact match.
- `check_faithfulness(raw_text, pred, gt, correct, q_type, method, status) -> label` — a **heuristic
  auto-labeller**. It splits off the reasoning *before* the final-answer tag, extracts what the
  reasoning itself concluded, and compares that to both ground truth and the stated final answer,
  emitting `1->1`, `1->0` (reasoned right, answered wrong = unfaithful), `0->1`, `0->0`, or `N/A`
  (for non-reasoning methods or incomplete outputs). The README's Gói-3 instructions say humans must
  verify these labels on a sample — this is a first-pass signal, not ground truth.

#### API-call helpers
- `make_config(model_id, temp, max_tok) -> GenerateContentConfig` — assembles the SDK config and adds
  a `ThinkingConfig` only for models whose `MODELS` entry sets `thinking`.
- `ask_llm(pool, model_id, prompt, temp, max_tok) -> dict` — the actual network call. Loops over keys
  via `pool.acquire`, sleeps the required gap, calls `client.models.generate_content`, and returns a
  dict of `text`, `tok_in`, `tok_out` (completion **plus** thinking tokens, since both are billed as
  output), `tok_think`, `lat` (pure API latency, excluding rate-limit sleep), `key`, and `finish`.
  On error it calls `pool.on_error` and retries another key; after exhausting retries it returns a
  record whose `text` starts with `ERR:`. **Affects:** the `pool` counters.

#### Record building & per-task execution
- `make_record(row, m_key, method, lang, pred, conf, agree, r, status, sc_runs) -> dict` — turns one
  graded result into the full flat row that lands in `1_Raw_Logs`. It computes `Correct` (via
  `is_match`), `Faithfulness` (via `check_faithfulness`), total tokens, and `Cost_1k_USD`
  (= tokens × price / 1000, i.e. cost per 1000 queries), and stamps metadata (Difficulty mapped to
  English, Leak_Group/Leak_Status, model id, timestamps, the raw output, etc.). `Task_Key`
  (`ID_model_lang_method`) is the unique key used for dedup and resume. **Affects:** nothing; returns a dict.
- `run_single(row, m_key, method, lang, pool, temp=0.0) -> record` — one prompt → one API call → parse
  → record. Picks the right token cap by method.
- `eval_group(row, m_key, lang, pool, done_map) -> list[record]` — runs **all six methods for one
  (question, model, language) triple** as a unit (this is the unit of parallelism and of checkpointing).
  Key behaviours:
  - Runs methods 1–4 via `run_single`, caching each in `cache` and skipping any already in `done_map`.
  - **Self-consistency (method 5):** fires up to 5 samples at T=0.7, lets only `OK` samples vote,
    stops early once any answer hits a 3/5 majority, counts truncations, aggregates real token/latency
    usage, and records the vote agreement %. On API failure it records `API_ERR`.
  - **Leaked (method 6):** if the question has *no* leak group, its prompt is identical to method 4, so
    it **clones the method-4 record** instead of paying for another call; otherwise it runs the leaked
    prompt. **Affects:** the `pool`; returns new records.

#### `main()` — the orchestrator
1. Loads keys, builds the `KeyPool`, sizes the thread pool at `max(4, 2×keys)`.
2. Loads the dataset (prefers `data_test` sheet), drops rows with no answer, dedups by `ID`.
3. Selects the 5 test IDs or all 150 depending on `TEST_MODE`.
4. Resumes from `checkpoint_v2.csv` if present (dropping `API_ERR` rows), building `done_map`.
5. Builds the list of **groups still needing work**, submits each to the thread pool via `eval_group`,
   shows a `tqdm` bar, and checkpoints to CSV every 5 groups and in a `finally` (so Ctrl-C still saves).
6. Builds the five output DataFrames and writes the timestamped `.xlsx`:
   - `1_Raw_Logs` — every task row.
   - `2_Summary` — accuracy %, avg tokens, latency, cost, confidence grouped by model × lang × method × difficulty.
   - `3_Leak_Compare` — methods 4 vs 6 side by side (unleaked vs leaked).
   - `4_Errors` — every wrong CoT row, with blank `Reasoning_State`/`Error_Type`/`Note` columns for humans to fill in (this is the Gói-3 hand-off).
   - `5_Faithfulness_Matrix` — counts and avg confidence per faithfulness label.

   **Affects:** the filesystem (`checkpoint_v2.csv` and the result `.xlsx`), stdout.

---

## 6. `analysis/` — the intervention experiment (Gói 3–4)

```
analysis/
├── run_six_method_experiment_20261106.py    # the 5-intervention runner (~1190 lines)
├── verify_reviewed.py                        # a tiny one-off sanity-check script
├── benchmark_results_20261106_reviewed.xlsx  # INPUT: human-reviewed errors + the 20 chosen cases
└── benchmark_experiment_evidence_20261106_20cases.xlsx  # OUTPUT-side evidence workbook
```

The idea: take the main benchmark's **failures**, have a human review and annotate them (in the
`4_Errors` sheet → producing `6_Selected_20` and `7_Confidence_Compare`), then try **five fix-it
interventions** on those 20 cases and see which ones flip a wrong answer to right and improve
confidence calibration.

### `analysis/benchmark_results_20261106_reviewed.xlsx` (input)
The reviewed workbook. The runner reads two sheets from it: `6_Selected_20` (headers on **row 4**,
the 20 chosen cases in rows 5–24) and `4_Errors` (full error log, keyed by `Task_Key`). Despite the
name "benchmark_results", this is the *annotated* artifact, not a raw benchmark output.

### `analysis/benchmark_experiment_evidence_20261106_20cases.xlsx` (output/evidence)
The compiled evidence workbook for the 20 cases — the presentable result of the intervention run.

### `analysis/verify_reviewed.py`
A throwaway inspection script. It opens a **hard-coded absolute path** on a different machine
(`C:\Users\HP\Documents\Codex\...`) and prints sheet shapes plus annotation counts to confirm the
review was done. **Do not expect this to run as-is** — the path is machine-specific and it's clearly
a personal spot-check, not part of the pipeline. Treat it as a scratch file.

### `analysis/run_six_method_experiment_20261106.py` — detailed walkthrough

Despite "six_method" in the filename, it runs **five** interventions (a sixth, RLAIF/DPO, was
explicitly skipped — see `main()`). Unlike the benchmark, this script talks to Gemini over **raw
`requests`**, not the `google-genai` SDK, and it keeps a **resumable JSON state file** instead of a CSV.

#### Module-level config
- `MODEL_PRICE` — same two models (`gemma-4-26b-a4b-it`, `gemini-3.5-flash-lite`) with aliases and
  prices. `TEMPERATURES = [0.2, 1.0]` is the sweep for calibration. `API_BASE` is the REST endpoint.
- `METHODS` — the five interventions:
  1. `Premise_Extraction` — force the model to list the explicit premises (relevant vs. irrelevant) before reasoning
  2. `Strict_CoT` — a rigid ≤4-step process with unit/arithmetic checks
  3. `PAL` (Program-Aided LM) — emit one safe Python arithmetic expression; **only for integer questions**
  4. `Postgen_NLI_Guardrail` — a second model pass that checks/repairs the *original* answer without seeing ground truth
  5. `Logprob_Temperature_Calibration` — pick the best temperature on train data and recalibrate confidence
- Module globals `ENV_PATH / ANNOTATED_PATH / DATASET_PATH / OUT_DIR / STATE_PATH` — left `None` and
  filled in by `main()` from CLI args (this script is **argparse-driven**, unlike the benchmark).

#### Utility functions
- `now_iso()` — local-tz ISO timestamp.
- `load_keys(env_path)` — more flexible than the benchmark's: reads keys from env vars
  (`GEMINI_API_KEY(S)`, `GOOGLE_API_KEY`, `GEMINI_API_KEY_*`) **and** an optional `.env` file, dedupes,
  and errors if none found.

#### `class GeminiREST` — the raw-HTTP client with key rotation
- `__init__(keys)` — per-key `next_free` timestamps and call counts, a `disabled_models` map, a lock.
- `_safe_message(msg)` — **redacts API keys** out of any error text before it's stored/printed (good
  hygiene — copy this habit).
- `generate(model_id, prompt, temperature, max_output_tokens, request_logprobs, thinking_level) -> dict`
  — the workhorse. Rotates keys round-robin honoring `MIN_KEY_GAP_S`, POSTs to
  `.../{model}:generateContent`, and returns a normalized dict (`raw_output`, `status`,
  `prompt/output/total_tokens`, `finish_reason`, `avg_logprobs`, `has_logprobs`, `key_slot`, `error`).
  It disables a model on 400/404 (treated as permanent), rotates on quota/transient codes, and can
  request logprobs (falling back cleanly if the model rejects them). **Affects:** `self.next_free`,
  `self.calls_by_key`, `self.disabled_models`.

#### Loading the cases
- `load_cases() -> (cases, train)` — reads the reviewed workbook (`6_Selected_20` with headers on row 4
  + `4_Errors` for category) and the dataset (`data_test` for type/answer, `data_train` for the
  calibration pool). Returns exactly **20** case dicts (raises if not 20) carrying the baseline
  prediction, baseline confidence, the human annotations (`reasoning_state`, `error_type`, `note`),
  the question text, ground truth, and the original raw output — plus the full `data_train` list.

#### Answer / confidence parsing (the analysis script's own copies)
- `norm_numeric(value)` — robust numeric normalizer (handles decimals, rounds near-integers).
- `is_numeric_question(case_or_type)` — true for integer/`nguyên`/number types.
- `extract_final_line(text)` — pulls the text after the `[Final Answer]/[Đáp án cuối cùng]` tag (or the last non-empty line).
- `normalize_answer(text, category)` / `normalize_ground_truth(value, category)` — canonicalize a
  prediction / a ground truth to a comparable integer or A–D letter.
- `parse_confidence(text)` — reads the stated confidence; **here it does default to 50.0** when absent
  (note the contrast with the benchmark, which returns `None`).
- `logprob_confidence(avg_logprobs)` — converts a geometric-mean log-prob into a 0–100 proxy
  (`100·exp(avg_logprob)`), to be calibrated before reporting.
- `is_correct(predicted, ground_truth, category) -> 0|1` — normalize-then-compare.

#### Prompt builders
- `answer_tail(lang)` — the mandatory 2-line footer.
- `make_prompt(case, method)` — the first-attempt prompt per intervention (VI + EN variants for each).
- `make_nli_prompt(case)` — builds the guardrail prompt that feeds the *original* candidate answer to
  a checker that must repair it using only the question (truncates very long candidates).
- `make_retry_prompt` / `make_retry3_prompt` — progressively **shorter, stricter** prompt formats used
  when an output was truncated or unparsable (the project fights truncation hard).

#### PAL safe execution (security-relevant — read this before touching it)
- `safe_parse_code(raw)` — extracts the `{"code": "print(...)", "confidence": ...}` JSON (or a bare
  `print(...)`), returning code + confidence.
- `ALLOWED_BINOPS` / `ALLOWED_UNARY` — the allow-list of arithmetic AST operators.
- `eval_numeric_node(node)` — recursively evaluates **only** numeric constants and allow-listed
  operators, bounding exponents and rejecting non-finite results.
- `execute_safe_print(code)` — parses the code, asserts it's exactly one `print(<expr>)` call, and
  evaluates the expression through `eval_numeric_node`. **This is a sandbox: no `eval`/`exec` of
  arbitrary model output, no imports, no names, no function calls.** If you ever extend PAL, keep this
  invariant — the model's code is untrusted.

#### Cost, records, and persistence
- `compute_cost(model_id, prompt_tokens, output_tokens)` — USD per **1M** tokens (note this differs
  from the benchmark's per-1000-query convention — don't compare the two cost columns naively).
- `raw_record(...)` — builds a verbose per-API-call row (one per attempt, including calibration calls),
  carrying logprob fields, confidence basis, temperature, finish reason, cost, redacted error, etc.
- `persist_state(state)` — atomic JSON checkpoint: writes a `.tmp` and `replace()`s it, with retries
  for the Windows "file briefly locked" case. **Affects:** `STATE_PATH` on disk.
- `export_csv(path, rows)` — writes a UTF-8-BOM CSV with a union of all row keys as headers.
- `call_and_store(...)` — the calibration-phase call wrapper. It **caches by (phase, id, model,
  method, temperature, confidence-basis)** so a resumed run never repeats a calibration call, appends
  the raw row to `state["raw_calls"]`, persists, and returns parsed prediction/confidence. **Affects:**
  `state`, disk.

#### Calibration
- `calibrate_temperature_and_confidence(client, state, train, model_ids)` — per model: runs the first
  15 train questions at each temperature to pick the best by held-out accuracy (tie → provider default
  1.0), then runs the next 15 to build reliability bins. Records whether logprobs were fully available
  (and labels the confidence basis honestly as a fallback if not). **Affects:** `state["calibration"]`, disk.
- `make_calibration_prompt(question, lang, category)` — the direct-answer prompt used during calibration.
- `expected_calibration_error(values)` — ECE over four confidence buckets.
- `make_reliability_bins(values)` — Laplace-smoothed empirical accuracy per confidence band.
- `calibrated_confidence(raw, bins)` — maps a raw confidence to the calibrated probability of its band.

#### Running the interventions on the 20 cases
- `save_case_result(state, case, method, row)` — stores one case×method result under
  `state["case_results"][task_key|method]` and persists.
- `method_case_record(...)` — the **before/after** comparison row: baseline prediction/confidence vs.
  the intervention's prediction/confidence, `delta_correct`, status, note, tokens, cost. This is the
  headline output of the whole analysis — "did the intervention fix it?".
- `run_one_case_method(client, calibration, case, method, max_output_tokens, phase) -> (case, method,
  raw, row)` — the per-cell executor. Chooses the prompt by phase (first try / retry / retry2 / retry3),
  sets temperature (calibrated for the logprob method, 1.0 otherwise), runs the call, and applies
  method-specific post-processing: PAL runs the sandbox and sets `PAL_PARSE_FAIL` / `PAL_REJECTED_CODE`
  statuses; the logprob method applies calibrated confidence and labels whether real logprobs were used;
  it skips PAL on non-numeric questions (`SKIPPED_UNSUPPORTED_TYPE`); marks `TRUNCATED` on MAX_TOKENS.
  Produces both a raw-call row and a before/after case row. **Affects:** nothing directly (caller persists).
- `run_case_methods(...)` — threaded fan-out (≤5 workers) over all pending (case, method) pairs that
  aren't already in `state`. **Affects:** `state`, disk.
- `retry_incomplete_methods` / `retry_unresolved_methods` / `retry_final_truncations` — three escalating
  retry passes for cells that came back `TRUNCATED`/`API_ERROR` (and PAL parse failures), each raising
  the output cap (2048 → 4096 → 8192) and switching to terser prompts; the final pass also forces
  Gemma "minimal" thinking. Earlier attempts are always retained in the raw log. **Affects:** `state`, disk.

#### `main(argv)` — the orchestrator
Parses CLI args (`--reviewed-results`, `--dataset`, `--env-file`, `--output-dir`, `--state-file`,
`--new-run`), resolves the global paths, loads the 20 cases + train set, loads or creates the JSON
state (and **explicitly drops any RLAIF/DPO results — that method is skipped**), runs calibration (if
not already done), then the four intervention/retry passes, and finally exports `case_results.csv`
and `raw_calls.csv` plus prints summary counts. **Affects:** `OUT_DIR` on disk, stdout.

> **Run contract to remember:** this script needs `--reviewed-results` and `--dataset` paths and will
> resume from `--output-dir/run_state.json` unless you pass `--new-run`. It also needs `requests`
> installed. Its outputs (state JSON + CSVs) go wherever `--output-dir` points — **make sure that is
> not inside `.venv/`**, or git will hide them (this is the root cause of the "outputs live in
> untracked `.venv`" note).

---

## 7. `docs/` — written deliverables

### `docs/Ma trận Faithfulness và phân tích Overconfidence Bias.docx`
The human-written analysis write-up: the faithfulness matrix and the overconfidence-bias discussion.
This is the narrative interpretation of the `5_Faithfulness_Matrix` sheet and the confidence
comparisons — i.e. the "so what" for the report/slides packages. `.gitkeep` just keeps the folder
tracked.

---

## 8. `app/` — the web app (your sprint, probably)

Only `app/.gitkeep` exists — the folder is a **placeholder**. `requirements.txt` already lists
`streamlit` and `plotly`, and the README's Gói-4 section spells out the intended design:

- Read `2_Summary`, `3_Leak_Compare`, and `4_Errors` straight from a results `.xlsx` to draw charts.
- For a "run a question live" feature, **import** `load_keys`, `KeyPool`, `make_prompt`, `ask_llm`,
  `parse_output` from [`benchmark/run_benchmark.py`](benchmark/run_benchmark.py) rather than
  reimplementing them.

The current branch name `sys/web` lines up with this being the active work.

---

## 9. Mental model / data flow (the one-paragraph version)

`data/dataset_150.xlsx` → **[benchmark]** `run_benchmark.py` fans out 150×2×2×6 tasks through a
rate-limited key pool, auto-grades each, and writes `benchmark_results_*.xlsx` (+ a resumable
`checkpoint_v2.csv`). A human opens the `4_Errors` sheet, annotates the failures, and curates the 20
worst into `*_reviewed.xlsx` (`6_Selected_20`). → **[analysis]** `run_six_method_experiment_*.py`
calibrates on `data_train`, applies five fix-it interventions to those 20 cases with heavy
retry/truncation handling and a PAL sandbox, and emits before/after CSVs + an evidence workbook. →
**[docs]** the faithfulness/overconfidence write-up interprets it. → **[app]** (next) a Streamlit
dashboard will read the result sheets and optionally re-run questions live.

---

## 10. Things to watch out for (sprint onboarding checklist)

- **Vietnamese column headers are the contract.** Both scripts index the dataset by exact strings
  like `Dạng câu hỏi`, `Đáp án`, `Nhóm_Màu_Leak`. Don't rename them.
- **Two different API clients, two different conventions.** Benchmark = `google-genai` SDK, cost per
  1000 queries, confidence defaults to `None`. Analysis = raw `requests`, cost per 1M tokens,
  confidence defaults to 50. Keep them straight.
- **README is stale** (model names + sheet count). Code is source of truth.
- **`requests` isn't in `requirements.txt`** but the analysis script needs it.
- **Never write outputs under `.venv/`** — `.gitignore` hides it; that's why prior outputs went
  missing from git.
- **Truncation is the recurring enemy.** Both scripts carry scar tissue around output-token caps and
  hidden "thinking" tokens; if a method's answers look empty, suspect `MAX_TOKENS` first.
- **PAL executes model-written code** — but only through a strict AST allow-list. Preserve that
  sandbox if you touch it.
- **`analysis/verify_reviewed.py` has a hard-coded foreign path** — it's a personal spot-check, not
  pipeline code.
- **Checkpoints make runs resumable** — delete `checkpoint_v2.csv` / the state JSON to force a clean
  re-run; otherwise both scripts happily continue.
```
