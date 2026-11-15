# CoT-Insight VN

An experimental toolkit that measures **how much Chain-of-Thought (CoT) prompting helps — and
when it does not** — on a fixed set of **150 reasoning questions** (Vietnamese + English), across
**3 models**, **6 prompting methods**, and **2 languages**. It includes automatic grading, error
analysis with five "fix-it" interventions, and a **Streamlit dashboard** with built-in statistics.

Research questions the project answers:

- **RQ1** — Does CoT help, and on which difficulty levels / at what token cost?
- **RQ2** — When a model is *wrong*, is it *overconfident*? (calibration / reliability)
- **RQ3** — Faithfulness: does the step-by-step reasoning match the final answer?
- **RQ4** — Contamination: how much does leaking the formula/context inflate accuracy?
- **RQ5** — Can targeted interventions fix the cases the baseline got wrong?

---

## 1. Repository layout

```
CoT-insight-vn/
├── data/dataset_150.xlsx            # INPUT: 150 questions (sheet data_test) + ~30 calibration (data_train)
├── benchmark/
│   ├── run_benchmark.py             # Benchmark for Small & Large (Google Gemini API)
│   ├── run_extra_llm.py             # Benchmark for Qwen_7B (self-hosted, OpenAI-compatible endpoint)
│   └── results/                     # Output workbooks (benchmark_results_*.xlsx) + checkpoints
├── analysis/
│   ├── run_six_method_experiment_20261106.py   # 5 interventions on 20 errors (Small/Large)
│   ├── run_qwen_five_methods.py                 # 5 interventions on 20 errors (Qwen)
│   ├── build_qwen_goi3.py                       # case_results.csv -> qwen_goi3_rerun.xlsx (clean)
│   ├── benchmark_results_20261106_reviewed.xlsx # human-reviewed errors + the 20 selected cases
│   └── *_20cases*.xlsx, qwen_goi3_rerun.xlsx    # intervention evidence workbooks
├── app/                             # Streamlit dashboard (main.py -> data.py -> charts.py + stats.py -> tabs/)
├── requirements.txt
└── .env                             # Gemini API keys (NOT committed; see Setup)
```

### The three models
| Label | Model ID | How it runs |
|-------|----------|-------------|
| `Small` | `gemma-4-26b-a4b-it` | Google Gemini API (`run_benchmark.py`) |
| `Large` | `gemini-3.5-flash-lite` | Google Gemini API (`run_benchmark.py`) |
| `Qwen_7B` | `qwen2.5:7b` | Self-hosted via Ollama/vLLM (`run_extra_llm.py`) |

### The six prompting methods
`1_Zero_Shot`, `2_Few_Shot` (examples, no reasoning), `3_Zero_Shot_CoT`, `4_Few_Shot_CoT`,
`5_Self_Consistency` (CoT ×5 at T=0.7, majority vote), `6_Few_Shot_CoT_Leaked` (method 4 + a leaked hint).

---

## 2. Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/yazfboy/CoT-insight-vn.git
cd CoT-insight-vn

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### API keys (only needed to RE-RUN the benchmark, not to view the dashboard)
Create a `.env` file in the project root with one Google Gemini API key per line:

```
GEMINI_API_KEY_1=AIza...key1
GEMINI_API_KEY_2=AIza...key2
```

Multiple keys are rotated to avoid rate limits. The file is git-ignored.

> **To only explore results, skip this section** — the dashboard reads the result workbooks that
> are already committed under `benchmark/results/` and `analysis/`.

---

## 3. Run the dashboard (no API needed)

```bash
streamlit run app/main.py
```

The app auto-loads the newest `benchmark/results/benchmark_results_*.xlsx`. Four tabs:

1. **Try it now** — replay a stored result (default) or call the API live; compare methods side by side.
2. **How much does CoT help?** — RQ1 (difficulty, cost, model, language) + RQ4 (leakage). Includes a
   paired **McNemar significance table** and 95% Wilson confidence intervals.
3. **When CoT is wrong** — RQ2 (overconfidence / reliability) + RQ3 (faithfulness) + error types.
4. **Intervention orchestrator** — RQ5 (five fixes), including the clean Qwen re-run.

The app is read-only unless you press **"Call API"** (which needs `.env`). Charts cannot be exported
to image in headless environments (no kaleido); view them in the browser.

### 3.1 Static web dashboard (GitHub Pages)

A static version of the dashboard lives in `web/` and is published at
<https://yazfboy.github.io/CoT-insight-vn/>. It needs no Python backend: the data is pre-built to JSON.

```bash
python scripts/build_web_data.py      # newest results .xlsx -> web/data/*.json (+ web/data/logs/<ID>.json)
python -m http.server 8000 -d web     # then open http://localhost:8000
```

`web/data/` is git-ignored; `.github/workflows/deploy.yml` rebuilds it and deploys `web/` on every push
to `main`. The "Live Test" mode in Tab 1 calls the Gemini REST API from the browser with a key the
visitor types in (kept in memory only).

---

## 4. Re-run the benchmark (optional)

### 4.1 Small & Large (Google Gemini API)
```bash
python benchmark/run_benchmark.py
```
- In `run_benchmark.py`, `TEST_MODE = True` runs a 5-question smoke test; `False` runs all 150.
- Resumable: the run checkpoints to `benchmark/results/checkpoint_v2.csv`. Delete it to start fresh.
- Output: `benchmark/results/benchmark_results_<timestamp>.xlsx` with 5 sheets (see below).

### 4.2 Qwen_7B (self-hosted)
Qwen runs against any OpenAI-compatible endpoint. The simplest host is **Ollama** (optionally on a
cloud GPU notebook exposed with a tunnel):

```bash
ollama serve
ollama pull qwen2.5:7b
```

```bash
VLLM_BASE_URL="http://localhost:11434/v1" \
python benchmark/run_extra_llm.py
```
This scores Qwen and merges it with the existing Small/Large rows into a single 3-model workbook.

### Result workbook — 5 sheets
1. **1_Raw_Logs** — one row per call: `150 × 3 models × 2 languages × 6 methods = 5400`.
2. **2_Summary** — accuracy %, tokens, latency, cost, confidence by model × language × method × difficulty.
3. **3_Leak_Compare** — `4_Few_Shot_CoT` (clean) vs `6_Few_Shot_CoT_Leaked` (leaked) on leaked items.
4. **4_Errors** — every wrong answer (`Correct == 0`) with blank columns for manual error labelling.
5. **5_Faithfulness_Matrix** — counts per faithfulness label (`1->1 / 0->1 / 1->0 / 0->0`) + avg confidence.

---

## 5. Error analysis & interventions (RQ5)

1. In `4_Errors`, select ≥ 20 wrong CoT cases and label them (`Reasoning_State`, `Error_Type` ∈
   {`Arithmetic`, `Logical_Leap`, `Incomplete`, `Misinterpretation`}, `Faithfulness`, `Note`).
2. Run the five interventions (Premise Extraction, Strict CoT, PAL, NLI Guardrail, Logprob/Temperature
   Calibration) on the selected cases:

   - **Small/Large:**
     ```bash
     python analysis/run_six_method_experiment_20261106.py \
       --reviewed-results analysis/benchmark_results_20261106_reviewed.xlsx \
       --dataset data/dataset_150.xlsx \
       --env-file .env --output-dir analysis_output
     ```
   - **Qwen** (point `--base-url` at the Ollama endpoint; `--baseline-results` at the clean 3-model
     workbook so the baseline is taken correctly):
     ```bash
     VLLM_BASE_URL="http://localhost:11434/v1" \
     python analysis/run_qwen_five_methods.py --new-run --model-id qwen2.5:7b \
       --reviewed-results analysis/benchmark_results_20261106_reviewed.xlsx \
       --dataset data/dataset_150.xlsx \
       --baseline-results benchmark/results/benchmark_results_20261110_232749.xlsx \
       --output-dir benchmark_output_qwen_v2
     ```
     Then aggregate the CSV into the workbook the app reads:
     ```bash
     python analysis/build_qwen_goi3.py benchmark_output_qwen_v2
     # -> analysis/qwen_goi3_rerun.xlsx
     ```

---

## 6. Notes

- Dataset column headers are in Vietnamese and are treated as an API (`Dạng câu hỏi`, `Đáp án`,
  `Mức độ`, `Nhóm_Màu_Leak`, …) — do not rename them.
- The answer parser tolerates models that print a bare answer followed by a separate confidence
  line; the same logic lives in `run_benchmark.py`, `run_extra_llm.py`, and the analysis scripts.
- Qwen is self-hosted, so its cost is an assumed price and its latency depends on hardware; token
  counts are not directly comparable across model families (different tokenizers). The dashboard
  states these caveats where relevant.
