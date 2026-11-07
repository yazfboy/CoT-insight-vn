"""Cầu nối sang các hàm chạy LLM trực tiếp (Live Demo - Tab 1 & Tab 4B).

Tái sử dụng hàm từ benchmark/ và analysis/ thay vì viết lại (theo README Gói 4).
Import đặt trong hàm để app vẫn mở được khi chưa cấu hình key / thiếu dep.
Chỉ chạy khi người dùng bấm nút "Gọi API thật"; mặc định các tab dùng chế độ phát lại.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(BASE_DIR, "benchmark"), os.path.join(BASE_DIR, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

# UI (ngắn) -> tên nội bộ trong script analysis.
FIX_INTERNAL = {
    "Premise": "Premise_Extraction",
    "Strict CoT": "Strict_CoT",
    "PAL": "PAL",
    "NLI": "Postgen_NLI_Guardrail",
    "Calibration": "Logprob_Temperature_Calibration",
}


# ============================ Tab 1: benchmark pipeline ============================
def _rb():
    import run_benchmark as rb  # noqa
    return rb


@st.cache_resource(show_spinner=False)
def get_pool():
    rb = _rb()
    keys = rb.load_keys(rb.ENV_PATH)
    if not keys:
        raise RuntimeError(f"Không tìm thấy API key trong {rb.ENV_PATH}")
    return rb.KeyPool(keys, min_gap=2.2)


def _run_sc(rb, row, model_key, lang, pool):
    """Self-Consistency: chạy CoT 5 lần ở T=0.7, bỏ phiếu đa số (theo Wang et al.)."""
    cfg = rb.MODELS[model_key]
    q_type = row["Dạng câu hỏi"]
    prompt = rb.make_prompt(row, "5_Self_Consistency", lang)
    ans_list, conf_list, logs = [], [], []
    agg = {"text": "", "tok_in": 0, "tok_out": 0, "tok_think": 0,
           "lat": 0.0, "key": 1, "finish": "STOP"}
    n_trunc = 0
    while len(logs) < rb.SC_RUNS:
        r = rb.ask_llm(pool, cfg["id"], prompt, 0.7, max_tok=rb.MAX_TOK_COT)
        pa, cf, stt = rb.parse_output(r["text"], q_type, r["finish"])
        if stt == "API_ERR":
            logs.append("API_ERR")
            break
        if stt == "OK":
            ans_list.append(pa)
            if cf is not None:
                conf_list.append(cf)
        elif stt == "TRUNCATED":
            n_trunc += 1
        logs.append(stt)
        for k in ("tok_in", "tok_out", "tok_think"):
            agg[k] += r[k]
        agg["lat"] += r["lat"]
        agg["key"] = r["key"]
        if ans_list and Counter(ans_list).most_common(1)[0][1] >= 3:
            break
    runs = len([x for x in logs if x != "API_ERR"])
    votes = dict(Counter(ans_list))
    if ans_list:
        pred = Counter(ans_list).most_common(1)[0][0]
        agree = round(votes[pred] / len(ans_list) * 100, 1)
        status = "OK"
    else:
        pred, agree, status = "", None, "NO_ANSWER"
    conf = round(sum(conf_list) / len(conf_list), 1) if conf_list else None
    agg["text"] = "Các lượt SC: " + ", ".join(logs)
    agg["finish"] = f"runs={runs},truncated={n_trunc}"
    rec = rb.make_record(row, model_key, "5_Self_Consistency", lang,
                         pred, conf, agree, agg, status, runs)
    rec["_votes"] = votes
    return rec


def run_method_live(row, model_key, lang, method, pool):
    rb = _rb()
    if method == "5_Self_Consistency":
        return _run_sc(rb, row, model_key, lang, pool)
    return rb.run_single(row, model_key, method, lang, pool)


def run_methods_live(row, model_key, lang, methods, pool):
    """Chạy 1 câu qua nhiều phương pháp -> {method: record}."""
    return {m: run_method_live(row, model_key, lang, m, pool) for m in methods}


# ============================ Tab 4B: analysis pipeline ============================
def run_intervention_live(task_key: str, method_short: str) -> dict:
    """Chạy lại 1 can thiệp lên 1 ca, tái dùng run_one_case_method của script analysis."""
    from pathlib import Path
    import run_six_method_experiment_20261106 as rb2  # noqa

    rb2.ANNOTATED_PATH = Path(os.path.join(
        BASE_DIR, "analysis", "benchmark_results_20261106_reviewed.xlsx"))
    rb2.DATASET_PATH = Path(os.path.join(BASE_DIR, "data", "dataset_150.xlsx"))
    rb2.ENV_PATH = Path(os.path.join(BASE_DIR, ".env"))

    cases, _ = rb2.load_cases()
    case = next((c for c in cases if c["task_key"] == task_key), None)
    if case is None:
        raise RuntimeError(f"Không tìm thấy ca {task_key} trong workbook đã review.")
    client = rb2.GeminiREST(rb2.load_keys(rb2.ENV_PATH))
    method = FIX_INTERNAL[method_short]
    _, _, _, row = rb2.run_one_case_method(client, {}, case, method,
                                           max_output_tokens=2048, phase="CASE_AFTER")
    return row
