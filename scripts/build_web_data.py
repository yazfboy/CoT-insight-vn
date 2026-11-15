"""Tiền xử lý dữ liệu cho Web Dashboard tĩnh (GitHub Pages).

Đọc file kết quả benchmark mới nhất (benchmark/results/benchmark_results_*.xlsx), bộ đề
data/dataset_150.xlsx và kết quả Gói 3 trong analysis/, rồi xuất JSON gọn vào web/data/:

  meta.json                 nguồn dữ liệu, thời điểm build, danh mục model/phương pháp
  questions_150.json        150 câu hỏi song ngữ + đáp án + nhóm leak
  records.json              5.400 lượt dạng cột (không có Raw_Output) cho biểu đồ & lọc
  summary.json              sheet 2_Summary
  leak_compare.json         sheet 3_Leak_Compare
  faithfulness.json         sheet 5_Faithfulness_Matrix
  errors.json               sheet 4_Errors (không có Raw_Output, tải lười theo câu)
  optimization.json         số liệu tối ưu hệ thống: Early Stopping SC, tái dùng DP, router, Gói 3
  logs/<ID>.json            Raw_Output của 36 lượt (3 model × 2 ngôn ngữ × 6 phương pháp) của 1 câu

Chạy:  python scripts/build_web_data.py
Không gọi API, không sửa file nguồn.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "benchmark", "results")
DATASET = os.path.join(BASE, "data", "dataset_150.xlsx")
EVIDENCE = os.path.join(BASE, "analysis", "benchmark_experiment_evidence_20261106_20cases.xlsx")
QWEN_RERUN = os.path.join(BASE, "analysis", "qwen_goi3_rerun.xlsx")
OUT = os.path.join(BASE, "web", "data")

METHODS = ["1_Zero_Shot", "2_Few_Shot", "3_Zero_Shot_CoT", "4_Few_Shot_CoT",
           "5_Self_Consistency", "6_Few_Shot_CoT_Leaked"]
MODELS = ["Small", "Large", "Qwen_7B"]
LANGS = ["VI", "EN"]
DIFFS = ["Easy", "Medium", "Hard"]
# Phương pháp được phép đưa vào router: 6_Leaked dùng gợi ý rò rỉ -> không triển khai thật được.
ROUTABLE = METHODS[:5]
# Router chọn phương pháp rẻ nhất có accuracy cách tốt nhất không quá ngưỡng này (điểm %).
ROUTER_TOL = 2.0


def latest_results_file() -> str:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_results_*.xlsx")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    if not files:
        sys.exit(f"Không tìm thấy benchmark_results_*.xlsx trong {RESULTS_DIR}")
    return files[-1]


def clean(v):
    """Giá trị Python thuần, JSON-safe (NaN -> None, numpy -> int/float)."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    if hasattr(v, "item"):  # numpy scalar
        return clean(v.item())
    return v


def df_records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def dump(name: str, obj) -> int:
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


def norm_answer(v) -> str:
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def build_questions(raw: pd.DataFrame) -> list[dict]:
    q = pd.read_excel(DATASET, sheet_name="data_test")
    diff = raw.drop_duplicates("ID").set_index("ID")["Difficulty"]
    out = []
    for r in q.to_dict("records"):
        out.append({
            "id": r["ID"],
            "difficulty": diff.get(r["ID"]),
            "level_vi": r["Mức độ"],
            "category": r["Dạng câu hỏi"],
            "leak_group": str(r["Nhóm_Màu_Leak"]).strip(),
            "q_vi": r["Câu hỏi Tiếng Việt"],
            "q_en": r["Câu hỏi Tiếng Anh"],
            "answer": norm_answer(r["Đáp án"]),
        })
    return out


def build_records(raw: pd.DataFrame, qids: list[str]) -> dict:
    """Dạng cột: mỗi khoá là một mảng dài 5.400; chuỗi lặp lại được mã hoá bằng chỉ số."""
    qidx = {q: i for i, q in enumerate(qids)}
    faith_vals = ["1->1", "1->0", "0->1", "0->0"]
    status_vals = sorted(raw["Status"].dropna().unique().tolist())
    cols = {
        "q": [qidx[i] for i in raw["ID"]],
        "m": [MODELS.index(x) for x in raw["Model"]],
        "l": [LANGS.index(x) for x in raw["Language"]],
        "k": [METHODS.index(x) for x in raw["Method"]],
        "c": raw["Correct"].astype(int).tolist(),
        "p": [None if pd.isna(x) else str(x) for x in raw["Predicted"]],
        "cf": [clean(x) for x in raw["Confidence"]],
        "sc": [clean(x) for x in raw["SC_Agree_Pct"]],
        "sr": raw["SC_Runs"].astype(int).tolist(),
        "f": [faith_vals.index(x) if x in faith_vals else None for x in raw["Faithfulness"]],
        "st": [status_vals.index(x) for x in raw["Status"]],
        "pt": raw["Prompt_Tokens"].astype(int).tolist(),
        "ct": raw["Comp_Tokens"].astype(int).tolist(),
        "tt": raw["Total_Tokens"].astype(int).tolist(),
        "lat": [clean(x) for x in raw["Latency_s"]],
        "cost": [clean(x) for x in raw["Cost_1k_USD"]],
    }
    return {
        "n": len(raw),
        "dict": {"q": qids, "m": MODELS, "l": LANGS, "k": METHODS, "f": faith_vals,
                 "st": status_vals},
        "cols": cols,
    }


def build_logs(raw: pd.DataFrame) -> int:
    logs_dir = os.path.join(OUT, "logs")
    if os.path.isdir(logs_dir):
        shutil.rmtree(logs_dir)
    total = 0
    for qid, g in raw.groupby("ID"):
        obj = {}
        for r in g.to_dict("records"):
            key = f"{r['Model']}|{r['Language']}|{r['Method']}"
            obj[key] = {
                "raw": "" if pd.isna(r["Raw_Output"]) else str(r["Raw_Output"]),
                "finish": clean(r["Finish_Reason"]),
                "model_id": r["Model_ID"],
            }
        total += dump(f"logs/{qid}.json", obj)
    return total


def build_optimization(raw: pd.DataFrame) -> dict:
    sc = raw[raw["Method"] == "5_Self_Consistency"]
    early = []
    for m in MODELS:
        s = sc[sc["Model"] == m]
        runs = s["SC_Runs"].astype(int)
        early.append({
            "model": m,
            "n": int(len(s)),
            "avg_runs": round(float(runs.mean()), 3),
            "dist": {str(k): int(v) for k, v in runs.value_counts().sort_index().items()},
            "calls_saved_pct": round((1 - runs.sum() / (5 * len(s))) * 100, 1),
        })

    # Tái dùng DP: ở cả 3 model, 6_Leaked trên câu Unleaked có prompt giống hệt 4_Few_Shot_CoT nên
    # được sao từ bảng DP (không gọi API). Với Qwen, luồng 1 của SC còn tái dùng 3_Zero_Shot_CoT
    # (SC_REUSE_ZCOT=True trong run_extra_llm.py).
    lk = raw[(raw["Method"] == "6_Few_Shot_CoT_Leaked") & (raw["Leak_Status"] == "Unleaked")]
    fc = raw[raw["Method"] == "4_Few_Shot_CoT"].set_index(["ID", "Model", "Language"])["Raw_Output"]
    copied = sum(1 for r in lk.itertuples() if fc.get((r.ID, r.Model, r.Language)) == r.Raw_Output)
    q = raw[raw["Model"] == "Qwen_7B"]
    # Nhãn "[Run 1 (DP Reused Z-CoT)]" đã bị ghi lại khi reparse, nên đếm bằng cách so nội dung
    # luồng 1 của SC với Raw_Output của 3_Zero_Shot_CoT cùng câu/ngôn ngữ.
    zc = q[q["Method"] == "3_Zero_Shot_CoT"].set_index(["ID", "Language"])["Raw_Output"]
    sc_rows = q[q["Method"] == "5_Self_Consistency"]
    reused = 0
    for r in sc_rows.itertuples():
        txt = str(r.Raw_Output or "")
        first = txt.split("\n\n[Run 2", 1)[0].split("\n", 1)[-1].strip()
        if first and first == str(zc.get((r.ID, r.Language), "")).strip():
            reused += 1
    sc_q = sc_rows["Raw_Output"]
    dp = {
        "leaked_unleaked": int(len(lk)),
        "leaked_copied": int(copied),
        "qwen_sc_total": int(len(sc_q)),
        "qwen_sc_reused_zcot": reused,
    }

    # Router: với mỗi model × độ khó, chọn phương pháp rẻ nhất (token TB) có accuracy
    # >= best - ROUTER_TOL. Chọn trên chính tập test -> kết quả là in-sample (ghi rõ trên web).
    g = (raw[raw["Method"].isin(ROUTABLE)]
         .groupby(["Model", "Difficulty", "Method"])
         .agg(acc=("Correct", "mean"), tok=("Total_Tokens", "mean"),
              lat=("Latency_s", "mean"), n=("Correct", "size"))
         .reset_index())
    g["acc"] = g["acc"] * 100
    rules = []
    for m in MODELS:
        for d in DIFFS:
            s = g[(g["Model"] == m) & (g["Difficulty"] == d)]
            best = s["acc"].max()
            pick = s[s["acc"] >= best - ROUTER_TOL].sort_values("tok").iloc[0]
            top = s.sort_values(["acc", "tok"], ascending=[False, True]).iloc[0]
            zs = s[s["Method"] == "1_Zero_Shot"].iloc[0]
            rules.append({
                "model": m, "difficulty": d,
                "method": pick["Method"], "acc": round(pick["acc"], 1), "tok": round(pick["tok"], 1),
                "lat": round(pick["lat"], 2),
                "best_method": top["Method"], "best_acc": round(top["acc"], 1),
                "best_tok": round(top["tok"], 1),
                "zs_acc": round(zs["acc"], 1), "zs_tok": round(zs["tok"], 1),
            })

    # Mô phỏng: router vs. luôn dùng 1 phương pháp (cùng 1 model, gộp 2 ngôn ngữ).
    sim = []
    for m in MODELS:
        pick = {r["difficulty"]: r["method"] for r in rules if r["model"] == m}
        s = raw[raw["Model"] == m]
        routed = s[s.apply(lambda r: r["Method"] == pick[r["Difficulty"]], axis=1)]
        row = {"model": m, "router_acc": round(routed["Correct"].mean() * 100, 1),
               "router_tok": round(routed["Total_Tokens"].mean(), 1), "fixed": {}}
        for k in ROUTABLE:
            t = s[s["Method"] == k]
            row["fixed"][k] = {"acc": round(t["Correct"].mean() * 100, 1),
                               "tok": round(t["Total_Tokens"].mean(), 1)}
        sim.append(row)

    # Gói 3: 5 can thiệp trên 20 ca sai.
    fixes = {"small_large": [], "qwen": []}
    if os.path.exists(EVIDENCE):
        e = pd.read_excel(EVIDENCE, sheet_name="Fix_Evidence", header=4)
        e = e[e["Method"].notna()]
        for r in e.to_dict("records"):
            fixes["small_large"].append({
                "method": r["Method"], "n": clean(r["N hợp lệ"]), "fixed": clean(r["Ca sửa được"]),
                "acc_after": clean(r["Acc sau (%)"]), "skipped": clean(r["Bỏ qua"]),
            })
    if os.path.exists(QWEN_RERUN):
        qm = pd.read_excel(QWEN_RERUN, sheet_name="Methods")
        fixes["qwen"] = df_records(qm)

    return {"early_stop": early, "dp": dp, "router_tol": ROUTER_TOL, "rules": rules,
            "router_sim": sim, "fixes": fixes}


def main():
    src = latest_results_file()
    print(f"Nguồn: {os.path.relpath(src, BASE)}")
    raw = pd.read_excel(src, sheet_name="1_Raw_Logs")
    raw = raw.sort_values(["ID", "Model", "Language", "Method"]).reset_index(drop=True)

    os.makedirs(OUT, exist_ok=True)
    sizes = {}
    questions = build_questions(raw)
    qids = [q["id"] for q in questions]
    missing = set(raw["ID"]) - set(qids)
    if missing:
        sys.exit(f"Raw_Logs có ID không nằm trong dataset: {sorted(missing)[:5]}")
    sizes["questions_150.json"] = dump("questions_150.json", questions)
    sizes["records.json"] = dump("records.json", build_records(raw, qids))

    for name, sheet in [("summary.json", "2_Summary"), ("leak_compare.json", "3_Leak_Compare"),
                        ("faithfulness.json", "5_Faithfulness_Matrix")]:
        sizes[name] = dump(name, df_records(pd.read_excel(src, sheet_name=sheet)))

    err = pd.read_excel(src, sheet_name="4_Errors")
    keep = ["Task_Key", "ID", "Difficulty", "Category", "Leak_Group", "Leak_Status", "Model",
            "Language", "Method", "Ground_Truth", "Predicted", "Status", "Faithfulness",
            "Confidence", "Total_Tokens", "Latency_s"]
    err = err[keep].copy()
    err["Ground_Truth"] = err["Ground_Truth"].map(norm_answer)
    sizes["errors.json"] = dump("errors.json", df_records(err))

    sizes["optimization.json"] = dump("optimization.json", build_optimization(raw))
    sizes["logs/*.json (150 file)"] = build_logs(raw)

    meta = {
        "source": os.path.basename(src),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_records": int(len(raw)),
        "n_questions": len(qids),
        "n_errors": int(len(err)),
        "models": MODELS,
        "model_ids": raw.drop_duplicates("Model").set_index("Model")["Model_ID"].to_dict(),
        "languages": LANGS,
        "methods": METHODS,
        "difficulties": DIFFS,
    }
    sizes["meta.json"] = dump("meta.json", meta)

    for k, v in sizes.items():
        print(f"  {k:<28} {v / 1024:8.1f} KB")
    print(f"Xong: {meta['n_records']} lượt, {meta['n_questions']} câu -> {os.path.relpath(OUT, BASE)}")


if __name__ == "__main__":
    main()
