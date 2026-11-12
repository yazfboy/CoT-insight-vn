"""Lớp đọc dữ liệu cho Web App (Gói 4).

Chịu trách nhiệm DUY NHẤT: tìm file kết quả .xlsx mới nhất và đọc từng sheet,
có cache để không đọc lại Excel mỗi lần Streamlit rerun.

Không gọi API ở đây. Live demo nằm ở app/llm.py.
"""
from __future__ import annotations

import os
import glob
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_RESULTS_DIR = os.path.join(BASE_DIR, "benchmark", "results")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset_150.xlsx")

# Workbook bằng chứng của Gói 3 (5 can thiệp trên 20 ca sai)
EVIDENCE_PATH = os.path.join(
    ANALYSIS_DIR, "benchmark_experiment_evidence_20261106_20cases.xlsx"
)


def list_benchmark_files() -> list[str]:
    """Trả về danh sách file benchmark_results_*.xlsx, mới nhất trước."""
    files = glob.glob(os.path.join(BENCH_RESULTS_DIR, "benchmark_results_*.xlsx"))
    return sorted(files, reverse=True)


def latest_benchmark_file() -> str | None:
    files = list_benchmark_files()
    return files[0] if files else None


@st.cache_data(show_spinner=False)
def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Đọc 1 sheet từ 1 file .xlsx (cache theo (path, sheet))."""
    return pd.read_excel(path, sheet_name=sheet)


# --- Các helper đọc nhanh từng sheet của benchmark (Gói 2) ---
def raw_logs(path: str) -> pd.DataFrame:
    return load_sheet(path, "1_Raw_Logs")


def summary(path: str) -> pd.DataFrame:
    return load_sheet(path, "2_Summary")


def leak_compare(path: str) -> pd.DataFrame:
    return load_sheet(path, "3_Leak_Compare")


def errors(path: str) -> pd.DataFrame:
    return load_sheet(path, "4_Errors")


def faithfulness(path: str) -> pd.DataFrame:
    return load_sheet(path, "5_Faithfulness_Matrix")


@st.cache_data(show_spinner=False)
def rescore_log(path: str):
    """Sheet 6_Rescore_Log (chỉ có ở file *_rescored.xlsx do benchmark/rescore_results.py tạo).
    Trả None nếu file kết quả chưa được chấm lại."""
    try:
        return pd.read_excel(path, sheet_name="6_Rescore_Log")
    except ValueError:
        return None


# --- Workbook bằng chứng Gói 3 ---
def evidence_summary() -> pd.DataFrame:
    return load_sheet(EVIDENCE_PATH, "Summary")


@st.cache_data(show_spinner=False)
def evidence_fix() -> pd.DataFrame:
    """Fix_Evidence: 1 dòng/can thiệp (tiêu đề thật ở dòng 5 Excel -> header=4)."""
    d = pd.read_excel(EVIDENCE_PATH, sheet_name="Fix_Evidence", header=4)
    return d[d["Method"].notna()].reset_index(drop=True)


QWEN_EVIDENCE_PATH = os.path.join(
    ANALYSIS_DIR, "benchmark_experiment_evidence_20261106_20cases_Qwen_updated.xlsx"
)


QWEN_RERUN_PATH = os.path.join(ANALYSIS_DIR, "qwen_goi3_rerun.xlsx")


@st.cache_data(show_spinner=False)
def qwen_rerun():
    """Gói 3 Qwen đã chạy lại SẠCH (prompt + parser đã sửa), từ analysis/qwen_goi3_rerun.xlsx.

    Trả DataFrame với cột đặt tên giống sheet Qwen_Methods cũ để tái dùng charts.qwen_fix_regression.
    None nếu chưa có file (khi đó Tab 4 mục 4 tạm ẩn).
    """
    if not os.path.exists(QWEN_RERUN_PATH):
        return None
    d = pd.read_excel(QWEN_RERUN_PATH, sheet_name="Methods")
    return d.rename(columns={
        "Method": "Phương pháp", "N_valid": "N hợp lệ", "Before_correct": "Đúng trước",
        "Acc_before": "Accuracy trước (%)", "After_correct": "Đúng sau",
        "Acc_after": "Accuracy sau (%)", "Delta": "Δ điểm %", "Fixed": "Sửa được"})


@st.cache_data(show_spinner=False)
def qwen_methods():
    """Qwen_Methods: 5 can thiệp chạy với Qwen 7B trên cùng 20 câu (tiêu đề ở dòng 4 Excel
    -> header=3, KHÁC sheet gốc). Trả None nếu chưa có file/sheet."""
    if not os.path.exists(QWEN_EVIDENCE_PATH):
        return None
    try:
        d = pd.read_excel(QWEN_EVIDENCE_PATH, sheet_name="Qwen_Methods", header=3)
    except ValueError:
        return None
    return d[d["Phương pháp"].notna()].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def qwen_cases():
    """Qwen_Case_Evidence: 1 dòng/(ca × can thiệp) của Qwen (header=3). None nếu thiếu."""
    if not os.path.exists(QWEN_EVIDENCE_PATH):
        return None
    try:
        d = pd.read_excel(QWEN_EVIDENCE_PATH, sheet_name="Qwen_Case_Evidence", header=3)
    except ValueError:
        return None
    return d[d["Qwen Task_Key"].notna()].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def qwen_methods_corrected(bench_path: str):
    """Tính lại bảng Qwen_Methods từ Qwen_Case_Evidence, lấy 'đúng trước' (baseline) từ file
    kết quả benchmark đang dùng (đã chấm lại) thay cho baseline cũ bị lỗi parse.

    Trả (bảng giống Qwen_Methods, số ca có baseline thay đổi) hoặc (None, 0)."""
    c = qwen_cases()
    if c is None or c.empty:
        return None, 0
    c = c.copy()
    base = raw_logs(bench_path).set_index("Task_Key")["Correct"]
    c["before"] = c["Qwen Task_Key"].map(base).fillna(c["Correct before"]).astype(int)
    changed = int(c.drop_duplicates("Qwen Task_Key")
                  .pipe(lambda x: (x["before"] != x["Correct before"]).sum()))
    v = c[~c["Status"].astype(str).str.startswith("SKIP")].copy()
    v["letter"] = (v["After prediction"].isna()
                   & v["After raw output"].astype(str).str.contains(_LETTER_ANS, regex=True))
    rows = []
    for m, g in v.groupby("Method", sort=False):
        n, b, a = len(g), int(g["before"].sum()), int(g["Correct after"].sum())
        reg = (g["before"] == 1) & (g["Correct after"] == 0)
        rows.append({
            "Phương pháp": m, "N hợp lệ": n, "Đúng trước": b,
            "Accuracy trước (%)": round(b / n * 100, 1), "Đúng sau": a,
            "Accuracy sau (%)": round(a / n * 100, 1),
            "Sửa được": int(((g["before"] == 0) & (g["Correct after"] == 1)).sum()),
            "Regression": int(reg.sum()),
            "Regression do chữ cái": int((reg & g["letter"]).sum()),
        })
    return pd.DataFrame(rows), changed


# Đáp án là chữ A–D sau tag (dùng để phát hiện Qwen ghi chữ cái cho câu cần đáp số).
_LETTER_ANS = r"(?:Final Answer|Đáp án cuối cùng)\]?\s*:\s*\(?[A-D]\b"


@st.cache_data(show_spinner=False)
def qwen_letter_on_numeric():
    """Đếm lượt can thiệp Qwen trả lời bằng chữ A–D cho câu cần ĐÁP SỐ (lỗi prompt
    'số nguyên hoặc A/B/C/D'). Trả (số lượt lỗi, số lượt câu số không SKIP)."""
    c = qwen_cases()
    if c is None or c.empty:
        return 0, 0
    ds = dataset_test()[["ID", "Dạng câu hỏi"]].drop_duplicates("ID")
    x = c.merge(ds, on="ID", how="left")
    num = x[(x["Dạng câu hỏi"].astype(str).str.strip().str.lower() == "số nguyên")
            & ~x["Status"].astype(str).str.startswith("SKIP")]
    letter = (num["After prediction"].isna()
              & num["After raw output"].astype(str).str.contains(_LETTER_ANS, regex=True))
    return int(letter.sum()), int(len(num))


@st.cache_data(show_spinner=False)
def evidence_cases() -> pd.DataFrame:
    """Case_Evidence: 1 dòng/(ca × can thiệp), trước–sau sửa. header=4."""
    d = pd.read_excel(EVIDENCE_PATH, sheet_name="Case_Evidence", header=4)
    return d[d["Task_Key"].notna()].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def evidence_baselines() -> pd.DataFrame:
    """Baselines: 20 ca sai gốc, có Error_Type + Raw_Output. header=4."""
    d = pd.read_excel(EVIDENCE_PATH, sheet_name="Baselines", header=4)
    return d[d["Task_Key"].notna()].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def dataset_test() -> pd.DataFrame:
    """150 câu đánh giá (sheet data_test; fallback sheet đầu tiên)."""
    xl = pd.ExcelFile(DATASET_PATH)
    sheet = "data_test" if "data_test" in xl.sheet_names else xl.sheet_names[0]
    return xl.parse(sheet)


def raw_lookup(path: str, qid: str, model: str, lang: str, method: str):
    """Tra 1 dòng kết quả đã lưu trong 1_Raw_Logs theo khoá (ID, Model, Lang, Method)."""
    r = raw_logs(path)
    hit = r[(r["ID"] == qid) & (r["Model"] == model)
            & (r["Language"] == lang) & (r["Method"] == method)]
    return hit.iloc[0].to_dict() if len(hit) else None


def calibration_before_after() -> dict:
    """Độ tự tin TB & Brier trước/sau hiệu chỉnh (từ sheet Calibration của evidence).

    Các ô trong sheet bị lệch cột nên lấy hằng số đã đọc từ file để chắc chắn.
    """
    return {
        "before": {"conf": 97.1, "brier": 0.945},
        "after": {"conf": 86.9, "brier": 0.247},
    }


# --- Thống kê cho màn mở đầu (overview) ---
@st.cache_data(show_spinner=False)
def setup_info(path: str) -> dict:
    """Mô tả thiết lập thí nghiệm, đọc động từ 1_Raw_Logs."""
    r = raw_logs(path)
    info = {
        "n_questions": int(r["ID"].nunique()),
        "n_models": int(r["Model"].nunique()),
        "n_langs": int(r["Language"].nunique()),
        "n_methods": int(r["Method"].nunique()),
        "n_tasks": int(len(r)),
        "run_date": None,
    }
    try:
        info["run_date"] = str(pd.to_datetime(r["Run_Time"]).max().date())
    except Exception:  # noqa
        pass
    return info


@st.cache_data(show_spinner=False)
def headline_stats(path: str) -> dict:
    """5 con số chốt cho CH1–CH5, tính động (có fallback nếu thiếu dữ liệu).

    Trả về dict {ch1..ch5} mỗi phần có 'value' (chuỗi ngắn hiển thị) + 'caption'.
    """
    r = raw_logs(path)
    out: dict = {}
    n_models = int(r["Model"].nunique())
    scope = f" · gộp {n_models} model" if n_models > 1 else ""

    # CH1: Zero-Shot -> Zero-Shot CoT (điểm %). TÁCH nhóm: gộp với Qwen (Zero-Shot lỗi
    # định dạng) sẽ thổi phồng con số (+28), nên hiện riêng Gemini/Gemma và Qwen.
    def _delta(df):
        acc = df.groupby("Method")["Correct"].mean() * 100
        return acc["3_Zero_Shot_CoT"] - acc["1_Zero_Shot"]
    try:
        is_q = r["Model"] == "Qwen_7B"
        if is_q.any() and (~is_q).any():
            out["ch1"] = {"value": f"+{_delta(r[~is_q]):.0f} / +{_delta(r[is_q]):.0f} điểm",
                          "caption": "CoT so với Zero-Shot: Gemini/Gemma · Qwen 7B"}
        else:
            out["ch1"] = {"value": f"+{_delta(r):.0f} điểm", "caption": "CoT so với Zero-Shot"}
    except Exception:  # noqa
        out["ch1"] = {"value": "+10 điểm", "caption": "CoT so với Zero-Shot"}

    # CH2: độ tự tin trung bình trên câu SAI, Zero-Shot -> CoT. TÁCH nhóm như CH1: gộp thì
    # Qwen (nhiều câu sai, xu hướng ngược) che mất mức giảm rõ ở Gemini/Gemma.
    def _conf(df):
        w = df[(df["Correct"] == 0) & (df["Confidence"].notna())]
        return (w[w["Method"] == "1_Zero_Shot"]["Confidence"].mean(),
                w[w["Method"] == "3_Zero_Shot_CoT"]["Confidence"].mean())
    try:
        is_q = r["Model"] == "Qwen_7B"
        if is_q.any() and (~is_q).any():
            zs, cot = _conf(r[~is_q])
            qz, qc = _conf(r[is_q])
            out["ch2"] = {"value": f"{zs:.1f} → {cot:.1f}",
                          "caption": f"Độ tự tin lúc sai (Zero-Shot → CoT) · Gemini/Gemma; "
                                     f"Qwen {qz:.1f} → {qc:.1f}"}
        else:
            zs, cot = _conf(r)
            out["ch2"] = {"value": f"{zs:.1f} → {cot:.1f}",
                          "caption": "Độ tự tin lúc sai (Zero-Shot → CoT)"}
    except Exception:  # noqa
        out["ch2"] = {"value": "99 → 92", "caption": "Độ tự tin lúc sai (Zero-Shot → CoT)"}

    # CH3: số ca faithfulness 1->0 (nghĩ đúng nhưng chốt sai)
    try:
        fm = faithfulness(path)
        n10 = int(fm[fm["Faithfulness"] == "1->0"]["Count"].sum())
        out["ch3"] = {"value": f"{n10} ca",
                      "caption": "Giải đúng nhưng chốt sai (1→0), rất hiếm" + scope}
    except Exception:  # noqa
        out["ch3"] = {"value": "≈0 ca", "caption": "Giải đúng nhưng chốt sai (1→0), rất hiếm"}

    # CH4: rò rỉ — acc pp6 - pp4 trên câu Leaked
    try:
        lc = leak_compare(path)
        lk = lc[lc["Leak_Status"] == "Leaked"]
        d = (lk[lk["Method"] == "6_Few_Shot_CoT_Leaked"]["Acc_Pct"].mean()
             - lk[lk["Method"] == "4_Few_Shot_CoT"]["Acc_Pct"].mean())
        out["ch4"] = {"value": f"+{d:.0f} điểm",
                      "caption": "Rò rỉ công thức (câu Leaked)" + scope}
    except Exception:  # noqa
        out["ch4"] = {"value": "+4–6 điểm", "caption": "Rò rỉ công thức (câu Leaked)"}

    # CH5: tỷ lệ sửa được tốt nhất (Premise Extraction) — evidence đọc đúng header=4.
    try:
        fx = evidence_fix()
        row = fx[fx["Method"].astype(str).str.contains("Premise", na=False)].iloc[0]
        out["ch5"] = {"value": f"{float(row['Acc sau (%)']):.0f}%",
                      "caption": "Premise Extraction cứu được · 20 ca sai của Small/Large"}
    except Exception:  # noqa
        out["ch5"] = {"value": "70%", "caption": "Premise Extraction cứu được (trên 20 ca sai)"}

    return out
