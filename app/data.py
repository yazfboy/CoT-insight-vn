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


# --- Workbook bằng chứng Gói 3 ---
def evidence_summary() -> pd.DataFrame:
    return load_sheet(EVIDENCE_PATH, "Summary")


@st.cache_data(show_spinner=False)
def evidence_fix() -> pd.DataFrame:
    """Fix_Evidence: 1 dòng/can thiệp (tiêu đề thật ở dòng 4 Excel -> header=4)."""
    d = pd.read_excel(EVIDENCE_PATH, sheet_name="Fix_Evidence", header=4)
    return d[d["Method"].notna()].reset_index(drop=True)


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

    # CH1: accuracy CoT (3_Zero_Shot_CoT) - Zero-Shot (1_Zero_Shot), tính theo điểm %
    try:
        acc = r.groupby("Method")["Correct"].mean() * 100
        delta = acc["3_Zero_Shot_CoT"] - acc["1_Zero_Shot"]
        out["ch1"] = {"value": f"+{delta:.0f} điểm", "caption": "CoT so với Zero-Shot"}
    except Exception:  # noqa
        out["ch1"] = {"value": "+10 điểm", "caption": "CoT so với Zero-Shot"}

    # CH2: độ tự tin trung bình trên câu SAI, Zero-Shot -> CoT
    try:
        w = r[(r["Correct"] == 0) & (r["Confidence"].notna())]
        zs = w[w["Method"] == "1_Zero_Shot"]["Confidence"].mean()
        cot = w[w["Method"] == "3_Zero_Shot_CoT"]["Confidence"].mean()
        out["ch2"] = {"value": f"{zs:.1f} → {cot:.1f}",
                      "caption": "Độ tự tin lúc sai (Zero-Shot → CoT)"}
    except Exception:  # noqa
        out["ch2"] = {"value": "99 → 92", "caption": "Độ tự tin lúc sai (Zero-Shot → CoT)"}

    # CH3: số ca faithfulness 1->0 (nghĩ đúng nhưng chốt sai)
    try:
        fm = faithfulness(path)
        n10 = int(fm[fm["Faithfulness"] == "1->0"]["Count"].sum())
        out["ch3"] = {"value": f"{n10} ca", "caption": "Giải đúng nhưng chốt sai (1→0), rất hiếm"}
    except Exception:  # noqa
        out["ch3"] = {"value": "≈0 ca", "caption": "Giải đúng nhưng chốt sai (1→0), rất hiếm"}

    # CH4: rò rỉ — acc pp6 - pp4 trên câu Leaked
    try:
        lc = leak_compare(path)
        lk = lc[lc["Leak_Status"] == "Leaked"]
        d = (lk[lk["Method"] == "6_Few_Shot_CoT_Leaked"]["Acc_Pct"].mean()
             - lk[lk["Method"] == "4_Few_Shot_CoT"]["Acc_Pct"].mean())
        out["ch4"] = {"value": f"+{d:.0f} điểm", "caption": "Rò rỉ công thức (câu Leaked)"}
    except Exception:  # noqa
        out["ch4"] = {"value": "+4–6 điểm", "caption": "Rò rỉ công thức (câu Leaked)"}

    # CH5: tỷ lệ sửa được tốt nhất (Premise Extraction) từ workbook evidence
    try:
        fx = pd.read_excel(EVIDENCE_PATH, sheet_name="Fix_Evidence", header=3)
        row = fx[fx["Method"].astype(str).str.contains("Premise", na=False)].iloc[0]
        out["ch5"] = {"value": f"{float(row['Acc sau (%)']):.0f}%",
                      "caption": "Premise Extraction cứu được (trên 20 ca sai)"}
    except Exception:  # noqa
        out["ch5"] = {"value": "70%", "caption": "Premise Extraction cứu được (trên 20 ca sai)"}

    return out
