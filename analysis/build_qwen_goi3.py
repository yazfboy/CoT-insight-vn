"""Gộp case_results.csv (output của run_qwen_five_methods.py) thành 1 workbook sạch cho web app.

Khác với workbook `..._Qwen_updated.xlsx` làm tay (tiêu đề ở dòng 4, tên cột tiếng Việt, cột
Question type trống): file này có header chuẩn ở dòng 1, tên cột ASCII rõ ràng, để app đọc
không cần header=3 và không bị cột rác.

Dùng:
    python analysis/build_qwen_goi3.py <output-dir có case_results.csv> [--out <xlsx>]
Mặc định out = analysis/qwen_goi3_rerun.xlsx
"""
import argparse
import os
import sys
import pandas as pd

# Tên nội bộ (trong CSV) -> nhãn ngắn hiển thị.
FIX_SHORT = {
    "Premise_Extraction": "Premise",
    "Strict_CoT": "Strict CoT",
    "PAL": "PAL",
    "Postgen_NLI_Guardrail": "NLI",
    "Logprob_Temperature_Calibration": "Calibration",
}
FIX_ORDER = ["Premise", "Strict CoT", "PAL", "NLI", "Calibration"]


def build(cases_csv: str, out_xlsx: str) -> None:
    df = pd.read_csv(cases_csv)
    df["M"] = df["method"].map(FIX_SHORT).fillna(df["method"])
    skipped = df["status"].astype(str).str.startswith("SKIP")

    # --- Sheet Cases: 1 dòng/(ca × can thiệp), cột sạch ---
    cases = df.copy()

    # --- Sheet Methods: tổng hợp theo can thiệp (loại SKIP khỏi mẫu số) ---
    rows = []
    for m in FIX_ORDER:
        g = df[(df["M"] == m)]
        valid = g[~g["status"].astype(str).str.startswith("SKIP")]
        n = len(valid)
        if n == 0:
            continue
        before = int(valid["before_correct"].sum())
        after = int(valid["after_correct"].sum())
        fixed = int(((valid["before_correct"] == 0) & (valid["after_correct"] == 1)).sum())
        reg = int(((valid["before_correct"] == 1) & (valid["after_correct"] == 0)).sum())
        rows.append({
            "Method": m, "N_valid": n, "Skipped": int((g["M"] == m).sum()) - n,
            "Before_correct": before, "After_correct": after,
            "Acc_before": round(before / n * 100, 1), "Acc_after": round(after / n * 100, 1),
            "Delta": round((after - before) / n * 100, 1),
            "Fixed": fixed, "Regression": reg,
        })
    methods = pd.DataFrame(rows)

    # --- Sheet Baseline: baseline/20 (lấy từ cột before_* của một method bất kỳ không-SKIP) ---
    base = (df[~skipped].drop_duplicates("id")[
        ["id", "language", "question_type", "ground_truth",
         "before_predicted", "before_correct"]].reset_index(drop=True))

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        methods.to_excel(w, sheet_name="Methods", index=False)
        cases.to_excel(w, sheet_name="Cases", index=False)
        base.to_excel(w, sheet_name="Baseline", index=False)

    print(f"Đọc:  {cases_csv}  ({len(df)} dòng)")
    print(f"Ghi:  {out_xlsx}")
    print(f"Baseline đúng: {int(base['before_correct'].sum())}/{len(base)}")
    print("\nMethods:")
    print(methods.to_string(index=False))
    # Cảnh báo nếu còn lượt câu số trả lời chữ cái (đáng lẽ bản sửa prompt đã hết)
    numeric = df[(df["question_type"].astype(str).str.lower() == "số nguyên") & ~skipped]
    letter = numeric[numeric["after_predicted"].isna()
                     & numeric["raw_output"].astype(str).str.contains(
                         r"(?:Final Answer|Đáp án cuối cùng)\]?\s*:\s*\(?[A-D]\b", regex=True)]
    print(f"\nLượt câu SỐ bị trả lời bằng chữ cái: {len(letter)}  (kỳ vọng 0 sau khi sửa prompt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="Thư mục chứa case_results.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "qwen_goi3_rerun.xlsx"))
    a = ap.parse_args()
    csv = os.path.join(a.output_dir, "case_results.csv")
    if not os.path.exists(csv):
        sys.exit(f"Không thấy {csv}")
    build(csv, a.out)


if __name__ == "__main__":
    main()
