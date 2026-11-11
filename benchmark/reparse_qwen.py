import os
import re
import glob
from collections import Counter
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "benchmark", "results")
EXTRA_CKPT_CSV = os.path.join(OUT_DIR, "checkpoint_extra_qwen.csv")
TARGET_XLSX = os.path.join(OUT_DIR, "benchmark_results_20261106_095454.xlsx")

# ==============================================================================
# BO PARSER DA TANG (MULTI-STAGE PARSER) - DAC TRI QWEN & LOI DONG CUOI
# ==============================================================================
CONF_LINE_RE = re.compile(
    r"^.*(?:Độ tự tin|Confidence(?:\s*Score)?|Mức độ tự tin)\s*[\]\*]*\s*[:：=].*$",
    re.I | re.M,
)
CONF_VAL_RE = re.compile(
    r"(?:Độ tự tin|Confidence(?:\s*Score)?|Mức độ tự tin)\s*[\]\*]*\s*[:：=]\s*[\*\[\<\s]*(\d+(?:[.,]\d+)?)",
    re.I,
)

# Tang 1: Bat tag dap an (ho tro ** ben trong/ngoai ngoac vuong, co hoac khong co dau :, hoac xuong dong)
EXPLICIT_TAG_RE = re.compile(
    r"(?:\[\s*\**(?:Đáp án cuối cùng|Final Answer|Đáp án|Kết quả|Answer)\**\s*\]"
    r"|\**(?:Đáp án cuối cùng|Final Answer)\**)"
    r"\s*\**\s*(?:[:：=\-]|là|is)?\s*\**\s*([^\n]+|\n\s*[^\n]+)",
    re.I,
)

# Tang 2: Bat LaTeX \boxed{...} dac trung cua ho Qwen khi giai CoT
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")

# Tang 3: Bat cac mau cau ket luan o cuoi bai
CONCLUSION_RE = re.compile(
    r"(?:(?:vậy|do đó|tóm lại|suy ra)\s+)?(?:đáp án|kết quả|phương án|lựa chọn)(?:\s+đúng|\s+cuối cùng|\s+chính xác)?\s*(?:[:：=]|là)\s*([^\n]+)"
    r"|(?:(?:therefore|thus|hence|so),?\s+)?(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|option|choice|result)\s*(?:[:：=]|is)\s*([^\n]+)"
    r"|(?:chọn|choose|select)\s+(?:đáp án|phương án|option|answer)?\s*[:：]?\s*([A-D]\b[^\n]*)",
    re.I,
)


def _clean_str_val(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "" if s.lower() == "nan" else s


def _extract_answer_from_Candidate(line, is_int):
    if not line:
        return ""
    line = str(line).strip()
    # Loai bo cac placeholder nhu <chỉ ghi ...> neu model copy lai de bai
    line = re.sub(r"<[^>\d]*>", "", line).strip()
    if not line:
        return ""

    if is_int:
        # Xoa dau phan cach hang nghin (1,200 hoac 1.200)
        s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", line)
        # Neu co phep tinh "27 + 12 = 39", uu tien lay so sau dau "="
        if "=" in s:
            after_eq = s.split("=")[-1]
            nums_eq = re.findall(r"-?\d+(?:\.\d+)?", after_eq)
            if nums_eq:
                v = float(nums_eq[-1])
                return str(int(v)) if v.is_integer() else str(v)
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not nums:
            return ""
        v = float(nums[-1])
        return str(int(v)) if v.is_integer() else str(v)
    else:
        # Cau hoi trac nghiem A, B, C, D
        # 1. Chi co duy nhat chu cai A-D (vd: "B", "**B**", "[B]", "<B>", "(B)", "B.")
        m = re.match(r"^[\s\[\(\<\*\"\']*([A-D])[\s\]\)\>\*\"\'\.\:]*$", line, re.I)
        if m:
            return m.group(1).upper()
        # 2. Bat dau bang "B. Noi dung..." hoac "B) Noi dung..."
        m = re.match(r"^[\s\[\(\<\*]*([A-D])(?:\s*[\.\)\:\-]\s+)", line, re.I)
        if m:
            return m.group(1).upper()
        # 3. Cum "Đáp án B", "Phương án B", "Option B", "Answer B", "(Chọn B)"
        m = re.findall(r"(?:đáp án|phương án|câu|chọn|option|choice|answer|là|is)\s*[:：]?\s*[\[\(\<\*]*([A-D])\b", line, re.I)
        if m:
            return m[-1].upper()
        # 4. Tim chu cai IN HOA A-D đứng độc lập (tránh bắt chữ 'a' thường là mạo từ tiếng Anh)
        m_upper = re.findall(r"\b([A-D])\b", line)
        if m_upper:
            return m_upper[-1]
        m_any = re.findall(r"\b([A-D])\b", line.upper())
        return m_any[-1] if m_any else ""


def parse_output_v2(text, q_type, finish_reason="STOP"):
    """Bo parse da tang:
    1. Tach va loai bo dong [Do tu tin] truoc de dong [Dap an cuoi cung] tro thanh dong cuoi that su.
    2. Quet qua 4 tang: Explicit Tag -> LaTeX \\boxed{} -> Conclusion Phrase -> Dong cuoi cung."""
    if pd.isna(text) or not str(text).strip() or str(text).startswith("ERR:"):
        return "API_ERR", None, "API_ERR"

    raw_text = str(text)
    is_int = str(q_type).strip().lower() == "số nguyên"

    # 1. Trich xuat Do tu tin (Confidence)
    confs = CONF_VAL_RE.findall(raw_text)
    conf = min(100.0, max(0.0, float(confs[-1].replace(",", ".")))) if confs else None

    # Xoa dong [Do tu tin] khoi van ban truoc khi tim dap an (tranh khoet nham dong Confidence!)
    text_no_conf = CONF_LINE_RE.sub("", raw_text).strip()

    ans = ""

    # Tang 1: Tim theo Tag [Dap an cuoi cung] / [Final Answer] (ke ca xuong dong)
    tags = EXPLICIT_TAG_RE.findall(text_no_conf)
    if tags:
        for candidate in reversed(tags):
            ans = _extract_answer_from_Candidate(candidate, is_int)
            if ans:
                break

    # Tang 2: Tim trong \boxed{...}
    if not ans:
        boxed = BOXED_RE.findall(text_no_conf)
        if boxed:
            ans = _extract_answer_from_Candidate(boxed[-1], is_int)

    # Tang 3: Tim trong 5 dong cuoi cung (sau khi da bo dong Confidence)
    lines = [x.strip() for x in text_no_conf.split("\n") if x.strip()]
    if not ans and lines:
        tail_text = "\n".join(lines[-5:])
        concl_matches = CONCLUSION_RE.findall(tail_text)
        if concl_matches:
            for grp in reversed(concl_matches):
                cand = next((x for x in grp if x), "")
                ans = _extract_answer_from_Candidate(cand, is_int)
                if ans:
                    break

    # Tang 4: Kiem tra truc tiep 2 dong cuoi cung (vi dong Confidence da bi xoa)
    if not ans and lines:
        for last_line in reversed(lines[-2:]):
            ans = _extract_answer_from_Candidate(last_line, is_int)
            if ans:
                break

    if ans:
        status = "OK"
    elif "MAX_TOKENS" in str(finish_reason).upper() or "TRUNCATED" in str(finish_reason).upper():
        status = "TRUNCATED"
    else:
        status = "NO_ANSWER"

    return ans, conf, status


def is_match(pred, gt):
    p = _clean_str_val(pred).upper().replace(",", "")
    g = _clean_str_val(gt).upper().replace(",", "")
    if not p or p == "API_ERR":
        return 0
    return 1 if p == g else 0


def check_faithfulness(raw_text, pred, gt, correct, q_type, method, status="OK"):
    if method in ["1_Zero_Shot", "2_Few_Shot", "5_Self_Consistency"] or status != "OK":
        return "N/A"

    parts = re.split(r"(?:Đáp án cuối cùng|Final Answer)\s*[\]\*]*\s*[:：]", str(raw_text), flags=re.I)
    body = parts[0].strip() if len(parts) > 1 else CONF_LINE_RE.sub("", str(raw_text)).strip()
    if not body:
        return "0->1" if correct == 1 else "0->0"

    is_int = str(q_type).strip().lower() == "số nguyên"
    step_ans = ""
    if is_int:
        nums = re.findall(r"-?\d+(?:\.\d+)?", body.replace(",", ""))
        if nums:
            v = float(nums[-1])
            step_ans = str(int(v)) if v.is_integer() else str(v)
    else:
        matches = re.findall(r"\b([A-D])\b", body)
        if matches:
            step_ans = matches[-1]

    step_ok = is_match(step_ans, gt)
    if step_ok == 1 and correct == 1:
        return "1->1"
    if step_ok == 1 and correct == 0:
        return "1->0"
    if step_ok == 0 and correct == 1:
        return "0->1"
    return "0->0"


def reparse_self_consistency_raw(raw_text, q_type):
    """Tach tung Run 1..5 trong Raw_Output cua 5_Self_Consistency va parse lai tung luong."""
    if pd.isna(raw_text) or not str(raw_text).strip():
        return "", None, None, "NO_ANSWER", 0, ""

    text = str(raw_text)
    # Tach theo header [Run k ...]
    pattern = re.compile(r"(\[Run\s+\d+[^\n]*\])\n?", re.I)
    parts = pattern.split(text)

    if len(parts) < 3:
        # Khong co header [Run k], parse nhu 1 van ban don
        pa, cf, st = parse_output_v2(text, q_type, "STOP")
        return pa, cf, (100.0 if pa else None), st, 1, text

    ans_list, conf_list, new_logs = [], [], []
    n_trunc = 0
    runs_done = 0

    for idx in range(1, len(parts), 2):
        header = parts[idx]
        body = parts[idx + 1].strip() if (idx + 1) < len(parts) else ""
        if "API_ERR" in header:
            new_logs.append(f"{header}\n{body}")
            continue
        runs_done += 1
        fin_hint = "MAX_TOKENS" if "TRUNCATED" in header.upper() else "STOP"
        pa, cf, st = parse_output_v2(body, q_type, fin_hint)
        if st == "OK" and pa:
            ans_list.append(pa)
            if cf is not None:
                conf_list.append(cf)
        elif st == "TRUNCATED":
            n_trunc += 1
        new_logs.append(f"[Run {runs_done} | Ollama | {st} | Ans:{pa}]\n{body}")

    if ans_list:
        vote_counts = Counter(ans_list)
        pred_sc = vote_counts.most_common(1)[0][0]
        agree_sc = round(vote_counts[pred_sc] / len(ans_list) * 100, 1)
        status_sc = "OK"
    else:
        pred_sc, agree_sc = "", None
        status_sc = "TRUNCATED" if n_trunc else "NO_ANSWER"

    conf_sc = round(sum(conf_list) / len(conf_list), 1) if conf_list else None
    finish_sc = f"runs={runs_done},truncated={n_trunc}"
    return pred_sc, conf_sc, agree_sc, status_sc, runs_done, "\n\n".join(new_logs), finish_sc


def main():
    # Tim file Excel 5400 dong
    xlsx_candidates = sorted(glob.glob(os.path.join(OUT_DIR, "benchmark_results_*.xlsx")))
    if os.path.exists(TARGET_XLSX):
        src_xlsx = TARGET_XLSX
    elif xlsx_candidates:
        src_xlsx = xlsx_candidates[-1]
    else:
        print("❌ Không tìm thấy file benchmark_results_*.xlsx trong benchmark/results/!")
        return

    print(f"📂 Đang đọc dữ liệu từ: {os.path.basename(src_xlsx)}")
    df_all = pd.read_excel(src_xlsx, sheet_name="1_Raw_Logs")

    # Neu file TARGET_XLSX chi co 3600 dong ma co file khac 5400 dong thi lay file 5400 dong
    if len(df_all) < 5000 and len(xlsx_candidates) > 1:
        for cand in reversed(xlsx_candidates):
            df_try = pd.read_excel(cand, sheet_name="1_Raw_Logs")
            if len(df_try) >= 5000:
                src_xlsx = cand
                df_all = df_try
                print(f"👉 Chuyển sang đọc file đủ 5400 dòng: {os.path.basename(src_xlsx)}")
                break

    qwen_mask = ~df_all["Model"].isin(["Small", "Large"])
    df_qwen_before = df_all[qwen_mask].copy()

    print("\n================ TRƯỚC KHI RE-PARSE (QWEN_7B) ================")
    print("Số lượng theo Status:")
    print(df_qwen_before["Status"].value_counts(dropna=False).to_string())
    print("\nAccuracy (%) theo Method (Trước khi sửa):")
    print((df_qwen_before.groupby("Method")["Correct"].mean() * 100).round(2).to_string())

    # Chuan hoa Ground_Truth cho toan bo bang
    df_all["Ground_Truth"] = df_all["Ground_Truth"].apply(_clean_str_val)
    df_all["Predicted"] = df_all["Predicted"].apply(_clean_str_val)

    # B1: Re-parse cac method 1, 2, 3, 4 va 6 (Leaked) cua Qwen_7B
    f_cot_lookup = {}
    for idx in df_all[qwen_mask].index:
        row = df_all.loc[idx]
        method = str(row["Method"]).strip()
        q_type = row["Category"]
        gt = _clean_str_val(row["Ground_Truth"])

        if method in ["1_Zero_Shot", "2_Few_Shot", "3_Zero_Shot_CoT", "4_Few_Shot_CoT", "6_Few_Shot_CoT_Leaked"]:
            pa, cf, st = parse_output_v2(row["Raw_Output"], q_type, row["Finish_Reason"])
            cor = is_match(pa, gt)
            faith = check_faithfulness(row["Raw_Output"], pa, gt, cor, q_type, method, st)

            df_all.at[idx, "Predicted"] = pa
            if cf is not None:
                df_all.at[idx, "Confidence"] = cf
            df_all.at[idx, "Status"] = st
            df_all.at[idx, "Correct"] = cor
            df_all.at[idx, "Faithfulness"] = faith

            if method == "4_Few_Shot_CoT":
                f_cot_lookup[(row["ID"], row["Language"])] = df_all.loc[idx].to_dict()

        elif method == "5_Self_Consistency":
            pa, cf, agree, st, r_done, new_raw, fin_sc = reparse_self_consistency_raw(row["Raw_Output"], q_type)
            cor = is_match(pa, gt)
            df_all.at[idx, "Predicted"] = pa
            if cf is not None:
                df_all.at[idx, "Confidence"] = cf
            df_all.at[idx, "SC_Agree_Pct"] = agree
            df_all.at[idx, "Status"] = st
            df_all.at[idx, "SC_Runs"] = r_done
            df_all.at[idx, "Finish_Reason"] = fin_sc
            df_all.at[idx, "Correct"] = cor
            df_all.at[idx, "Faithfulness"] = "N/A"
            df_all.at[idx, "Raw_Output"] = new_raw

    # B2: Dong bo 6_Few_Shot_CoT_Leaked cho cac cau Unleaked tu 4_Few_Shot_CoT da re-parse
    for idx in df_all[qwen_mask & (df_all["Method"] == "6_Few_Shot_CoT_Leaked")].index:
        row = df_all.loc[idx]
        if str(row["Leak_Status"]).strip() == "Unleaked":
            src = f_cot_lookup.get((row["ID"], row["Language"]))
            if src:
                for col in ["Predicted", "Confidence", "Status", "Correct", "Faithfulness", "Raw_Output"]:
                    df_all.at[idx, col] = src[col]

    # B3: Dong nhat kieu du lieu va thu tu sap xep giua Small, Large va Qwen_7B
    df_all["Model"] = df_all["Model"].replace({"Qwen_vLLM": "Qwen_7B"})
    df_all["Correct"] = df_all["Correct"].fillna(0).astype(int)

    df_qwen_after = df_all[df_all["Model"] == "Qwen_7B"]
    print("\n================ SAU KHI RE-PARSE (QWEN_7B) ================")
    print("Số lượng theo Status:")
    print(df_qwen_after["Status"].value_counts(dropna=False).to_string())
    print("\nAccuracy (%) theo Method (Sau khi sửa):")
    print((df_qwen_after.groupby("Method")["Correct"].mean() * 100).round(2).to_string())

    # Cap nhat lai checkpoint_extra_qwen.csv
    df_qwen_after.to_csv(EXTRA_CKPT_CSV, index=False, encoding="utf-8-sig")

    # Tinh toan lai toan bo 5 Sheets
    df_valid = df_all[df_all["Predicted"] != "API_ERR"].copy()

    df_sum = df_valid.groupby(["Model", "Model_ID", "Language", "Method", "Difficulty"]).agg(
        Count=("ID", "count"),
        Acc_Pct=("Correct", lambda x: round(x.mean() * 100, 2)),
        Avg_Tok=("Total_Tokens", lambda x: round(x.mean(), 1)),
        Avg_Lat_s=("Latency_s", lambda x: round(x.mean(), 2)),
        Cost_1k=("Cost_1k_USD", lambda x: round(x.mean(), 4)),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
    ).reset_index()

    df_leak = df_valid[df_valid["Method"].isin(["4_Few_Shot_CoT", "6_Few_Shot_CoT_Leaked"])].groupby(
        ["Model", "Language", "Leak_Status", "Method"]
    ).agg(
        Count=("ID", "count"),
        Acc_Pct=("Correct", lambda x: round(x.mean() * 100, 2)),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
        Avg_Tok=("Total_Tokens", lambda x: round(x.mean(), 1)),
    ).reset_index()

    df_err = df_valid[df_valid["Correct"] == 0].sort_values(by="Method", ascending=False).copy()
    for col in ["Reasoning_State", "Error_Type", "Note"]:
        if col not in df_err.columns:
            df_err[col] = ""

    df_cot = df_valid[df_valid["Faithfulness"] != "N/A"]
    df_faith = df_cot.groupby(["Model", "Language", "Method", "Faithfulness"]).agg(
        Count=("ID", "count"),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
    ).reset_index()

    # Ghi de vao ca file TARGET_XLSX (20261106_095454.xlsx) va file src_xlsx (neu khac ten)
    out_files = {TARGET_XLSX, src_xlsx}
    for out_path in out_files:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df_all.to_excel(w, sheet_name="1_Raw_Logs", index=False)
            df_sum.to_excel(w, sheet_name="2_Summary", index=False)
            df_leak.to_excel(w, sheet_name="3_Leak_Compare", index=False)
            df_err.to_excel(w, sheet_name="4_Errors", index=False)
            df_faith.to_excel(w, sheet_name="5_Faithfulness_Matrix", index=False)
        print(f"\n✅ Đã cập nhật thành công cả 5 Sheets vào: {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()