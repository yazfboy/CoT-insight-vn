"""Tab 2 - CoT giúp đến đâu? (CH1 + CH4).

Kết luận: với Gemini/Gemma, CoT tăng ~+10 điểm, dồn vào câu Trung bình/Khó, câu Dễ gần
như không đổi; với Qwen 7B, CoT tăng nhiều hơn và cũng tăng dần theo độ khó.
File kết quả đã được chấm lại offline (benchmark/rescore_results.py) vì lỗi parse ở Qwen.
Self-Consistency tốn ~3x token mà chỉ thêm 0-5 điểm. Rò rỉ công thức đẩy accuracy lên.

Bố cục mỗi khối: khẳng định -> loại biểu đồ -> biểu đồ -> Đọc -> Kết luận.
Nguồn: 2_Summary, 3_Leak_Compare, 1_Raw_Logs. So sánh các model & VI/EN song song.
Các con số trong Đọc/Kết luận tính từ dữ liệu để luôn khớp với file kết quả.
"""
import pandas as pd
import streamlit as st
from .. import data, charts, stats

QWEN = "Qwen_7B"

# Các so sánh chính (ghép cặp trên cùng câu). (nhãn, phương pháp A, phương pháp B, chỉ câu Leaked)
COMPARISONS = [
    ("Zero-Shot → Zero-Shot CoT", "1_Zero_Shot", "3_Zero_Shot_CoT", False),
    ("Few-Shot vs Zero-Shot", "1_Zero_Shot", "2_Few_Shot", False),
    ("Zero-CoT → Self-Consistency", "3_Zero_Shot_CoT", "5_Self_Consistency", False),
    ("Rò rỉ: Few-CoT → Leaked (câu Leaked)", "4_Few_Shot_CoT", "6_Few_Shot_CoT_Leaked", True),
]


def _acc(s, model, method, **flt):
    d = s[(s["Model"] == model) & (s["Method"] == method)]
    for k, v in flt.items():
        d = d[d[k] == v]
    return float(d["Acc_Pct"].mean()) if len(d) else float("nan")


def _series(raw, model, method, leak=False):
    """Series Correct (0/1) của 1 model×phương pháp, đánh index theo câu (ID_ngôn ngữ)."""
    d = raw[(raw["Model"] == model) & (raw["Method"] == method)]
    if leak:
        d = d[d["Leak_Status"] == "Leaked"]
    d = d.assign(item=d["ID"].astype(str) + "_" + d["Language"].astype(str))
    return d.set_index("item")["Correct"]


def _stats_table(raw, models):
    """Tính 1 LẦN mọi so sánh × model; hiệu chỉnh Holm trên toàn bộ. Trả DataFrame hiển thị +
    dict tra cứu {(tên so sánh, model): {delta, p, p_holm, sig}}."""
    rows, pv = [], []
    for name, ma, mb, leak in COMPARISONS:
        for m in models:
            r = stats.mcnemar(_series(raw, m, ma, leak), _series(raw, m, mb, leak))
            rows.append({"So sánh": name, "Model": m,
                         "Δ điểm": round(r["acc_b"] - r["acc_a"], 1), "n": r["n"],
                         "lệch (B/A)": f"{r['b10']}/{r['b01']}", "_p": r["p"]})
            pv.append(r["p"])
    adj = stats.holm(pv)
    look = {}
    for row, a in zip(rows, adj):
        look[(row["So sánh"], row["Model"])] = {
            "delta": row["Δ điểm"], "p": row["_p"], "p_holm": a, "sig": a < 0.05}
        row["p"] = f"{row.pop('_p'):.2g}"
        row["p (Holm)"] = f"{a:.2g}"
        row["Kết luận"] = stats.verdict(a)
    return pd.DataFrame(rows), look


def _sig(look, comp_name, models):
    """Danh sách model mà so sánh đó có ý nghĩa (sau Holm)."""
    return [m for m in models if look.get((comp_name, m), {}).get("sig")]


def render(ctx):
    path = ctx["bench_path"]
    s = data.summary(path)
    raw = data.raw_logs(path)
    models = charts._models(s)
    base = [m for m in models if m != QWEN]          # Gemini/Gemma
    has_q = QWEN in models

    st.subheader("CoT giúp đến đâu, và khi nào thì không?")

    # Hàng chỉ số: TÁCH theo model (gộp sẽ bị Qwen kéo lệch).
    cols = st.columns(len(models) + 1)
    for c, m in zip(cols, models):
        zs, zc = _acc(s, m, "1_Zero_Shot"), _acc(s, m, "3_Zero_Shot_CoT")
        c.metric(f"{m}: Zero-Shot → Zero-CoT", f"{zs:.0f} → {zc:.0f}%", f"+{zc - zs:.0f} điểm")
    tok = s.groupby("Method")["Avg_Tok"].mean()
    tok_ratio = tok["5_Self_Consistency"] / tok["3_Zero_Shot_CoT"]
    cols[-1].metric("Token: SC / Zero-CoT", f"{tok_ratio:.1f}×")

    if has_q:
        st.caption("Qwen_7B tự host (qwen2.5:7b, Ollama). Mọi so sánh tách theo model; con số "
                   "tóm tắt tính riêng từng model để không bị kéo lệch khi gộp.")

    st.divider()

    # Lớp kiểm định thống kê — tính 1 lần, dùng chung cho mọi kết luận bên dưới.
    stt_tbl, look = _stats_table(raw, models)
    with st.expander("Kiểm định thống kê — McNemar ghép cặp (đọc trước khi tin các chênh lệch)"):
        st.caption("Mỗi câu được chạy qua mọi phương pháp nên kết quả KHÔNG độc lập → dùng kiểm "
                   "định ghép cặp. 'lệch (B/A)' = số câu chỉ B đúng / chỉ A đúng. p đã hiệu chỉnh "
                   "đa so sánh (Holm). 'không đáng kể' = chưa đủ bằng chứng (p≥0.05), KHÔNG phải "
                   "bằng nhau.")
        st.dataframe(stt_tbl, width="stretch", hide_index=True)

    # ---- 1. Theo độ khó ----
    st.markdown("**1. CoT giúp *khi nào*?** — theo độ khó, từng model.")
    st.markdown("*Biểu đồ cột nhóm — accuracy (%) theo phương pháp × độ khó, tách theo model.*")
    st.plotly_chart(charts.acc_by_method_difficulty(s), width="stretch")
    easy_gain = max((_acc(s, m, "3_Zero_Shot_CoT", Difficulty="Easy")
                     - _acc(s, m, "1_Zero_Shot", Difficulty="Easy")) for m in base)
    st.caption("Đọc: so từng cụm model. Với " + "/".join(base) + ", cột **Dễ** đã kịch trần "
               f"(CoT thêm tối đa +{easy_gain:.0f}đ) còn khoảng cách nới rộng ở **Trung bình** "
               "và **Khó**.")
    msg = ("**Kết luận:** với Gemini/Gemma, CoT chỉ đáng dùng cho câu Trung bình/Khó; câu Dễ "
           "không thêm điểm mà chỉ tốn token.")
    if has_q:
        q_easy = (_acc(s, QWEN, "3_Zero_Shot_CoT", Difficulty="Easy")
                  - _acc(s, QWEN, "1_Zero_Shot", Difficulty="Easy"))
        msg += (f" Với Qwen 7B thì khác: CoT giúp ở **mọi** độ khó, kể cả câu Dễ "
                f"(+{q_easy:.0f}đ) — model yếu được lợi nhiều hơn.")
    st.success(msg)

    # ---- 2. Chi phí ----
    st.markdown("**2. Cái giá của CoT** — token đổi lấy accuracy.")
    st.markdown("*Biểu đồ phân tán — accuracy (%) theo số token trung bình mỗi câu.*")
    st.plotly_chart(charts.cost_vs_accuracy(s), width="stretch")
    gains = {m: _acc(s, m, "5_Self_Consistency") - _acc(s, m, "3_Zero_Shot_CoT") for m in models}
    st.caption(f"Đọc: điểm càng sang phải càng tốn token. Self-Consistency tốn ~{tok_ratio:.0f}× "
               "token so với Zero-Shot CoT; mức tăng so với Zero-CoT: "
               + ", ".join(f"{m} {g:+.0f}đ" for m, g in gains.items()) + ".")
    if has_q:
        st.warning("**So sánh chi phí chỉ hợp lệ TRONG cùng một model.** Qwen 7B **tự host** nên "
                   "giá (USD) là mức giả định và độ trễ phụ thuộc phần cứng (≈22s/câu vs ~2–9s "
                   "của API); số token cũng không so trực tiếp được giữa hai họ model vì "
                   "tokenizer khác nhau. Trục token chỉ dùng để so phương pháp *bên trong* một "
                   "model, không so Qwen với Gemini/Gemma.")
    sig_sc = _sig(look, "Zero-CoT → Self-Consistency", models)
    msg = ("**Kết luận:** Zero-Shot CoT là điểm ngọt chi phí–hiệu quả. Self-Consistency tốn "
           f"~{tok_ratio:.0f}× token nhưng sau hiệu chỉnh đa so sánh, mức tăng so với CoT "
           + (f"chỉ có ý nghĩa ở {', '.join(sig_sc)}" if sig_sc
              else "**không có ý nghĩa ở model nào**")
           + " → lãi rất mỏng, không đáng chi phí.")
    st.success(msg)

    # ---- 3. Heatmap Δ ----
    st.markdown("**3. Mức tăng của CoT theo độ khó** (Zero-Shot → Zero-Shot CoT).")
    st.markdown("*Bản đồ nhiệt — mức tăng Δ (điểm) theo độ khó × (model·ngôn ngữ).*")
    st.plotly_chart(charts.delta_heatmap(s), width="stretch")
    qd = {}
    if has_q:
        qd = {lv: _acc(s, QWEN, "3_Zero_Shot_CoT", Difficulty=lv)
              - _acc(s, QWEN, "1_Zero_Shot", Difficulty=lv) for lv in ("Easy", "Medium", "Hard")}
    st.caption("Đọc: ô càng **đậm (xanh tối)**, CoT tăng càng nhiều điểm. Với Gemini/Gemma, hàng "
               "**Dễ** nhạt gần như bằng 0, hàng **Trung bình/Khó** đậm rõ ở cả hai ngôn ngữ."
               + (f" Ở Qwen: Dễ {qd['Easy']:+.0f} · Trung bình {qd['Medium']:+.0f} · "
                  f"Khó {qd['Hard']:+.0f} điểm." if has_q else ""))
    msg = ("**Kết luận:** với Gemini/Gemma, lợi ích của CoT tỉ lệ thuận với độ khó và ổn định "
           "qua hai ngôn ngữ — không phải hiệu ứng ngẫu nhiên.")
    if has_q:
        if qd["Easy"] < qd["Medium"] < qd["Hard"]:
            msg += (f" Qwen cũng tăng dần theo độ khó nhưng ở mức cao hơn hẳn (Dễ "
                    f"{qd['Easy']:+.0f} → Khó {qd['Hard']:+.0f}) — model yếu được lợi nhiều hơn.")
        else:
            msg += (" Ở Qwen, mức tăng không đi theo độ khó (Dễ "
                    f"{qd['Easy']:+.0f} · Trung bình {qd['Medium']:+.0f} · Khó {qd['Hard']:+.0f}).")
    st.success(msg)

    # ---- 4. VI vs EN ----
    st.markdown("**4. Tiếng Việt vs Tiếng Anh.**")
    st.markdown("*Biểu đồ quả tạ — accuracy (%) của VI và EN theo từng phương pháp, tách theo model.*")
    st.plotly_chart(charts.lang_dumbbell(s), width="stretch")
    gap_cot = max(abs(_acc(s, m, "3_Zero_Shot_CoT", Language="EN")
                      - _acc(s, m, "3_Zero_Shot_CoT", Language="VI")) for m in models)
    st.caption("Đọc: mỗi thanh nối điểm VI và EN của một phương pháp; thanh càng dài, chênh "
               "lệch ngôn ngữ càng lớn (rõ nhất ở các phương pháp không-CoT).")
    st.success(f"**Kết luận:** với Zero-Shot CoT, chênh Việt–Anh chỉ còn ≤{gap_cot:.0f} điểm ở cả "
               f"{len(models)} model — suy luận từng bước giúp cân bằng hai ngôn ngữ.")

    # ---- 5. Few-shot ----
    st.markdown("**5. Không phải cứ cho ví dụ là tốt.**")
    st.markdown("*Biểu đồ cột (tách theo model) — accuracy (%) theo từng phương pháp.*")
    st.plotly_chart(charts.method_family_bar(s, raw), width="stretch")
    below = [m for m in models if _acc(s, m, "2_Few_Shot") < _acc(s, m, "1_Zero_Shot")]
    above = [m for m in models if m not in below]
    st.caption("Đọc: đường chấm là mức Zero-Shot của từng model; thanh **│** trên mỗi cột là "
               "khoảng tin cậy 95% (Wilson, n=300). Few-Shot *chỉ cho đáp án, không kèm lời giải* "
               + (f"**thấp hơn** Zero-Shot ở {'/'.join(below)}" if below else "")
               + ("; " if below and above else "")
               + (f"cao hơn ở {'/'.join(above)}" if above else "")
               + ". Xem bảng kiểm định ở đầu tab để biết chênh lệch nào có ý nghĩa.")
    sig_fs = _sig(look, "Few-Shot vs Zero-Shot", models)
    if not above:
        msg = (f"**Kết luận:** hướng đều nhất quán — Few-Shot *chỉ-đáp-án* thấp hơn Zero-Shot ở "
               f"cả {len(models)} model")
    else:
        msg = ("**Kết luận:** Few-Shot chỉ-đáp-án thấp hơn ở " + "/".join(below or ["—"])
               + f", cao hơn ở {'/'.join(above)}")
    msg += ("; nhưng sau hiệu chỉnh đa so sánh, khác biệt chỉ **có ý nghĩa** ở "
            + (", ".join(sig_fs) if sig_fs else "không model nào")
            + ". Dù sao, chỉ ví dụ **kèm lời giải** (Few-CoT) mới cho điểm cao ở mọi model.")
    st.success(msg)

    # ---- 6. Rò rỉ ----
    lc = data.leak_compare(path)
    lk = lc[lc["Leak_Status"] == "Leaked"]
    pv = lk.pivot_table(index=["Model", "Language"], columns="Method", values="Acc_Pct")
    dl = (pv["6_Few_Shot_CoT_Leaked"] - pv["4_Few_Shot_CoT"])
    dvi = dl.xs("VI", level="Language").mean()
    den = dl.xs("EN", level="Language").mean()
    n_leak = int(lk["Count"].max()) if "Count" in lk.columns else 49
    st.markdown("**6. CH4 — Rò rỉ công thức (contamination).**")
    st.markdown("*Biểu đồ cột nhóm — accuracy (%): prompt sạch vs rò rỉ, trên câu Leaked.*")
    st.plotly_chart(charts.leak_effect(lc), width="stretch")
    comp_leak = "Rò rỉ: Few-CoT → Leaked (câu Leaked)"
    sig_leak = _sig(look, comp_leak, models)
    pstr = ", ".join(f"{m} p={look[(comp_leak, m)]['p_holm']:.2g}" for m in models)
    st.caption(f"Đọc: chỉ tính trên {n_leak} câu thuộc nhóm rò rỉ. Lộ công thức (pp6) thay đổi "
               f"accuracy {dl.min():+.0f} đến {dl.max():+.0f} điểm so với pp4 (trung bình "
               f"{dl.mean():+.0f}). Câu không rò rỉ có Δ=0 (pp6 dùng lại pp4) làm đối chứng.")
    if sig_leak:
        st.success(f"**Kết luận:** rò rỉ làm tăng accuracy có ý nghĩa ở {'/'.join(sig_leak)} "
                   f"(McNemar-Holm; {pstr}).")
    else:
        st.warning("**Kết luận (thận trọng):** chênh lệch do rò rỉ đều **dương** nhưng cỡ mẫu "
                   f"nhỏ (n={n_leak}) nên **chưa đạt ý nghĩa thống kê ở từng model** (Holm: {pstr}). "
                   "Chỉ nên nói 'có dấu hiệu nhạy với rò rỉ', không khẳng định '+4–6 điểm'. "
                   "Cần thêm câu rò rỉ để kết luận chắc.")

    with st.expander("Giải thích & giới hạn"):
        st.markdown(
            "- **Vì sao CoT giúp câu khó hơn câu dễ (Gemini/Gemma):** câu dễ model trả lời đúng "
            "ngay nên không còn chỗ để cải thiện (trần ~100%); câu khó cần nhiều bước trung "
            "gian — đây đúng là chỗ 'suy nghĩ từng bước' phát huy.\n"
            f"- **Self-Consistency:** chạy CoT tối đa 5 lần ở T=0.7 rồi lấy đa số phiếu (dừng "
            f"sớm khi đạt 3/5); tốn ~{tok_ratio:.0f}× token so với CoT đơn.\n"
            "- **Qwen 7B:** chạy trên server tự host. Kết quả đã được chấm lại offline từ "
            "Raw_Output do bộ parse cũ bỏ sót đáp án ghi trần (xem hộp lưu ý đầu tab); vẫn còn "
            "vài lượt dùng định dạng lạ (`\\boxed{}`, `[54]`) bị tính là không có đáp án.\n"
            "- **Rò rỉ:** là proxy cho nhiễm bẩn benchmark — điểm cao cần đọc thận trọng.\n"
            "- Mọi con số gộp trung bình theo nhóm; xem bảng dưới để tra chi tiết."
        )
        st.dataframe(s, width="stretch")
