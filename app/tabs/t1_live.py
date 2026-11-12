"""Tab 1 - Thử ngay (Live Demo).

Cho hội đồng tự thấy cùng 1 câu hỏi, trả lời thẳng vs suy nghĩ từng bước ra kết quả
khác nhau. Mặc định PHÁT LẠI kết quả đã lưu trong 1_Raw_Logs (chạy ngay, không lỗi
mạng); nút "Gọi API thật" chạy trực tiếp qua app/llm.
"""
import pandas as pd
import streamlit as st
from .. import data, charts, llm

SHORTS = [charts.METHOD_SHORT[m] for m in charts.METHOD_ORDER]
# Model gọi live được qua benchmark/run_benchmark.py (Qwen_7B tự host -> chỉ phát lại).
LIVE_MODELS = {"Small", "Large"}
SHORT2KEY = {charts.METHOD_SHORT[m]: m for m in charts.METHOD_ORDER}


def _card(c, method_key, rec, live):
    short = charts.METHOD_SHORT[method_key]
    c.markdown(f"**{short}**")
    if rec is None:
        c.caption("— chưa có kết quả —")
        return
    pred = rec.get("Predicted", "")
    correct = rec.get("Correct")
    mark = "✅" if correct == 1 else ("❌" if correct == 0 else "")
    if pred is None or (isinstance(pred, float) and pd.isna(pred)) or str(pred).strip() == "":
        status = rec.get("Status") or "NO_ANSWER"
        c.markdown(f"Đáp án: *không trích được* {mark}")
        why = {"TRUNCATED": "câu trả lời bị cắt vì hết giới hạn token",
               "NO_ANSWER": "model không ghi đúng dòng đáp án cuối",
               "API_ERR": "lỗi gọi API"}.get(str(status), "không đọc được đáp án")
        c.caption(f"Trạng thái: `{status}` ({why})")
    else:
        c.markdown(f"Đáp án: **{pred}** {mark}")
    conf = rec.get("Confidence")
    tok = rec.get("Total_Tokens") or 0
    lat = rec.get("Latency_s")
    c.caption(f"Tự tin: {('—' if pd.isna(conf) else conf)} · "
              f"{int(tok)} token · {('—' if lat is None else f'{lat:.2f}s')}")
    faith = rec.get("Faithfulness")
    if faith and str(faith) != "N/A":
        c.caption(f"Faithfulness: `{faith}`")
    if method_key == "5_Self_Consistency":
        votes = rec.get("_votes")
        if votes:
            c.plotly_chart(charts.sc_vote_bar(votes), width="stretch",
                           key=f"sc_{short}_{live}")
        elif rec.get("SC_Agree_Pct") is not None:
            c.caption(f"Đồng thuận: {rec.get('SC_Agree_Pct')}%")
    with c.expander("Lời giải"):
        st.text(str(rec.get("Raw_Output", "")).strip())


def _render_results(res):
    gt = res.get("gt")
    if gt not in (None, "", "nan"):
        st.caption(f"Đáp án chuẩn: **{gt}**")
    keys = res["order"]
    recs = res["records"]
    live = res["live"]
    for i in range(0, len(keys), 3):
        chunk = keys[i:i + 3]
        cols = st.columns(len(chunk))
        for col, k in zip(cols, chunk):
            _card(col, k, recs.get(k), live)


def render(ctx):
    path = ctx["bench_path"]
    ds = data.dataset_test()

    st.subheader("Thử ngay: trả lời thẳng vs suy nghĩ từng bước")

    src = st.radio("Nguồn câu hỏi", ["Chọn từ 150 câu", "Tự nhập câu mới"],
                   horizontal=True, key="t1_src")
    col_m, col_l = st.columns(2)
    # Danh sách model lấy từ file kết quả; chỉ model có trong run_benchmark mới gọi live được.
    models = charts._models(data.raw_logs(path))
    model = col_m.radio("Mô hình", models, horizontal=True, key="t1_model")
    lang = col_l.radio("Ngôn ngữ", ["VI", "EN"], horizontal=True, key="t1_lang")
    can_live = model in LIVE_MODELS
    if not can_live:
        col_m.caption(f"{model} chạy trên server tự host → chỉ **phát lại** kết quả đã lưu.")
    picked = st.multiselect("Phương pháp (so sánh song song)", SHORTS,
                            default=["Zero-Shot", "Zero-CoT", "Self-Consist"], key="t1_meth")
    sel = [m for m in charts.METHOD_ORDER if charts.METHOD_SHORT[m] in picked]

    custom = src == "Tự nhập câu mới"
    row = None
    if custom:
        qtext = st.text_area("Nhập câu hỏi", key="t1_qtext",
                             placeholder="VD: Một cửa hàng... Hỏi tổng cộng bao nhiêu?")
        c1, c2 = st.columns(2)
        qtype = c1.selectbox("Dạng câu hỏi", ["số nguyên", "A/B/C/D"], key="t1_qtype")
        gt = c2.text_input("Đáp án chuẩn (tuỳ chọn)", key="t1_gt")
        row = pd.Series({
            "ID": "CUSTOM", "Mức độ": "Trung bình", "Dạng câu hỏi": qtype,
            "Câu hỏi Tiếng Việt": qtext, "Câu hỏi Tiếng Anh": qtext,
            "Đáp án": gt, "Nhóm_Màu_Leak": "Khong_Lo_Unleaked"})
        st.info("Câu tự nhập chỉ chạy được bằng **Gọi API thật** (không có sẵn để phát lại).")
    else:
        levels = {"Dễ": "Easy", "Trung bình": "Medium", "Khó": "Hard"}
        lv = st.radio("Độ khó", list(levels), horizontal=True, key="t1_lv")
        pool_ids = ds[ds["Mức độ"] == lv]["ID"].tolist()
        qid = st.selectbox("Chọn câu", pool_ids, key="t1_qid")
        row = ds[ds["ID"] == qid].iloc[0]
        qtext = row["Câu hỏi Tiếng Việt"] if lang == "VI" else row["Câu hỏi Tiếng Anh"]
        st.info(qtext)

    b1, b2 = st.columns(2)
    do_replay = b2.button("Phát lại kết quả đã lưu", disabled=custom or not sel,
                          width="stretch", key="t1_replay")
    do_live = b1.button("Gọi API thật", type="primary", disabled=(not sel) or (not can_live),
                        width="stretch", key="t1_live_btn")
    if custom and not can_live:
        st.warning(f"Câu tự nhập cần gọi API thật, mà {model} không hỗ trợ live — "
                   "hãy chọn Small hoặc Large.")

    if do_replay and not custom:
        recs = {k: data.raw_lookup(path, row["ID"], model, lang, k) for k in sel}
        st.session_state["t1_res"] = {
            "records": recs, "order": sel, "gt": str(row["Đáp án"]), "live": False}

    if do_live:
        try:
            pool = llm.get_pool()
            with st.spinner("Đang gọi API..."):
                recs = llm.run_methods_live(row, model, lang, sel, pool)
            st.session_state["t1_res"] = {
                "records": recs, "order": sel, "gt": str(row["Đáp án"]), "live": True}
        except Exception as e:  # noqa
            st.error(f"Không gọi được API: {e}. Dùng 'Phát lại kết quả đã lưu' để xem dữ liệu sẵn có.")

    res = st.session_state.get("t1_res")
    if res:
        st.divider()
        st.caption("Nguồn: **chạy trực tiếp**" if res["live"] else "Nguồn: **kết quả đã lưu (phát lại)**")
        _render_results(res)
    else:
        st.caption("Chọn câu + phương pháp rồi bấm **Phát lại** (hoặc **Gọi API thật**).")
