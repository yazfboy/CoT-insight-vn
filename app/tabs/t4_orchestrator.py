"""Tab 4 - Bộ điều phối CoT (CH5).

Kết luận: trên 20 ca vốn sai, Premise Extraction sửa 70%, PAL 8/10 câu số, còn lại
~65%; hiệu chỉnh độ tự tin đưa Brier 0.945 -> 0.247.

Phần A - Kết quả đã chạy (tĩnh, từ workbook evidence Gói 3).
Phần B - Bật/tắt trực tiếp (Giai đoạn 4): phát lại + gọi API thật.
"""
import pandas as pd
import streamlit as st
from .. import data, charts, llm

# UI ngắn -> tên Method trong Case_Evidence (phát lại).
FIX_DISPLAY = {
    "Premise": "Premise Extraction",
    "Strict CoT": "Strict CoT",
    "PAL": "PAL",
    "NLI": "Post-generation NLI guardrail",
    "Calibration": "Logprob & Temperature Calibration",
}
FIX_LIST = ["Premise", "Strict CoT", "PAL", "NLI", "Calibration"]


def render(ctx):
    st.subheader("Có sửa được những câu CoT làm sai không?")
    tab_a, tab_b = st.tabs(["A. Kết quả đã chạy", "B. Bật/tắt trực tiếp (demo)"])

    with tab_a:
        st.markdown("**1. Mỗi can thiệp cứu được bao nhiêu trong 20 ca sai?**")
        st.plotly_chart(charts.fix_rate_by_method(data.evidence_fix()), width="stretch")
        st.caption("Đọc: nhãn trên cột là *số ca sửa được / số ca áp dụng được*. PAL chỉ chạy "
                   "trên 10 câu số (10 câu trắc nghiệm bị bỏ qua) nên mẫu số là 10.")
        st.success("**Kết luận:** trích tiền đề (Premise) hiệu quả nhất (70%); PAL chính xác "
                   "nhất trên phần nó áp dụng được (8/10) nhưng phạm vi hẹp.")

        st.markdown("**2. Kỹ thuật nào hợp với kiểu lỗi nào?**")
        st.markdown("*Ma trận nhiệt — tỷ lệ sửa được (%) theo can thiệp × kiểu lỗi.*")
        st.plotly_chart(charts.fix_by_errortype(data.evidence_cases()), width="stretch")
        st.caption("Đọc: mỗi ô = % số ca thuộc kiểu lỗi đó được can thiệp lật thành đúng "
                   "(kèm số ô sửa/áp dụng). Ô càng **đậm xanh** = kỹ thuật càng hợp với kiểu "
                   "lỗi đó; '–' = không có ca áp dụng được (PAL trên câu trắc nghiệm).")
        st.success("**Kết luận:** không kỹ thuật nào mạnh đều ở mọi kiểu lỗi — mỗi kiểu lỗi có "
                   "can thiệp phù hợp riêng, gợi ý nên phối hợp thay vì dùng một kỹ thuật duy nhất.")

        st.markdown("**3. Hiệu chỉnh độ tự tin có đáng không?**")
        st.plotly_chart(charts.calibration_bars(data.calibration_before_after()),
                        width="stretch")
        st.caption("Đọc: Brier càng thấp càng tốt (độ tự tin khớp thực tế hơn); độ tự tin "
                   "trung bình kéo từ ~97 xuống ~87 trên tập 20 ca vốn sai.")
        st.success("**Kết luận:** hiệu chỉnh nhiệt độ + calibration cải thiện mạnh độ tin cậy "
                   "của điểm tự tin (Brier 0.945 → 0.247) dù accuracy chỉ nhích nhẹ.")

        with st.expander("Giải thích & giới hạn"):
            st.markdown(
                "- 20 ca là tập *chọn lọc toàn câu sai* → 'accuracy sau' là **tỷ lệ khắc "
                "phục**, không phải accuracy tổng quát.\n"
                "- NLI guardrail dùng chính model cùng họ làm 'checker', không phải một bộ "
                "phân loại NLI độc lập.\n"
                "- Logprob thật không lấy được → dùng self-report làm fallback; cả hai model "
                "chọn nhiệt độ 1.0 do hoà.\n"
                "- DPO/RLAIF bị bỏ qua theo yêu cầu; chỉ 5 can thiệp này được chạy."
            )

    with tab_b:
        _render_demo()


def _is_integer_q(category) -> bool:
    s = str(category).lower()
    return ("nguyên" in s) or ("số" in s) or ("integer" in s)


def _after_card(c, short, live, replay_row=None, live_row=None):
    c.markdown(f"**{short}**")
    if live and live_row is not None:
        pred = live_row.get("after_predicted")
        correct = live_row.get("after_correct")
        cb, ca = live_row.get("before_confidence"), live_row.get("after_confidence")
        status, note = live_row.get("status"), live_row.get("note")
        raw = live_row.get("raw_output")
    elif replay_row is not None:
        pred = replay_row.get("After prediction")
        correct = replay_row.get("Correct after")
        cb, ca = replay_row.get("Confidence before"), replay_row.get("Confidence after")
        status, note = replay_row.get("Status"), replay_row.get("Note")
        raw = replay_row.get("After raw output")
    else:
        c.caption("— không có dữ liệu —")
        return
    mark = "✅" if correct == 1 else ("❌" if correct == 0 else "")
    c.markdown(f"Sau sửa: **{pred}** {mark}")
    fmt = lambda v: "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{float(v):.0f}"
    c.caption(f"Tự tin: {fmt(cb)} → {fmt(ca)}")
    if status:
        c.caption(f"Trạng thái: `{status}`")
    with c.expander("Lời giải sau sửa"):
        st.text(str(raw).strip() if raw is not None else "")
        if note:
            st.caption(f"Ghi chú: {note}")


def _render_demo():
    base = data.evidence_baselines()
    labels = [f"{r['ID']} · {r['Model']}/{r['Language']} · {r['Error_Type']}"
              for _, r in base.iterrows()]
    i = st.selectbox("Chọn 1 trong 20 câu vốn sai", range(len(labels)),
                     format_func=lambda k: labels[k], key="t4b_case")
    case = base.iloc[i]
    task_key = case["Task_Key"]

    st.markdown(f"**Câu hỏi:** {case['Question']}")
    m1, m2 = st.columns(2)
    m1.metric("Đáp án chuẩn", str(case["Ground_Truth"]))
    m2.metric("Baseline trả lời (❌ sai)", str(case["Predicted"]))
    with st.expander("Lời giải gốc (baseline)"):
        st.text(str(case["Raw_Output"]).strip())

    is_int = _is_integer_q(case["Category"])
    st.markdown("**Bật công tắc tối ưu để áp dụng:**")
    cols = st.columns(5)
    toggles = {}
    for col, short in zip(cols, FIX_LIST):
        disabled = (short == "PAL" and not is_int)
        toggles[short] = col.checkbox(short, value=(short == "Premise"),
                                      disabled=disabled, key=f"t4b_{short}",
                                      help="PAL chỉ dùng cho câu số" if disabled else None)
    sel = [s for s, on in toggles.items() if on]

    b1, b2 = st.columns(2)
    do_live = b1.button("Gọi API thật", type="primary", disabled=not sel,
                        width="stretch", key="t4b_live")
    do_replay = b2.button("Phát lại kết quả đã lưu", disabled=not sel,
                          width="stretch", key="t4b_replay")

    if do_replay:
        ce = data.evidence_cases()
        rows = {}
        for s in sel:
            hit = ce[(ce["Task_Key"] == task_key) & (ce["Method"] == FIX_DISPLAY[s])]
            rows[s] = hit.iloc[0].to_dict() if len(hit) else None
        st.session_state["t4b_res"] = {"order": sel, "replay": rows, "live": None}

    if do_live:
        try:
            out = {}
            with st.spinner("Đang chạy lại các can thiệp..."):
                for s in sel:
                    out[s] = llm.run_intervention_live(task_key, s)
            st.session_state["t4b_res"] = {"order": sel, "replay": None, "live": out}
        except Exception as e:  # noqa
            st.error(f"Không gọi được API: {e}. Dùng 'Phát lại kết quả đã lưu'.")

    res = st.session_state.get("t4b_res")
    if res:
        st.divider()
        live = res["live"] is not None
        st.caption("Nguồn: **chạy trực tiếp**" if live else "Nguồn: **kết quả đã lưu (phát lại)**")
        keys = res["order"]
        for j in range(0, len(keys), 3):
            chunk = keys[j:j + 3]
            cc = st.columns(len(chunk))
            for col, s in zip(cc, chunk):
                _after_card(col, s, live,
                            replay_row=(res["replay"] or {}).get(s),
                            live_row=(res["live"] or {}).get(s))
