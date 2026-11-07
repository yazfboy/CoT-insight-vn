"""Tab 3 - Khi CoT sai: sai thế nào? (CH2 + CH3 + phân loại lỗi).

Kết luận: Zero-Shot sai mà vẫn tự tin ~99.5; CoT kéo xuống ~92 nhưng chưa khắc phục
hẳn (76% câu sai vẫn ≥90). Model gần như không có ca 1→0 (nghĩ đúng chốt sai);
lỗi nằm gần như toàn bộ ở bước suy luận.

Bố cục mỗi khối: khẳng định -> biểu đồ -> "Đọc" -> "Kết luận".
Nguồn: 1_Raw_Logs, 5_Faithfulness_Matrix, 4_Errors; Error_Type từ Baselines (Gói 3).
"""
import pandas as pd
import streamlit as st
from .. import data, charts


def render(ctx):
    path = ctx["bench_path"]
    raw = data.raw_logs(path)

    st.subheader("Khi CoT sai thì sai thế nào?")

    # --- CH2: overconfidence ---
    st.markdown("**1. CH2 — CoT có làm model 'sai một cách tự tin' không?**")
    st.markdown("*Biểu đồ hộp (box plot) — phân bố độ tự tin trên các câu SAI theo phương pháp.*")
    st.plotly_chart(charts.confidence_on_wrong(raw), width="stretch")
    st.caption("Đọc: mỗi điểm là một câu **trả lời sai** (jitter ngang cho khỏi chồng); hộp "
               "tóm tắt tứ phân vị. Đường đỏ = ngưỡng 90. Zero-Shot gần như toàn bộ điểm dồn "
               "ở 100 (hộp dẹt ở đỉnh) → luôn cực tự tin dù sai; Zero-CoT trải xuống thấp hơn.")
    st.success("**Kết luận:** CoT *giảm* sự mù quáng — độ tự tin trung bình lúc sai từ ~99.5 "
               "xuống ~92 — nhưng chưa chữa khỏi: vẫn ~76% câu sai được chấm ≥90.")

    # --- CH3: faithfulness ---
    st.markdown("**2. CH3 — Lời giải từng bước có khớp với đáp án cuối không?**")
    st.markdown("*Ma trận nhiệt 2×2 — số ca theo (lập luận đúng/sai) × (đáp án cuối đúng/sai).*")
    st.plotly_chart(charts.faithfulness_matrix(data.faithfulness(path)), width="stretch")
    st.caption("Đọc: 4 ô = (lập luận đúng/sai) × (đáp án cuối đúng/sai). Ô **1→0** là trường "
               "hợp 'nghĩ đúng nhưng chốt sai' — đáng lo nhất, và ở đây rất nhỏ.")
    st.success("**Kết luận:** gần như không có ca 1→0 → model hiếm khi phản bội chính lời "
               "giải của nó; sai chủ yếu vì *suy luận sai* (0→0), không phải vì lệch khâu chốt.")

    # --- Phân loại lỗi 20 ca ---
    st.markdown("**3. 20 câu sai tiêu biểu sai vì kiểu lỗi gì?**")
    st.markdown("*Biểu đồ cột nhóm — số ca theo kiểu lỗi × model.*")
    st.plotly_chart(charts.error_type_breakdown(data.evidence_baselines()), width="stretch")
    st.caption("Đọc: đếm theo nhãn lỗi do người gắn, tách theo model.")
    st.success("**Kết luận:** lỗi *hiểu sai đề* (Misinterpretation) và *làm dở dang* "
               "(Incomplete) chiếm đa số — không phải lỗi số học thuần.")

    # --- Bảng tra lỗi ---
    st.markdown("**4. Tra cứu từng câu sai** (yêu cầu bắt buộc: phân tích ≥20 câu sai).")
    st.markdown("*Bảng — danh sách câu CoT trả lời sai, kèm trích lời giải rút gọn.*")
    err = data.errors(path).reset_index(drop=True)
    methods = st.multiselect("Lọc theo phương pháp",
                             sorted(err["Method"].unique()), default=[], key="t3_method")
    view = (err[err["Method"].isin(methods)] if methods else err).reset_index(drop=True)

    disp = pd.DataFrame({
        "ID": view["ID"],
        "Độ khó": view["Difficulty"],
        "Model": view["Model"],
        "Ngôn ngữ": view["Language"],
        "Phương pháp": view["Method"].map(charts.METHOD_SHORT).fillna(view["Method"]),
        "Đáp án đúng": view["Ground_Truth"].astype(str),
        "Model trả lời": view["Predicted"].astype(str),
        "Tự tin": pd.to_numeric(view["Confidence"], errors="coerce"),
    })
    st.dataframe(
        disp, width="stretch", height=340, hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(width="small"),
            "Tự tin": st.column_config.NumberColumn(
                format="%d", help="Độ tự tin model tự báo (0–100)"),
        },
    )
    st.caption(f"{len(view)} câu CoT sai. Chọn một câu để xem lời giải đầy đủ:")

    # Xem chi tiết một câu (lời giải đầy đủ, giữ xuống dòng cho dễ đọc).
    labels = [f"{r.ID} · {charts.METHOD_SHORT.get(r.Method, r.Method)} · "
              f"{r.Model}/{r.Language}" for r in view.itertuples()]
    if labels:
        i = st.selectbox("Xem chi tiết", range(len(labels)),
                         format_func=lambda k: labels[k], key="t3_pick")
        row = view.iloc[i]
        a, b, c = st.columns(3)
        a.metric("Đáp án đúng", str(row["Ground_Truth"]))
        b.metric("Model trả lời", str(row["Predicted"]))
        conf = pd.to_numeric(pd.Series([row["Confidence"]]), errors="coerce").iloc[0]
        c.metric("Độ tự tin", "—" if pd.isna(conf) else f"{conf:.0f}")
        with st.expander("Lời giải đầy đủ của model", expanded=True):
            st.text(str(row["Raw_Output"]).strip())

    with st.expander("Giải thích & giới hạn"):
        st.markdown(
            "- So sánh độ tự tin chỉ **mô tả**, chưa chứng minh CoT *gây ra* khác biệt: "
            "tập câu sai của Zero-Shot và CoT khác nhau về model/ngôn ngữ/độ khó.\n"
            "- Nhãn faithfulness do heuristic tự động → còn nhiễu (nhiều ca 0→1 là do "
            "heuristic chấm lập luận 'sai' nhưng đáp án vẫn đúng); cần người kiểm lại.\n"
            "- Ca **MEDIUM_069**: phép tính ra 3.2 nhưng đáp án nguồn ghi A=3.5 — một khả "
            "năng lệch đáp án gốc, cần lưu ý khi diễn giải."
        )
