"""Tab 3 - Khi CoT sai: sai thế nào? (CH2 + CH3 + phân loại lỗi).

Kết luận: Zero-Shot sai mà vẫn rất tự tin; CoT kéo độ tự tin lúc sai xuống nhưng chưa
khắc phục hẳn (phần lớn câu sai vẫn ≥90). Ca 1→0 (nghĩ đúng chốt sai) rất hiếm;
lỗi nằm gần như toàn bộ ở bước suy luận. Các con số hiển thị tính từ dữ liệu.

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
    st.markdown("*Biểu đồ hộp (box plot) — phân bố độ tự tin trên các câu SAI theo phương pháp, "
                "tách theo model.*")
    st.plotly_chart(charts.confidence_on_wrong(raw), width="stretch")
    w = raw[(raw["Correct"] == 0) & (raw["Confidence"].notna())]
    stats = {}   # model -> (tự tin TB ZS, CoT, %≥90 ZS, CoT)
    for m in charts._models(w):
        wz = w[(w["Model"] == m) & (w["Method"] == "1_Zero_Shot")]["Confidence"]
        wc = w[(w["Model"] == m) & (w["Method"] == "3_Zero_Shot_CoT")]["Confidence"]
        if len(wz) and len(wc):
            stats[m] = (wz.mean(), wc.mean(), (wz >= 90).mean() * 100, (wc >= 90).mean() * 100)
    down = [m for m, v in stats.items() if v[3] < v[2] - 5]      # CoT giảm rõ tỷ lệ ≥90
    other = [m for m in stats if m not in down]
    st.caption("Đọc: mỗi panel là một model; mỗi điểm là một câu **trả lời sai** (jitter ngang "
               "cho khỏi chồng), hộp tóm tắt tứ phân vị. Đường đỏ = ngưỡng 90; dưới trục ghi % "
               "câu sai vẫn tự tin ≥90 và số câu sai (n).")
    parts = [f"{m}: {v[2]:.0f}% → {v[3]:.0f}% câu sai ≥90" for m, v in stats.items()]
    msg = "**Kết luận:** "
    if down:
        msg += (f"ở {'/'.join(down)}, CoT *giảm* rõ sự mù quáng nhưng chưa chữa khỏi")
    if other:
        msg += ("; " if down else "") + f"ở {'/'.join(other)} thì CoT **không** giảm"
    msg += " (" + "; ".join(parts) + ")."
    st.success(msg)

    # --- CH3: faithfulness ---
    st.markdown("**2. CH3 — Lời giải từng bước có khớp với đáp án cuối không?**")
    st.markdown("*Ma trận nhiệt 2×2 — số ca theo (lập luận đúng/sai) × (đáp án cuối đúng/sai).*")
    st.plotly_chart(charts.faithfulness_matrix(data.faithfulness(path)), width="stretch")
    st.caption("Đọc: 4 ô = (lập luận đúng/sai) × (đáp án cuối đúng/sai). Ô **1→0** là trường "
               "hợp 'nghĩ đúng nhưng chốt sai' — đáng lo nhất, và ở đây rất nhỏ.")
    fm = data.faithfulness(path)
    n10 = int(fm[fm["Faithfulness"] == "1->0"]["Count"].sum())
    n_all = int(fm["Count"].sum())
    st.success(f"**Kết luận:** chỉ {n10}/{n_all} lượt CoT rơi vào ô 1→0 → model hiếm khi phản bội "
               "chính lời giải của nó; sai chủ yếu vì *suy luận sai* (0→0), không phải vì lệch "
               "khâu chốt.")

    # --- Phân loại lỗi 20 ca ---
    st.markdown("**3. 20 câu sai tiêu biểu sai vì kiểu lỗi gì?**")
    st.markdown("*Biểu đồ cột nhóm — số ca theo kiểu lỗi × model.*")
    st.plotly_chart(charts.error_type_breakdown(data.evidence_baselines()), width="stretch")
    st.caption("Đọc: đếm theo nhãn lỗi do người gắn, tách theo model. Phạm vi: 20 ca sai được "
               "chọn từ Small/Large (Gói 3).")
    st.success("**Kết luận:** lỗi *hiểu sai đề* (Misinterpretation) và *làm dở dang* "
               "(Incomplete) chiếm đa số — không phải lỗi số học thuần.")

    # --- Bảng tra lỗi ---
    st.markdown("**4. Tra cứu từng câu sai** (yêu cầu bắt buộc: phân tích ≥20 câu sai).")
    st.markdown("*Bảng — mọi câu trả lời sai của cả 6 phương pháp (sheet 4_Errors); chọn một "
                "câu bên dưới để xem lời giải.*")
    err = data.errors(path).reset_index(drop=True)
    f1, f2 = st.columns(2)
    methods = f1.multiselect("Lọc theo phương pháp",
                             sorted(err["Method"].unique()), default=[], key="t3_method")
    mdl = f2.multiselect("Lọc theo model", charts._models(err), default=[], key="t3_model")
    view = err
    if methods:
        view = view[view["Method"].isin(methods)]
    if mdl:
        view = view[view["Model"].isin(mdl)]
    view = view.reset_index(drop=True)

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
    st.caption(f"{len(view)} câu trả lời sai. Lọc theo phương pháp/model ở trên; chọn một câu "
               "để xem lời giải đầy đủ:")

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
