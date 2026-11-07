"""Tab 2 - CoT giúp đến đâu? (CH1 + CH4).

Kết luận: CoT tăng ~+10 điểm, lợi ích dồn vào câu Trung bình/Khó; câu Dễ gần như
không đổi; Self-Consistency tốn ~5x chi phí mà chỉ thêm 0-2 điểm. Rò rỉ công thức
làm accuracy nhảy +4-6 điểm.

Bố cục mỗi khối: khẳng định -> biểu đồ -> caption "đọc thế nào".
Nguồn: 2_Summary, 3_Leak_Compare. So sánh Small/Large & VI/EN song song.
"""
import streamlit as st
from .. import data, charts


def render(ctx):
    path = ctx["bench_path"]
    s = data.summary(path)

    st.subheader("CoT giúp đến đâu, và khi nào thì không?")

    # Hàng chỉ số chốt (gộp cả 2 model, 2 ngôn ngữ).
    acc = s.groupby("Method")["Acc_Pct"].mean()
    tok = s.groupby("Method")["Avg_Tok"].mean()
    zs, zcot = acc["1_Zero_Shot"], acc["3_Zero_Shot_CoT"]
    best = acc.drop(["1_Zero_Shot", "2_Few_Shot"]).max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zero-Shot", f"{zs:.0f}%")
    c2.metric("CoT (tốt nhất)", f"{best:.0f}%", f"+{best - zs:.0f} điểm")
    c3.metric("Zero-Shot → Zero-CoT", f"+{zcot - zs:.0f} điểm")
    c4.metric("Token: SC / Zero-CoT",
              f"{tok['5_Self_Consistency'] / tok['3_Zero_Shot_CoT']:.1f}×")

    st.divider()

    st.markdown("**1. CoT giúp *khi nào*?** — lợi ích dồn vào câu Trung bình/Khó.")
    st.markdown("*Biểu đồ cột nhóm — accuracy (%) theo phương pháp × độ khó.*")
    st.plotly_chart(charts.acc_by_method_difficulty(s), width="stretch")
    st.caption("Đọc: so hai cụm Small/Large. Cột **Dễ** đã kịch trần ~98–100 ở mọi phương "
               "pháp → CoT gần như vô ích; khoảng cách nới rộng ở **Trung bình** và **Khó**.")
    st.success("**Kết luận:** CoT chỉ đáng dùng cho câu Trung bình/Khó; với câu Dễ nó không "
               "thêm điểm mà chỉ tốn token.")

    st.markdown("**2. Cái giá của CoT** — token và thời gian đổi lấy accuracy.")
    st.markdown("*Biểu đồ phân tán — accuracy (%) theo số token trung bình mỗi câu.*")
    st.plotly_chart(charts.cost_vs_accuracy(s), width="stretch")
    st.caption("Đọc: điểm càng sang phải càng tốn token. Self-Consistency nằm xa bên phải "
               "nhưng không cao hơn Zero-Shot CoT bao nhiêu → lãi biên rất mỏng.")
    st.success("**Kết luận:** Zero-Shot CoT là điểm ngọt chi phí–hiệu quả; Self-Consistency "
               "không đáng đồng tiền ở bộ dữ liệu này.")

    st.markdown("**3. Mức tăng của CoT theo độ khó** (Zero-Shot → Zero-Shot CoT).")
    st.markdown("*Bản đồ nhiệt — mức tăng Δ (điểm) theo độ khó × (model·ngôn ngữ).*")
    st.plotly_chart(charts.delta_heatmap(s), width="stretch")
    st.caption("Đọc: ô càng **đậm (xanh tối)**, CoT tăng càng nhiều điểm. Hàng **Dễ** nhạt "
               "gần như bằng 0; hàng **Trung bình/Khó** đậm rõ ở cả hai model và hai ngôn ngữ.")
    st.success("**Kết luận:** lợi ích của CoT tỉ lệ thuận với độ khó và ổn định qua cả hai "
               "model lẫn hai ngôn ngữ — không phải hiệu ứng ngẫu nhiên.")

    st.markdown("**4. Tiếng Việt vs Tiếng Anh.**")
    st.markdown("*Biểu đồ quả tạ — accuracy (%) của VI và EN theo từng phương pháp.*")
    st.plotly_chart(charts.lang_dumbbell(s), width="stretch")
    st.caption("Đọc: mỗi thanh nối điểm VI và EN của một phương pháp; thanh càng dài, chênh "
               "lệch ngôn ngữ càng lớn (rõ nhất ở các phương pháp không-CoT).")
    st.success("**Kết luận:** CoT thu hẹp khoảng cách Việt–Anh (rõ nhất ở model Small) — "
               "suy luận từng bước giúp cân bằng giữa hai ngôn ngữ.")

    st.markdown("**5. Không phải cứ cho ví dụ là tốt.**")
    st.markdown("*Biểu đồ cột — accuracy (%) trung bình theo từng phương pháp.*")
    st.plotly_chart(charts.method_family_bar(s), width="stretch")
    st.caption("Đọc: Few-Shot *chỉ cho đáp án, không kèm lời giải* lại **thấp hơn** Zero-Shot "
               "— ví dụ khiến model bắt chước 'nhảy thẳng tới đáp án'.")
    st.success("**Kết luận:** few-shot chỉ có ích khi ví dụ KÈM lời giải; ví dụ chỉ-đáp-án "
               "phản tác dụng. Giá trị nằm ở *cách suy luận*, không phải ở ví dụ.")

    st.markdown("**6. CH4 — Rò rỉ công thức (contamination).**")
    st.markdown("*Biểu đồ cột nhóm — accuracy (%): prompt sạch vs rò rỉ, trên câu Leaked.*")
    st.plotly_chart(charts.leak_effect(data.leak_compare(path)), width="stretch")
    st.caption("Đọc: chỉ tính trên 49 câu thuộc nhóm rò rỉ. Lộ công thức (pp6) đẩy accuracy "
               "lên +4–6 điểm so với pp4. Câu không rò rỉ có Δ=0 (pp6 clone pp4) làm đối chứng.")
    st.success("**Kết luận:** điểm benchmark nhạy với rò rỉ dữ liệu (+4–6đ, rõ hơn ở Tiếng "
               "Việt) → phải đọc các con số accuracy cao một cách thận trọng.")

    with st.expander("Giải thích & giới hạn"):
        st.markdown(
            "- **Vì sao CoT giúp câu khó hơn câu dễ:** câu dễ model trả lời đúng ngay nên "
            "không còn chỗ để cải thiện (trần ~100%); câu khó cần nhiều bước trung gian — "
            "đây đúng là chỗ 'suy nghĩ từng bước' phát huy.\n"
            "- **Self-Consistency:** chạy CoT 5 lần ở T=0.7 rồi lấy đa số phiếu; tốn ~5× "
            "token/thời gian nhưng ở bộ này chỉ hơn CoT đơn 0–2 điểm.\n"
            "- **Rò rỉ:** là proxy cho nhiễm bẩn benchmark — điểm cao cần đọc thận trọng.\n"
            "- Mọi con số gộp trung bình theo nhóm; xem bảng dưới để tra chi tiết."
        )
        st.dataframe(s, width="stretch")
