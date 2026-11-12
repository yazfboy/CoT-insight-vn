"""CoT-Insight VN - Web App (Gói 4).

Chạy:  streamlit run app/main.py
Cấu trúc: main.py (khung + màn mở đầu) -> data.py -> charts.py -> tabs/.

Thiết kế theo góc nhìn LLM researcher: ưu tiên trực quan hoá + giải thích, so sánh
các model (Small/Large/Qwen_7B) và VI/EN song song. Không có bộ lọc chung; tự lấy file
kết quả mới nhất.

4 Tab tương ứng các câu hỏi nghiên cứu trong PROJECT.md:
  Tab 1 Thử ngay          - mở đầu, live demo
  Tab 2 CoT giúp đến đâu  - CH1 (độ khó, chi phí, model, ngôn ngữ) + CH4 (rò rỉ)
  Tab 3 Khi CoT sai       - CH2 (overconfidence) + CH3 (faithfulness) + phân loại lỗi
  Tab 4 Bộ điều phối      - CH5 (5 can thiệp sửa lỗi)
"""
import os
import sys
import streamlit as st

# Cho phép "from app import ..." khi chạy bằng `streamlit run app/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import data, style
from app.tabs import t1_live, t2_benchmark, t3_errors, t4_orchestrator

st.set_page_config(page_title="CoT-Insight VN", layout="wide")
style.inject()
style.header()

# Tự lấy file kết quả benchmark mới nhất (không cho người dùng chọn).
path = data.latest_benchmark_file()
if not path:
    st.error("Chưa tìm thấy file `benchmark/results/benchmark_results_*.xlsx`. "
             "Hãy chạy benchmark trước.")
    st.stop()

# --- Màn mở đầu: Câu chuyện thí nghiệm ---
info = data.setup_info(path)
st.caption(
    f"Thiết lập: **{info['n_questions']} câu** × {info['n_models']} mô hình × "
    f"{info['n_langs']} ngôn ngữ × {info['n_methods']} phương pháp = "
    f"**{info['n_tasks']} lượt gọi**"
    + (f" · chạy {info['run_date']}" if info['run_date'] else "")
)
st.caption("Mô hình: **Small** = gemma-4-26b-a4b-it · **Large** = gemini-3.5-flash-lite "
           "(API Google) · **Qwen_7B** = qwen2.5:7b (tự host, Ollama). Nhãn ngắn dùng xuyên biểu đồ.")

hs = data.headline_stats(path)
style.section_label("Năm phát hiện chính")
cols = st.columns(5)
_cards = [
    ("CH1 · CoT giúp", "ch1"),
    ("CH2 · Tự tin khi sai", "ch2"),
    ("CH3 · Trung thực", "ch3"),
    ("CH4 · Rò rỉ", "ch4"),
    ("CH5 · Sửa lỗi tốt nhất", "ch5"),
]
for col, (label, key) in zip(cols, _cards):
    with col:
        st.metric(label, hs[key]["value"])
        st.caption(hs[key]["caption"])

st.divider()

ctx = {"bench_path": path}

t1, t2, t3, t4 = st.tabs([
    "1. Thử ngay",
    "2. CoT giúp đến đâu?",
    "3. Khi CoT sai",
    "4. Bộ điều phối CoT",
])
with t1:
    t1_live.render(ctx)
with t2:
    t2_benchmark.render(ctx)
with t3:
    t3_errors.render(ctx)
with t4:
    t4_orchestrator.render(ctx)
