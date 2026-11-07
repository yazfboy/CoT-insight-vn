"""Trang trí layout bên ngoài cho dashboard — vibe nghiên cứu khoa học.

Chỉ CSS + khối header; KHÔNG đụng tới nội dung biểu đồ. Nạp 1 lần ở đầu main.py.
"""
import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink:#1a2233; --muted:#5b6472; --accent:#26406b; --accent-2:#0f766e;
  --line:#e4e7ec; --panel:#f7f9fc;
}

/* Nền & khung đọc như một trang paper */
.stApp { background:#ffffff; }
.block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:3rem; }

/* Thân chữ sans, tiêu đề serif */
html, body, [class*="css"] { font-family:'Inter', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family:'Source Serif 4', Georgia, serif !important;
  color:var(--ink); letter-spacing:-0.01em; }

/* Header kiểu trang bìa paper */
.paper-header { border-bottom:2px solid var(--ink); padding-bottom:0.9rem; margin-bottom:0.4rem; }
.paper-header .eyebrow { font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:600;
  letter-spacing:0.18em; text-transform:uppercase; color:var(--accent); margin-bottom:0.35rem; }
.paper-header h1 { font-size:2.15rem; font-weight:700; margin:0; line-height:1.1; }
.paper-header .subtitle { font-family:'Source Serif 4',serif; font-style:italic;
  font-size:1.08rem; color:var(--muted); margin:0.5rem 0 0; }

/* Nhãn mục nhỏ (section label) */
.section-label { font-family:'Inter',sans-serif; font-size:0.74rem; font-weight:600;
  letter-spacing:0.14em; text-transform:uppercase; color:var(--accent);
  border-left:3px solid var(--accent); padding-left:0.55rem; margin:0.2rem 0 0.6rem; }

/* Thẻ chỉ số (metric) kiểu ô bảng số liệu */
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line);
  border-top:3px solid var(--accent); border-radius:6px; padding:0.7rem 0.9rem; }
[data-testid="stMetricValue"] { font-family:'Source Serif 4',serif; color:var(--ink);
  font-size:1.55rem; line-height:1.2; white-space:normal; overflow-wrap:anywhere; }
[data-testid="stMetricLabel"] p { font-family:'Inter',sans-serif; font-size:0.78rem;
  font-weight:600; color:var(--muted); letter-spacing:0.02em; }

/* Tabs: gạch chân accent cho tab đang chọn */
.stTabs [data-baseweb="tab-list"] { gap:0.3rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family:'Inter',sans-serif; font-weight:600;
  font-size:0.95rem; color:var(--muted); padding:0.5rem 0.9rem; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; }

/* Hộp thông báo (Kết luận / Giải thích): viền trái accent, nền nhạt */
[data-testid="stAlert"] { border-radius:6px; border-left-width:4px; }

/* Caption nghiêng, nhạt — như chú thích hình trong paper */
[data-testid="stCaptionContainer"], .stCaption { color:var(--muted); font-size:0.86rem; }

/* Divider mảnh */
hr { border-color:var(--line); }

/* Bảng dữ liệu viền gọn */
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; }

/* Nút bấm */
.stButton button { font-family:'Inter',sans-serif; font-weight:600; border-radius:6px; }
</style>
"""

_HEADER = """
<div class="paper-header">
  <div class="eyebrow">Chain-of-Thought · Phân tích thực nghiệm</div>
  <h1>CoT-Insight VN</h1>
  <p class="subtitle">Yêu cầu LLM &ldquo;suy nghĩ từng bước&rdquo; giúp ích đến đâu, và khi nào thì không?</p>
</div>
"""


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


def header():
    st.markdown(_HEADER, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)
