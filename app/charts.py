"""Các hàm vẽ biểu đồ Plotly cho dashboard CoT-Insight VN.

Mỗi biểu đồ = 1 hàm, nhận DataFrame -> trả về go.Figure. Giữ LOGIC VẼ ở đây,
tách khỏi tab để dễ test và tái sử dụng.

Thiết kế theo skill `dataviz`:
- So sánh là mặc định: facet theo Model, cặp VI/EN cạnh nhau (không lọc ẩn).
- Màu phân loại theo Method giữ CỐ ĐỊNH xuyên các tab (bảng màu đã validate CVD).
- Độ khó dùng thang xanh theo thứ tự (ordinal): Dễ nhạt -> Khó đậm.
- Tiêu đề nói Ý, có chú giải/đường ngưỡng ngay trên biểu đồ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Nhãn ngắn cho 5 can thiệp (Gói 3) + thứ tự hiển thị.
FIX_SHORT = {
    "Premise Extraction": "Premise",
    "Strict CoT": "Strict CoT",
    "PAL": "PAL",
    "Post-generation NLI guardrail": "NLI",
    "Logprob & Temperature Calibration": "Calibration",
}
FIX_ORDER = ["Premise", "Strict CoT", "PAL", "NLI", "Calibration"]

# --- Bảng màu (đã chạy scripts/validate_palette.js, đạt mọi gate cứng) ---
METHOD_ORDER = [
    "1_Zero_Shot", "2_Few_Shot", "3_Zero_Shot_CoT",
    "4_Few_Shot_CoT", "5_Self_Consistency", "6_Few_Shot_CoT_Leaked",
]
METHOD_SHORT = {
    "1_Zero_Shot": "Zero-Shot",
    "2_Few_Shot": "Few-Shot",
    "3_Zero_Shot_CoT": "Zero-CoT",
    "4_Few_Shot_CoT": "Few-CoT",
    "5_Self_Consistency": "Self-Consist",
    "6_Few_Shot_CoT_Leaked": "Leaked",
}
_METHOD_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
METHOD_COLORS = {METHOD_SHORT[m]: h for m, h in zip(METHOD_ORDER, _METHOD_HEX)}
SHORT_ORDER = [METHOD_SHORT[m] for m in METHOD_ORDER]

# Độ khó: thang xanh tuần tự (ordinal), bắt đầu không nhạt hơn step 250.
DIFF_ORDER = ["Easy", "Medium", "Hard"]
DIFF_COLORS = {"Easy": "#86b6ef", "Medium": "#3987e5", "Hard": "#184f95"}

_FAMILY_HEX = {"Không CoT": "#898781", "CoT / biến thể": "#2a78d6"}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _short(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["M"] = df["Method"].map(METHOD_SHORT).fillna(df["Method"])
    return df


def _layout(fig: go.Figure, height: int = 380) -> go.Figure:
    # Không đặt tiêu đề trong figure: tiêu đề/khẳng định do tab (markdown) đảm nhiệm,
    # tránh trùng lặp. Chừa ~30px trên cho legend ngang.
    fig.update_layout(
        title=None,
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(size=12),
    )
    return fig


# 1) CH1 — accuracy theo phương pháp × độ khó, so sánh 2 model.
def acc_by_method_difficulty(summary_df: pd.DataFrame) -> go.Figure:
    df = _short(summary_df)
    g = (df.groupby(["Model", "M", "Difficulty"], as_index=False)["Acc_Pct"].mean())
    fig = px.bar(
        g, x="M", y="Acc_Pct", color="Difficulty", barmode="group",
        facet_col="Model",
        category_orders={"M": SHORT_ORDER, "Difficulty": DIFF_ORDER},
        color_discrete_map=DIFF_COLORS,
    )
    fig.update_yaxes(range=[0, 105], title="Accuracy (%)")
    fig.update_xaxes(title="")
    # Làm nổi tên 2 model (Small/Large) để thấy rõ đây là trục so sánh.
    fig.for_each_annotation(lambda a: a.update(
        text="<b>" + a.text.replace("Model=", "Model: ") + "</b>",
        font=dict(size=15, color="#2a78d6")))
    return _layout(fig)


# 2) CH1 — đánh đổi chi phí (token) ↔ accuracy; Self-Consistency đắt mà lãi mỏng.
def cost_vs_accuracy(summary_df: pd.DataFrame) -> go.Figure:
    df = _short(summary_df)
    g = (df.groupby(["Model", "Method", "M"], as_index=False)
         .agg(Acc=("Acc_Pct", "mean"), Tok=("Avg_Tok", "mean")))
    g = g.sort_values("Method")
    fig = px.scatter(
        g, x="Tok", y="Acc", color="M", symbol="Model",
        category_orders={"M": SHORT_ORDER}, color_discrete_map=METHOD_COLORS,
    )
    fig.update_traces(marker=dict(size=13))
    fig.update_xaxes(title="Token trung bình / câu (↑ = tốn hơn)")
    fig.update_yaxes(title="Accuracy (%)")
    # Chú giải: trỏ đúng điểm Self-Consistency (hồng) lệch xa nhất sang phải.
    sc = g[g["Method"] == "5_Self_Consistency"]
    if not sc.empty:
        far = sc.loc[sc["Tok"].idxmax()]
        fig.add_annotation(x=float(far["Tok"]), y=float(far["Acc"]),
                           text="Đắt ~5× mà<br>không cao hơn",
                           showarrow=True, arrowhead=2, ax=-60, ay=30,
                           font=dict(size=10, color="#52514e"))
    fig = _layout(fig, height=470)
    # Legend nằm ngang, dàn phẳng thành khối chữ nhật dưới biểu đồ (không che điểm).
    fig.update_layout(
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=-0.38, yanchor="top", font=dict(size=9), title_text=""),
        margin=dict(l=10, r=10, t=30, b=120))
    return fig


# 3) CH1 — mức tăng của CoT so với Zero-Shot (heatmap theo độ khó × model·ngôn ngữ).
def delta_heatmap(summary_df: pd.DataFrame) -> go.Figure:
    df = summary_df.copy()
    piv = df.pivot_table(index=["Model", "Language", "Difficulty"],
                         columns="Method", values="Acc_Pct", aggfunc="mean")
    delta = (piv["3_Zero_Shot_CoT"] - piv["1_Zero_Shot"]).reset_index(name="Delta")
    delta["Col"] = delta["Model"] + "·" + delta["Language"]
    mat = delta.pivot(index="Difficulty", columns="Col", values="Delta")
    mat = mat.reindex(DIFF_ORDER)
    fig = px.imshow(
        mat, text_auto=".0f", aspect="auto",
        color_continuous_scale="Blues", zmin=0,
        labels=dict(color="Δ điểm"),
    )
    fig.update_xaxes(title="", side="bottom")
    fig.update_yaxes(title="")
    fig.update_coloraxes(colorbar_title="Δ điểm")
    return _layout(fig)


# 4) CH1 — chênh accuracy giữa Tiếng Việt và Tiếng Anh (dumbbell theo từng model).
def lang_dumbbell(summary_df: pd.DataFrame) -> go.Figure:
    df = _short(summary_df)
    g = (df.groupby(["Model", "M", "Language"], as_index=False)["Acc_Pct"].mean())
    models = ["Small", "Large"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=models, horizontal_spacing=0.08)
    lang_color = {"VI": "#2a78d6", "EN": "#eb6834"}
    for ci, model in enumerate(models, start=1):
        sub = g[g["Model"] == model]
        for _, m in enumerate(SHORT_ORDER):
            row = sub[sub["M"] == m]
            if row.empty:
                continue
            vi = row[row["Language"] == "VI"]["Acc_Pct"]
            en = row[row["Language"] == "EN"]["Acc_Pct"]
            if vi.empty or en.empty:
                continue
            vi, en = float(vi.iloc[0]), float(en.iloc[0])
            fig.add_trace(go.Scatter(
                x=[vi, en], y=[m, m], mode="lines",
                line=dict(color="#c3c2b7", width=2), showlegend=False,
                hoverinfo="skip"), row=1, col=ci)
            for lang, val in (("VI", vi), ("EN", en)):
                fig.add_trace(go.Scatter(
                    x=[val], y=[m], mode="markers",
                    marker=dict(color=lang_color[lang], size=11),
                    name=lang, legendgroup=lang,
                    showlegend=(ci == 1 and m == SHORT_ORDER[0]),
                    hovertemplate=f"{lang} {m}: %{{x:.0f}}%<extra></extra>"),
                    row=1, col=ci)
    fig.update_xaxes(range=[60, 103], title="Accuracy (%)")
    fig.update_yaxes(categoryorder="array", categoryarray=SHORT_ORDER[::-1])
    return _layout(fig, height=420)


# 5) CH1 — accuracy tổng theo phương pháp; làm bật Few-Shot (không CoT) < Zero-Shot.
def method_family_bar(summary_df: pd.DataFrame) -> go.Figure:
    df = _short(summary_df)
    g = df.groupby(["M", "Method"], as_index=False)["Acc_Pct"].mean().sort_values("Method")
    g["Family"] = g["Method"].apply(
        lambda m: "Không CoT" if m in ("1_Zero_Shot", "2_Few_Shot") else "CoT / biến thể")
    fig = px.bar(
        g, x="M", y="Acc_Pct", color="Family", text="Acc_Pct",
        category_orders={"M": SHORT_ORDER}, color_discrete_map=_FAMILY_HEX,
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_yaxes(range=[0, 105], title="Accuracy (%)")
    fig.update_xaxes(title="")
    zs = float(g[g["Method"] == "1_Zero_Shot"]["Acc_Pct"].iloc[0])
    few = float(g[g["Method"] == "2_Few_Shot"]["Acc_Pct"].iloc[0])
    fig.add_hline(y=zs, line_dash="dot", line_color="#52514e",
                  annotation_text="mức Zero-Shot", annotation_position="top right")
    # Chú thích đỏ đặt hẳn dưới đường nét đứt để không chồng lên nó.
    fig.add_annotation(x="Few-Shot", y=few,
                       text="Few-Shot KHÔNG kèm lời giải<br>còn thấp hơn Zero-Shot",
                       showarrow=True, arrowhead=2, ax=55, ay=60,
                       align="left", font=dict(size=10, color="#eda100"))
    return _layout(fig)


# 6) CH4 — rò rỉ công thức: Few-Shot CoT (pp4) vs Leaked (pp6) trên tập câu Leaked.
def leak_effect(leak_df: pd.DataFrame) -> go.Figure:
    df = leak_df[leak_df["Leak_Status"] == "Leaked"].copy()
    df["Col"] = df["Model"] + "·" + df["Language"]
    df["Loại"] = df["Method"].map({
        "4_Few_Shot_CoT": "pp4 (sạch)",
        "6_Few_Shot_CoT_Leaked": "pp6 (rò rỉ)"})
    fig = px.bar(
        df, x="Col", y="Acc_Pct", color="Loại", barmode="group",
        text="Acc_Pct",
        color_discrete_map={"pp4 (sạch)": "#898781", "pp6 (rò rỉ)": "#2a78d6"},
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_yaxes(range=[80, 103], title="Accuracy (%)")
    fig.update_xaxes(title="")
    return _layout(fig)


# ===================== Giai đoạn 3: Tab 3 (CH2+CH3) + Tab 4A =====================

# CH2 — box plot độ tự tin trên câu SAI: Zero-Shot vs Zero-CoT (nền trong + điểm jitter).
def confidence_on_wrong(raw_df: pd.DataFrame) -> go.Figure:
    w = raw_df[(raw_df["Correct"] == 0) & (raw_df["Confidence"].notna())].copy()
    w = w[w["Method"].isin(["1_Zero_Shot", "3_Zero_Shot_CoT"])]
    order = [("Zero-Shot", "1_Zero_Shot"), ("Zero-CoT", "3_Zero_Shot_CoT")]
    fig = go.Figure()
    ticktext = []
    for short, method in order:
        col = w[w["Method"] == method]["Confidence"]
        color = METHOD_COLORS[short]
        fig.add_trace(go.Box(
            y=col, name=short, boxpoints="all", jitter=0.6, pointpos=0,
            fillcolor=_rgba(color, 0.15), line=dict(color=color, width=2),
            marker=dict(color=color, size=5, opacity=0.55),
            whiskerwidth=0.4, hovertemplate="%{y:.0f}<extra></extra>"))
        p90 = (col >= 90).mean() * 100 if len(col) else 0
        ticktext.append(
            f"{short}<br><span style='color:#d03b3b'>{p90:.0f}% câu sai vẫn ≥90</span>")
    # Hộp Zero-Shot dẹt ở 100 (phần lớn câu sai ở mức 100) nhưng điểm jitter vẫn hiện rõ.
    fig.add_hline(y=90, line_dash="dot", line_color="#d03b3b",
                  annotation_text="ngưỡng 90", annotation_position="right")
    fig.update_xaxes(tickmode="array", tickvals=["Zero-Shot", "Zero-CoT"],
                     ticktext=ticktext, title="")
    fig.update_yaxes(range=[75, 103], title="Độ tự tin model tự báo (0–100)")
    fig.update_layout(showlegend=False)
    return _layout(fig, height=380)


# CH3 — ma trận faithfulness 2×2 (giải đúng/sai × đáp án đúng/sai).
def faithfulness_matrix(faith_df: pd.DataFrame) -> go.Figure:
    c = faith_df.groupby("Faithfulness")["Count"].sum()
    def g(k): return int(c.get(k, 0))
    rows = ["Giải đúng", "Giải sai"]          # lập luận
    cols = ["Đáp án đúng", "Đáp án sai"]       # đáp án cuối
    z = [[g("1->1"), g("1->0")],
         [g("0->1"), g("0->0")]]
    txt = [[f"1→1<br><b>{z[0][0]}</b>", f"1→0<br><b>{z[0][1]}</b> (hiếm)"],
           [f"0→1<br><b>{z[1][0]}</b>", f"0→0<br><b>{z[1][1]}</b>"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=rows, text=txt, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
        hovertemplate="%{y} · %{x}: %{z} ca<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Lập luận (reasoning)")
    fig.update_xaxes(title="Đáp án cuối", side="top")
    return _layout(fig, height=320)


# Phân loại 20 ca sai theo kiểu lỗi (tách theo model).
def error_type_breakdown(baselines_df: pd.DataFrame) -> go.Figure:
    g = (baselines_df.groupby(["Error_Type", "Model"]).size()
         .reset_index(name="n"))
    order = (baselines_df["Error_Type"].value_counts().index.tolist())
    fig = px.bar(
        g, x="Error_Type", y="n", color="Model", barmode="group", text="n",
        category_orders={"Error_Type": order},
        color_discrete_map={"Small": "#eb6834", "Large": "#2a78d6"},
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(title="Số ca", range=[0, 7])
    fig.update_xaxes(title="")
    return _layout(fig, height=340)


# CH5 — tỷ lệ sửa được của 5 can thiệp (ghi rõ mẫu số; PAL chỉ 10 câu số).
def fix_rate_by_method(fix_df: pd.DataFrame) -> go.Figure:
    d = fix_df.copy()
    d["M"] = d["Method"].map(FIX_SHORT).fillna(d["Method"])
    d["label"] = d.apply(
        lambda r: f"{int(r['Ca sửa được'])}/{int(r['N hợp lệ'])}", axis=1)
    fig = px.bar(
        d, x="M", y="Acc sau (%)", text="label",
        category_orders={"M": FIX_ORDER},
        color_discrete_sequence=["#1baf7a"],
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 100], title="Tỷ lệ sửa được (%)")
    fig.update_xaxes(title="")
    fig.add_annotation(x="PAL", y=82, showarrow=False, font=dict(size=9, color="#52514e"),
                       text="PAL chỉ 10 câu số")
    return _layout(fig, height=340)


# Ma trận nhiệt: can thiệp × kiểu lỗi -> tỷ lệ sửa được (%) + số ô sửa/áp dụng.
def fix_by_errortype(cases_df: pd.DataFrame) -> go.Figure:
    d = cases_df.copy()
    d["M"] = d["Method"].map(FIX_SHORT).fillna(d["Method"])
    d = d[~d["Status"].astype(str).str.startswith("SKIPPED")]  # bỏ ô không áp dụng được
    etypes = (d["Error_Type"].value_counts().index.tolist())
    appl = d.pivot_table(index="M", columns="Error_Type", values="Correct after",
                         aggfunc="size").reindex(index=FIX_ORDER, columns=etypes)
    fixed = d.pivot_table(index="M", columns="Error_Type", values="Correct after",
                          aggfunc="sum").reindex(index=FIX_ORDER, columns=etypes)
    rate = (fixed / appl * 100)
    text = [[("–" if (pd.isna(appl.iloc[i, j]) or appl.iloc[i, j] == 0)
              else f"{rate.iloc[i, j]:.0f}%<br><span style='font-size:9px'>"
                   f"{int(fixed.iloc[i, j])}/{int(appl.iloc[i, j])}</span>")
             for j in range(len(etypes))] for i in range(len(FIX_ORDER))]
    fig = go.Figure(go.Heatmap(
        z=rate.values, x=etypes, y=FIX_ORDER, text=text, texttemplate="%{text}",
        colorscale="Greens", zmin=0, zmax=100, xgap=3, ygap=3,
        colorbar=dict(title="% sửa", ticksuffix="%"),
        hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(side="top", title="")
    return _layout(fig, height=360)


# Hiệu chỉnh độ tự tin: Brier & độ tự tin TB trước–sau (2 ô cạnh nhau).
def calibration_bars(cal: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Brier score (↓ tốt)",
                                                        "Độ tự tin TB (%)"),
                        horizontal_spacing=0.18)
    stages = ["Trước", "Sau"]
    bcol = ["#898781", "#1baf7a"]
    fig.add_trace(go.Bar(x=stages, y=[cal["before"]["brier"], cal["after"]["brier"]],
                         marker_color=bcol, text=[f"{cal['before']['brier']:.3f}",
                         f"{cal['after']['brier']:.3f}"], textposition="outside",
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=stages, y=[cal["before"]["conf"], cal["after"]["conf"]],
                         marker_color=bcol, text=[f"{cal['before']['conf']:.0f}",
                         f"{cal['after']['conf']:.0f}"], textposition="outside",
                         showlegend=False), row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 105], row=1, col=2)
    return _layout(fig, height=330)


# Self-Consistency: phân bố 5 phiếu bầu (dùng cho Tab 1 khi chạy live).
def sc_vote_bar(votes: dict) -> go.Figure:
    if not votes:
        return _layout(go.Figure(), height=240)
    items = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    xs = [str(k) for k, _ in items]
    ys = [v for _, v in items]
    colors = ["#1baf7a"] + ["#c3c2b7"] * (len(xs) - 1)  # đáp án thắng = xanh
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=colors, text=ys,
                           textposition="outside"))
    fig.update_yaxes(title="Số phiếu", dtick=1, range=[0, max(ys) + 1])
    fig.update_xaxes(title="Đáp án")
    return _layout(fig, height=260)
