"""Thống kê cho dashboard — thuần tính toán, không phụ thuộc Streamlit.

- wilson_ci: khoảng tin cậy 95% cho một tỷ lệ (dùng cho thanh sai số trên biểu đồ accuracy).
- mcnemar: kiểm định GHÉP CẶP cho hai phương pháp trên CÙNG bộ câu (cùng model) — đúng thiết
  kế thí nghiệm, vì mỗi câu được chạy qua mọi phương pháp nên các kết quả không độc lập.
- holm: hiệu chỉnh đa so sánh (step-down Holm) để tránh "có ý nghĩa" do may rủi khi so nhiều cặp.
"""
from __future__ import annotations

import math
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho p=k/n, trả (lo, hi) theo đơn vị % (0–100)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Kiểm định McNemar chính xác (sign test trên các cặp lệch) cho hai Series 0/1 cùng index.

    b01 = số câu A đúng & B sai; b10 = số câu A sai & B đúng. Trả p hai phía chính xác.
    """
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    av, bv = j.iloc[:, 0].astype(int), j.iloc[:, 1].astype(int)
    b01 = int(((av == 1) & (bv == 0)).sum())
    b10 = int(((av == 0) & (bv == 1)).sum())
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        lo = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"b01": b01, "b10": b10, "n_disc": n, "p": p,
            "acc_a": float(av.mean() * 100), "acc_b": float(bv.mean() * 100),
            "n": int(len(av))}


def holm(pvals: list[float]) -> list[float]:
    """Hiệu chỉnh Holm, giữ nguyên thứ tự đầu vào. p_adj đơn điệu không giảm theo p tăng."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        run = max(run, val)
        adj[i] = run
    return adj


def verdict(p: float, alpha: float = 0.05) -> str:
    return "có ý nghĩa" if p < alpha else "không đáng kể"
