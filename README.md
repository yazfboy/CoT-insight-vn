# CoT-Insight VN

Bộ công cụ chạy thực nghiệm tự động đánh giá kỹ thuật **Chain-of-Thought (CoT)** trên 150 bài
toán tiếng Việt + tiếng Anh, kèm phân tích lỗi và **web app trực quan hoá** kết quả.

## 1. Cấu trúc thư mục

- `data/dataset_150.xlsx` — bộ 150 câu hỏi (50 Dễ, 50 Trung bình, 50 Khó) + sheet `data_train`.
- `benchmark/run_benchmark.py` — chạy benchmark 6 phương pháp + chấm điểm tự động.
- `benchmark/results/benchmark_results_<ngày_giờ>.xlsx` — kết quả đầy đủ (5 sheet).
- `benchmark/results/checkpoint_v2.csv` — checkpoint để chạy tiếp khi bị ngắt.
- `analysis/` — thực nghiệm 5 can thiệp sửa lỗi (Gói 3) + workbook bằng chứng.
- `app/` — **web app Streamlit** (Gói 4) đọc kết quả và trực quan hoá.
- `.env` — danh sách Gemini API Keys (mỗi dòng 1 key), đặt ở gốc dự án.

## 2. Cài đặt

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 3. Chạy benchmark

```bash
python benchmark/run_benchmark.py
```

- `TEST_MODE = True`: chạy thử 5 câu (EASY_001, EASY_011, MEDIUM_069, MEDIUM_085, HARD_144).
- `TEST_MODE = False`: chạy toàn bộ 150 câu. Muốn chạy lại từ đầu, xoá `checkpoint_v2.csv`.

### Thành phần chính trong `run_benchmark.py`
- **KeyPool**: xoay tua nhiều API key từ `.env`, giãn cách request để tránh lỗi 429, khoá key khi hết hạn mức ngày.
- **MODELS**: 3 mô hình đối sánh (2 qua API Google trong `run_benchmark.py`, 1 tự host):
  - `Small` → **gemma-4-26b-a4b-it** (thinking `minimal`)
  - `Large` → **gemini-3.5-flash-lite**
  - `Qwen_7B` → **qwen2.5:7b**, tự host (vLLM/Ollama), chạy bằng `benchmark/run_extra_llm.py`
- **make_prompt**: tạo prompt cho 6 phương pháp (`1_Zero_Shot`, `2_Few_Shot`, `3_Zero_Shot_CoT`, `4_Few_Shot_CoT`, `5_Self_Consistency`, `6_Few_Shot_CoT_Leaked`).
- **parse_output & is_match**: tách `[Đáp án cuối cùng]` + `[Độ tự tin]` bằng regex, đối chiếu đáp án gốc để chấm 1/0.

### 5 sheet trong file Excel kết quả
1. **1_Raw_Logs** — log chi tiết từng lượt. File mới nhất `20261110_232749` có 5400 dòng (150 × 3 model × 2 ngôn ngữ × 6 phương pháp); file `20261106_095454` có 3600 dòng (2 model).
2. **2_Summary** — % Accuracy, token, thời gian, chi phí, độ tự tin theo model × ngôn ngữ × phương pháp × độ khó.
3. **3_Leak_Compare** — so sánh `4_Few_Shot_CoT` (sạch) vs `6_Few_Shot_CoT_Leaked` (rò rỉ).
4. **4_Errors** — mọi câu trả lời sai (Correct == 0) của cả 6 phương pháp + cột trống để phân tích lỗi thủ công.
5. **5_Faithfulness_Matrix** — thống kê nhãn faithfulness (1→1 / 0→1 / 1→0 / 0→0) + độ tự tin trung bình.

## 4. Chạy web app (Gói 4)

```bash
streamlit run app/main.py
```

App tự lấy file `benchmark_results_*.xlsx` mới nhất, gồm 4 tab:
1. **Thử ngay** — phát lại / gọi API thật cho 1 câu, so sánh các phương pháp cạnh nhau.
2. **CoT giúp đến đâu?** — CH1 (độ khó, chi phí, model, ngôn ngữ) + CH4 (rò rỉ).
3. **Khi CoT sai** — CH2 (overconfidence) + CH3 (faithfulness) + phân loại lỗi.
4. **Bộ điều phối CoT** — CH5 (5 can thiệp sửa lỗi), phát lại + gọi API thật.

Tính năng gọi API thật dùng lại `load_keys`, `KeyPool`, `make_prompt`, `ask_llm`, `parse_output`
từ `benchmark/run_benchmark.py` và `run_one_case_method` từ script trong `analysis/`.

## 5. Phân tích lỗi + can thiệp (Gói 3)

Mở `4_Errors`, chọn ≥ 20 câu sai, đọc `Raw_Output` và điền:
- **Reasoning_State**: `Chinh_xac_1_phan` | `Sai_hoan_toan`
- **Error_Type**: `Arithmetic` | `Logical_Leap` | `Incomplete` | `Misinterpretation`
- **Faithfulness**: `0->0` (giải sai → đáp án sai) | `1->0` (giải đúng nhưng chốt sai)
- **Note**: lý do sai.

Script `analysis/run_six_method_experiment_20261106.py` chạy 5 can thiệp (Premise Extraction,
Strict CoT, PAL, NLI Guardrail, Logprob/Temperature Calibration) trên 20 ca đã chọn.
