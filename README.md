# CoT Benchmark

Bộ công cụ chạy thực nghiệm tự động đánh giá kỹ thuật Chain-of-Thought (CoT) trên 150 bài toán tiếng Việt và tiếng Anh.

## 1. Cấu trúc file

- `.env`: Chứa danh sách Gemini API Keys (mỗi dòng 1 key).
- `dataset_150.csv` (hoặc `.xlsx`): Bộ 150 câu hỏi đầu vào (50 Dễ, 50 Trung bình, 50 Khó).
- `run_benchmark.py`: Script chạy thực nghiệm và chấm điểm tự động.
- `checkpoint.csv`: File lưu tạm tự động khi chạy full 150 câu (dùng để chạy tiếp nếu bị ngắt giữa chừng).
- `test_results_<ngay_gio>.xlsx`: Kết quả chạy thử 3 câu (`TEST_MODE = True`).
- `benchmark_results_<ngay_gio>.xlsx`: Kết quả chạy toàn bộ 150 câu (`TEST_MODE = False`).

## 2. Cách chạy

```bash
# 1. Tạo và bật môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate

# 2. Cài thư viện
pip install google-genai pandas openpyxl tqdm

# 3. Chạy thực nghiệm
python run_benchmark.py
Lưu ý trong run_benchmark.py:
TEST_MODE = True: Chạy thử 3 câu (EASY_001, EASY_026, HARD_144).
TEST_MODE = False: Chạy toàn bộ 150 câu. Nếu muốn chạy lại từ đầu hoàn toàn, xóa file checkpoint.csv trước khi chạy.
3. Cấu trúc Code & Dữ liệu
Các thành phần chính trong run_benchmark.py
KeyPool: Xoay tua 5 API keys từ .env, tự động giãn cách 4.2s/req trên mỗi key để tránh lỗi Rate Limit (429), đếm số token đã dùng và khóa key nếu hết hạn mức ngày.
MODELS: Chỉ sử dụng 2 mô hình đối sánh:
Small: gemini-3.5-flash-lite
Large: gemini-3.5-flash
make_prompt: Tạo prompt cho 6 kịch bản (1_Zero_Shot, 2_Few_Shot, 3_Zero_Shot_CoT, 4_Few_Shot_CoT, 5_Self_Consistency, 6_Few_Shot_CoT_Leaked).
parse_output & is_match: Dùng Regex tách [Đáp án cuối cùng] và [Độ tự tin], đối chiếu với đáp án gốc để chấm 1 (Đúng) hoặc 0 (Sai).
Cấu trúc 4 Sheet trong file Excel kết quả
1_Raw_Logs: Log chi tiết từng câu trả lời (Run_Time, ID, Difficulty, Leak_Status, Model, Language, Method, Ground_Truth, Predicted, Correct, Confidence, Total_Tokens, Latency_s, Cost_1k_USD, Key_ID, Raw_Output).
2_Summary: Bảng thống kê sẵn % Accuracy, Token trung bình, Thời gian (s), Chi phí ($/1k) và Độ tự tin (%) theo từng độ khó.
3_Leak_Compare: Bảng so sánh kết quả giữa Không lộ đề (4_Few_Shot_CoT) và Lộ một phần ngữ cảnh/công thức (6_Few_Shot_CoT_Leaked).
4_Errors: Lọc sẵn các câu CoT trả lời sai (Correct == 0) kèm 4 cột trống cuối bảng để phân tích lỗi.
4. Hướng dẫn cho các gói công việc tiếp theo
Gói 3 (Phân tích lỗi & Mở rộng):
Mở Sheet 4_Errors, chọn >= 20 câu sai, đọc cột Raw_Output và điền vào 4 cột cuối:
Reasoning_State: Chinh_xac_1_phan hoặc Sai_hoan_toan
Error_Type: Arithmetic (Tính sai) | Logical_Leap (Suy diễn sai/Overthinking) | Incomplete (Cụt giữa chừng) | Misinterpretation (Hiểu sai đề)
Faithfulness: 0->0 (Giải sai -> Đáp án sai) | 1->0 (Giải ra đúng nhưng chốt đáp án sai)
Note: Ghi chú ngắn gọn lý do sai.
Mở Sheet 1_Raw_Logs, lọc Correct == 0 để so sánh cột Confidence giữa 1_Zero_Shot và 3_Zero_Shot_CoT (phục vụ ý "Sai tự tin hơn").
Gói 4 (Làm Web App Streamlit):
Đọc trực tiếp Sheet 2_Summary, 3_Leak_Compare và 4_Errors từ file Excel kết quả để vẽ biểu đồ.
Với tính năng chạy trực tiếp câu hỏi trên Web, import các hàm load_keys, KeyPool, make_prompt, ask_llm, parse_output từ run_benchmark.py.
Gói 5 (Báo cáo & Slide):
Lấy bảng số liệu tổng hợp tại Sheet 2_Summary và 3_Leak_Compare.
Lấy ví dụ minh họa lỗi tại Sheet 4_Errors (sau khi Gói 3 điền xong).