# CLAUDE.md — CoT-Insight VN

Đề tài: yêu cầu LLM "suy nghĩ từng bước" (CoT) giúp đến đâu, và khi nào thì không.
Benchmark 150 câu (VI + EN, 50 Dễ / 50 Trung bình / 50 Khó) × 6 phương pháp prompt × 3 model.
Kế hoạch công việc đang làm nằm ở [PLAN.md](PLAN.md). Kiến trúc code chi tiết ở [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Bản đồ thư mục

- `data/dataset_150.xlsx`: sheet `data_test` (150 câu) và `data_train` (chỉ dùng để hiệu chỉnh ở Gói 3).
- `benchmark/run_benchmark.py`: Small = gemma-4-26b-a4b-it, Large = gemini-3.5-flash-lite (API Google).
- `benchmark/run_extra_llm.py`: Qwen_7B = qwen2.5:7b, **tự host** (endpoint vLLM/Ollama kiểu OpenAI).
- `benchmark/results/benchmark_results_*.xlsx`: 5 sheet `1_Raw_Logs`, `2_Summary`, `3_Leak_Compare`, `4_Errors`, `5_Faithfulness_Matrix`. Bản `20261110_232749` có đủ 3 model (5400 dòng). **App dùng bản `20261110_232749_rescored.xlsx`**: bản này được chấm lại offline, có thêm sheet `6_Rescore_Log` ghi các dòng đã đổi.
- `benchmark/rescore_results.py`: chấm lại một file kết quả từ `Raw_Output` bằng `parse_output` hiện tại, không gọi API và không ghi đè file gốc. Mỗi khi sửa bộ parse thì chạy lại script này.
- `analysis/`: 5 can thiệp sửa lỗi trên 20 ca sai (Gói 3). Có workbook evidence gốc và bản `_Qwen_updated`.
- `app/`: web app Streamlit (Gói 4). `main.py` → `data.py` (loader) → `charts.py` (vẽ) → `tabs/t1..t4`. `llm.py` là cầu nối gọi live, `style.py` chứa CSS.

## Chạy & kiểm tra

- Dùng Python của venv: `.venv/Scripts/python.exe` (Python hệ thống không có streamlit).
- Chạy app: `.venv/Scripts/python.exe -m streamlit run app/main.py`.
- Sau **mọi** thay đổi trong `app/`, chạy AppTest. Kết quả phải là `exc: []`:
  ```
  .venv/Scripts/python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('app/main.py',default_timeout=120).run(); print('exc:',[str(e) for e in at.exception],'errors',len(at.error))"
  ```
- Khi in tiếng Việt ra terminal, đặt `PYTHONIOENCODING=utf-8`.
- Môi trường này không chụp ảnh biểu đồ được (không có kaleido). Phần kiểm tra bằng mắt thì nhờ người dùng xem trên trình duyệt.

## Quy ước dữ liệu (dễ làm hỏng)

- Header tiếng Việt của dataset (`Dạng câu hỏi`, `Đáp án`, `Mức độ`, `Nhóm_Màu_Leak`, …) là **API**, không đổi tên.
- Các sheet gốc của workbook evidence (`Fix_Evidence`, `Case_Evidence`, `Baselines`, `Calibration`) có tiêu đề thật ở **dòng 5 Excel**, nên đọc bằng `header=4`.
- Ngoại lệ: các sheet `Qwen_*` trong workbook `_Qwen_updated` đọc bằng **`header=3`**, tên cột khác bản gốc ("Phương pháp", "Sửa được", "Regression", "Qwen Task_Key"), và gọi tên `NLI Guardrails` (có "s"). Baseline của Qwen đã đúng sẵn 7/20 ca, nên phải tính cả regression, không chỉ tỷ lệ sửa.
- App tự lấy file `benchmark_results_*.xlsx` mới nhất theo tên file. Không thêm ô chọn file vào giao diện.
- Không ghi output vào `.venv/`, vì `.venv/` bị gitignore nên file sẽ biến mất khỏi git.
- **Bộ parse đáp án** (`parse_output`, có trong cả `run_benchmark.py` và `run_extra_llm.py`, hai bản phải giữ giống nhau). Qwen hay ghi đáp án trần rồi mới tới dòng `[Confidence]: 100%` hoặc `80%`, mà bộ parse cũ chỉ xét dòng cuối nên báo `NO_ANSWER` sai ở 204 lượt (Qwen Zero-Shot 19.7% → 62.3% sau khi sửa). Bản sửa bỏ **một** dòng độ tự tin ở cuối rồi mới dò đáp án. Trước khi kết luận gì về một model mới, luôn xem `Raw_Output` của các lượt `NO_ANSWER`.
- **Gói 3 của Qwen đã chạy lại SẠCH** (12/11). Script `analysis/run_qwen_five_methods.py` đã sửa (parser + prompt theo đúng dạng câu) — commit `3ea7c56` trên main. Chạy lại qua Ollama/tunnel → `benchmark_output_qwen_v2/case_results.csv` → `analysis/build_qwen_goi3.py` → **`analysis/qwen_goi3_rerun.xlsx`** (3 sheet header chuẩn: Methods/Cases/Baseline). App đọc file này qua `data.qwen_rerun()`; **không dùng** các sheet `Qwen_*` của workbook cũ nữa. Kết quả sạch: baseline 8/20, 0 lượt câu số bị trả lời chữ cái; Premise +20 (tốt nhất), Calibration −15 (tệ nhất).
- Các loader cũ `qwen_methods/qwen_cases/qwen_methods_corrected/qwen_letter_on_numeric` trong `data.py` đọc workbook cũ (lỗi) — **không còn tab nào dùng**, giữ tạm, có thể xoá khi dọn.
- Kết luận viết trong app phải **tính từ dữ liệu** và phải khớp chiều với từng model. Nhận định nào đã viết sẵn phải kiểm lại bằng số trước khi đưa lên.

## Quy ước app (người dùng đã chốt)

- Chuỗi hiển thị viết **tiếng Việt có dấu**. Tên hàm, tên biến, key, tên sheet giữ ASCII.
- **So sánh là mặc định**: dùng facet hoặc màu theo `Model` và `Language`, không thêm bộ lọc ẩn dữ liệu. Không có sidebar filter.
- **Không** đặt tiêu đề bên trong figure (`_layout` để `title=None`). Tiêu đề là dòng markdown in đậm trong tab.
- Mỗi biểu đồ theo khuôn: **khẳng định in đậm → dòng in nghiêng "loại biểu đồ — quan hệ giữa đại lượng nào" → biểu đồ → `st.caption("Đọc: …")` → `st.success("**Kết luận:** …")`**.
- Màu theo phương pháp cố định trong `METHOD_COLORS` (đã chạy validator CVD), không tự đổi. Độ khó dùng thang xanh ordinal.
- Legend và annotation không được che điểm dữ liệu. Mũi tên phải trỏ đúng một điểm dữ liệu thật.
- Box plot dữ liệu dồn cục: nền hộp trong suốt (`_rgba(color, 0.15)`) + điểm jitter, để hộp không thành khối đặc.
- Tab 1 và Tab 4B: **Phát lại** (đọc kết quả đã lưu) là mặc định. **Gọi API thật** chỉ chạy sau nút bấm và phải bắt lỗi gọn gàng.
- Trang trí (CSS, theme) chỉ sửa trong `style.py` và `.streamlit/config.toml`. Không đụng nội dung biểu đồ khi người dùng chỉ yêu cầu chỉnh layout.

## Git

- Nhánh làm việc: `sys/web`, remote `origin` = github.com/yazfboy/CoT-insight-vn. Nhánh chính: `main`.
- Cập nhật từ main bằng **`git merge origin/main`**, không rebase, vì nhánh đã push và có collaborator.
- Chỉ commit hoặc push khi người dùng yêu cầu.
- Commit message nhiều dòng: ghi ra file rồi chạy `git commit -F <file>`. Không dùng here-string PowerShell `@'…'@` trong Bash, vì sẽ dính ký tự `@` vào message.
