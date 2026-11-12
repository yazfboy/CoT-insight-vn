# PLAN — Sửa analysis + chạy lại Qwen Gói 3 + build lại UI (sys/web)

> Quy ước: [CLAUDE.md](CLAUDE.md). Đánh `[x]` khi xong.

## Luồng tổng (người dùng đã chốt)

```
Sửa parser + prompt của ANALYSIS  →  push lên main  →  chạy lại Qwen Gói 3
   →  output analysis mới (evidence Qwen sạch) lên main  →  build lại UI trên sys/web
```
- **Benchmark giữ nguyên** (dữ liệu đã sạch; đã hoàn tác mọi sửa đổi benchmark về đúng `main`).
- Chỉ **analysis (Gói 3)** được sửa và **chỉ Qwen** được chạy lại. Small/Large Gói 3 không chạy
  lại (evidence của chúng đã đúng — đã kiểm 0/80 dòng lệch).

## Bối cảnh dữ liệu

- Benchmark 3 model (`benchmark_results_20261110_232749.xlsx` = `..._095454.xlsx`, hai file nay
  trùng nhau) đã sạch. App tự lấy file này sau khi đã xoá bản `_rescored` local.
- 3 lỗi của Gói 3 Qwen (tất cả ở `analysis/..._Qwen_updated.xlsx`):
  - 17 lượt câu số bị Qwen trả lời chữ cái (sheet `Qwen_Case_Evidence`, dòng 10,11,14,16,19,21,
    24,56,64,75,76,79,84,96,99,101,104) — **lỗi prompt**.
  - Baseline 7/20 thay vì 8/20 (sheet `Qwen_Baseline`, dòng 18, MEDIUM_068).
  - 2 ca đọc nhầm "100" (sheet `Qwen_Case_Evidence`, dòng 20 và 95) — **lỗi parser**.

### Số kiểm định đã đo (viết kết luận UI cho đúng) — n=300/ô, McNemar ghép cặp
- Zero-Shot→CoT: có ý nghĩa ở cả 3 model (p<1e-5).
- Few-Shot < Zero-Shot: Small p=5e-4, Large p=0.035, Qwen p=0.020.
- Zero-CoT→Self-Consistency: chỉ Qwen có ý nghĩa (p=0.013); Small/Large không.
- **Rò rỉ: KHÔNG model nào đạt ý nghĩa khi xét riêng** (p=0.25/0.06/0.27) → bỏ câu "+4–6đ chắc chắn".

---

## Giai đoạn 1 — Sửa ANALYSIS (đã làm ở local, cần đóng gói để push)

- [x] **1.1** `analysis/run_qwen_five_methods.py`: sửa **parser** (`extract_final_line`/
  `normalize_answer`/`parse_confidence` đồng bộ với benchmark — bỏ dòng độ tự tin ở cuối, câu số
  lấy số cuối, nhận `### Final Answer]`/`**…**`). Sửa **prompt** (`answer_tail(lang, numeric)` ghi
  định dạng theo đúng dạng câu, áp cho cả 7 nơi: Premise/Strict/PAL/Logprob/NLI/retry/retry3/
  calibration). → sửa lỗi 17 chữ-cái + 2 ca "100".
- [x] **1.2** `analysis/run_six_method_experiment_20261106.py`: sửa **parser** y hệt (đồng bộ).
  *Chỉ parser* — Small/Large không chạy lại nên không đụng prompt; để consistency.
- [x] **1.3** Hoàn tác mọi sửa đổi `benchmark/*` về `main`; xoá `rescore_results.py` và file
  `_rescored.xlsx`.
- [ ] **1.4** Kiểm tra lại: dựng thử prompt cho câu số và câu trắc nghiệm (đã làm, đạt);
  unit-test parser (9/9 đạt); so trên dữ liệu thật: Small/Large 0 dòng đổi, Qwen 3 dòng sai→đúng.
- [ ] **1.5** Commit lên `sys/web` **hoặc** nhánh để merge vào main (người dùng sẽ push lên main).
  Commit **chỉ gồm 2 file analysis**, message nêu 3 lỗi đã sửa.

**Xong khi:** 2 file analysis đã sửa + kiểm tra, sẵn sàng để người dùng push lên main.

## Giai đoạn 2 — Chạy lại Qwen Gói 3 (người có server Qwen, sau khi main có bản sửa)

- [ ] **2.1** Chạy:
  ```
  python analysis/run_qwen_five_methods.py --new-run \
    --base-url <endpoint vLLM/Ollama> \
    --reviewed-results analysis/benchmark_results_20261106_reviewed.xlsx \
    --dataset data/dataset_150.xlsx \
    --baseline-results benchmark/results/benchmark_results_20261110_232749.xlsx \
    --output-dir <thư mục MỚI ngoài .venv>
  ```
  - **Bắt buộc `--new-run`** (nếu không sẽ resume kết quả cũ).
  - `--baseline-results` trỏ file benchmark đã sạch → baseline tự ra 8/20.
- [ ] **2.2** Người tổng hợp dựng lại các sheet `Qwen_*` trong workbook evidence từ output mới,
  **ghi cột `Question type`** (trước bị trống). Push file evidence mới lên main.

**Xong khi:** `analysis/..._Qwen_updated.xlsx` trên main có dữ liệu Qwen sạch (0 lượt chữ-cái ở
câu số, baseline 8/20).

## Giai đoạn 3 — Build lại UI trên sys/web (sau khi main có output mới)

- [ ] **3.1** `git merge origin/main` vào `sys/web` để lấy evidence Qwen mới.
- [ ] **3.2** **Bật lại Tab 4 mục 4** (Qwen Gói 3): bỏ `st.warning` lỗi prompt, cho kết luận
  tính động bình thường. (Tới lúc này dữ liệu đã sạch.)
- [ ] **3.3** Nguồn dữ liệu: xác nhận app lấy `...232749.xlsx` của main; sửa hộp "đã chấm lại"
  ở `t2_benchmark.py` (đọc `6_Rescore_Log` không còn) thành 1 dòng ghi nguồn.
- [ ] **3.4** **Lớp thống kê** (việc chính của UI):
  - Thêm `app/stats.py`: `wilson_ci`, `mcnemar` (ghép theo `ID_Language`), `holm`.
  - Thêm thanh khoảng tin cậy Wilson vào biểu đồ accuracy (`error_y`).
  - Tab 2: bảng kiểm định (model · Δ · n · p-Holm · nhãn ý nghĩa) trong expander.
  - **Viết lại kết luận cho khớp kiểm định** — đặc biệt bỏ "rò rỉ +4–6đ chắc chắn".
- [ ] **3.5** **Ghi chú công bằng khi so model** (caption, không đụng dữ liệu):
  - SC của Qwen dùng lại Zero-CoT làm phiếu 1 (nhưng cả 3 model SC~Zero-CoT nên ảnh hưởng nhỏ).
  - Chi phí/độ trễ Qwen tự host → tách hoặc ghi chú; không so token tuyệt đối giữa 2 họ.
  - Tên model thật (gemma-4-26b / gemini-3.5-flash-lite / qwen2.5:7b).

**Xong khi:** UI dùng dữ liệu sạch cả benchmark lẫn Gói 3, có thống kê, kết luận khớp kiểm định.

## Giai đoạn 4 — Kiểm tra & push

- [ ] **4.1** AppTest `exc: []`; bấm thử Tab 1 (3 model) + Tab 4B phát lại.
- [ ] **4.2** Người dùng xem trình duyệt (CI, bảng p-value, Tab 4 mục 4 đã sạch).
- [ ] **4.3** Cập nhật `README.md` + `CLAUDE.md`. Push `sys/web`.

---

## Trạng thái ngay lúc này (chờ bước tiếp)

- Benchmark: đã về `main`, không đụng.
- Analysis: 2 file đã sửa ở working tree, **chưa commit**.
- UI: **Tab 4 mục 4 tạm ẩn** trong lúc chờ output Qwen mới (Giai đoạn 3.2 bật lại). Lớp thống kê
  (3.4) có thể làm song song ngay vì chỉ dùng dữ liệu benchmark (đã sạch).

## Việc có thể làm NGAY mà không cần chờ chạy lại Qwen
1. [x] Đẩy bản sửa analysis lên main (commit `3ea7c56`).
2. [x] Dọn nguồn dữ liệu: xoá `_rescored.xlsx` + `rescore_results.py`; app dùng `...232749.xlsx`;
   bỏ hộp "đã chấm lại" ở Tab 2 (A3).
3. [x] Ẩn Tab 4 mục 4 (A4) — có caption "tạm ẩn, chờ chạy lại".
4. [x] Lớp thống kê (3.4): thêm `app/stats.py` (Wilson/McNemar/Holm); bảng kiểm định trong
   expander đầu Tab 2; thanh CI 95% trên biểu đồ 5; **kết luận mục 2/5/6 lấy verdict từ bảng
   Holm** (SC không ý nghĩa ở model nào; Few-Shot chỉ Small; rò rỉ không model nào → đã bỏ
   "+4–6đ chắc chắn", chuyển sang cảnh báo thận trọng).

5. [x] **Chạy lại Qwen Gói 3** qua tunnel (135 lượt, 0 lỗi) bằng script đã sửa → `case_results.csv`.
   - `analysis/build_qwen_goi3.py` gộp CSV → `analysis/qwen_goi3_rerun.xlsx` (3 sheet header chuẩn).
   - Kiểm: baseline **8/20**, **0** lượt câu số bị trả lời chữ cái (prompt đã ăn).
6. [x] **Bật lại Tab 4 mục 4** dùng `data.qwen_rerun()` (đọc file sạch, bỏ coupling header=3):
   - Premise +20 (60%, tốt nhất), Calibration −15 (tệ nhất); sửa 15 / hỏng 11; chỉ **NLI** 0 regression.
   - Bỏ cảnh báo "chưa tin cậy"; kết luận tính động từ số sạch.

### Còn lại của UI
- [x] **C1** Ghi chú SC của Qwen dùng lại Zero-CoT (ảnh hưởng nhỏ vì SC≈Zero-CoT ở cả 3 model).
- [x] **C2** Cảnh báo chi phí/độ trễ/token Qwen tự host không so xuyên model.
- [x] **C5** Chú thích tên model thật ở màn mở đầu.
- [x] **Commit + push `sys/web`** — commit `fffec8f` (`9a26f39..fffec8f`).

### Thực sự còn lại (tuỳ chọn / việc của người khác)
- [ ] **4.2** Người dùng tự xem trình duyệt (CI, bảng p-value, Tab 4 mục 4).
- [ ] (tuỳ chọn) Thêm CI cho biểu đồ 1 (độ khó); dọn các loader Qwen cũ không dùng trong `data.py`.
- [ ] (tuỳ chọn) Mở PR `sys/web` → `main`.

> Toàn bộ Giai đoạn 1–4 ở trên đã hoàn tất (checkbox cũ còn `[ ]` là do chưa tick lúc làm);
> cách thực hiện khác plan ở chỗ **không dựng sheet `Qwen_*` tay** mà dùng `build_qwen_goi3.py`
> → `analysis/qwen_goi3_rerun.xlsx` (app đọc qua `data.qwen_rerun`), và **không cần merge main
> vào sys/web** vì file sạch + bản sửa analysis đã commit thẳng trên `sys/web`.
