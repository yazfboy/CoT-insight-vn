import os
import re
import glob
import time
import threading
from datetime import datetime
import requests
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_MODE = False  # False = chay full 150 cau | True = test 5 cau

XLSX_PATH = os.path.join(BASE_DIR, "data", "dataset_150.xlsx")
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_150.csv")
DATA_PATH = XLSX_PATH if os.path.exists(XLSX_PATH) else CSV_PATH

OUT_DIR = os.path.join(BASE_DIR, "benchmark", "results")
os.makedirs(OUT_DIR, exist_ok=True)

EXTRA_CKPT_CSV = os.path.join(OUT_DIR, "checkpoint_extra_qwen.csv")
BASE_CKPT_CSV = os.path.join(OUT_DIR, "checkpoint_v2.csv")

# ==============================================================================
# CAU HINH SERVER OLLAMA / vLLM & THAM SO TOI UU DP + DA LUONG
# ==============================================================================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").strip()
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "gdg-nhom2").strip()

EXTRA_MODEL_KEY = "Qwen_7B"
EXTRA_MODEL_CFG = {
    "id": "qwen2.5:7b",
    "in_price": 0.05,   # USD / 1M prompt tokens
    "out_price": 0.10,  # USD / 1M completion tokens
}

WORKERS = 6

MAX_TOK_DIRECT = 256   # Zero-shot / Few-shot
MAX_TOK_COT = 1024     # CoT / SC / Leaked
SC_REUSE_ZCOT = True   # Tai su dung ket qua 3_Zero_Shot_CoT lam luong 1 cua 5_Self_Consistency

METHODS = [
    "1_Zero_Shot",
    "2_Few_Shot",
    "3_Zero_Shot_CoT",
    "4_Few_Shot_CoT",
    "5_Self_Consistency",
    "6_Few_Shot_CoT_Leaked",
]
LANGS = ["VI", "EN"]
SC_RUNS = 5

HTTP_SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=WORKERS * 2, pool_maxsize=WORKERS * 2)
HTTP_SESSION.mount("https://", adapter)
HTTP_SESSION.mount("http://", adapter)

# ==============================================================================
# GIU NGUYEN 100% PROMPT, EXAMPLES, LEAK HINTS & REGEX TU RUN_BENCHMARK.PY
# ==============================================================================
EXAMPLES = {
    "VI": {
        "direct": (
            "Ví dụ 1:\nCâu hỏi: Tiệm bánh nướng 45 chiếc, bán 18 chiếc rồi nướng thêm 12 chiếc. Còn mấy chiếc?\n[Đáp án cuối cùng]: 39\n[Độ tự tin]: 95%\n\n"
            "Ví dụ 2:\nCâu hỏi: Mọi loài cá thở bằng mang. Cá voi thở bằng phổi. Câu nào đúng? A. Cá voi là cá B. Cá voi không phải cá C. Cá thở bằng phổi D. Không rõ\n[Đáp án cuối cùng]: B\n[Độ tự tin]: 100%"
        ),
        "cot": (
            "Ví dụ 1:\nCâu hỏi: Tiệm bánh nướng 45 chiếc, bán 18 chiếc rồi nướng thêm 12 chiếc. Còn mấy chiếc?\n"
            "Lập luận:\n- Bước 1: Còn lại 45 - 18 = 27 chiếc.\n- Bước 2: Nướng thêm 12 chiếc -> 27 + 12 = 39 chiếc.\n[Đáp án cuối cùng]: 39\n[Độ tự tin]: 98%\n\n"
            "Ví dụ 2:\nCâu hỏi: Mọi loài cá thở bằng mang. Cá voi thở bằng phổi. Câu nào đúng? A. Cá voi là cá B. Cá voi không phải cá C. Cá thở bằng phổi D. Không rõ\n"
            "Lập luận:\n- Bước 1: Cá phải thở bằng mang.\n- Bước 2: Cá voi thở bằng phổi nên không phải cá (Chọn B).\n[Đáp án cuối cùng]: B\n[Độ tự tin]: 100%"
        ),
    },
    "EN": {
        "direct": (
            "Example 1:\nQuestion: A bakery baked 45 croissants, sold 18, and baked 12 more. How many left?\n[Final Answer]: 39\n[Confidence]: 95%\n\n"
            "Example 2:\nQuestion: All fish use gills. Whales use lungs. Which is true? A. Whales are fish B. Whales are not fish C. Fish use lungs D. Unknown\n[Final Answer]: B\n[Confidence]: 100%"
        ),
        "cot": (
            "Example 1:\nQuestion: A bakery baked 45 croissants, sold 18, and baked 12 more. How many left?\n"
            "Reasoning:\n- Step 1: 45 - 18 = 27.\n- Step 2: 27 + 12 = 39.\n[Final Answer]: 39\n[Confidence]: 98%\n\n"
            "Example 2:\nQuestion: All fish use gills. Whales use lungs. Which is true? A. Whales are fish B. Whales are not fish C. Fish use lungs D. Unknown\n"
            "Reasoning:\n- Step 1: Fish must use gills.\n- Step 2: Whales use lungs so they are not fish (B).\n[Final Answer]: B\n[Confidence]: 100%"
        ),
    },
}

LEAK_HINTS = {
    "Leak_01_PhamVanDong": {
        "VI": "Ví dụ cùng đoạn văn:\nCâu hỏi: Chức vụ từ 1981 của Phạm Văn Đồng và Chủ nhiệm tại Quế Lâm? A. Tô B. Chủ tịch HĐ Bộ trưởng (Chủ nhiệm là Hồ Học Lãm) C. Tổng Bí thư D. Không có nhắc tới\nLập luận: Đối chiếu sát văn bản; nếu chi tiết không được đề cập trong đoạn văn thì chọn D.\n[Đáp án cuối cùng]: B\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nQuestion: Pham Van Dong's title from 1981? A. To B. Chairman of Council of Ministers C. General Secretary D. Not mentioned\nReasoning: Match text directly; if unmentioned in passage, choose D.\n[Final Answer]: B\n[Confidence]: 100%",
    },
    "Leak_02_MaMayNhiPhan": {
        "VI": "Ví dụ cùng đoạn văn:\nCâu hỏi: Mã máy nhị phân là ngôn ngữ thế hệ mấy? A. Thế hệ đầu tiên B. Thế hệ hai C. Thế hệ ba D. Không có nhắc tới\nLập luận: Mã nhị phân là thế hệ đầu tiên, khó viết vì quản lý từng bit; nếu hỏi mã thập phân hay tên hợp ngữ không có trong bài thì chọn D.\n[Đáp án cuối cùng]: A\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nQuestion: Generation of binary code? A. First B. Second C. Third D. Not mentioned\nReasoning: First-gen, hard to manage bits; choose D if asked about unmentioned decimal/assembly names.\n[Final Answer]: A\n[Confidence]: 100%",
    },
    "Leak_03_DanSoNhatBan": {
        "VI": "Ví dụ cùng đoạn văn:\nCâu hỏi: Thay đổi dân số Nhật gây ra gì? A. Thay đổi dân số làm giảm lao động, tăng chi phí phúc lợi, xu hướng giảm vì người trẻ không kết hôn/sinh con (ngoài bài chọn D)\n[Đáp án cuối cùng]: A\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nQuestion: Japan population trend? A. Population change reduces labor, raises pension costs, downward trend (choose D if unmentioned)\n[Final Answer]: A\n[Confidence]: 100%",
    },
    "Leak_04_BongDaNhatBan": {
        "VI": "Ví dụ cùng đoạn văn:\nCâu hỏi: Bóng đá Nhật? B. Giải chuyên nghiệp lập 1992, Nhật tổ chức Cúp liên lục địa 1981-2004, 2 nước đồng chủ nhà 2002, 4 lần vô địch châu Á.\n[Đáp án cuối cùng]: B\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nQuestion: Japan football facts? B. League in 1992, Japan hosted 1981-2004, 2 co-hosts in 2002, 4 Asian Cups.\n[Final Answer]: B\n[Confidence]: 100%",
    },
    "Leak_05_DongHoCat": {
        "VI": "Ví dụ cùng đoạn văn:\nCâu hỏi: Đồng hồ cát đi biển? A. Tàu lắc không ảnh hưởng, vượt trội đồng hồ nước vì nước dễ đông cứng, dùng xác định kinh độ (hỏi bẫy ngoài bài chọn D).\n[Đáp án cuối cùng]: A\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nQuestion: Marine sandglass? A. Ship motion doesn't affect it, superior to water clock (freezes), measures longitude (trick questions -> D).\n[Final Answer]: A\n[Confidence]: 100%",
    },
    "Leak_06_HoiNghiQuenBiet": {
        "VI": "Ví dụ cùng cấu trúc:\nCâu hỏi: 8 người họp (1 quen 3, 3 quen 2, 4 quen 1)? Lập luận: Tổng bậc = 13 (lẻ) -> quen một chiều qua truyền hình/giới thiệu.\n[Đáp án cuối cùng]: C\n[Độ tự tin]: 100%",
        "EN": "Pattern Example:\nQuestion: 8 people (1 knows 3, 3 know 2, 4 know 1)? Reasoning: Sum = 13 (odd) -> one-way acquaintance via TV/briefing.\n[Final Answer]: C\n[Confidence]: 100%",
    },
    "Leak_07_NguyBienLuanDiem": {
        "VI": "Ví dụ cùng cấu trúc:\nCâu hỏi: Ngụy biện vòng quanh? Lập luận: Câu kẻ phản bội -> C (Trái Đất hình cầu); câu Aristotle -> D (Ý kiến anh sai nên tôi đúng).\n[Đáp án cuối cùng]: C\n[Độ tự tin]: 98%",
        "EN": "Pattern Example:\nQuestion: Circular fallacy? Reasoning: Traitor -> C (Earth is a sphere); Aristotle -> D (Your opinion is wrong).\n[Final Answer]: C\n[Confidence]: 98%",
    },
    "Leak_08_GiaThuyetYeuTo": {
        "VI": "Ví dụ cùng cấu trúc:\nCâu hỏi: Làm suy yếu kết luận lạm phát/tự tử Greenland? Lập luận: Số liệu gốc bị nhiễu (năm ngoái tạm rơi dưới 4% / năm 1960 không ghi hồ sơ) -> D.\n[Đáp án cuối cùng]: D\n[Độ tự tin]: 100%",
        "EN": "Pattern Example:\nQuestion: Weaken inflation/Greenland conclusion? Reasoning: Baseline distortion -> D.\n[Final Answer]: D\n[Confidence]: 100%",
    },
    "Leak_09_ChuyenDongGapNhau": {
        "VI": "Ví dụ công thức:\n(1) Đua 1000m thắng 48m hoặc 12s -> v_B=4m/s, t_A=250-12=238s (A). (2) Tàu đi 25h và 16h sau gặp -> v1/v2=sqrt(16/25)=4:5 (D). (3) Đua vòng tròn 600m (10m/s, 15m/s) -> BCNN(60,40)=120s (A).\n[Đáp án cuối cùng]: A\n[Độ tự tin]: 100%",
        "EN": "Formula Example:\n(1) 1000m race 48m/12s -> 238s (A). (2) Trains 25h/16h -> 4:5 (D). (3) 600m track -> LCM(60,40)=120s (A).\n[Final Answer]: A\n[Confidence]: 100%",
    },
    "Leak_10_NangSuatCongViec": {
        "VI": "Ví dụ công thức:\n(1) A=4h, B+C=3h, A+C=2h -> B=12h (C). (2) Máy in 20+10=30 bản/p cho 900 bản -> 30p (D). (3) 3 máy bơm vs 15 máy bơm -> A (3.5).\n[Đáp án cuối cùng]: C\n[Độ tự tin]: 100%",
        "EN": "Formula Example:\n(1) A=4h, B+C=3h, A+C=2h -> 12h (C). (2) Copiers 900/30 -> 30m (D). (3) 15 pumps -> A (3.5).\n[Final Answer]: C\n[Confidence]: 100%",
    },
    "Leak_11_LichSuQuanSu": {
        "VI": "Ví dụ cùng ngữ cảnh:\nKhông quân (15 trung đoàn) > Công binh (0) -> A; Dalmatia 18 - 20 = 3559 - 3110 = 449; Hạ Lý = 68; Phòng không lớn nhất = 1992.\n[Đáp án cuối cùng]: 449\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nAir Force > Engineers -> A; Dalmatia diff = 449; Ha Ly = 68; max regiment = 1992.\n[Final Answer]: 449\n[Confidence]: 100%",
    },
    "Leak_12_ThucVatHoc": {
        "VI": "Ví dụ cùng ngữ cảnh:\nMào lông đầu hạt = 3,5 cm (C); Hoa lớn dài = 4,5 - 5,4 cm (C).\n[Đáp án cuối cùng]: C\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nSeed hair tuft = 3.5 cm (C); Large flowers = 4.5 - 5.4 cm (C).\n[Final Answer]: C\n[Confidence]: 100%",
    },
    "Leak_13_PhienBanHeThong": {
        "VI": "Ví dụ cùng ngữ cảnh:\nTổng phiên bản UDF (1.00, 1.01, 1.02, 1.50) = 4; Drupal cũ nhất (5, 6, 7, 8) = 5.\n[Đáp án cuối cùng]: 4\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nTotal UDF versions = 4; earliest Drupal = 5.\n[Final Answer]: 4\n[Confidence]: 100%",
    },
    "Leak_14_YTeDichTe": {
        "VI": "Ví dụ cùng ngữ cảnh:\nTử vong mẹ Mỹ gấp Phần Lan = 3 lần; Tử vong nCoV Anh tăng thêm = 428.\n[Đáp án cuối cùng]: 3\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nUS vs Finland = 3; UK additional deaths = 428.\n[Final Answer]: 3\n[Confidence]: 100%",
    },
    "Leak_15_DiaGioiHanhChinh": {
        "VI": "Ví dụ cùng ngữ cảnh:\nSố xã mới lập (Ia Khai, Ia Nhin, Ia Ly) = 3; Diện tích xã Hà Phong = 267,2 ha (B).\n[Đáp án cuối cùng]: 3\n[Độ tự tin]: 100%",
        "EN": "Context Example:\nNew communes = 3; Ha Phong area = 267.2 ha (B).\n[Final Answer]: 3\n[Confidence]: 100%",
    },
}


def make_prompt(row, method, lang):
    is_int = str(row["Dạng câu hỏi"]).strip().lower() == "số nguyên"
    q = row["Câu hỏi Tiếng Việt"] if lang == "VI" else row["Câu hỏi Tiếng Anh"]
    leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()

    if lang == "VI":
        rule = "một số nguyên duy nhất" if is_int else "một chữ cái (A, B, C hoặc D)"
        tail = f"BẮT BUỘC kết thúc bằng đúng 2 dòng:\n[Đáp án cuối cùng]: <chỉ ghi {rule}>\n[Độ tự tin]: <0-100>%"
        if method == "1_Zero_Shot":
            return f"Trả lời trực tiếp không giải thích.\nCâu hỏi: {q}\n\n{tail}"
        if method == "2_Few_Shot":
            return f"Trả lời theo mẫu (không giải thích):\n\n{EXAMPLES['VI']['direct']}\n\nCâu hỏi: {q}\n\n{tail}"
        if method in ["3_Zero_Shot_CoT", "5_Self_Consistency"]:
            return f"Câu hỏi: {q}\n\nHãy suy nghĩ từng bước trước khi chốt đáp án.\n{tail}"
        if method == "4_Few_Shot_CoT":
            return f"Giải từng bước theo mẫu:\n\n{EXAMPLES['VI']['cot']}\n\nCâu hỏi: {q}\nLập luận:\n{tail}"
        if method == "6_Few_Shot_CoT_Leaked":
            hint = LEAK_HINTS.get(leak_grp, {}).get("VI", EXAMPLES["VI"]["cot"])
            return f"Giải từng bước theo mẫu:\n\n{EXAMPLES['VI']['cot']}\n\n{hint}\n\nCâu hỏi: {q}\nLập luận:\n{tail}"
    else:
        rule = "a single integer" if is_int else "a single letter (A, B, C, or D)"
        tail = f"MUST end with exact 2 lines:\n[Final Answer]: <{rule}>\n[Confidence]: <0-100>%"
        if method == "1_Zero_Shot":
            return f"Answer directly without explanation.\nQuestion: {q}\n\n{tail}"
        if method == "2_Few_Shot":
            return f"Follow the format below (no explanation):\n\n{EXAMPLES['EN']['direct']}\n\nQuestion: {q}\n\n{tail}"
        if method in ["3_Zero_Shot_CoT", "5_Self_Consistency"]:
            return f"Question: {q}\n\nLet's think step by step.\n{tail}"
        if method == "4_Few_Shot_CoT":
            return f"Solve step by step following examples:\n\n{EXAMPLES['EN']['cot']}\n\nQuestion: {q}\nReasoning:\n{tail}"
        if method == "6_Few_Shot_CoT_Leaked":
            hint = LEAK_HINTS.get(leak_grp, {}).get("EN", EXAMPLES["EN"]["cot"])
            return f"Solve step by step following examples:\n\n{EXAMPLES['EN']['cot']}\n\n{hint}\n\nQuestion: {q}\nReasoning:\n{tail}"


ANS_TAG_RE = re.compile(r"(?:Đáp án cuối cùng|Final Answer)\s*\]?\s*\**\s*[:：]\s*\**\s*([^\n]*)", re.I)
CONF_TAG_RE = re.compile(r"(?:Độ tự tin|Confidence)\s*\]?\s*\**\s*[:：]\s*\**\s*(\d+(?:[.,]\d+)?)", re.I)
BARE_ANS_RE = re.compile(r"^\W*(?:(?:đáp án|answer)\s*[:：]?\s*)?(-?[\d.,]+|[A-D])\W*$", re.I)


def _extract_answer(line, is_int):
    if is_int:
        s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", line)
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not nums:
            return ""
        v = float(nums[-1])
        return str(int(v)) if v.is_integer() else str(v)
    m = re.search(r"\b([A-D])\b", line.upper())
    return m.group(1) if m else ""


def parse_output(text, q_type, finish_reason="STOP"):
    if not text or str(text).startswith("ERR:"):
        return "API_ERR", None, "API_ERR"

    is_int = str(q_type).strip().lower() == "số nguyên"

    confs = CONF_TAG_RE.findall(text)
    conf = min(100.0, max(0.0, float(confs[-1].replace(",", ".")))) if confs else None

    ans = ""
    tags = ANS_TAG_RE.findall(text)
    if tags:
        ans = _extract_answer(tags[-1].strip(), is_int)
    else:
        lines = [x.strip() for x in str(text).strip().split("\n") if x.strip()]
        if lines and BARE_ANS_RE.match(lines[-1]):
            ans = _extract_answer(lines[-1], is_int)

    if ans:
        status = "OK"
    elif str(finish_reason).upper().endswith("MAX_TOKENS"):
        status = "TRUNCATED"
    else:
        status = "NO_ANSWER"
    return ans, conf, status


def is_match(pred, gt):
    if pred == "API_ERR" or not pred:
        return 0
    p = str(pred).strip().upper().replace(",", "")
    g = str(gt).strip().upper().replace(",", "")
    if g.endswith(".0"):
        g = g[:-2]
    return 1 if (p and p == g) else 0


def check_faithfulness(raw_text, pred, gt, correct, q_type, method, status="OK"):
    if method in ["1_Zero_Shot", "2_Few_Shot", "5_Self_Consistency"] or status != "OK":
        return "N/A"

    parts = re.split(r"(?:Đáp án cuối cùng|Final Answer)\s*\]?\s*\**\s*[:：]", str(raw_text), flags=re.I)
    body = parts[0].strip() if len(parts) > 1 else ""
    if not body:
        return "0->1" if correct == 1 else "0->0"

    is_int = str(q_type).strip().lower() == "số nguyên"
    step_ans = ""
    if is_int:
        nums = re.findall(r"-?\d+(?:\.\d+)?", body.replace(",", ""))
        if nums:
            v = float(nums[-1])
            step_ans = str(int(v)) if v.is_integer() else str(v)
    else:
        matches = re.findall(r"\b([A-D])\b", body.upper())
        if matches:
            step_ans = matches[-1]

    step_ok = is_match(step_ans, gt)
    if step_ok == 1 and correct == 1:
        return "1->1"
    if step_ok == 1 and correct == 0:
        return "1->0"
    if step_ok == 0 and correct == 1:
        return "0->1"
    return "0->0"


# ==============================================================================
# CAU CHI NGAT MACH TU DONG (CIRCUIT BREAKER) & BANG QUY HOACH DONG (DP TABLE)
# ==============================================================================
class CircuitBreaker:
    """Ngat mach ngay lap tuc neu Colab/Cloudflare bi rot ket noi lien tiep."""
    def __init__(self, max_consecutive_failures=8):
        self.lock = threading.Lock()
        self.failures = 0
        self.max_failures = max_consecutive_failures
        self.tripped = False

    def record_success(self):
        with self.lock:
            self.failures = 0

    def record_failure(self):
        with self.lock:
            self.failures += 1
            if self.failures >= self.max_failures and not self.tripped:
                self.tripped = True
                tqdm.write("\n⚠️ [NGẮT MẠCH TỰ ĐỘNG] Phát hiện Colab/Cloudflare đã rớt kết nối! Đang dừng an toàn và lưu Checkpoint...")

    def is_open(self):
        with self.lock:
            return self.tripped


class DPMemoTable:
    """Bang luu trang thai cac bai toan con da giai (Thread-safe Memoization)."""
    def __init__(self, initial_records=None):
        self.lock = threading.Lock()
        self.memo = {}
        self.new_count = 0
        if initial_records:
            for r in initial_records:
                if r.get("Predicted") != "API_ERR" and pd.notna(r.get("Predicted")):
                    self.memo[r["Task_Key"]] = r

    def get(self, task_key):
        with self.lock:
            return self.memo.get(task_key)

    def put(self, record, ckpt_path=None, flush_every=10):
        if record.get("Predicted") == "API_ERR" or record.get("Status") == "API_ERR":
            return
        with self.lock:
            self.memo[record["Task_Key"]] = record
            self.new_count += 1
            if ckpt_path and (self.new_count % flush_every == 0):
                pd.DataFrame(list(self.memo.values())).to_csv(ckpt_path, index=False, encoding="utf-8-sig")

    def flush(self, ckpt_path):
        with self.lock:
            if self.memo and ckpt_path:
                pd.DataFrame(list(self.memo.values())).to_csv(ckpt_path, index=False, encoding="utf-8-sig")

    def all_records(self):
        with self.lock:
            return list(self.memo.values())

    def size(self):
        with self.lock:
            return len(self.memo)


def ask_vllm(base_url, api_key, model_id, prompt, temp, breaker, max_tok=MAX_TOK_COT):
    if breaker.is_open():
        return {"text": "ERR: CircuitBreaker Open", "tok_in": 0, "tok_out": 0, "tok_think": 0, "lat": 0.0, "key": 1, "finish": "ERROR"}

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temp),
        "max_tokens": int(max_tok),
    }

    last_err = ""
    for attempt in range(3):
        if breaker.is_open():
            break
        try:
            t0 = time.time()
            resp = HTTP_SESSION.post(url, headers=headers, json=payload, timeout=90)
            lat = round(time.time() - t0, 3)

            if resp.status_code in (502, 503, 504, 524, 530):
                last_err = f"Cloudflare/Colab Tunnel Down (HTTP {resp.status_code})"
                time.sleep(1.0)
                continue
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
                time.sleep(1.0)
                continue

            data = resp.json()
            choice = data["choices"][0]
            txt = choice["message"].get("content") or ""
            raw_fin = str(choice.get("finish_reason") or "stop").lower()
            fin = "MAX_TOKENS" if raw_fin == "length" else "STOP"

            usage = data.get("usage") or {}
            tok_in = int(usage.get("prompt_tokens") or 0)
            tok_out = int(usage.get("completion_tokens") or 0)

            breaker.record_success()
            return {
                "text": txt,
                "tok_in": tok_in,
                "tok_out": tok_out,
                "tok_think": 0,
                "lat": lat,
                "key": 1,
                "finish": fin,
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0)

    breaker.record_failure()
    return {
        "text": f"ERR: {last_err}",
        "tok_in": 0,
        "tok_out": 0,
        "tok_think": 0,
        "lat": 0.0,
        "key": 1,
        "finish": "ERROR",
    }


def make_record(row, m_key, cfg, method, lang, pred, conf, agree, r, status, sc_runs=1):
    q_id = row["ID"]
    lvl_map = {"Dễ": "Easy", "Trung bình": "Medium", "Khó": "Hard"}
    lvl = lvl_map.get(str(row["Mức độ"]).strip(), row["Mức độ"])
    q_type = row["Dạng câu hỏi"]
    leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()
    leak_flag = "Unleaked" if leak_grp == "Khong_Lo_Unleaked" else "Leaked"
    gt = str(row["Đáp án"]).strip()
    raw = r["text"]
    tok_in, tok_out = r["tok_in"], r["tok_out"]

    correct = is_match(pred, gt)
    faith = check_faithfulness(raw, pred, gt, correct, q_type, method, status)
    tot_tok = tok_in + tok_out
    cost = round((tok_in * cfg["in_price"] + tok_out * cfg["out_price"]) / 1000.0, 5)

    return {
        "Task_Key": f"{q_id}_{m_key}_{lang}_{method}",
        "Run_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ID": q_id,
        "Difficulty": lvl,
        "Category": q_type,
        "Leak_Group": leak_grp,
        "Leak_Status": leak_flag,
        "Model": m_key,
        "Model_ID": cfg["id"],
        "Language": lang,
        "Method": method,
        "Ground_Truth": gt,
        "Predicted": pred,
        "Correct": correct,
        "Status": status,
        "Finish_Reason": r["finish"],
        "Faithfulness": faith,
        "Confidence": conf,
        "SC_Agree_Pct": agree,
        "SC_Runs": sc_runs,
        "Prompt_Tokens": tok_in,
        "Comp_Tokens": tok_out,
        "Thought_Tokens": r["tok_think"],
        "Total_Tokens": tot_tok,
        "Latency_s": round(r["lat"], 3),
        "Cost_1k_USD": cost,
        "Key_ID": r["key"],
        "Raw_Output": raw,
    }


def run_single(row, m_key, cfg, method, lang, base_url, api_key, breaker, temp=0.0):
    max_t = MAX_TOK_DIRECT if method in ["1_Zero_Shot", "2_Few_Shot"] else MAX_TOK_COT
    r = ask_vllm(base_url, api_key, cfg["id"], make_prompt(row, method, lang), temp, breaker, max_tok=max_t)
    pred, conf, status = parse_output(r["text"], row["Dạng câu hỏi"], r["finish"])
    return make_record(row, m_key, cfg, method, lang, pred, conf, None, r, status)


def eval_group(row, m_key, cfg, lang, base_url, api_key, dp_table, ckpt_path, breaker):
    if breaker.is_open():
        return row["ID"], lang

    q_id = row["ID"]
    q_type = row["Dạng câu hỏi"]
    leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()

    # 1. Giai 4 bai toan co so (chi chay cac method chua co trong bang DP)
    for m in ["1_Zero_Shot", "2_Few_Shot", "3_Zero_Shot_CoT", "4_Few_Shot_CoT"]:
        if breaker.is_open():
            return q_id, lang
        tk = f"{q_id}_{m_key}_{lang}_{m}"
        if dp_table.get(tk) is None:
            rec = run_single(row, m_key, cfg, m, lang, base_url, api_key, breaker)
            dp_table.put(rec, ckpt_path)

    # 2. Quy hoach dong + Cat tia nhanh can cho 5_Self_Consistency
    tk_sc = f"{q_id}_{m_key}_{lang}_5_Self_Consistency"
    if not breaker.is_open() and dp_table.get(tk_sc) is None:
        prompt_sc = make_prompt(row, "5_Self_Consistency", lang)
        ans_list, conf_list, logs = [], [], []
        agg = {"text": "", "tok_in": 0, "tok_out": 0, "tok_think": 0, "lat": 0.0, "key": 1, "finish": "STOP"}
        n_trunc = 0

        z_cot = dp_table.get(f"{q_id}_{m_key}_{lang}_3_Zero_Shot_CoT")
        if SC_REUSE_ZCOT and z_cot and z_cot.get("Status") == "OK":
            ans_list.append(str(z_cot["Predicted"]))
            if pd.notna(z_cot.get("Confidence")):
                conf_list.append(float(z_cot["Confidence"]))
            logs.append(f"[Run 1 (DP Reused Z-CoT) | Ans:{z_cot['Predicted']}]\n{z_cot['Raw_Output']}")
            agg["tok_in"] += int(z_cot["Prompt_Tokens"])
            agg["tok_out"] += int(z_cot["Comp_Tokens"])
            agg["tok_think"] += int(z_cot.get("Thought_Tokens", 0) or 0)
            agg["lat"] += float(z_cot["Latency_s"])

        api_failed = False
        while len(logs) < SC_RUNS:
            if breaker.is_open():
                api_failed = True
                break
            r = ask_vllm(base_url, api_key, cfg["id"], prompt_sc, 0.7, breaker, max_tok=MAX_TOK_COT)
            pa, cf, st = parse_output(r["text"], q_type, r["finish"])
            if st == "API_ERR":
                api_failed = True
                logs.append(f"[Run {len(logs)+1} | Ollama | API_ERR]\n{r['text']}")
                break
            if st == "OK":
                ans_list.append(pa)
                if cf is not None:
                    conf_list.append(cf)
            elif st == "TRUNCATED":
                n_trunc += 1
            logs.append(f"[Run {len(logs)+1} | Ollama | {st} | Ans:{pa}]\n{r['text']}")
            for k in ("tok_in", "tok_out", "tok_think"):
                agg[k] += r[k]
            agg["lat"] += r["lat"]

            if ans_list:
                top_cnt = Counter(ans_list).most_common(1)[0][1]
                if top_cnt >= 3:
                    break

        if not api_failed:
            runs_done = len([x for x in logs if "API_ERR" not in x.split("\n", 1)[0]])
            if ans_list:
                vote_counts = Counter(ans_list)
                pred_sc = vote_counts.most_common(1)[0][0]
                agree_sc = round(vote_counts[pred_sc] / len(ans_list) * 100, 1)
                status_sc = "OK"
            else:
                pred_sc, agree_sc = "", None
                status_sc = "TRUNCATED" if n_trunc else "NO_ANSWER"
            conf_sc = round(sum(conf_list) / len(conf_list), 1) if conf_list else None
            agg["text"] = "\n\n".join(logs)
            agg["finish"] = f"runs={runs_done},truncated={n_trunc}"
            rec_sc = make_record(row, m_key, cfg, "5_Self_Consistency", lang, pred_sc, conf_sc, agree_sc, agg, status_sc, runs_done)
            dp_table.put(rec_sc, ckpt_path)

    # 3. Quy hoach dong cho 6_Few_Shot_CoT_Leaked (O(1) lookup neu Unleaked)
    tk_lk = f"{q_id}_{m_key}_{lang}_6_Few_Shot_CoT_Leaked"
    if not breaker.is_open() and dp_table.get(tk_lk) is None:
        f_cot = dp_table.get(f"{q_id}_{m_key}_{lang}_4_Few_Shot_CoT")
        if leak_grp == "Khong_Lo_Unleaked" and f_cot and f_cot.get("Status") != "API_ERR":
            rec_lk = dict(f_cot)
            rec_lk["Task_Key"] = tk_lk
            rec_lk["Method"] = "6_Few_Shot_CoT_Leaked"
            dp_table.put(rec_lk, ckpt_path)
        elif not breaker.is_open():
            rec_lk = run_single(row, m_key, cfg, "6_Few_Shot_CoT_Leaked", lang, base_url, api_key, breaker)
            dp_table.put(rec_lk, ckpt_path)

    return q_id, lang


def load_and_normalize_extra_ckpt():
    """Doc checkpoint_extra_qwen.csv, quy doi toan bo Qwen_vLLM -> Qwen_7B de cuu 100% du lieu da chay."""
    if not os.path.exists(EXTRA_CKPT_CSV):
        return []
    df_ck = pd.read_csv(EXTRA_CKPT_CSV)
    if df_ck.empty:
        return []

    # Loc bo cac dong API_ERR
    df_ck = df_ck[df_ck["Predicted"].notna() & (df_ck["Predicted"] != "API_ERR") & (df_ck["Status"] != "API_ERR")].copy()
    if df_ck.empty:
        return []

    # Dong bo ten Model va Task_Key ve chung EXTRA_MODEL_KEY ("Qwen_7B")
    df_ck["Model"] = EXTRA_MODEL_KEY
    df_ck["Model_ID"] = EXTRA_MODEL_CFG["id"]
    df_ck["Task_Key"] = (
        df_ck["ID"].astype(str).str.strip()
        + "_"
        + EXTRA_MODEL_KEY
        + "_"
        + df_ck["Language"].astype(str).str.strip()
        + "_"
        + df_ck["Method"].astype(str).str.strip()
    )
    df_ck = df_ck.drop_duplicates(subset=["Task_Key"], keep="last").reset_index(drop=True)
    # Ghi lai ngay file checkpoint da chuan hoa
    df_ck.to_csv(EXTRA_CKPT_CSV, index=False, encoding="utf-8-sig")
    return df_ck.to_dict("records")


def load_existing_base_results():
    """Chi lay dung 3600 dong goc cua Small va Large, xoa file tam 4611 dong bi thieu."""
    bad_file = os.path.join(OUT_DIR, "benchmark_results_20261110_220526.xlsx")
    if os.path.exists(bad_file):
        try:
            os.remove(bad_file)
            print(f"🧹 Đã dọn file xuất dở dang: {os.path.basename(bad_file)}")
        except Exception:
            pass

    orig_xlsx = os.path.join(OUT_DIR, "benchmark_results_20261106_095454.xlsx")
    if os.path.exists(orig_xlsx):
        print(f"📂 Đang đọc 3600 dòng gốc (Small & Large) từ: {os.path.basename(orig_xlsx)}")
        df_base = pd.read_excel(orig_xlsx, sheet_name="1_Raw_Logs")
        return df_base[df_base["Model"].isin(["Small", "Large"])].drop_duplicates(subset=["Task_Key"], keep="last")

    if os.path.exists(BASE_CKPT_CSV):
        print(f"📂 Đang đọc dữ liệu gốc từ: {os.path.basename(BASE_CKPT_CSV)}")
        df_base = pd.read_csv(BASE_CKPT_CSV)
        return df_base[df_base["Model"].isin(["Small", "Large"])].drop_duplicates(subset=["Task_Key"], keep="last")

    return pd.DataFrame()


def main():
    # Buoc 1: Chuan hoa & nap ngay toan bo cac cau da giai tu ca 2 lan chay truoc
    initial_extra = [] if TEST_MODE else load_and_normalize_extra_ckpt()
    dp_table = DPMemoTable(initial_extra)
    ckpt_target = None if TEST_MODE else EXTRA_CKPT_CSV

    if DATA_PATH.endswith(".xlsx"):
        xl = pd.ExcelFile(DATA_PATH)
        sh = "data_test" if "data_test" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(DATA_PATH, sheet_name=sh)
    else:
        df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Đáp án"]).drop_duplicates(subset=["ID"], keep="last").reset_index(drop=True)
    df_run = df[df["ID"].isin(["EASY_001", "EASY_011", "MEDIUM_069", "MEDIUM_085", "HARD_144"])].copy() if TEST_MODE else df

    # Lan truyen DP truoc khi goi API: Tu dong dien Method 6 cho cac cau Unleaked da co Method 4
    propagated = 0
    for l in LANGS:
        for _, row in df_run.iterrows():
            q_id = row["ID"]
            leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()
            tk_4 = f"{q_id}_{EXTRA_MODEL_KEY}_{l}_4_Few_Shot_CoT"
            tk_6 = f"{q_id}_{EXTRA_MODEL_KEY}_{l}_6_Few_Shot_CoT_Leaked"
            f_cot = dp_table.get(tk_4)
            if dp_table.get(tk_6) is None and leak_grp == "Khong_Lo_Unleaked" and f_cot and f_cot.get("Status") != "API_ERR":
                rec_lk = dict(f_cot)
                rec_lk["Task_Key"] = tk_6
                rec_lk["Method"] = "6_Few_Shot_CoT_Leaked"
                dp_table.put(rec_lk, None)
                propagated += 1
    if ckpt_target and propagated > 0:
        dp_table.flush(ckpt_target)

    print(f"♻️ Đã gộp & cứu thành công {dp_table.size()}/1800 tasks hợp lệ từ các lần chạy trước!")

    groups = []
    for l in LANGS:
        for _, r in df_run.iterrows():
            if any(dp_table.get(f"{r['ID']}_{EXTRA_MODEL_KEY}_{l}_{m}") is None for m in METHODS):
                groups.append((r, l))

    missing_tasks = sum(
        1 for (r, l) in groups for m in METHODS if dp_table.get(f"{r['ID']}_{EXTRA_MODEL_KEY}_{l}_{m}") is None
    )
    print(f"🚀 Số cụm câu hỏi còn thiếu cần chạy tiếp: {len(groups)}/300 (Chỉ còn {missing_tasks}/1800 tasks!)")

    if groups:
        base_url = VLLM_BASE_URL
        if not base_url:
            base_url = input("🔗 Nhập link API từ Colab (VD: https://xxxx.trycloudflare.com/v1): ").strip()
        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
            base_url = base_url.rstrip("/") + "/v1"

        print(f"🔍 Đang kiểm tra kết nối tới {base_url}/models ...")
        try:
            r_test = HTTP_SESSION.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=15,
            )
            if r_test.status_code == 200:
                models_data = r_test.json().get("data", [])
                if models_data:
                    detected_id = models_data[0].get("id")
                    if detected_id:
                        EXTRA_MODEL_CFG["id"] = detected_id
                print(f"✅ Kết nối Server thành công! Model_ID: {EXTRA_MODEL_CFG['id']}")
            else:
                print(f"⚠️ Server trả về HTTP {r_test.status_code}. Hãy kiểm tra lại link Colab!")
                return
        except Exception as e:
            print(f"❌ Không kết nối được tới Server ({e}). Hãy lấy link Colab mới!")
            return

        breaker = CircuitBreaker(max_consecutive_failures=8)
        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [
                    ex.submit(eval_group, r, EXTRA_MODEL_KEY, EXTRA_MODEL_CFG, l, base_url, VLLM_API_KEY, dp_table, ckpt_target, breaker)
                    for (r, l) in groups
                ]
                pbar = tqdm(as_completed(futs), total=len(futs), desc=f"Resuming {EXTRA_MODEL_KEY}", ncols=95)
                for fut in pbar:
                    q_id, lang = fut.result()
                    pbar.set_postfix_str(f"{q_id}_{lang} | DP_Size: {dp_table.size()}/1800")
        except KeyboardInterrupt:
            print("\n[TẠM DỪNG] Nhấn Ctrl+C -> Đang lưu bảng trạng thái DP...")
        finally:
            if ckpt_target:
                dp_table.flush(ckpt_target)

        if breaker.is_open() and dp_table.size() < 1800:
            print(f"\n📌 Tiến độ hiện tại đã lưu an toàn: {dp_table.size()}/1800 tasks.")
            print("👉 Hãy khởi động lại ô code trên Colab lấy link mới rồi chạy lại lệnh này để hoàn tất nốt phần còn lại!")
            return

    # ==========================================================================
    # GOP DU LIEU CU (3600 DONG) + DU LIEU MODEL MOI (1800 DONG) = 5400 DONG
    # ==========================================================================
    df_extra = pd.DataFrame(dp_table.all_records()).drop_duplicates(subset=["Task_Key"], keep="last")
    df_base = load_existing_base_results()

    if not df_base.empty and not df_extra.empty:
        df_all = pd.concat([df_base, df_extra], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["Task_Key"], keep="last").reset_index(drop=True)
    elif not df_base.empty:
        df_all = df_base.reset_index(drop=True)
    else:
        df_all = df_extra.reset_index(drop=True)

    if "Faithfulness" not in df_all.columns:
        df_all["Faithfulness"] = df_all.apply(
            lambda r: check_faithfulness(
                r["Raw_Output"], r["Predicted"], r["Ground_Truth"], r["Correct"], r["Category"], r["Method"]
            ),
            axis=1,
        )

    df_valid = df_all[df_all["Predicted"] != "API_ERR"]

    df_sum = df_valid.groupby(["Model", "Model_ID", "Language", "Method", "Difficulty"]).agg(
        Count=("ID", "count"),
        Acc_Pct=("Correct", lambda x: round(x.mean() * 100, 2)),
        Avg_Tok=("Total_Tokens", lambda x: round(x.mean(), 1)),
        Avg_Lat_s=("Latency_s", lambda x: round(x.mean(), 2)),
        Cost_1k=("Cost_1k_USD", lambda x: round(x.mean(), 4)),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
    ).reset_index()

    df_leak = df_valid[df_valid["Method"].isin(["4_Few_Shot_CoT", "6_Few_Shot_CoT_Leaked"])].groupby(
        ["Model", "Language", "Leak_Status", "Method"]
    ).agg(
        Count=("ID", "count"),
        Acc_Pct=("Correct", lambda x: round(x.mean() * 100, 2)),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
        Avg_Tok=("Total_Tokens", lambda x: round(x.mean(), 1)),
    ).reset_index()

    df_err = df_valid[df_valid["Correct"] == 0].sort_values(by="Method", ascending=False).copy()
    for col in ["Reasoning_State", "Error_Type", "Note"]:
        if col not in df_err.columns:
            df_err[col] = ""

    df_cot = df_valid[df_valid["Faithfulness"] != "N/A"]
    df_faith = df_cot.groupby(["Model", "Language", "Method", "Faithfulness"]).agg(
        Count=("ID", "count"),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
    ).reset_index()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"test_extra_{ts}.xlsx" if TEST_MODE else f"benchmark_results_{ts}.xlsx"
    out = os.path.join(OUT_DIR, fname)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="1_Raw_Logs", index=False)
        df_sum.to_excel(w, sheet_name="2_Summary", index=False)
        df_leak.to_excel(w, sheet_name="3_Leak_Compare", index=False)
        df_err.to_excel(w, sheet_name="4_Errors", index=False)
        df_faith.to_excel(w, sheet_name="5_Faithfulness_Matrix", index=False)

    print(f"\n🎉 Đã gộp xong hoàn chỉnh và lưu tại: {out}")
    print(f"📊 Tổng số dòng hợp lệ: {len(df_valid)}/5400 (Gồm Small, Large và {EXTRA_MODEL_KEY})")


if __name__ == "__main__":
    main()