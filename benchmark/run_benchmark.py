import os
import re
import time
import threading
from datetime import datetime
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
TEST_MODE = False  # False = chay full 150 cau | True = test 5 cau

XLSX_PATH = os.path.join(BASE_DIR, "data", "dataset_150.xlsx")
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_150.csv")
DATA_PATH = XLSX_PATH if os.path.exists(XLSX_PATH) else CSV_PATH

OUT_DIR = os.path.join(BASE_DIR, "benchmark", "results")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_CSV = os.path.join(OUT_DIR, "checkpoint.csv")

MODELS = {
    "Small": {"id": "gemma-4-26b-a4b-it", "in_price": 0.04, "out_price": 0.08},
    "Large": {"id": "gemini-3.5-flash-lite", "in_price": 0.075, "out_price": 0.30},
}

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


def load_keys(path=".env"):
    if not os.path.exists(path):
        return []
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" in s:
                s = s.split("=", 1)[1].strip()
            k = s.strip("\"', ")
            if k:
                keys.append(k)
    return keys


class KeyPool:
    def __init__(self, keys, min_gap=2.2):
        self.clients = [genai.Client(api_key=k) for k in keys]
        self.n = len(self.clients)
        self.idx = {m["id"]: 0 for m in MODELS.values()}
        self.min_gap = min_gap
        self.lock = threading.Lock()
        self.dead = {(i, m["id"]): False for i in range(self.n) for m in MODELS.values()}
        self.next_free = {(i, m["id"]): 0.0 for i in range(self.n) for m in MODELS.values()}
        self.toks = [0] * self.n
        self.reqs = [0] * self.n

    def acquire(self, model_id):
        with self.lock:
            now = time.time()
            alive = [i for i in range(self.n) if not self.dead[(i, model_id)]]
            if not alive:
                raise RuntimeError(f"Tat ca {self.n} key deu da het quota cho model {model_id}.")

            best_i = None
            min_wait = float("inf")
            for _ in range(self.n):
                i = self.idx[model_id]
                self.idx[model_id] = (i + 1) % self.n
                if self.dead[(i, model_id)]:
                    continue
                wait = max(0.0, self.next_free[(i, model_id)] - now)
                if wait == 0.0:
                    best_i = i
                    min_wait = 0.0
                    break
                if wait < min_wait:
                    min_wait = wait
                    best_i = i

            self.next_free[(best_i, model_id)] = max(now, self.next_free[(best_i, model_id)]) + self.min_gap
            return best_i, self.clients[best_i], min_wait

    def add_usage(self, i, tok):
        with self.lock:
            self.toks[i] += tok
            self.reqs[i] += 1

    def on_error(self, i, model_id, err):
        with self.lock:
            msg = err.lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                if "requests per day" in msg or "per_day" in msg or "daily" in msg or "limit: 0" in msg:
                    if not self.dead[(i, model_id)]:
                        self.dead[(i, model_id)] = True
                        tqdm.write(f"[Key #{i+1}] Het han muc ngay ({model_id}) -> khoa key")
                else:
                    self.next_free[(i, model_id)] = max(time.time(), self.next_free[(i, model_id)]) + 8.0
            elif "invalid" in msg or "403" in msg or "401" in msg:
                for m in MODELS.values():
                    self.dead[(i, m["id"])] = True
                tqdm.write(f"[Key #{i+1}] Key loi -> bo qua")

    def summary(self):
        print("\nTrang thai API keys:")
        for i in range(self.n):
            dead_models = [k for k, v in MODELS.items() if self.dead[(i, v["id"])]]
            st = f"HET_QUOTA({','.join(dead_models)})" if dead_models else "OK"
            print(f"  Key #{i+1}: {self.reqs[i]} reqs | {self.toks[i]} tokens | {st}")


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


def parse_output(text, q_type):
    if not text or text.startswith("ERR:"):
        return "API_ERR", 0.0

    conf = 50.0
    m_conf = re.search(r"\[?(?:Độ tự tin|Confidence)\]?\s*:\s*(\d+(?:\.\d+)?)", text, re.I)
    if m_conf:
        conf = min(100.0, max(0.0, float(m_conf.group(1))))

    m_ans = re.search(r"\[?(?:Đáp án cuối cùng|Final Answer)\]?\s*:\s*(.+)", text, re.I)
    if m_ans:
        line = m_ans.group(1).strip()
    else:
        lines = [x.strip() for x in text.strip().split("\n") if x.strip()]
        line = lines[-1] if lines else ""

    is_int = str(q_type).strip().lower() == "số nguyên"
    ans = ""
    if is_int:
        nums = re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", ""))
        if nums:
            v = float(nums[-1])
            ans = str(int(v)) if v.is_integer() else str(v)
    else:
        m = re.search(r"\b([A-D])\b", line.upper())
        if m:
            ans = m.group(1)

    return ans, conf


def is_match(pred, gt):
    if pred == "API_ERR" or not pred:
        return 0
    p = str(pred).strip().upper().replace(",", "")
    g = str(gt).strip().upper().replace(",", "")
    if g.endswith(".0"):
        g = g[:-2]
    return 1 if (p and p == g) else 0


def check_faithfulness(raw_text, pred, gt, correct, q_type, method):
    if method in ["1_Zero_Shot", "2_Few_Shot"] or pred == "API_ERR":
        return "N/A"

    parts = re.split(r"\[?(?:Đáp án cuối cùng|Final Answer)\]?\s*:", str(raw_text), flags=re.I)
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


def ask_llm(pool, model_id, prompt, temp, max_tok=512):
    t0 = time.time()
    k_idx = 0
    for step in range(pool.n + 2):
        try:
            k_idx, client, wait_s = pool.acquire(model_id)
            if wait_s > 0:
                time.sleep(wait_s)
            res = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temp, max_output_tokens=max_tok),
            )
            lat = round(time.time() - t0, 3)
            txt = res.text or ""
            u = res.usage_metadata
            tok_in = u.prompt_token_count if u and u.prompt_token_count else 0
            tok_out = u.candidates_token_count if u and u.candidates_token_count else 0
            pool.add_usage(k_idx, tok_in + tok_out)
            return txt, tok_in, tok_out, lat, k_idx + 1
        except RuntimeError as re_err:
            return f"ERR: {re_err}", 0, 0, round(time.time() - t0, 3), k_idx + 1
        except Exception as e:
            pool.on_error(k_idx, model_id, str(e))
            if step == pool.n + 1:
                return f"ERR: {e}", 0, 0, round(time.time() - t0, 3), k_idx + 1
            time.sleep(0.8)


def make_record(row, m_key, method, lang, pred, conf, agree, tok_in, tok_out, lat, k_num, raw):
    cfg = MODELS[m_key]
    q_id = row["ID"]
    lvl_map = {"Dễ": "Easy", "Trung bình": "Medium", "Khó": "Hard"}
    lvl = lvl_map.get(str(row["Mức độ"]).strip(), row["Mức độ"])
    q_type = row["Dạng câu hỏi"]
    leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()
    leak_flag = "Unleaked" if leak_grp == "Khong_Lo_Unleaked" else "Leaked"
    gt = str(row["Đáp án"]).strip()

    correct = is_match(pred, gt)
    faith = check_faithfulness(raw, pred, gt, correct, q_type, method)
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
        "Faithfulness": faith,
        "Confidence": conf,
        "SC_Agree_Pct": agree,
        "Prompt_Tokens": tok_in,
        "Comp_Tokens": tok_out,
        "Total_Tokens": tot_tok,
        "Latency_s": round(lat, 3),
        "Cost_1k_USD": cost,
        "Key_ID": k_num,
        "Raw_Output": raw,
    }


def eval_group(row, m_key, lang, pool, done_map):
    cfg = MODELS[m_key]
    q_id = row["ID"]
    q_type = row["Dạng câu hỏi"]
    leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()
    group_res = []
    cache = {}

    for m in ["1_Zero_Shot", "2_Few_Shot", "3_Zero_Shot_CoT", "4_Few_Shot_CoT"]:
        tk = f"{q_id}_{m_key}_{lang}_{m}"
        if tk in done_map:
            cache[m] = done_map[tk]
            continue
        max_t = 64 if m in ["1_Zero_Shot", "2_Few_Shot"] else 512
        prompt = make_prompt(row, m, lang)
        raw, ti, to, lat, kn = ask_llm(pool, cfg["id"], prompt, 0.0, max_tok=max_t)
        pred, conf = parse_output(raw, q_type)
        rec = make_record(row, m_key, m, lang, pred, conf, 100.0, ti, to, lat, kn, raw)
        cache[m] = rec
        group_res.append(rec)

    tk_sc = f"{q_id}_{m_key}_{lang}_5_Self_Consistency"
    if tk_sc not in done_map:
        prompt_sc = make_prompt(row, "5_Self_Consistency", lang)
        ans_list, conf_list, logs = [], [], []
        tok_in, tok_out, lat, k_num = 0, 0, 0.0, 1

        z_cot = cache.get("3_Zero_Shot_CoT")
        if z_cot and z_cot["Predicted"] != "API_ERR" and z_cot["Predicted"] != "":
            p0 = str(z_cot["Predicted"])
            c0 = float(z_cot["Confidence"])
            ans_list.append(p0)
            conf_list.append(c0)
            logs.append(f"[Run 1 (Reused Z-CoT) | Ans:{p0}]\n{z_cot['Raw_Output']}")
            tok_in += int(z_cot["Prompt_Tokens"])
            tok_out += int(z_cot["Comp_Tokens"])
            lat += float(z_cot["Latency_s"])
            k_num = int(z_cot["Key_ID"])

        while len(logs) < SC_RUNS:
            r_txt, ti, to, l_s, kn = ask_llm(pool, cfg["id"], prompt_sc, 0.7, max_tok=512)
            k_num = kn
            pa, cf = parse_output(r_txt, q_type)
            if pa == "API_ERR":
                logs.append(f"[Run {len(logs)+1} | Key#{kn} | Ans:API_ERR]\n{r_txt}")
                break
            if pa:
                ans_list.append(pa)
                conf_list.append(cf)
            logs.append(f"[Run {len(logs)+1} | Key#{kn} | Ans:{pa}]\n{r_txt}")
            tok_in += ti
            tok_out += to
            lat += l_s

            if ans_list:
                top_ans, top_cnt = Counter(ans_list).most_common(1)[0]
                if top_cnt >= 3:
                    runs_done = len(logs)
                    scale = SC_RUNS / runs_done
                    tok_in = int(round(tok_in * scale))
                    tok_out = int(round(tok_out * scale))
                    lat = round(lat * scale, 3)
                    break

        if ans_list:
            # Bau cu 3 tang: So phieu -> Tong do tu tin -> Uu tien Luong 1 (T=0)
            vote_counts = Counter(ans_list)
            conf_sums = {a: sum(cf for x, cf in zip(ans_list, conf_list) if x == a) for a in vote_counts}
            pred_sc = max(vote_counts.keys(), key=lambda a: (vote_counts[a], conf_sums[a]))
            cnt_sc = vote_counts[pred_sc]
            agree_sc = round((cnt_sc / len(ans_list)) * 100, 1)
        else:
            pred_sc, agree_sc = "API_ERR", 0.0
        conf_sc = round(sum(conf_list) / len(conf_list), 1) if conf_list else 0.0
        raw_sc = "\n\n".join(logs)
        rec_sc = make_record(row, m_key, "5_Self_Consistency", lang, pred_sc, conf_sc, agree_sc, tok_in, tok_out, lat, k_num, raw_sc)
        group_res.append(rec_sc)

    tk_lk = f"{q_id}_{m_key}_{lang}_6_Few_Shot_CoT_Leaked"
    if tk_lk not in done_map:
        f_cot = cache.get("4_Few_Shot_CoT")
        if leak_grp == "Khong_Lo_Unleaked" and f_cot and f_cot["Predicted"] != "API_ERR":
            rec_lk = dict(f_cot)
            rec_lk["Task_Key"] = tk_lk
            rec_lk["Method"] = "6_Few_Shot_CoT_Leaked"
            group_res.append(rec_lk)
        else:
            prompt_lk = make_prompt(row, "6_Few_Shot_CoT_Leaked", lang)
            raw_lk, ti, to, lat, kn = ask_llm(pool, cfg["id"], prompt_lk, 0.0, max_tok=512)
            pred_lk, conf_lk = parse_output(raw_lk, q_type)
            rec_lk = make_record(row, m_key, "6_Few_Shot_CoT_Leaked", lang, pred_lk, conf_lk, 100.0, ti, to, lat, kn, raw_lk)
            group_res.append(rec_lk)

    return group_res


def main():
    keys = load_keys(ENV_PATH)
    if not keys:
        print(f"Khong tim thay key trong {ENV_PATH}")
        return

    pool = KeyPool(keys, min_gap=2.2)
    workers = max(4, pool.n * 2)  # 7 keys -> 14 workers
    print(f"Loaded {pool.n} keys tu {ENV_PATH} (Workers: {workers})")
    print(f"Models dang dung: Small='{MODELS['Small']['id']}' | Large='{MODELS['Large']['id']}'")

    if DATA_PATH.endswith(".xlsx"):
        xl = pd.ExcelFile(DATA_PATH)
        sh = "data_test" if "data_test" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(DATA_PATH, sheet_name=sh)
    else:
        df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Đáp án"]).drop_duplicates(subset=["ID"], keep="last").reset_index(drop=True)

    if TEST_MODE:
        ids = ["EASY_001", "EASY_011", "MEDIUM_069", "MEDIUM_085", "HARD_144"]
        df_run = df[df["ID"].isin(ids)].copy()
        print(f"Mode: TEST ({len(df_run)} cau)")
    else:
        df_run = df
        print(f"Mode: FULL ({len(df_run)} cau)")

    done_map, rows = {}, []
    if not TEST_MODE and os.path.exists(CKPT_CSV):
        df_ck = pd.read_csv(CKPT_CSV)
        df_ck = df_ck[df_ck["Predicted"] != "API_ERR"].drop_duplicates(subset=["Task_Key"], keep="last")
        rows = df_ck.to_dict("records")
        done_map = {r["Task_Key"]: r for r in rows}
        print(f"Resumed {len(done_map)}/3600 tasks hop le tu {CKPT_CSV}")

    groups = []
    for _, r in df_run.iterrows():
        for mk in MODELS:
            for l in LANGS:
                if any(f"{r['ID']}_{mk}_{l}_{m}" not in done_map for m in METHODS):
                    groups.append((r, mk, l))

    missing_tasks = sum(
        1 for (r, mk, l) in groups for m in METHODS if f"{r['ID']}_{mk}_{l}_{m}" not in done_map
    )
    print(f"Question groups can chay tiep: {len(groups)} (Con thieu {missing_tasks} tasks)")

    if groups:
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(eval_group, r, mk, l, pool, done_map) for (r, mk, l) in groups]
                pbar = tqdm(as_completed(futs), total=len(futs), desc="Fast-Running", ncols=95)
                for idx_f, fut in enumerate(pbar, 1):
                    new_recs = fut.result()
                    valid_new = [x for x in new_recs if x["Predicted"] != "API_ERR"]
                    rows.extend(valid_new)
                    if new_recs:
                        last = new_recs[-1]
                        ki = last["Key_ID"] - 1
                        pbar.set_postfix_str(f"Key#{last['Key_ID']}({pool.toks[ki]}t) | {last['ID']} ({len(rows)}/3600)")
                    if not TEST_MODE and idx_f % 5 == 0 and rows:
                        pd.DataFrame(rows).drop_duplicates(subset=["Task_Key"], keep="last").to_csv(
                            CKPT_CSV, index=False, encoding="utf-8-sig"
                        )
        except KeyboardInterrupt:
            print("\n[TAM DUNG] Nguoi dung nhan Ctrl+C -> Luu checkpoint va xuat Excel...")
        finally:
            if not TEST_MODE and rows:
                pd.DataFrame(rows).drop_duplicates(subset=["Task_Key"], keep="last").to_csv(
                    CKPT_CSV, index=False, encoding="utf-8-sig"
                )

    pool.summary()

    df_all = pd.DataFrame(rows).drop_duplicates(subset=["Task_Key"], keep="last").reset_index(drop=True)
    if "Faithfulness" not in df_all.columns:
        df_all["Faithfulness"] = df_all.apply(
            lambda r: check_faithfulness(r["Raw_Output"], r["Predicted"], r["Ground_Truth"], r["Correct"], r["Category"], r["Method"]),
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
        df_err[col] = ""

    df_cot = df_valid[df_valid["Faithfulness"] != "N/A"]
    df_faith = df_cot.groupby(["Model", "Language", "Method", "Faithfulness"]).agg(
        Count=("ID", "count"),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
    ).reset_index()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"test_results_{ts}.xlsx" if TEST_MODE else f"benchmark_results_{ts}.xlsx"
    out = os.path.join(OUT_DIR, fname)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="1_Raw_Logs", index=False)
        df_sum.to_excel(w, sheet_name="2_Summary", index=False)
        df_leak.to_excel(w, sheet_name="3_Leak_Compare", index=False)
        df_err.to_excel(w, sheet_name="4_Errors", index=False)
        df_faith.to_excel(w, sheet_name="5_Faithfulness_Matrix", index=False)

    print(f"Saved: {out} (Total valid: {len(df_valid)}/3600 rows | Errors: {len(df_err)} rows)")


if __name__ == "__main__":
    main()