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
TEST_MODE = True  # Doi sang False khi chay full 150 cau

XLSX_PATH = os.path.join(BASE_DIR, "data", "dataset_150.xlsx")
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_150.csv")
DATA_PATH = XLSX_PATH if os.path.exists(XLSX_PATH) else CSV_PATH

OUT_DIR = os.path.join(BASE_DIR, "benchmark", "results")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_CSV = os.path.join(OUT_DIR, "checkpoint.csv")

# Chi dung dung 2 model nay: 1 Nho va 1 Lon
MODELS = {
    "Small": {"id": "gemini-3.5-flash-lite", "in_price": 0.075, "out_price": 0.30},
    "Large": {"id": "gemini-3.5-flash", "in_price": 0.15, "out_price": 0.60},
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
WORKERS = 5


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
    def __init__(self, keys, min_gap=4.2):
        self.clients = [genai.Client(api_key=k) for k in keys]
        self.n = len(self.clients)
        self.idx = 0
        self.min_gap = min_gap
        self.lock = threading.Lock()
        self.dead = [False] * self.n
        self.next_free = [0.0] * self.n
        self.toks = [0] * self.n
        self.reqs = [0] * self.n

    def acquire(self):
        with self.lock:
            now = time.time()
            alive = [i for i in range(self.n) if not self.dead[i]]
            if not alive:
                raise RuntimeError("Tat ca key deu da het token.")

            best_i = None
            min_wait = float("inf")
            for _ in range(self.n):
                i = self.idx
                self.idx = (self.idx + 1) % self.n
                if self.dead[i]:
                    continue
                wait = max(0.0, self.next_free[i] - now)
                if wait == 0.0:
                    best_i = i
                    min_wait = 0.0
                    break
                if wait < min_wait:
                    min_wait = wait
                    best_i = i

            self.next_free[best_i] = max(now, self.next_free[best_i]) + self.min_gap
            return best_i, self.clients[best_i], min_wait

    def add_usage(self, i, tok):
        with self.lock:
            self.toks[i] += tok
            self.reqs[i] += 1

    def on_error(self, i, err):
        with self.lock:
            msg = err.lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                if "per_day" in msg or "daily" in msg or "limit: 0" in msg:
                    self.dead[i] = True
                    tqdm.write(f"[Key #{i+1}] Het token trong ngay -> khoa key")
                else:
                    self.next_free[i] = max(time.time(), self.next_free[i]) + 20.0
                    tqdm.write(f"[Key #{i+1}] Cham rate limit ({self.toks[i]} tok) -> doi key")
            elif "invalid" in msg or "403" in msg or "401" in msg:
                self.dead[i] = True
                tqdm.write(f"[Key #{i+1}] Key loi -> bo qua")

    def summary(self):
        print("\nTrang thai API keys:")
        for i in range(self.n):
            st = "HET_TOKEN" if self.dead[i] else "OK"
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
    if not text:
        return "", 50.0

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
    p = str(pred).strip().upper().replace(",", "")
    g = str(gt).strip().upper().replace(",", "")
    if g.endswith(".0"):
        g = g[:-2]
    return 1 if (p and p == g) else 0


def ask_llm(pool, model_id, prompt, temp):
    t0 = time.time()
    for step in range(5):
        k_idx, client, wait_s = pool.acquire()
        if wait_s > 0:
            time.sleep(wait_s)
        try:
            res = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temp, max_output_tokens=1024),
            )
            lat = round(time.time() - t0, 3)
            txt = res.text or ""
            u = res.usage_metadata
            tok_in = u.prompt_token_count if u and u.prompt_token_count else 0
            tok_out = u.candidates_token_count if u and u.candidates_token_count else 0
            pool.add_usage(k_idx, tok_in + tok_out)
            return txt, tok_in, tok_out, lat, k_idx + 1
        except Exception as e:
            pool.on_error(k_idx, str(e))
            if step == 4:
                return f"ERR: {e}", 0, 0, round(time.time() - t0, 3), k_idx + 1
            time.sleep(1.5)


def eval_row(row, m_key, method, lang, pool):
    cfg = MODELS[m_key]
    q_id = row["ID"]
    lvl_map = {"Dễ": "Easy", "Trung bình": "Medium", "Khó": "Hard"}
    lvl = lvl_map.get(str(row["Mức độ"]).strip(), row["Mức độ"])
    q_type = row["Dạng câu hỏi"]
    leak_grp = str(row.get("Nhóm_Màu_Leak", "Khong_Lo_Unleaked")).strip()
    leak_flag = "Unleaked" if leak_grp == "Khong_Lo_Unleaked" else "Leaked"
    gt = str(row["Đáp án"]).strip()

    prompt = make_prompt(row, method, lang)

    if method != "5_Self_Consistency":
        raw, tok_in, tok_out, lat, k_num = ask_llm(pool, cfg["id"], prompt, 0.0)
        pred, conf = parse_output(raw, q_type)
        agree = 100.0
    else:
        ans_list, conf_list, logs = [], [], []
        tok_in, tok_out, lat, k_num = 0, 0, 0.0, 1
        for i in range(SC_RUNS):
            r_txt, ti, to, l_s, kn = ask_llm(pool, cfg["id"], prompt, 0.7)
            k_num = kn
            pa, cf = parse_output(r_txt, q_type)
            if pa:
                ans_list.append(pa)
            conf_list.append(cf)
            logs.append(f"[Run {i+1} | Key#{kn} | Ans:{pa}]\n{r_txt}")
            tok_in += ti
            tok_out += to
            lat += l_s

        if ans_list:
            pred, cnt = Counter(ans_list).most_common(1)[0]
            agree = round((cnt / SC_RUNS) * 100, 1)
        else:
            pred, agree = "", 0.0
        conf = round(sum(conf_list) / len(conf_list), 1) if conf_list else 50.0
        raw = "\n\n".join(logs)

    correct = is_match(pred, gt)
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


def main():
    keys = load_keys(ENV_PATH)
    if not keys:
        print(f"Khong tim thay key trong {ENV_PATH}")
        return

    pool = KeyPool(keys, min_gap=4.2)
    print(f"Loaded {pool.n} keys tu {ENV_PATH}")
    print(f"Models dang dung: Small='{MODELS['Small']['id']}' | Large='{MODELS['Large']['id']}'")

    if DATA_PATH.endswith(".xlsx"):
        xl = pd.ExcelFile(DATA_PATH)
        sh = "data_test" if "data_test" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(DATA_PATH, sheet_name=sh)
    else:
        df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Đáp án"]).drop_duplicates(subset=["ID"], keep="last").reset_index(drop=True)

    if TEST_MODE:
        ids = ["EASY_001", "EASY_026", "HARD_144"]
        df_run = df[df["ID"].isin(ids)].copy()
        print(f"Mode: TEST ({len(df_run)} cau x 2 models x 2 ngon ngu x 6 phuong phap)")
    else:
        df_run = df
        print(f"Mode: FULL ({len(df_run)} cau)")

    done, rows = set(), []
    if not TEST_MODE and os.path.exists(CKPT_CSV):
        df_ck = pd.read_csv(CKPT_CSV)
        done = set(df_ck["Task_Key"].tolist())
        rows = df_ck.to_dict("records")
        print(f"Resumed {len(done)} tasks tu {CKPT_CSV}")

    tasks = [
        (r, mk, m, l)
        for _, r in df_run.iterrows()
        for mk in MODELS
        for l in LANGS
        for m in METHODS
        if f"{r['ID']}_{mk}_{l}_{m}" not in done
    ]
    print(f"Tasks to run: {len(tasks)}")

    if tasks:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(eval_row, r, mk, m, l, pool) for (r, mk, m, l) in tasks]
            pbar = tqdm(as_completed(futs), total=len(futs), desc="Running", ncols=95)
            for fut in pbar:
                res = fut.result()
                rows.append(res)
                ki = res["Key_ID"] - 1
                pbar.set_postfix_str(f"Key#{res['Key_ID']}({pool.toks[ki]}t) | {res['ID']}")
                if not TEST_MODE and len(rows) % 15 == 0:
                    pd.DataFrame(rows).to_csv(CKPT_CSV, index=False, encoding="utf-8-sig")

        if not TEST_MODE:
            pd.DataFrame(rows).to_csv(CKPT_CSV, index=False, encoding="utf-8-sig")

    pool.summary()

    df_all = pd.DataFrame(rows)
    df_sum = df_all.groupby(["Model", "Model_ID", "Language", "Method", "Difficulty"]).agg(
        Count=("ID", "count"),
        Acc_Pct=("Correct", lambda x: round(x.mean() * 100, 2)),
        Avg_Tok=("Total_Tokens", lambda x: round(x.mean(), 1)),
        Avg_Lat_s=("Latency_s", lambda x: round(x.mean(), 2)),
        Cost_1k=("Cost_1k_USD", lambda x: round(x.mean(), 4)),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
    ).reset_index()

    df_leak = df_all[df_all["Method"].isin(["4_Few_Shot_CoT", "6_Few_Shot_CoT_Leaked"])].groupby(
        ["Model", "Language", "Leak_Status", "Method"]
    ).agg(
        Count=("ID", "count"),
        Acc_Pct=("Correct", lambda x: round(x.mean() * 100, 2)),
        Avg_Conf=("Confidence", lambda x: round(x.mean(), 1)),
        Avg_Tok=("Total_Tokens", lambda x: round(x.mean(), 1)),
    ).reset_index()

    df_err = df_all[(df_all["Correct"] == 0) & (df_all["Method"].str.contains("CoT"))].copy()
    for col in ["Reasoning_State", "Error_Type", "Faithfulness", "Note"]:
        df_err[col] = ""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"test_results_{ts}.xlsx" if TEST_MODE else f"benchmark_results_{ts}.xlsx"
    out = os.path.join(OUT_DIR, fname)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="1_Raw_Logs", index=False)
        df_sum.to_excel(w, sheet_name="2_Summary", index=False)
        df_leak.to_excel(w, sheet_name="3_Leak_Compare", index=False)
        df_err.to_excel(w, sheet_name="4_Errors", index=False)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()