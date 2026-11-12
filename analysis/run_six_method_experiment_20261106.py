from __future__ import annotations

import ast
import argparse
import csv
import json
import math
import os
import operator
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from openpyxl import load_workbook


ENV_PATH: Path | None = None
ANNOTATED_PATH: Path | None = None
DATASET_PATH: Path | None = None
OUT_DIR: Path | None = None
STATE_PATH: Path | None = None

MODEL_PRICE = {
    "gemma-4-26b-a4b-it": {"alias": "Small", "in_price": 0.04, "out_price": 0.08},
    "gemini-3.5-flash-lite": {"alias": "Large", "in_price": 0.075, "out_price": 0.30},
}
MODEL_IDS = list(MODEL_PRICE)
TEMPERATURES = [0.2, 1.0]
MIN_KEY_GAP_S = 2.2
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

METHODS = [
    "Premise_Extraction",
    "Strict_CoT",
    "PAL",
    "Postgen_NLI_Guardrail",
    "Logprob_Temperature_Calibration",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_keys(env_path: Path | None) -> list[str]:
    keys: list[str] = []
    env_key_values = [
        value for name, value in os.environ.items()
        if name.upper() in {"GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"}
        or name.upper().startswith("GEMINI_API_KEY_")
    ]
    for raw_value in env_key_values:
        for value in raw_value.replace(";", ",").split(","):
            key = value.strip().strip("\"'")
            if key and key not in keys:
                keys.append(key)
    if env_path is not None:
        if not env_path.exists():
            raise FileNotFoundError(f"API-key file not found: {env_path}")
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" in line:
                name, value = line.split("=", 1)
                if not name.strip().upper().startswith(("GEMINI_API_KEY", "GEMINI_API_KEYS", "GOOGLE_API_KEY", "API_KEY")):
                    continue
                candidates = value.replace(";", ",").split(",")
            else:
                # Also accepts the legacy format: one raw API key per line.
                candidates = [line]
            for value in candidates:
                key = value.strip().strip("\"', ")
                if key and key not in keys:
                    keys.append(key)
    if not keys:
        raise ValueError("No API key found. Set GEMINI_API_KEY(S) or provide --env-file.")
    return keys


class GeminiREST:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.idx = 0
        self.next_free = [0.0] * len(keys)
        self.calls_by_key = [0] * len(keys)
        self.disabled_models: dict[str, str] = {}
        self.key_lock = threading.Lock()

    def _safe_message(self, msg: str) -> str:
        for key in self.keys:
            msg = msg.replace(key, "[REDACTED]")
        return msg[:500]

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        request_logprobs: bool = False,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        last_error = "unknown"
        attempts = min(max(len(self.keys), 1), 5)
        for attempt in range(attempts):
            with self.key_lock:
                if model_id in self.disabled_models:
                    return {
                        "raw_output": "", "status": "API_ERROR", "http_status": 0, "latency_s": 0.0,
                        "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "finish_reason": "",
                        "avg_logprobs": None, "has_logprobs": False, "key_slot": "",
                        "error": self.disabled_models[model_id],
                    }
                slot = self.idx
                self.idx = (self.idx + 1) % len(self.keys)
                now = time.monotonic()
                scheduled = max(now, self.next_free[slot])
                wait_for = scheduled - now
                self.next_free[slot] = scheduled + MIN_KEY_GAP_S
            if wait_for > 0:
                time.sleep(wait_for)
            start = time.perf_counter()
            cfg: dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            }
            if thinking_level:
                cfg["thinkingConfig"] = {"thinkingLevel": thinking_level}
            if request_logprobs:
                cfg["responseLogprobs"] = True
                cfg["logprobs"] = 5
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": cfg,
            }
            try:
                response = requests.post(
                    f"{API_BASE}/{model_id}:generateContent",
                    params={"key": self.keys[slot]},
                    json=body,
                    timeout=90,
                )
            except requests.RequestException as exc:
                self.next_free[slot] = time.monotonic() + MIN_KEY_GAP_S
                last_error = type(exc).__name__
                continue
            latency = time.perf_counter() - start
            self.next_free[slot] = time.monotonic() + MIN_KEY_GAP_S
            with self.key_lock:
                self.calls_by_key[slot] += 1
            if response.ok:
                data = response.json()
                candidate = (data.get("candidates") or [{}])[0]
                parts = candidate.get("content", {}).get("parts", []) or []
                text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
                usage = data.get("usageMetadata", {}) or {}
                return {
                    "raw_output": text,
                    "status": "OK",
                    "http_status": response.status_code,
                    "latency_s": round(latency, 3),
                    "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                    "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                    "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
                    "finish_reason": candidate.get("finishReason", ""),
                    "avg_logprobs": candidate.get("avgLogprobs"),
                    "has_logprobs": bool(candidate.get("logprobsResult")),
                    "key_slot": f"K{slot + 1}",
                    "error": "",
                }
            try:
                error = response.json().get("error", {}) or {}
                code = error.get("code", response.status_code)
                status = error.get("status", "API_ERROR")
                message = self._safe_message(str(error.get("message", "")))
            except Exception:
                code, status, message = response.status_code, "API_ERROR", "unparsed response"
            last_error = f"HTTP_{code}_{status}: {message}"
            if request_logprobs and response.status_code == 400 and "logprob" in message.lower():
                break
            if response.status_code in (400, 404):
                with self.key_lock:
                    self.disabled_models[model_id] = last_error
                break
            # Rotate on quota, auth, and transient errors. A permanent model/input error
            # will be repeated only against a different key to avoid printing secrets.
            if response.status_code not in (400, 401, 403, 404, 429, 500, 502, 503, 504):
                break
        return {
            "raw_output": "",
            "status": "API_ERROR",
            "http_status": 0,
            "latency_s": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "",
            "avg_logprobs": None,
            "has_logprobs": False,
            "key_slot": "",
            "error": last_error,
        }


def load_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if ANNOTATED_PATH is None or DATASET_PATH is None:
        raise RuntimeError("Input paths are not configured; call main() first.")
    if not ANNOTATED_PATH.is_file():
        raise FileNotFoundError(f"Reviewed workbook not found: {ANNOTATED_PATH}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset workbook not found: {DATASET_PATH}")
    reviewed = load_workbook(ANNOTATED_PATH, data_only=True, read_only=False)
    sample = reviewed["6_Selected_20"]
    errors = reviewed["4_Errors"]
    sample_headers = {sample.cell(4, col).value: col for col in range(1, sample.max_column + 1)}
    err_headers = {errors.cell(1, col).value: col for col in range(1, errors.max_column + 1)}
    err_by_key: dict[str, dict[str, Any]] = {}
    for r in range(2, errors.max_row + 1):
        key = errors.cell(r, err_headers["Task_Key"]).value
        if key:
            err_by_key[str(key)] = {h: errors.cell(r, c).value for h, c in err_headers.items()}

    dataset = load_workbook(DATASET_PATH, data_only=True, read_only=False)
    test_sheet = dataset["data_test"]
    train_sheet = dataset["data_train"]
    test_headers = {test_sheet.cell(1, c).value: c for c in range(1, test_sheet.max_column + 1) if test_sheet.cell(1, c).value}
    test_by_id: dict[str, dict[str, Any]] = {}
    for r in range(2, test_sheet.max_row + 1):
        qid = test_sheet.cell(r, test_headers["ID"]).value
        if qid:
            test_by_id[str(qid)] = {
                "question_type": str(test_sheet.cell(r, test_headers["Dạng câu hỏi"]).value or ""),
                "answer": str(test_sheet.cell(r, test_headers["Đáp án"]).value or ""),
            }

    cases: list[dict[str, Any]] = []
    for r in range(5, 25):
        rec = {h: sample.cell(r, c).value for h, c in sample_headers.items()}
        task_key = str(rec.get("Task_Key") or "")
        err = err_by_key.get(task_key)
        if not err:
            raise ValueError(f"Selected Task_Key missing from 4_Errors: {task_key}")
        qid = str(rec.get("ID") or "")
        ds_case = test_by_id.get(qid)
        if not ds_case:
            raise ValueError(f"Selected ID missing from dataset_150.xlsx: {qid}")
        baseline_confidence = rec.get("Confidence (0–100)")
        try:
            baseline_confidence = float(baseline_confidence) if baseline_confidence not in (None, "") else None
        except (TypeError, ValueError):
            baseline_confidence = None
        cases.append({
            "row_no": int(rec.get("No.") or (r - 4)),
            "task_key": task_key,
            "id": qid,
            "difficulty": str(rec.get("Difficulty") or ""),
            "language": str(rec.get("Language") or ""),
            "model_id": str(rec.get("Model_ID") or ""),
            "model": str(rec.get("Model") or ""),
            "original_method": str(rec.get("Method") or ""),
            "question": str(rec.get("Question") or ""),
            "ground_truth": str(rec.get("Ground_Truth") or ""),
            "baseline_predicted": str(rec.get("Predicted") or ""),
            "baseline_confidence": baseline_confidence,
            "faithfulness": str(rec.get("Faithfulness") or ""),
            "reasoning_state": str(rec.get("Reasoning_State") or ""),
            "error_type": str(rec.get("Error_Type") or ""),
            "note": str(rec.get("Note") or ""),
            "category": str(err.get("Category") or ""),
            "baseline_raw": str(rec.get("Raw_Output") or ""),
            "question_type": ds_case["question_type"],
        })

    train_headers = {train_sheet.cell(1, c).value: c for c in range(1, train_sheet.max_column + 1) if train_sheet.cell(1, c).value}
    train: list[dict[str, Any]] = []
    for r in range(2, train_sheet.max_row + 1):
        qid = train_sheet.cell(r, train_headers["ID"]).value
        if not qid:
            continue
        train.append({
            "id": str(qid),
            "question_type": str(train_sheet.cell(r, train_headers["Dạng câu hỏi"]).value or ""),
            "q_vi": str(train_sheet.cell(r, train_headers["Câu hỏi Tiếng Việt"]).value or ""),
            "q_en": str(train_sheet.cell(r, train_headers["Câu hỏi Tiếng Anh"]).value or ""),
            "answer": str(train_sheet.cell(r, train_headers["Đáp án"]).value or ""),
        })
    if len(cases) != 20:
        raise ValueError(f"Expected 20 selected cases, found {len(cases)}")
    return cases, train

def norm_numeric(value: str) -> str | None:
    s = str(value).strip().replace(",", "")
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", s)
    if not m:
        return None
    try:
        n = float(m.group(0))
    except ValueError:
        return None
    if math.isfinite(n) and abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    return format(n, ".12g")


def is_numeric_question(case_or_type: dict[str, Any] | str) -> bool:
    value = case_or_type.get("question_type", "") if isinstance(case_or_type, dict) else case_or_type
    value = str(value).lower()
    return "nguyên" in value or "integer" in value or value.strip().lower() in {"number", "numeric"}


# --- Parse đáp án: đồng bộ với benchmark/run_benchmark.parse_output ---
# Lỗi cũ: thiếu tag thì lấy luôn DÒNG CUỐI. Model nhỏ (Qwen) hay ghi đáp án trần rồi mới tới
# "[Confidence]: 100%" -> dòng cuối là dòng độ tự tin -> câu số bị lấy nhầm "100" làm đáp án.
# Ngoài ra câu số lấy số ĐẦU dòng (benchmark lấy số CUỐI) -> hai nơi chấm khác nhau.
ANS_TAG_RE = re.compile(
    r"(?:Đáp\s+án\s+cuối\s+cùng|Final\s+Answer)\s*\]?\s*\**\s*[:：]\s*\**\s*([^\r\n]*)", re.IGNORECASE)
BARE_ANS_RE = re.compile(r"^\W*(?:(?:đáp án|answer)\s*[:：]?\s*)?(-?[\d.,]+|[A-D])\W*$", re.IGNORECASE)
CONF_LINE_RE = re.compile(
    r"^\W*(?:\[?\s*(?:độ\s*tự\s*tin|confidence)\s*\]?\s*\**\s*[:：]?)?\s*\**\s*(\d{1,3}(?:[.,]\d+)?)\s*%\W*$",
    re.IGNORECASE)
CONF_WORD_RE = re.compile(r"độ\s*tự\s*tin|confidence", re.IGNORECASE)


def _strip_trailing_conf(lines: list[str]) -> tuple[list[str], float | None]:
    """Bỏ 1 dòng độ tự tin ở CUỐI nếu dòng đó có tag, hoặc ngay trước nó là dòng chỉ chứa
    đáp án (vd "B" rồi "80%"). Trả (các dòng còn lại, confidence hoặc None)."""
    if not lines:
        return lines, None
    m = CONF_LINE_RE.match(lines[-1])
    if m and (CONF_WORD_RE.search(lines[-1]) or (len(lines) > 1 and BARE_ANS_RE.match(lines[-2]))):
        return lines[:-1], float(m.group(1).replace(",", "."))
    return lines, None


def extract_final_line(text: str) -> str:
    """Phần chứa đáp án cuối: nội dung sau tag [Final Answer]/[Đáp án cuối cùng] cuối cùng; nếu
    không có (hoặc tag trống) thì dòng cuối CHỈ chứa đáp án, sau khi bỏ 1 dòng độ tự tin ở cuối.
    Trả "" nếu không tìm thấy — không đoán từ câu văn để tránh bắt nhầm số trung gian."""
    tags = ANS_TAG_RE.findall(text or "")
    if tags and tags[-1].strip(" *"):
        return tags[-1].strip(" *")
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    body, _ = _strip_trailing_conf(lines)
    if body and BARE_ANS_RE.match(body[-1]):
        return body[-1]
    return ""


def normalize_answer(text: str, category: str) -> str | None:
    final = extract_final_line(text)
    if not final:
        return None
    if is_numeric_question(category):
        s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", final)  # bỏ dấu phân cách hàng nghìn: 1,200
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        return norm_numeric(nums[-1]) if nums else None   # số CUỐI, giống benchmark
    m = re.search(r"\b([A-D])\b", final.upper())
    return m.group(1) if m else None


def normalize_ground_truth(value: str, category: str) -> str | None:
    if is_numeric_question(category):
        return norm_numeric(value)
    m = re.search(r"\b([A-D])\b", str(value).upper())
    return m.group(1) if m else str(value).strip().upper() or None


def parse_confidence(text: str) -> float:
    matches = re.findall(r"(?:\[\s*(?:Confidence|Độ\s*tự\s*tin)\s*\]|(?:Confidence|Độ\s*tự\s*tin))\s*[:：]\s*(\d+(?:[.,]\d+)?)\s*%?", text or "", re.IGNORECASE)
    if matches:
        try:
            return float(matches[-1].replace(",", "."))
        except ValueError:
            pass
    # Không có tag: nhận dòng độ tự tin trần ở cuối (vd "B" rồi "80%").
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    _, conf = _strip_trailing_conf(lines)
    if conf is not None:
        return min(100.0, max(0.0, conf))
    return 50.0


def logprob_confidence(avg_logprobs: Any) -> float | None:
    if avg_logprobs is None:
        return None
    try:
        value = float(avg_logprobs)
        if not math.isfinite(value):
            return None
        # Geometric-mean token probability proxy; calibrate it on train data before reporting.
        return round(100.0 * math.exp(max(-30.0, min(0.0, value))), 2)
    except (TypeError, ValueError, OverflowError):
        return None


def is_correct(predicted: str | None, ground_truth: str, category: str) -> int:
    p = normalize_answer(predicted or "", category) if predicted is not None else None
    g = normalize_ground_truth(ground_truth, category)
    return int(p is not None and g is not None and p == g)


def answer_tail(lang: str) -> str:
    if lang == "VI":
        return "BẮT BUỘC kết thúc bằng đúng 2 dòng:\n[Đáp án cuối cùng]: <một số nguyên hoặc A/B/C/D>\n[Độ tự tin]: <0-100>%"
    return "MUST end with exactly these 2 lines:\n[Final Answer]: <one integer or A/B/C/D>\n[Confidence]: <0-100>%"


def make_prompt(case: dict[str, Any], method: str) -> str:
    q = case["question"]
    lang = case["language"]
    tail = answer_tail(lang)
    if method == "Premise_Extraction":
        if lang == "VI":
            return f"Trích xuất 2-5 tiền đề nêu rõ trong đề; phân biệt dữ kiện liên quan và không liên quan; ghi điều đề hỏi. Không tự thêm dữ kiện. Suy luận ngắn, kiểm tra đơn vị/phép tính, rồi chốt.\n\nĐề bài:\n{q}\n\n{tail}"
        return f"Extract 2-5 explicit premises from the question; separate relevant from irrelevant facts; state what is being asked. Do not add facts. Give a concise derivation and check units/arithmetic before concluding.\n\nQuestion:\n{q}\n\n{tail}"
    if method == "Strict_CoT":
        if lang == "VI":
            return f"Giải theo quy trình nghiêm ngặt, tối đa 4 bước ngắn: (1) nêu dữ kiện và điều cần tìm; (2) lập quan hệ/phép tính; (3) tính hoặc đối chiếu từng lựa chọn; (4) kiểm tra điều kiện, đơn vị và đáp án. Mỗi bước phải dựa trên đề; không nhảy bước, không suy diễn ngoài đề.\n\nĐề bài:\n{q}\n\n{tail}"
        return f"Solve with a strict process of at most 4 concise steps: (1) list givens and target; (2) set up the relation/calculation; (3) compute or compare each choice; (4) check conditions, units, and answer. Every step must follow from the question; do not skip steps or infer beyond it.\n\nQuestion:\n{q}\n\n{tail}"
    if method == "PAL":
        if lang == "VI":
            return f"Chuyển bài toán số nguyên thành đúng một biểu thức Python số học an toàn để tính câu trả lời. Chỉ dùng hằng số và +, -, *, /, //, %, **; viết code dạng print(<biểu thức>), không import, không biến, không gọi hàm khác. Trả về duy nhất JSON: {{\"code\":\"print(...)\",\"confidence\":<0-100>}}. Đề bài: {q}"
        return f"Translate this integer problem into exactly one safe Python arithmetic expression. Use only constants and +, -, *, /, //, %, **; write code as print(<expression>), with no imports, variables, or other function calls. Return only JSON: {{\"code\":\"print(...)\",\"confidence\":<0-100>}}. Question: {q}"
    if method == "Logprob_Temperature_Calibration":
        if lang == "VI":
            return f"Trả lời trực tiếp, không giải thích. Chỉ dùng thông tin trong đề và kiểm tra điều kiện cuối cùng. Tự báo confidence trung thực, không mặc định 100%.\nĐề bài: {q}\n\n{tail}"
        return f"Answer directly without explanation. Use only the question and check the final condition. Report confidence honestly; do not default to 100%.\nQuestion: {q}\n\n{tail}"
    return ""


def make_nli_prompt(case: dict[str, Any]) -> str:
    if case["language"] == "VI":
        intro = "Bạn là bộ kiểm tra suy diễn sau sinh (post-generation NLI guardrail). Đánh giá xem câu trả lời dựa đúng vào đề hay không; không được xem đáp án chuẩn và không dùng kiến thức ngoài đề. Nếu câu trả lời thiếu bước, mâu thuẫn, suy diễn vượt tiền đề hoặc chốt sai so với chính lập luận, hãy từ chối bản nháp và viết câu trả lời đã sửa. Nếu hợp lệ, giữ nguyên. Nêu thật ngắn lý do kiểm tra."
        qlabel, clabel = "Đề bài", "Bản nháp cần kiểm tra"
    else:
        intro = "You are a post-generation NLI guardrail. Check whether the response is supported by the question; do not see a reference answer and do not use outside facts. If the response skips a necessary step, contradicts itself, overreaches the premises, or its conclusion conflicts with its reasoning, reject and repair it. Otherwise keep it. Give a very brief check rationale."
        qlabel, clabel = "Question", "Candidate response to verify"
    candidate = case["baseline_raw"]
    if len(candidate) > 12000:
        candidate = candidate[:10000] + "\n...[truncated candidate for checker input]...\n" + candidate[-2000:]
    return f"{intro}\n\n{qlabel}:\n{case['question']}\n\n{clabel}:\n{candidate}\n\n{answer_tail(case['language'])}"


def make_retry_prompt(case: dict[str, Any], method: str) -> str:
    """Short-format retry prompts for outputs that were truncated or unparsable."""
    q = case["question"]
    lang = case["language"]
    tail = answer_tail(lang)
    if method == "PAL":
        if lang == "VI":
            return f"Chỉ giải bài toán bằng một biểu thức số học Python. Dùng hằng số và + - * / // % **; chỉ viết JSON một dòng đúng dạng {{\"code\":\"print(biểu_thức)\",\"confidence\":70}}. Không markdown, không giải thích, không biến/import/hàm khác. Câu hỏi: {q}"
        return f"Solve using one Python arithmetic expression only. Use constants and + - * / // % **. Return exactly one JSON line: {{\"code\":\"print(expression)\",\"confidence\":70}}. No markdown or explanation; no variables/imports/other functions. Question: {q}"
    if method == "Postgen_NLI_Guardrail":
        candidate = case["baseline_raw"]
        if len(candidate) > 8000:
            candidate = candidate[:6500] + "\n...[candidate excerpt]...\n" + candidate[-1200:]
        if lang == "VI":
            return f"Guardrail: chỉ dùng đề và bản nháp, không xem đáp án chuẩn/kiến thức ngoài. Nếu nháp mâu thuẫn hoặc không được đề hỗ trợ, sửa ngắn; nếu ổn thì giữ. Tối đa 2 câu ngắn, rồi đúng 2 dòng kết thúc theo mẫu.\nĐề: {q}\nNháp: {candidate}\n{tail}"
        return f"Guardrail: use only the question and draft, never a reference answer or outside facts. Briefly repair unsupported/contradictory reasoning; otherwise keep it. Max 2 short sentences, then end with exactly the required final lines.\nQuestion: {q}\nDraft: {candidate}\n{tail}"
    if method == "Premise_Extraction":
        if lang == "VI":
            return f"Nêu tối đa 3 dữ kiện liên quan đúng như đề, một dòng phép tính/suy luận, rồi kết luận. Không thêm giả định; tối đa 5 dòng.\nĐề: {q}\n{tail}"
        return f"List at most 3 relevant premises stated in the question, one calculation/inference line, then conclude. Add no assumptions; use at most 5 lines.\nQuestion: {q}\n{tail}"
    if method == "Strict_CoT":
        if lang == "VI":
            return f"Giải đúng 3 bước ngắn: dữ kiện/mục tiêu; phép tính hoặc đối chiếu; kiểm tra và kết luận. Mỗi bước một dòng, không thêm giả định.\nĐề: {q}\n{tail}"
        return f"Solve in exactly 3 short steps: givens/target; calculation or option check; verification and answer. One line per step; add no assumptions.\nQuestion: {q}\n{tail}"
    if lang == "VI":
        return f"Trả lời trực tiếp, không giải thích. Kiểm tra điều kiện cuối; confidence trung thực.\nĐề: {q}\n{tail}"
    return f"Answer directly without explanation. Check the final condition; report honest confidence.\nQuestion: {q}\n{tail}"


def make_retry3_prompt(case: dict[str, Any], method: str) -> str:
    q = case["question"]
    lang = case["language"]
    tail = answer_tail(lang)
    if method == "Premise_Extraction":
        if lang == "VI":
            return f"Chỉ xuất 3 dòng ngắn: Tiền đề liên quan: ... / Quan hệ: ... / Kết quả: ... . Không thêm bình luận.\nĐề: {q}\n{tail}"
        return f"Output only 3 short lines: Relevant premises: ... / Relation: ... / Result: ... . No extra commentary.\nQuestion: {q}\n{tail}"
    if lang == "VI":
        return f"Giải đúng 3 dòng: (1) dữ kiện và mục tiêu; (2) phép tính/đối chiếu; (3) kiểm tra và kết luận. Không thêm giải thích khác.\nĐề: {q}\n{tail}"
    return f"Use exactly 3 short lines: (1) givens and target; (2) calculation/option check; (3) verification and answer. No other explanation.\nQuestion: {q}\n{tail}"


def safe_parse_code(raw: str) -> tuple[str | None, float, str]:
    conf = parse_confidence(raw)
    obj: dict[str, Any] | None = None
    try:
        obj = json.loads(raw.strip())
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw or "")
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        code = obj.get("code", "")
        try:
            conf = max(0.0, min(100.0, float(obj.get("confidence", conf))))
        except (TypeError, ValueError):
            pass
        return (str(code) if code else None), conf, ""
    m = re.search(r"print\s*\([^\n]+\)", raw or "")
    return (m.group(0) if m else None), conf, ""


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_numeric_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
        return ALLOWED_UNARY[type(node.op)](eval_numeric_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        left = eval_numeric_node(node.left)
        right = eval_numeric_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 10 or abs(left) > 1e8):
            raise ValueError("exponent out of safe range")
        result = ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(result, complex) or not math.isfinite(float(result)):
            raise ValueError("non-finite result")
        return result
    raise ValueError("disallowed syntax")


def execute_safe_print(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("expected one print expression")
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "print":
        raise ValueError("expected print(expression)")
    if len(call.args) != 1 or call.keywords:
        raise ValueError("print must have one positional expression")
    value = eval_numeric_node(call.args[0])
    if isinstance(value, int) or abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return format(float(value), ".12g")


def compute_cost(model_id: str, prompt_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICE.get(model_id)
    if not p:
        return 0.0
    # Configured rates use USD per 1M tokens, matching the supplied model-price convention.
    return round(prompt_tokens / 1_000_000 * p["in_price"] + output_tokens / 1_000_000 * p["out_price"], 10)


def raw_record(
    result: dict[str, Any],
    *,
    phase: str,
    case_id: str,
    difficulty: str,
    category: str,
    model_id: str,
    lang: str,
    method: str,
    ground_truth: str,
    predicted: str | None,
    confidence: float,
    temperature: float,
    status: str | None = None,
    raw_output: str | None = None,
    calibrated_confidence: float | None = None,
) -> dict[str, Any]:
    correct = is_correct(predicted, ground_truth, category) if predicted is not None else 0
    return {
        "Run time": now_iso(),
        "Phase": phase,
        "ID": case_id,
        "Difficulty": difficulty,
        "Category": category,
        "Model": MODEL_PRICE.get(model_id, {}).get("alias", model_id),
        "Model ID": model_id,
        "Lang": lang,
        "Method": method,
        "Ground truth": ground_truth,
        "Predicted": predicted or "",
        "Correct": correct,
        "Confidence": round(float(confidence or 0), 2),
        "Calibrated confidence": "" if calibrated_confidence is None else round(float(calibrated_confidence), 2),
        "Prompt tokens": result.get("prompt_tokens", 0),
        "Output tokens": result.get("output_tokens", 0),
        "Total tokens": result.get("total_tokens", 0),
        "Latency (s)": result.get("latency_s", 0),
        "API calls": 1 if result.get("http_status") == 200 else 0,
        "Key slot": result.get("key_slot", ""),
        "Status": status or result.get("status", ""),
        "HTTP status": result.get("http_status", 0),
        "Finish reason": result.get("finish_reason", ""),
        "Temperature": temperature,
        "Avg logprob": result.get("avg_logprobs"),
        "Has logprobs": result.get("has_logprobs", False),
        "Output cap": 1024,
        "Cost USD": compute_cost(model_id, result.get("prompt_tokens", 0), result.get("output_tokens", 0)),
        "Error": result.get("error", ""),
        "Raw output": raw_output if raw_output is not None else result.get("raw_output", ""),
    }


def persist_state(state: dict[str, Any]) -> None:
    if STATE_PATH is None:
        raise RuntimeError("State path is not configured; call main() first.")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    temp.write_text(payload, encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(STATE_PATH)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    # Last-resort write if another Windows process briefly holds the destination handle.
    STATE_PATH.write_text(payload, encoding="utf-8")


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def call_and_store(
    client: GeminiREST,
    state: dict[str, Any],
    *,
    model_id: str,
    prompt: str,
    phase: str,
    case_id: str,
    difficulty: str,
    category: str,
    lang: str,
    method: str,
    ground_truth: str,
    temperature: float,
    max_output_tokens: int = 1024,
    request_logprobs: bool = False,
) -> dict[str, Any]:
    expected_basis = {"avg_logprobs geometric-mean proxy", "self-report fallback"} if request_logprobs else {"self-report"}
    cached = next((x for x in reversed(state.get("raw_calls", []))
                   if x.get("Phase") == phase and x.get("ID") == case_id
                   and x.get("Model ID") == model_id and x.get("Method") == method
                   and round(float(x.get("Temperature", -999) or -999), 4) == round(float(temperature), 4)
                   and x.get("Confidence basis") in expected_basis), None)
    if cached is not None:
        http_status = int(cached.get("HTTP status", 0) or 0)
        result = {
            "raw_output": cached.get("Raw output", "") or "",
            "status": "OK" if http_status == 200 else "API_ERROR",
            "http_status": http_status,
            "latency_s": float(cached.get("Latency (s)", 0) or 0),
            "prompt_tokens": int(cached.get("Prompt tokens", 0) or 0),
            "output_tokens": int(cached.get("Output tokens", 0) or 0),
            "total_tokens": int(cached.get("Total tokens", 0) or 0),
            "finish_reason": cached.get("Finish reason", "") or "",
            "avg_logprobs": cached.get("Avg logprob"),
            "has_logprobs": bool(cached.get("Has logprobs")),
            "key_slot": cached.get("Key slot", "") or "",
            "error": cached.get("Error", "") or "",
            "logprob_request_error": cached.get("Logprob request error", "") or "",
        }
        self_conf = float(cached.get("Self-reported confidence") if cached.get("Self-reported confidence") not in (None, "") else 50.0)
        lp_val = cached.get("Logprob confidence proxy")
        lp_conf = float(lp_val) if lp_val not in (None, "") else None
        conf = float(cached.get("Confidence") if cached.get("Confidence") not in (None, "") else 50.0)
        pred = cached.get("Predicted") or None
        correct = int(cached.get("Correct", 0) or 0)
        return {"result": result, "predicted": pred, "confidence": conf, "self_confidence": self_conf, "logprob_confidence": lp_conf, "correct": correct, "raw_record": cached}
    result = client.generate(model_id, prompt, temperature, max_output_tokens, request_logprobs=request_logprobs)
    logprob_request_error = ""
    if request_logprobs and result.get("status") != "OK":
        logprob_request_error = str(result.get("error", "Logprobs request failed"))
        result = client.generate(model_id, prompt, temperature, max_output_tokens, request_logprobs=False)
        result["logprob_request_error"] = logprob_request_error
    pred = normalize_answer(result.get("raw_output", ""), category) if result.get("raw_output") else None
    self_conf = parse_confidence(result.get("raw_output", "")) if result.get("raw_output") else 50.0
    lp_conf = logprob_confidence(result.get("avg_logprobs")) if result.get("has_logprobs") else None
    conf = lp_conf if request_logprobs and lp_conf is not None else self_conf
    row = raw_record(
        result,
        phase=phase,
        case_id=case_id,
        difficulty=difficulty,
        category=category,
        model_id=model_id,
        lang=lang,
        method=method,
        ground_truth=ground_truth,
        predicted=pred,
        confidence=conf,
        temperature=temperature,
    )
    row["Output cap"] = max_output_tokens
    row["Self-reported confidence"] = round(float(self_conf), 2)
    row["Logprob confidence proxy"] = "" if lp_conf is None else lp_conf
    row["Confidence basis"] = "avg_logprobs geometric-mean proxy" if request_logprobs and lp_conf is not None else ("self-report fallback" if request_logprobs else "self-report")
    row["Logprob request error"] = logprob_request_error
    if result.get("status") != "OK":
        row["Status"] = result.get("status", "API_ERROR")
    state["raw_calls"].append(row)
    persist_state(state)
    return {"result": result, "predicted": pred, "confidence": conf, "self_confidence": self_conf, "logprob_confidence": lp_conf, "correct": row["Correct"], "raw_record": row}


def calibrate_temperature_and_confidence(
    client: GeminiREST,
    state: dict[str, Any],
    train: list[dict[str, Any]],
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    if len(train) < 30:
        raise ValueError(f"Expected 30 training examples, found {len(train)}")
    dev_rows = train[:15]
    cal_rows = train[15:30]
    results: dict[str, Any] = dict(state.get("calibration", {}))
    for model_id in (model_ids or MODEL_IDS):
        temp_eval: dict[str, list[dict[str, Any]]] = {str(t): [] for t in TEMPERATURES}
        for t in TEMPERATURES:
            for i, tr in enumerate(dev_rows):
                lang = "VI" if i % 2 == 0 else "EN"
                q = tr["q_vi"] if lang == "VI" else tr["q_en"]
                cat = tr["question_type"]
                prompt = make_calibration_prompt(q, lang, cat)
                run = call_and_store(
                    client, state, model_id=model_id, prompt=prompt,
                    phase="TEMP_SELECTION", case_id=tr["id"], difficulty="Train",
                    category=cat, lang=lang, method="Logprob_Temperature_Calibration",
                    ground_truth=tr["answer"], temperature=t, max_output_tokens=512, request_logprobs=True,
                )
                temp_eval[str(t)].append({
                    "id": tr["id"], "correct": run["correct"], "confidence": run["confidence"],
                    "predicted": run["predicted"], "status": run["result"]["status"],
                    "has_logprobs": bool(run["result"].get("has_logprobs")),
                })
                if len(state["raw_calls"]) % 10 == 0:
                    print(f"PROGRESS api_calls={len(state['raw_calls'])}", flush=True)
        scores = {}
        for t, vals in temp_eval.items():
            usable = [x for x in vals if x["status"] == "OK"]
            accuracy = sum(x["correct"] for x in usable) / len(usable) if usable else 0.0
            ece = expected_calibration_error(usable)
            scores[t] = {"n": len(usable), "accuracy": accuracy, "ece": ece}
        # Select by held-out training-split accuracy; prefer the provider default 1.0 on ties.
        selected = sorted(TEMPERATURES, key=lambda t: (scores[str(t)]["accuracy"], t == 1.0), reverse=True)[0]

        cal_data: list[dict[str, Any]] = []
        for i, tr in enumerate(cal_rows):
            idx = i + 15
            lang = "VI" if idx % 2 == 0 else "EN"
            q = tr["q_vi"] if lang == "VI" else tr["q_en"]
            cat = tr["question_type"]
            run = call_and_store(
                client, state, model_id=model_id,
                prompt=make_calibration_prompt(q, lang, cat),
                phase="CONFIDENCE_CALIBRATION", case_id=tr["id"], difficulty="Train",
                category=cat, lang=lang, method="Logprob_Temperature_Calibration",
                ground_truth=tr["answer"], temperature=selected, max_output_tokens=512, request_logprobs=True,
            )
            cal_data.append({"confidence": run["confidence"], "correct": run["correct"], "status": run["result"]["status"], "has_logprobs": bool(run["result"].get("has_logprobs"))})
            if len(state["raw_calls"]) % 10 == 0:
                print(f"PROGRESS api_calls={len(state['raw_calls'])}", flush=True)

        bins = make_reliability_bins(cal_data)
        all_calibration_rows = [x for vals in temp_eval.values() for x in vals] + cal_data
        logprob_success_n = sum(1 for x in all_calibration_rows if x.get("has_logprobs"))
        logprob_total_n = len(all_calibration_rows)
        logprobs_available = logprob_total_n > 0 and logprob_success_n == logprob_total_n
        results[model_id] = {
            "selected_temperature": selected,
            "temperature_scores": scores,
            "confidence_bins": bins,
            "calibration_n": sum(1 for x in cal_data if x["status"] == "OK"),
            "calibration_accuracy": (
                sum(x["correct"] for x in cal_data if x["status"] == "OK")
                / max(1, sum(1 for x in cal_data if x["status"] == "OK"))
            ),
            "logprobs_available": logprobs_available,
            "logprob_success_n": logprob_success_n,
            "logprob_total_n": logprob_total_n,
            "confidence_basis": "avg_logprobs geometric-mean proxy calibrated on train" if logprobs_available else "self-reported confidence fallback; logprobs unavailable or incomplete",
        }
    state["calibration"] = results
    persist_state(state)
    return results


def make_calibration_prompt(question: str, lang: str, category: str) -> str:
    if lang == "VI":
        return f"Trả lời trực tiếp, không giải thích. Nếu thiếu dữ kiện thì chọn phương án phù hợp theo đề. Tự nêu confidence trung thực.\nCâu hỏi: {question}\n\n[Đáp án cuối cùng]: <số nguyên hoặc A/B/C/D>\n[Độ tự tin]: <0-100>%"
    return f"Answer directly without explanation. If the question is underspecified, choose based only on the prompt. Report confidence honestly.\nQuestion: {question}\n\n[Final Answer]: <integer or A/B/C/D>\n[Confidence]: <0-100>%"


def expected_calibration_error(values: list[dict[str, Any]]) -> float:
    if not values:
        return 1.0
    buckets = [(0, 49.999), (50, 69.999), (70, 84.999), (85, 100.001)]
    n = len(values)
    ece = 0.0
    for lo, hi in buckets:
        bucket = [x for x in values if lo <= float(x["confidence"]) < hi]
        if bucket:
            conf = sum(float(x["confidence"]) for x in bucket) / len(bucket) / 100
            acc = sum(int(x["correct"]) for x in bucket) / len(bucket)
            ece += len(bucket) / n * abs(conf - acc)
    return ece


def make_reliability_bins(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges = [(0, 59.999, "0-59"), (60, 84.999, "60-84"), (85, 100.001, "85-100")]
    usable = [x for x in values if x["status"] == "OK"]
    global_p = (sum(int(x["correct"]) for x in usable) + 1) / (len(usable) + 2) if usable else 0.5
    out = []
    for lo, hi, label in ranges:
        group = [x for x in usable if lo <= float(x["confidence"]) < hi]
        p = (sum(int(x["correct"]) for x in group) + 1) / (len(group) + 2) if group else global_p
        out.append({"bin": label, "n": len(group), "correct": sum(int(x["correct"]) for x in group), "calibrated_probability": p})
    return out


def calibrated_confidence(raw: float, bins: list[dict[str, Any]]) -> float:
    if raw < 60:
        label = "0-59"
    elif raw < 85:
        label = "60-84"
    else:
        label = "85-100"
    chosen = next((x for x in bins if x["bin"] == label), None)
    return 50.0 if chosen is None else round(100 * float(chosen["calibrated_probability"]), 2)


def save_case_result(state: dict[str, Any], case: dict[str, Any], method: str, row: dict[str, Any]) -> None:
    key = f"{case['task_key']}|{method}"
    state["case_results"][key] = row
    persist_state(state)


def method_case_record(
    case: dict[str, Any],
    method: str,
    predicted: str | None,
    confidence: float,
    calibrated: float | None,
    status: str,
    note: str,
    raw_output: str,
    api_result: dict[str, Any] | None,
    temperature: float,
) -> dict[str, Any]:
    correct = is_correct(predicted, case["ground_truth"], case["category"]) if predicted is not None else 0
    return {
        "task_key": case["task_key"],
        "id": case["id"],
        "method": method,
        "model": case["model"],
        "model_id": case["model_id"],
        "language": case["language"],
        "question_type": case["question_type"],
        "error_type": case["error_type"],
        "ground_truth": case["ground_truth"],
        "before_predicted": case["baseline_predicted"],
        "before_correct": 0,
        "before_confidence": case["baseline_confidence"],
        "after_predicted": predicted or "",
        "after_correct": correct,
        "delta_correct": correct,
        "after_confidence": round(confidence, 2),
        "calibrated_confidence": "" if calibrated is None else round(calibrated, 2),
        "status": status,
        "note": note,
        "raw_output": raw_output,
        "temperature": temperature,
        "prompt_tokens": (api_result or {}).get("prompt_tokens", 0),
        "output_tokens": (api_result or {}).get("output_tokens", 0),
        "latency_s": (api_result or {}).get("latency_s", 0),
        "api_calls": 1 if (api_result or {}).get("http_status") == 200 else 0,
        "cost_usd": compute_cost(case["model_id"], (api_result or {}).get("prompt_tokens", 0), (api_result or {}).get("output_tokens", 0)),
    }


def run_one_case_method(
    client: GeminiREST,
    calibration: dict[str, Any],
    case: dict[str, Any],
    method: str,
    max_output_tokens: int = 1024,
    phase: str = "CASE_AFTER",
) -> tuple[dict[str, Any], str, dict[str, Any] | None, dict[str, Any]]:
    if method == "PAL" and not is_numeric_question(case):
        row = method_case_record(
            case, method, None, 0, None, "SKIPPED_UNSUPPORTED_TYPE",
            "PAL chỉ áp dụng cho câu số nguyên; bỏ qua câu trắc nghiệm.", "", None, 0.0,
        )
        return case, method, None, row
    if phase == "CASE_AFTER_RETRY3":
        prompt = make_retry3_prompt(case, method)
        temperature = 0.2
    elif phase == "CASE_AFTER_RETRY2":
        prompt = make_retry_prompt(case, method) if method != "Postgen_NLI_Guardrail" else make_retry_prompt(case, method)
        temperature = 1.0 if method == "Postgen_NLI_Guardrail" else (float(calibration.get(case["model_id"], {}).get("selected_temperature", 1.0)) if method == "Logprob_Temperature_Calibration" else 1.0)
    elif method == "Postgen_NLI_Guardrail":
        prompt = make_nli_prompt(case)
        temperature = 1.0
    else:
        prompt = make_prompt(case, method)
        temperature = float(calibration.get(case["model_id"], {}).get("selected_temperature", 1.0)) if method == "Logprob_Temperature_Calibration" else 1.0
    request_logs = method == "Logprob_Temperature_Calibration" and bool(calibration.get(case["model_id"], {}).get("logprobs_available"))
    thinking_level = "minimal" if phase == "CASE_AFTER_RETRY3" and case["model_id"].startswith("gemma-") else None
    result = client.generate(case["model_id"], prompt, temperature, max_output_tokens, request_logprobs=request_logs, thinking_level=thinking_level)
    logprob_request_error = ""
    if request_logs and result.get("status") != "OK":
        logprob_request_error = str(result.get("error", "Logprobs request failed"))
        result = client.generate(case["model_id"], prompt, temperature, max_output_tokens, request_logprobs=False)
        result["logprob_request_error"] = logprob_request_error
    text = result.get("raw_output", "")
    status = result.get("status", "API_ERROR")
    predicted = normalize_answer(text, case["category"]) if text else None
    self_confidence = parse_confidence(text) if text else 50.0
    lp_confidence = logprob_confidence(result.get("avg_logprobs")) if result.get("has_logprobs") else None
    confidence = lp_confidence if request_logs and lp_confidence is not None else self_confidence
    cal_conf: float | None = None
    note = ""
    if method == "PAL" and status == "OK":
        code, confidence, _ = safe_parse_code(text)
        if not code:
            status = "PAL_PARSE_FAIL"
            predicted = None
            note = "Không trích xuất được print(expression) để chạy AST an toàn."
        else:
            try:
                predicted = execute_safe_print(code)
                status = "OK"
                note = "Code số học chạy qua AST allowlist; không thực thi Python tùy ý."
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
                status = "PAL_REJECTED_CODE"
                predicted = None
                note = f"AST từ chối biểu thức: {type(exc).__name__}."
    elif method == "Logprob_Temperature_Calibration":
        bins = calibration.get(case["model_id"], {}).get("confidence_bins", [])
        cal_conf = calibrated_confidence(confidence, bins)
        if request_logs and lp_confidence is not None:
            note = "Temperature chọn trên train; hiệu chỉnh proxy exp(avgLogprobs) bằng 15 câu train tách biệt; held-out 20 câu không dùng để fit."
            status = "OK_LOGPROB" if status == "OK" else status
        else:
            note = "Logprobs unavailable/incomplete; dùng temperature sweep và self-reported confidence calibration trên train như fallback, không gọi là logprob calibration."
            if status == "OK":
                status = "PARTIAL_LOGPROB_UNAVAILABLE"
    elif method == "Postgen_NLI_Guardrail":
        note = "Bộ kiểm tra chỉ thấy đề và Raw_Output gốc; không thấy ground truth."
    elif method == "Premise_Extraction":
        note = "Yêu cầu liệt kê tiền đề liên quan/không liên quan trước khi suy luận."
    elif method == "Strict_CoT":
        note = "Giới hạn 4 bước, yêu cầu kiểm tra điều kiện và phép tính; rationale ngắn hiển thị."
    if status in {"OK", "OK_LOGPROB", "PARTIAL_LOGPROB_UNAVAILABLE"} and result.get("finish_reason") == "MAX_TOKENS":
        status = "TRUNCATED"
    if result.get("status") != "OK" and not note:
        note = result.get("error", "API call failed")
    raw = raw_record(
        result, phase=phase, case_id=case["id"], difficulty=case["difficulty"],
        category=case["category"], model_id=case["model_id"], lang=case["language"],
        method=method, ground_truth=case["ground_truth"], predicted=predicted,
        confidence=confidence, temperature=temperature, status=status,
        raw_output=text, calibrated_confidence=cal_conf,
    )
    raw["Task_Key"] = case["task_key"]
    raw["Output cap"] = max_output_tokens
    raw["Self-reported confidence"] = round(float(self_confidence), 2)
    raw["Logprob confidence proxy"] = "" if lp_confidence is None else lp_confidence
    raw["Confidence basis"] = "avg_logprobs geometric-mean proxy" if request_logs and lp_confidence is not None else ("self-report fallback" if method == "Logprob_Temperature_Calibration" else "self-report")
    raw["Logprob request error"] = logprob_request_error or result.get("logprob_request_error", "")
    raw["Thinking level"] = thinking_level or "model default"
    row = method_case_record(case, method, predicted, confidence, cal_conf, status, note, text, result, temperature)
    return case, method, raw, row


def run_case_methods(client: GeminiREST, state: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    calibration = state.get("calibration", {})
    methods_to_run = ["Premise_Extraction", "Strict_CoT", "PAL", "Postgen_NLI_Guardrail", "Logprob_Temperature_Calibration"]
    pending = [
        (case, method)
        for case in cases
        for method in methods_to_run
        if f"{case['task_key']}|{method}" not in state["case_results"]
    ]
    total = len(pending)
    if not pending:
        return
    done = 0
    workers = min(5, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one_case_method, client, calibration, case, method) for case, method in pending]
        for future in as_completed(futures):
            case, method, raw, row = future.result()
            if raw is not None:
                state["raw_calls"].append(raw)
            save_case_result(state, case, method, row)
            done += 1
            if done % 5 == 0 or done == total:
                print(f"PROGRESS case_methods={done}/{total} api_calls={len(state['raw_calls'])}", flush=True)


def retry_incomplete_methods(client: GeminiREST, state: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    calibration = state.get("calibration", {})
    already_retried = {
        f"{x.get('Task_Key')}|{x.get('Method')}"
        for x in state.get("raw_calls", []) if x.get("Phase") == "CASE_AFTER_RETRY"
    }
    pending=[]
    for case in cases:
        for method in ["Premise_Extraction", "Strict_CoT", "PAL", "Postgen_NLI_Guardrail", "Logprob_Temperature_Calibration"]:
            key=f"{case['task_key']}|{method}"
            previous=state.get("case_results",{}).get(key)
            if previous and previous.get("status") in {"TRUNCATED","API_ERROR"} and key not in already_retried:
                pending.append((case,method,previous))
    if not pending:
        return
    print(f"RETRY_INCOMPLETE pending={len(pending)} output_cap=2048",flush=True)
    done=0
    with ThreadPoolExecutor(max_workers=min(5,len(pending))) as pool:
        futures={pool.submit(run_one_case_method,client,calibration,case,method,2048,"CASE_AFTER_RETRY"):(case,method,previous) for case,method,previous in pending}
        for future in as_completed(futures):
            case,method,previous=futures[future]
            c,m,raw,row=future.result()
            if raw is not None:
                state["raw_calls"].append(raw)
            key=f"{case['task_key']}|{method}"
            if row.get("status")=="API_ERROR":
                final=dict(previous)
                final["note"]=(str(final.get("note", ""))+f" Retry at 2048 output tokens also failed ({row.get('status')}); original attempt retained. {row.get('note','')}").strip()
                save_case_result(state,case,method,final)
            else:
                row["note"]=(f"Retry at 2048 output tokens after {previous.get('status')}; first attempt is retained in Raw_After. "+str(row.get("note",""))).strip()
                save_case_result(state,case,method,row)
            done+=1
            if done%5==0 or done==len(pending):
                print(f"PROGRESS retry={done}/{len(pending)} api_calls={len(state['raw_calls'])}",flush=True)


def retry_unresolved_methods(client: GeminiREST, state: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    calibration = state.get("calibration", {})
    already_retried = {
        f"{x.get('Task_Key')}|{x.get('Method')}"
        for x in state.get("raw_calls", []) if x.get("Phase") == "CASE_AFTER_RETRY2"
    }
    pending = []
    for case in cases:
        for method in ["Premise_Extraction", "Strict_CoT", "PAL", "Postgen_NLI_Guardrail", "Logprob_Temperature_Calibration"]:
            key = f"{case['task_key']}|{method}"
            previous = state.get("case_results", {}).get(key)
            retryable = {"TRUNCATED", "API_ERROR"} | ({"PAL_PARSE_FAIL"} if method == "PAL" else set())
            if previous and previous.get("status") in retryable and key not in already_retried:
                pending.append((case, method, previous))
    if not pending:
        return
    print(f"RETRY_UNRESOLVED pending={len(pending)} output_cap=4096 concise_format=on", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=min(5, len(pending))) as pool:
        futures = {
            pool.submit(run_one_case_method, client, calibration, case, method, 4096, "CASE_AFTER_RETRY2"): (case, method, previous)
            for case, method, previous in pending
        }
        for future in as_completed(futures):
            case, method, previous = futures[future]
            _, _, raw, row = future.result()
            if raw is not None:
                state["raw_calls"].append(raw)
            key = f"{case['task_key']}|{method}"
            if row.get("status") == "API_ERROR":
                final = dict(previous)
                final["note"] = (str(final.get("note", "")) + f" Concise-format retry at 4096 tokens failed; earlier outputs retained in Raw_After. {row.get('note', '')}").strip()
                save_case_result(state, case, method, final)
            else:
                row["note"] = (f"Retried unresolved {previous.get('status')} with concise format at 4096 tokens; earlier attempts remain in Raw_After. " + str(row.get("note", ""))).strip()
                save_case_result(state, case, method, row)
            done += 1
            if done % 5 == 0 or done == len(pending):
                print(f"PROGRESS retry2={done}/{len(pending)} api_calls={len(state['raw_calls'])}", flush=True)


def retry_final_truncations(client: GeminiREST, state: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    already_retried = {
        f"{x.get('Task_Key')}|{x.get('Method')}"
        for x in state.get("raw_calls", []) if x.get("Phase") == "CASE_AFTER_RETRY3"
    }
    pending = []
    for case in cases:
        for method in ["Premise_Extraction", "Strict_CoT"]:
            key = f"{case['task_key']}|{method}"
            previous = state.get("case_results", {}).get(key)
            if previous and previous.get("status") in {"TRUNCATED", "API_ERROR"} and key not in already_retried:
                pending.append((case, method, previous))
    if not pending:
        return
    print(f"RETRY_FINAL_TRUNCATIONS pending={len(pending)} output_cap=8192 thinking=minimal-for-Gemma", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=min(5, len(pending))) as pool:
        futures = {
            pool.submit(run_one_case_method, client, state.get("calibration", {}), case, method, 8192, "CASE_AFTER_RETRY3"): (case, method, previous)
            for case, method, previous in pending
        }
        for future in as_completed(futures):
            case, method, previous = futures[future]
            _, _, raw, row = future.result()
            if raw is not None:
                state["raw_calls"].append(raw)
            if row.get("status") == "API_ERROR":
                row = dict(previous)
                row["note"] = (str(row.get("note", "")) + " Final short-format retry failed; earlier attempts retained in Raw_After.").strip()
            else:
                row["note"] = ("Final retry used temperature 0.2, concise answer format and Gemma minimal thinking; earlier attempts retained in Raw_After. " + str(row.get("note", ""))).strip()
            save_case_result(state, case, method, row)
            done += 1
            if done == len(pending):
                print(f"PROGRESS retry3={done}/{len(pending)} api_calls={len(state['raw_calls'])}", flush=True)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five faithfulness interventions on the selected benchmark cases."
    )
    parser.add_argument("--reviewed-results", type=Path, required=True,
                        help="Reviewed XLSX containing 6_Selected_20 and 4_Errors sheets.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset XLSX containing data_test and data_train sheets.")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="Optional .env file; defaults to .env in the current directory if present.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"),
                        help="Directory for resumable state JSON (default: ./benchmark_output).")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Optional checkpoint path; defaults to <output-dir>/run_state.json.")
    parser.add_argument("--new-run", action="store_true",
                        help="Use a new timestamped checkpoint instead of resuming the default checkpoint.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global ENV_PATH, ANNOTATED_PATH, DATASET_PATH, OUT_DIR, STATE_PATH
    args = parse_args(argv)
    ANNOTATED_PATH = args.reviewed_results.expanduser().resolve()
    DATASET_PATH = args.dataset.expanduser().resolve()
    OUT_DIR = args.output_dir.expanduser().resolve()
    env_arg = args.env_file
    if env_arg is None:
        cwd_env = Path.cwd() / ".env"
        env_arg = cwd_env if cwd_env.is_file() else None
    ENV_PATH = env_arg.expanduser().resolve() if env_arg is not None else None
    if args.state_file is not None:
        STATE_PATH = args.state_file.expanduser().resolve()
    else:
        state_name = f"run_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" if args.new_run else "run_state.json"
        STATE_PATH = OUT_DIR / state_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases, train = load_cases()
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state["case_results"] = {k: v for k, v in state.get("case_results", {}).items() if "|RLAIF_DPO_Faithfulness" not in k}
        state["excluded_methods"] = {"RLAIF_DPO_Faithfulness": "Skipped at user request"}
        if len(state.get("cases", [])) != 20:
            raise ValueError("Existing checkpoint does not match selected 20 cases")
        expected_keys = [case["task_key"] for case in cases]
        state_keys = [case.get("task_key") for case in state.get("cases", [])]
        if state_keys != expected_keys:
            raise ValueError("Existing checkpoint Task_Key sequence differs from the current reviewed workbook")
    else:
        state = {"started_at": now_iso(), "cases": cases, "raw_calls": [], "case_results": {}, "calibration": {}}
        persist_state(state)
    keys = load_keys(ENV_PATH)
    client = GeminiREST(keys)
    print(f"START cases={len(cases)} train={len(train)} unique_keys={len(keys)} models={len(MODEL_IDS)}", flush=True)
    if not state.get("calibration"):
        calibrate_temperature_and_confidence(client, state, train)
    run_case_methods(client, state, cases)
    retry_incomplete_methods(client, state, cases)
    retry_unresolved_methods(client, state, cases)
    retry_final_truncations(client, state, cases)
    state["finished_at"] = now_iso()
    key_counts = Counter(str(x.get("Key slot")) for x in state.get("raw_calls", []) if int(x.get("API calls", 0) or 0) > 0 and x.get("Key slot"))
    state["api_calls_by_key_alias"] = {f"K{i+1}": key_counts.get(f"K{i+1}", 0) for i in range(len(keys))}
    state["dpo_status"] = "SKIPPED_AT_USER_REQUEST"
    persist_state(state)
    export_csv(OUT_DIR / "case_results.csv", [
        {"Task_Key": key.split("|", 1)[0], "Method": key.split("|", 1)[1], **value}
        for key, value in state["case_results"].items()
    ])
    export_csv(OUT_DIR / "raw_calls.csv", state.get("raw_calls", []))
    print(f"DONE api_calls={len(state['raw_calls'])} state={STATE_PATH}", flush=True)
    print(f"CSV_RESULTS={OUT_DIR / 'case_results.csv'}", flush=True)
    print(f"CSV_RAW={OUT_DIR / 'raw_calls.csv'}", flush=True)
    print("METHOD_COUNTS=" + json.dumps(dict(Counter(x["method"] for x in state["case_results"].values())), ensure_ascii=True), flush=True)
    print("KEY_CALL_COUNTS=" + json.dumps(state["api_calls_by_key_alias"], ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
