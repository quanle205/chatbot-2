"""Sinh câu trả lời có trích dẫn (Prompt Engineering cho BCTC) + công cụ tính toán an toàn + kiểm tra hallucination."""
import ast, bisect, json, operator, os, re

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai (OpenAI-compatible, vd. Gemini)
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5" if PROVIDER == "anthropic" else "gpt-4o-mini")
COMPANY = os.getenv("COMPANY", "doanh nghiệp")

SYSTEM = """Bạn là chuyên gia phân tích tài chính doanh nghiệp, trả lời bằng tiếng Việt cho nhà đầu tư, cổ đông và chuyên viên phân tích.
Bạn chỉ được dùng các đoạn NGỮ CẢNH (trích từ báo cáo tài chính, thuyết minh, bản tin nhà đầu tư của {company}) làm bằng chứng.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên NGỮ CẢNH. Không dùng kiến thức bên ngoài hay trí nhớ để bịa/ước lượng số liệu.
2. Sau MỖI số liệu và MỖI khẳng định phải ghi nguồn dạng [#n] (n là số thứ tự đoạn ngữ cảnh). Chỉ dùng n có trong ngữ cảnh.
3. Nếu ngữ cảnh không đủ, nói rõ: "Tài liệu được cung cấp không có thông tin này." (hoặc nêu phần còn thiếu). Tuyệt đối không đoán.
4. Xác định đúng chỉ tiêu, KỲ/NGÀY (vd. 31/03/2026 so với 31/12/2025; Quý I/2026 so với cùng kỳ), PHẠM VI (báo cáo riêng/công ty mẹ hay hợp nhất) và CỘT số liệu; nêu rõ chúng trong câu trả lời. Nếu ngữ cảnh có nhiều kỳ/phạm vi lẫn nhau, đừng nhầm cột.
5. Giữ nguyên đơn vị (VND, triệu đồng, tỷ đồng...) như trong tài liệu. Có thể quy đổi sang tỷ đồng để dễ đọc nhưng phải tính bằng công cụ.
6. KHÔNG tự tính nhẩm. Mọi phép tính (tăng trưởng %, biên lợi nhuận, tỷ số, tổng/hiệu) phải gọi công cụ `calculate`
   với số liệu lấy từ ngữ cảnh (viết số thuần, không dấu phân cách, ví dụ 1234567.8), rồi trình bày công thức + kết quả.
7. Với câu hỏi "nguyên nhân/lý do/do đâu": chỉ nêu nguyên nhân khi ngữ cảnh (thuyết minh, nhận định của ban lãnh đạo...) có nói rõ.
   Nếu chỉ có số liệu mà không có giải thích, hãy nêu số liệu rồi ghi: "Tài liệu không nêu rõ nguyên nhân." — không suy diễn.

ĐỊNH DẠNG: ngắn gọn, rõ ràng.
- Dòng đầu: kết luận trực tiếp (số liệu chính + kỳ + phạm vi) kèm [#n].
- Tiếp theo (nếu cần): công thức/phép tính, bảng markdown khi so sánh nhiều chỉ tiêu, và phần "Nguyên nhân/ghi chú" nếu được hỏi."""

BASELINE_SYSTEM = "Bạn là trợ lý tài chính. Trả lời ngắn gọn bằng tiếng Việt."  # LLM đơn thuần, không ngữ cảnh (đối chứng hallucination)

TOOL_DESC = "Tính biểu thức số học (+ - * / ** và ngoặc). Dùng cho mọi phép tính số liệu tài chính."


def safe_calc(expr: str) -> str:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}

    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10: raise ValueError("số mũ quá lớn")
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.operand))
        raise ValueError("biểu thức không hợp lệ")
    try:
        return repr(round(ev(ast.parse(expr.strip(), mode="eval")), 6))
    except Exception as e:
        return f"Lỗi: {e}"


def format_context(hits):
    return "\n\n".join(f"[#{i}] (file: {c.doc}, trang {c.page}, {'BẢNG' if c.kind == 'table' else 'văn bản'})\n{c.text}"
                       for i, (c, _) in enumerate(hits, 1))


def answer(question, hits, history=None, max_steps=5):
    """Trả về (text, calc_log). history: list {role, content} các lượt trước (chỉ text)."""
    user = f"NGỮ CẢNH:\n{format_context(hits)}\n\nCÂU HỎI: {question}"
    msgs = list(history or []) + [{"role": "user", "content": user}]
    system = SYSTEM.format(company=COMPANY)
    return (_anthropic if PROVIDER == "anthropic" else _openai)(msgs, max_steps, system)


def answer_baseline(question):
    """LLM đơn thuần (không truy xuất, không công cụ) — dùng để so sánh mức độ hallucination với RAG."""
    return complete(BASELINE_SYSTEM, question, max_tokens=800)


def complete(system, user, max_tokens=1000):
    """Gọi LLM một lượt, không công cụ (dùng cho baseline và LLM-as-judge)."""
    if PROVIDER == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=max_tokens, temperature=0, system=system,
                                                  messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import OpenAI
    r = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None).chat.completions.create(
        model=MODEL, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def _anthropic(msgs, max_steps, system):
    import anthropic
    client, log = anthropic.Anthropic(), []
    tools = [{"name": "calculate", "description": TOOL_DESC,
              "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}]
    for _ in range(max_steps):
        r = client.messages.create(model=MODEL, max_tokens=1500, temperature=0, system=system, tools=tools, messages=msgs)
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), log
        msgs.append({"role": "assistant", "content": r.content})
        res = []
        for b in r.content:
            if b.type == "tool_use":
                out = safe_calc(b.input.get("expression", ""))
                log.append((b.input.get("expression", ""), out))
                res.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        msgs.append({"role": "user", "content": res})
    return "Không hoàn tất được phép tính, vui lòng thử lại.", log


def _openai(msgs, max_steps, system):
    from openai import OpenAI
    client, log = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None), []
    tools = [{"type": "function", "function": {"name": "calculate", "description": TOOL_DESC,
              "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
    m = [{"role": "system", "content": system}] + msgs
    for _ in range(max_steps):
        r = client.chat.completions.create(model=MODEL, messages=m, tools=tools, max_tokens=1500, temperature=0)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", log
        m.append(msg)
        for tc in msg.tool_calls:
            expr = json.loads(tc.function.arguments).get("expression", "")
            out = safe_calc(expr); log.append((expr, out))
            m.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "Không hoàn tất được phép tính, vui lòng thử lại.", log


# ---------- Kiểm tra hallucination bằng quy tắc (không cần gọi LLM) ----------
_NUM_TOKEN = re.compile(r"\d[\d.,]*\d|\d")
_THOUSANDS = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")


def parse_numbers(text):
    """Trích mọi số trong văn bản (hiểu định dạng Việt Nam 1.234.567 và 12,5) thành list float."""
    out = []
    for tok in _NUM_TOKEN.findall(text):
        try:
            if _THOUSANDS.match(tok):
                out.append(float(re.sub(r"\D", "", tok)))
            elif "." in tok and "," in tok:  # 33.067,7 -> dấu cuối là thập phân, dấu còn lại là ngăn nghìn
                dec = max(tok.rfind("."), tok.rfind(","))
                out.append(float(re.sub(r"\D", "", tok[:dec]) + "." + re.sub(r"\D", "", tok[dec + 1:])))
            else:
                t = tok.replace(",", ".")
                if t.count(".") > 1:  # nhiều dấu giống nhau mà không đúng nhóm nghìn -> bỏ
                    continue
                out.append(float(t))
        except ValueError:
            pass
    return out


def _matches(n, sorted_vals, tol):
    """n khớp một giá trị trong ngữ cảnh (cho phép làm tròn và đổi đơn vị 1e3/1e6/1e9/1e12)."""
    for scale in (1, 1e3, 1e6, 1e9, 1e12):
        x = n * scale
        i = bisect.bisect_left(sorted_vals, x * (1 - tol))
        if i < len(sorted_vals) and sorted_vals[i] <= x * (1 + tol):
            return True
    return False


def grounding_check(text, hits, calcs=None, tol=0.005):
    """Đối chiếu số liệu trong câu trả lời với ngữ cảnh truy xuất (+ kết quả công cụ tính).
    Chỉ xét số có ≥4 chữ số hoặc số thập phân; bỏ qua năm, ngày tháng và số thứ tự trích dẫn [#n].
    Trả về dict: checked, ungrounded (list), rate (tỷ lệ số không có căn cứ), invalid_citations (list [#n] ngoài phạm vi)."""
    clean = re.sub(r"\[#\d+\]", " ", text)
    clean = re.sub(r"\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}", " ", clean)  # ngày/tháng/năm -> không phải số liệu
    vals = []
    for c, _ in hits:
        vals += parse_numbers(c.text)
    for _, r in (calcs or []):
        vals += parse_numbers(str(r))
    vals = sorted(abs(v) for v in vals if v)
    checked, bad = 0, []
    for tok in _NUM_TOKEN.findall(clean):
        v = parse_numbers(tok)
        if not v:
            continue
        n = abs(v[0])
        is_year = float(n).is_integer() and 1900 <= n <= 2100
        if is_year or not (len(re.sub(r"\D", "", tok)) >= 4 or "." in tok or "," in tok):
            continue
        checked += 1
        if not vals or not _matches(n, vals, tol):
            bad.append(tok)
    nmax = len(hits)
    invalid = sorted({int(n) for n in re.findall(r"\[#(\d+)\]", text) if not 0 < int(n) <= nmax})
    return {"checked": checked, "ungrounded": bad, "rate": (len(bad) / checked) if checked else 0.0,
            "invalid_citations": invalid}
