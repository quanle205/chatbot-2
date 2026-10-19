"""Đánh giá hệ thống theo kế hoạch (Chương 4, Tuần 7-8).

  4.2  Thành phần IR      : Precision@k, Recall@k, F1, MRR, Hit@k — so sánh BM25 / Dense(cosine) / Hybrid / +Re-rank
  4.3  Chất lượng sinh    : Faithfulness & Answer Relevance (LLM-as-judge), độ đúng số liệu
  Mục tiêu "giảm 95% hallucination": so tỷ lệ câu trả lời chứa số liệu KHÔNG có căn cứ trong tài liệu giữa RAG và LLM thuần.

Chạy:
  python eval.py --data data --questions tests/questions.csv --k 6                      # chỉ đánh giá IR (không tốn API)
  python eval.py --data data --questions tests/questions.csv --gen --judge --out report # + sinh câu trả lời, hallucination, judge

File câu hỏi (CSV, UTF-8; 50-100 câu theo kế hoạch) — cột:
  question          (bắt buộc)
  relevant_snippets đoạn/số liệu PHẢI xuất hiện trong chunk đúng, phân tách bằng '|'  (vd. 33.067.716.327.546)
  relevant_pages    tuỳ chọn, 'file:trang' hoặc 'trang', phân tách bằng ';'  (vd. FPT.pdf:4;5)
  gold_numbers      các số đúng mà câu trả lời phải chứa, phân tách bằng '|'  (dùng chấm độ đúng)
  gold_answer       đáp án tham chiếu (tuỳ chọn, để người đọc/judge)
Một chunk được coi là liên quan nếu khớp relevant_pages HOẶC chứa một trong relevant_snippets (so khớp không phụ thuộc
dấu chấm/phẩy ngăn cách nghìn, hoa/thường)."""
import argparse, csv, io, json, os, re, sys, time
from collections import defaultdict

MODES = {"bm25": dict(mode="bm25"), "dense": dict(mode="dense"), "hybrid": dict(mode="hybrid"),
         "hybrid+rerank": dict(mode="hybrid", rerank=True), "dense+rerank": dict(mode="dense", rerank=True)}


# ---------- Câu hỏi kiểm thử ----------
def load_questions(src):
    """src: đường dẫn CSV hoặc bytes/str nội dung CSV. Trả về list dict (đã tách |, ;)."""
    if isinstance(src, (bytes, bytearray)):
        src = src.decode("utf-8-sig", errors="replace")
    text = open(src, encoding="utf-8-sig").read() if (isinstance(src, str) and "\n" not in src and os.path.exists(src)) else src
    rows = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text)), 1):
        q = (r.get("question") or "").strip()
        if not q:
            continue
        rows.append({
            "id": (r.get("id") or str(i)).strip(), "question": q,
            "snippets": [s.strip() for s in (r.get("relevant_snippets") or "").split("|") if s.strip()],
            "pages": [s.strip() for s in (r.get("relevant_pages") or "").split(";") if s.strip()],
            "numbers": [s.strip() for s in (r.get("gold_numbers") or "").split("|") if s.strip()],
            "gold_answer": (r.get("gold_answer") or "").strip(),
        })
    return rows


# ---------- Xác định chunk liên quan ----------
def _digits(s):
    return re.sub(r"\D", "", s)


def _is_numeric(s):
    return bool(re.fullmatch(r"[\d\s.,]+", s)) and len(_digits(s)) >= 3


def _norm(s):
    return " ".join(s.lower().split())


def chunk_matches(chunk, row):
    for p in row["pages"]:
        doc, _, pg = p.rpartition(":")
        if pg.strip().isdigit() and int(pg) == chunk.page and (not doc or os.path.basename(doc).lower() == chunk.doc.lower()):
            return True
    text, digit_tokens = _norm(chunk.text), None
    for s in row["snippets"]:
        if _is_numeric(s):
            if digit_tokens is None:
                digit_tokens = {_digits(t) for t in re.findall(r"\d[\d.,]*\d|\d", chunk.text)}
            if _digits(s) in digit_tokens:
                return True
        elif _norm(s) in text:
            return True
    return False


def relevant_ids(index, row):
    return {c.id for c in index.chunks if chunk_matches(c, row)}


# ---------- Độ đo IR (mục 1.2.4) ----------
def ir_metrics(retrieved_ids, relevant, k):
    """Precision@k, Recall@k (mẫu số = min(#chunk liên quan, k) để không phạt khi có nhiều chunk cùng đúng), F1, Hit@k, RR."""
    got = retrieved_ids[:k]
    rel = [i in relevant for i in got]
    tp = sum(rel)
    p = tp / max(len(got), 1)
    r = tp / max(min(len(relevant), k), 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rr = next((1 / (n + 1) for n, ok in enumerate(rel) if ok), 0.0)
    return {"precision": p, "recall": r, "f1": f1, "hit": float(tp > 0), "rr": rr}


def retrieval_eval(index, rows, k=6, modes=("bm25", "dense", "hybrid"), alpha=0.5, progress=None):
    """Trả về (bảng tổng hợp theo chế độ, chi tiết theo câu, danh sách câu bị bỏ vì không tìm được đáp án trong kho)."""
    usable, skipped = [], []
    for row in rows:
        rel = relevant_ids(index, row)
        (usable if rel else skipped).append((row, rel))
    modes = [m for m in modes if not (m.startswith(("dense", "hybrid")) and index.emb is None)]
    summary, detail = [], []
    for mi, m in enumerate(modes):
        agg = defaultdict(float)
        for qi, (row, rel) in enumerate(usable):
            hits = index.search(row["question"], top_k=k, alpha=alpha, **MODES[m])
            met = ir_metrics([h.chunk.id for h in hits], rel, k)
            for key, v in met.items():
                agg[key] += v
            detail.append({"mode": m, "id": row["id"], "question": row["question"], "n_relevant": len(rel),
                           "first_rank": next((n + 1 for n, h in enumerate(hits) if h.chunk.id in rel), None), **met})
            if progress:
                progress(f"{m}: câu {qi + 1}/{len(usable)}", (mi + (qi + 1) / len(usable)) / len(modes))
        n = max(len(usable), 1)
        summary.append({"mode": m, f"P@{k}": agg["precision"] / n, f"R@{k}": agg["recall"] / n, "F1": agg["f1"] / n,
                        f"Hit@{k}": agg["hit"] / n, "MRR": agg["rr"] / n, "n_questions": len(usable)})
    return summary, detail, [r["id"] + ": " + r["question"] for r, _ in skipped]


# ---------- Chất lượng sinh văn bản (mục 4.3) ----------
def numbers_correct(answer, gold_numbers, tol=0.005):
    """Tỷ lệ số đúng (gold_numbers) xuất hiện trong câu trả lời (cho phép làm tròn / đổi đơn vị)."""
    import llm
    if not gold_numbers:
        return None
    vals = sorted(abs(v) for v in llm.parse_numbers(answer) if v)
    ok = 0
    for g in gold_numbers:
        gv = llm.parse_numbers(g)
        if not gv or not vals:
            continue
        gold = abs(gv[0])  # khớp theo cả hai chiều đổi đơn vị: đáp án "33.067,7 tỷ" vs gold VND, hoặc đáp án VND vs gold triệu đồng
        if llm._matches(gold, vals, tol) or any(llm._matches(a, [gold], tol) for a in vals):
            ok += 1
    return ok / len(gold_numbers)


def _json_from(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def judge_faithfulness(question, answer, hits):
    """RAGAS-style: LLM tách câu trả lời thành các khẳng định, kiểm tra từng khẳng định có được ngữ cảnh ủng hộ không."""
    import llm
    ctx = llm.format_context(hits)
    out = llm.complete(
        "Bạn là giám khảo nghiêm khắc. Chỉ trả về JSON.",
        f"NGỮ CẢNH:\n{ctx}\n\nCÂU HỎI: {question}\n\nCÂU TRẢ LỜI:\n{answer}\n\n"
        'Tách câu trả lời thành các khẳng định độc lập (số liệu, sự kiện, nguyên nhân). Với mỗi khẳng định, xác định nó có được '
        'NGỮ CẢNH ủng hộ trực tiếp hay không. Trả về: {"claims":[{"claim":"...","supported":true|false}]}', 1200)
    claims = _json_from(out).get("claims") or []
    if not claims:
        return None
    return sum(bool(c.get("supported")) for c in claims) / len(claims)


def judge_relevance(question, answer):
    """Answer Relevance: câu trả lời có trả lời đúng trọng tâm câu hỏi không (1-5 -> 0..1)."""
    import llm
    out = llm.complete("Bạn là giám khảo. Chỉ trả về JSON.",
                       f"CÂU HỎI: {question}\n\nCÂU TRẢ LỜI:\n{answer}\n\n"
                       'Chấm mức độ câu trả lời đáp ứng đúng trọng tâm câu hỏi (không xét đúng/sai số liệu): 1 (lạc đề) đến 5 (đầy đủ, '
                       'trọng tâm). Trả về: {"score": <1-5>}', 200)
    s = _json_from(out).get("score")
    return (float(s) - 1) / 4 if isinstance(s, (int, float)) and 1 <= s <= 5 else None


def generation_eval(index, rows, k=6, mode="hybrid", rerank=False, alpha=0.5, judge=False, baseline=True, progress=None):
    """Chạy RAG (và LLM thuần) trên từng câu hỏi; trả về (chi tiết theo câu, tổng hợp)."""
    import llm
    detail = []
    for qi, row in enumerate(rows):
        hits = index.search(row["question"], top_k=k, alpha=alpha, mode=mode, rerank=rerank)
        rec = {"id": row["id"], "question": row["question"], "gold_answer": row["gold_answer"]}
        for tag in (["rag", "llm"] if baseline else ["rag"]):
            try:
                if tag == "rag":
                    text, calcs = llm.answer(row["question"], hits)
                else:
                    text, calcs = llm.answer_baseline(row["question"]), []
            except Exception as e:  # lỗi API: ghi nhận và đi tiếp
                text, calcs = f"[LỖI LLM] {e}", []
            g = llm.grounding_check(text, hits, calcs)  # LLM thuần cũng được đối chiếu với đúng ngữ cảnh đó
            rec.update({f"{tag}_answer": text, f"{tag}_numbers_checked": g["checked"], f"{tag}_ungrounded": g["ungrounded"],
                        f"{tag}_halluc": float(bool(g["ungrounded"])), f"{tag}_correct": numbers_correct(text, row["numbers"]),
                        f"{tag}_invalid_cites": g["invalid_citations"]})
            if judge and not text.startswith("[LỖI"):
                rec[f"{tag}_faithfulness"] = judge_faithfulness(row["question"], text, hits)
                rec[f"{tag}_relevance"] = judge_relevance(row["question"], text)
        detail.append(rec)
        if progress:
            progress(f"Sinh câu trả lời {qi + 1}/{len(rows)}", (qi + 1) / len(rows))
    return detail, summarize_generation(detail, baseline)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_generation(detail, baseline=True):
    out = {}
    for tag in (["rag", "llm"] if baseline else ["rag"]):
        out[tag] = {
            "n": len(detail),
            "halluc_rate": _mean(d[f"{tag}_halluc"] for d in detail),  # % câu trả lời có ≥1 số liệu không có căn cứ
            "numbers_correct": _mean(d.get(f"{tag}_correct") for d in detail),
            "faithfulness": _mean(d.get(f"{tag}_faithfulness") for d in detail),
            "answer_relevance": _mean(d.get(f"{tag}_relevance") for d in detail),
        }
    if baseline and out["llm"]["halluc_rate"]:
        out["halluc_reduction"] = 1 - out["rag"]["halluc_rate"] / out["llm"]["halluc_rate"]
    return out


# ---------- Báo cáo ----------
def _fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def md_table(rows):
    if not rows:
        return ""
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows])


def write_csv(path, rows):
    if rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--modes", default="bm25,dense,hybrid,hybrid+rerank")
    ap.add_argument("--no-dense", action="store_true", help="không dùng embedding (chỉ BM25)")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--chunk-overlap", type=int)
    ap.add_argument("--gen", action="store_true", help="chạy cả bước sinh câu trả lời + hallucination (tốn API)")
    ap.add_argument("--gen-mode", default="hybrid")
    ap.add_argument("--rerank", action="store_true", help="dùng re-ranking ở bước sinh")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge: Faithfulness & Answer Relevance (tốn API)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="eval_report")
    a = ap.parse_args()

    import glob
    from rag import build_index, SUPPORTED_EXT
    files = {}
    for p in sorted(glob.glob(os.path.join(a.data, "*"))):
        if p.lower().endswith(SUPPORTED_EXT):
            files[os.path.basename(p)] = open(p, "rb").read()
    index, warns, stats = build_index(files, not a.no_dense, progress=lambda m, f: print(f"\r{m[:70]:70}", end="", file=sys.stderr),
                                      chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap)
    print(file=sys.stderr)
    for w in warns:
        print("[WARN]", w, file=sys.stderr)
    rows = load_questions(a.questions)[: a.limit]
    print(f"{len(files)} file, {len(index.chunks)} chunk, {len(rows)} câu hỏi; vector DB: {index.backend}", file=sys.stderr)

    t0 = time.time()
    summary, detail, skipped = retrieval_eval(index, rows, a.k, [m.strip() for m in a.modes.split(",")], a.alpha)
    report = [f"# Báo cáo đánh giá\n\n- Tài liệu: {', '.join(files)} ({len(index.chunks)} chunk), vector DB: {index.backend}",
              f"- Số câu hỏi dùng được: {len(rows) - len(skipped)}/{len(rows)}; k = {a.k}",
              "\n## 4.2 Đánh giá thành phần IR\n", md_table(summary)]
    if skipped:
        report.append("\nCâu hỏi bị bỏ (không tìm thấy đáp án gold trong kho — kiểm tra relevant_snippets/OCR):\n" + "\n".join("- " + s for s in skipped))
    write_csv(a.out + "_ir_detail.csv", detail)

    if a.gen:
        usable = [r for r in rows if r["id"] not in {s.split(":")[0] for s in skipped}] or rows
        gdetail, gsum = generation_eval(index, usable, a.k, a.gen_mode, a.rerank, a.alpha, a.judge, not a.no_baseline,
                                        progress=lambda m, f: print(f"\r{m:40}", end="", file=sys.stderr))
        print(file=sys.stderr)
        tbl = [{"hệ thống": "RAG" if t == "rag" else "LLM thuần", **{k: v for k, v in gsum[t].items()}} for t in gsum if t in ("rag", "llm")]
        report += ["\n## 4.3 Chất lượng sinh văn bản & hallucination\n", md_table(tbl)]
        if "halluc_reduction" in gsum:
            report.append(f"\n**Giảm hallucination so với LLM thuần: {gsum['halluc_reduction']:.1%}** (mục tiêu kế hoạch: ≥ 95%)")
        report.append("\nhalluc_rate = tỷ lệ câu trả lời có ≥1 số liệu không khớp ngữ cảnh truy xuất (kiểm tra bằng quy tắc, dung sai 0,5%).")
        write_csv(a.out + "_gen_detail.csv", [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in d.items()} for d in gdetail])
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nĐã ghi {a.out}.md và các file chi tiết (mất {time.time() - t0:.0f}s).", file=sys.stderr)


if __name__ == "__main__":
    main()
