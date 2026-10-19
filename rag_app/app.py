import glob, os, re

import streamlit as st


# ---- Cấu hình từ Streamlit Secrets (phải đặt TRƯỚC khi import rag/llm) ----
def _secret(name, default=""):
    try:
        return str(st.secrets[name])
    except Exception:
        return os.environ.get(name, default)


_provider = _secret("LLM_PROVIDER", "openai")  # "openai" (dùng cho Gemini) hoặc "anthropic"
os.environ["LLM_PROVIDER"] = _provider
os.environ["LLM_MODEL"] = _secret("LLM_MODEL", "gemini-3.8-flash")
os.environ["COMPANY"] = _secret("COMPANY", "doanh nghiệp")  # vd. "Công ty Cổ phần Sữa Việt Nam (Vinamilk)"
for _k in ("EMBED_MODEL", "RERANK_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "VECTOR_DB"):
    if _secret(_k):
        os.environ[_k] = _secret(_k)
if _provider == "anthropic":
    os.environ["ANTHROPIC_API_KEY"] = _secret("LLM_API_KEY")
else:
    os.environ["OPENAI_API_KEY"] = _secret("LLM_API_KEY")
    os.environ["OPENAI_BASE_URL"] = _secret("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

import pymupdf
from rag import build_index, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL
import llm
import eval as ev

COMPANY = os.environ["COMPANY"]
st.set_page_config(page_title="Chatbot phân tích BCTC", page_icon="📊", layout="wide")
st.title("📊 Chatbot hỏi đáp Báo cáo tài chính (RAG)")
st.caption(f"Phạm vi dữ liệu: {COMPANY} · Truy xuất ngữ nghĩa (BERT embedding + Vector DB + Cosine) kết hợp sinh câu trả lời có trích dẫn nguồn")

# ---- (Tuỳ chọn) khoá bằng mật khẩu để người lạ không dùng hết hạn mức API ----
_pwd = _secret("APP_PASSWORD")
if _pwd and st.text_input("Mật khẩu", type="password") != _pwd:
    st.info("Nhập mật khẩu để sử dụng.")
    st.stop()

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLES = ["Biên lợi nhuận gộp quý I/2026 là bao nhiêu và nguyên nhân do đâu?",
           "Doanh thu thuần quý I là bao nhiêu?", "Lợi nhuận sau thuế là bao nhiêu?",
           "Tổng tài sản tại ngày kết thúc kỳ là bao nhiêu?", "Tính tăng trưởng doanh thu so với cùng kỳ năm trước",
           "Dòng tiền thuần từ hoạt động kinh doanh là bao nhiêu?"]
MODE_LABELS = {"Hybrid (Cosine + BM25)": "hybrid", "Dense (Cosine, Vector DB)": "dense", "BM25 (từ khoá)": "bm25"}


def _default_files():
    exts = ("*.pdf", "*.docx", "*.csv", "*.xlsx", "*.xls", "*.html", "*.htm", "*.txt")
    # README nội bộ không phải dữ liệu BCTC -> đặt tên khác đuôi hỗ trợ (.md) để khỏi bị lẫn vào index
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(DATA, e)))
    return {os.path.basename(p): open(p, "rb").read() for p in paths}


def _do_build(files, use_dense, chunk_size, chunk_overlap):
    bar = st.progress(0.0, text="Đang xử lý (file scan cần OCR nên có thể mất vài phút)...")
    idx, warns, stats = build_index(files, use_dense, progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg),
                                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    bar.empty()
    st.session_state.update(index=idx, files=files, history=[], chat=[], warns=warns, stats=stats,
                            cfg=(use_dense, chunk_size, chunk_overlap))


with st.sidebar:
    st.header("Dữ liệu")
    use_dense = st.checkbox("Dùng embedding BERT + Vector DB", os.getenv("USE_DENSE", "1") == "1",
                            help=f"Mô hình: {EMBED_MODEL}. Trên Streamlit Cloud (1 GB RAM) có thể thiếu RAM → bỏ chọn để chỉ dùng BM25")
    chunk_size = st.slider("Kích thước chunk (ký tự)", 300, 2000, CHUNK_SIZE, 100,
                           help="Tinh chỉnh ở Tuần 8. Đổi giá trị này cần bấm 'Xây dựng lại index'")
    chunk_overlap = st.slider("Overlap giữa các chunk", 0, 400, CHUNK_OVERLAP, 50)
    ups = st.file_uploader("Tải thêm file (PDF, Word, Excel, CSV, HTML, TXT)",
                           type=["pdf", "docx", "csv", "xlsx", "xls", "html", "htm", "txt"], accept_multiple_files=True)
    merge = st.checkbox("Gộp với file mặc định trong `data/`", True,
                        help="Tắt nếu chỉ muốn dùng riêng các file vừa tải lên")
    rebuild = st.button("Xây dựng lại index", type="primary")
    if rebuild:
        files = dict(_default_files()) if merge else {}
        files.update({u.name: u.getvalue() for u in ups or []})  # file trùng tên sẽ ghi đè bản mặc định
        _do_build(files, use_dense, chunk_size, chunk_overlap)

if "index" not in st.session_state:
    files = _default_files()
    if files:
        with st.spinner("Đang đọc file mặc định, tách bảng, tạo embedding và index..."):
            _do_build(files, use_dense, chunk_size, chunk_overlap)

if "index" not in st.session_state:
    st.info("Chưa có dữ liệu: đặt file vào thư mục `data/` của repo (PDF/Word/Excel/CSV/HTML/TXT), hoặc tải file lên ở thanh bên.")
    st.stop()

index = st.session_state.index
with st.sidebar:
    if st.session_state.get("cfg") != (use_dense, chunk_size, chunk_overlap):
        st.warning("Cấu hình dữ liệu đã đổi — bấm 'Xây dựng lại index' để áp dụng.")
    st.header("Truy xuất")
    has_dense = index.emb is not None
    labels = list(MODE_LABELS) if has_dense else ["BM25 (từ khoá)"]
    mode = MODE_LABELS[st.radio("Chế độ tìm kiếm", labels, help="Dense dùng độ tương đồng Cosine trên vector embedding")]
    top_k = st.slider("Số đoạn truy xuất (Top-K)", 3, 12, 6)
    alpha = st.slider("Trọng số dense vs BM25", 0.0, 1.0, 0.5, 0.1, disabled=mode != "hybrid")
    rerank = st.checkbox("Re-ranking (cross-encoder)", os.getenv("USE_RERANK", "0") == "1",
                         help="Sắp xếp lại các đoạn ứng viên để tăng độ chính xác; cần tải thêm model, chậm hơn")
    compare = st.checkbox("So sánh với LLM thuần (không RAG)", False,
                          help="Gọi thêm LLM không có ngữ cảnh để thấy rõ hiện tượng ảo giác số liệu")
    st.caption(f"Model: {llm.PROVIDER} / {llm.MODEL}")

    for w in st.session_state.get("warns", []):
        st.warning(w)
    ch = index.chunks
    st.success(f"Đã index {len(ch)} đoạn ({sum(c.kind == 'table' for c in ch)} bảng) từ {len(st.session_state.files)} file")
    st.caption("Vector DB: " + (index.backend if has_dense else "không dùng (chỉ BM25)"))
    stats = st.session_state.get("stats") or []
    if stats:
        with st.expander(f"Chi tiết {len(stats)} file đã nạp"):
            st.dataframe(
                [{"File": s["name"], "Trang": s["pages"], "Đoạn": s["chunks"], "Bảng": s["tables"],
                  "Trang OCR": s["ocr_pages"]} for s in stats],
                hide_index=True, use_container_width=True,
            )
    st.header("Câu hỏi mẫu")
    pick = None
    for s in SAMPLES:
        if st.button(s, use_container_width=True):
            pick = s

view = st.radio("Chế độ", ["💬 Chatbot", "📈 Đánh giá IR"], horizontal=True, label_visibility="collapsed")

# ---------------- Trang đánh giá thành phần IR (mục 4.2 của đề cương) ----------------
if view.endswith("Đánh giá IR"):
    st.subheader("Đánh giá thành phần truy xuất (IR)")
    st.markdown("Tải file CSV câu hỏi kiểm thử (50-100 câu theo kế hoạch). Cột: `question`, `relevant_snippets` (phân tách `|`), "
                "`relevant_pages` (vd. `file.pdf:4;5`), `gold_numbers`, `gold_answer`. Xem mẫu `tests/questions_sample.csv`.")
    qf = st.file_uploader("File câu hỏi (CSV)", type=["csv"])
    avail = [m for m in ("bm25", "dense", "hybrid", "dense+rerank", "hybrid+rerank") if has_dense or m == "bm25"]
    modes = st.multiselect("Các chế độ cần so sánh", avail, default=[m for m in ("bm25", "dense", "hybrid") if m in avail])
    if qf and st.button("Chạy đánh giá", type="primary"):
        rows = ev.load_questions(qf.getvalue())
        bar = st.progress(0.0)
        summary, detail, skipped = ev.retrieval_eval(index, rows, top_k, modes, alpha, progress=lambda m, f: bar.progress(min(f, 1.0), text=m))
        bar.empty()
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption(f"Precision@{top_k}, Recall@{top_k} (mẫu số = min(#đoạn đúng, k)), F1, Hit@{top_k}, MRR — trung bình trên "
                   f"{summary[0]['n_questions'] if summary else 0}/{len(rows)} câu.")
        if skipped:
            st.warning("Bỏ qua câu không tìm thấy đáp án gold trong kho (kiểm tra `relevant_snippets` hoặc chất lượng OCR):\n\n- " + "\n- ".join(skipped))
        st.expander("Chi tiết theo câu hỏi").dataframe(detail, hide_index=True, use_container_width=True)
    st.info("Đánh giá sinh văn bản (Faithfulness, Answer Relevance, giảm hallucination so với LLM thuần) chạy bằng lệnh: "
            "`python eval.py --data data --questions tests/questions.csv --gen --judge`")
    st.stop()


def render_sources(hits, text, key):
    cited = sorted({int(n) for n in re.findall(r"\[#(\d+)\]", text) if 0 < int(n) <= len(hits)})
    for n in cited or range(1, min(3, len(hits)) + 1):
        h = hits[n - 1]
        c = h.chunk
        scores = " · ".join(x for x in (f"cosine {h.cos:.3f}" if h.cos is not None else "",
                                        f"rerank {h.rerank:.2f}" if h.rerank is not None else "") if x)
        with st.expander(f"[#{n}] {c.doc} — trang {c.page} ({'bảng' if c.kind == 'table' else 'văn bản'})" + (f" · {scores}" if scores else "")):
            st.markdown(c.text)
            if c.doc.lower().endswith(".pdf") and st.checkbox("Xem trang PDF gốc", key=f"pg{key}_{n}"):
                pdf = pymupdf.open(stream=st.session_state.files[c.doc], filetype="pdf")
                st.image(pdf[c.page - 1].get_pixmap(dpi=110).tobytes("png"))


def render_grounding(g):
    if g["ungrounded"]:
        st.warning(f"⚠️ {len(g['ungrounded'])}/{g['checked']} số liệu trong câu trả lời không đối chiếu được với ngữ cảnh truy xuất: "
                   + ", ".join(g["ungrounded"][:8]) + ". Hãy kiểm tra lại với tài liệu gốc.")
    if g["invalid_citations"]:
        st.warning("Trích dẫn không hợp lệ: " + ", ".join(f"[#{n}]" for n in g["invalid_citations"]))


for i, m in enumerate(st.session_state.chat):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("grounding"):
            render_grounding(m["grounding"])
        if m.get("hits"):
            render_sources(m["hits"], m["content"], i)
        if m.get("baseline"):
            with st.expander("🔬 LLM thuần (không RAG)"):
                st.markdown(m["baseline"])
                st.caption(m["baseline_note"])

q = st.chat_input("Hỏi về báo cáo tài chính...") or pick
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        h = st.session_state.history
        search_q = q if len(q.split()) > 8 or not h else h[-2]["content"].split("CÂU HỎI:")[-1].strip() + " " + q
        hits = index.search(search_q, top_k=top_k, alpha=alpha, mode=mode, rerank=rerank)
        if rerank and index.rerank_error:
            st.caption(f"Không bật được re-ranking ({index.rerank_error}); dùng thứ tự truy xuất gốc.")
        llm_failed = False
        try:
            with st.spinner("Đang phân tích..."):
                text, calcs = llm.answer(q, hits, h[-6:])
        except Exception as e:
            llm_failed = True
            text, calcs = f"Lỗi gọi LLM: {e}\n\nKiểm tra `LLM_API_KEY` và `LLM_MODEL` trong Secrets.", []
        st.markdown(text)
        grounding = None if llm_failed else llm.grounding_check(text, hits, calcs)  # không kiểm tra số liệu trên thông báo lỗi
        if grounding:
            render_grounding(grounding)
        if calcs:
            with st.expander("🧮 Phép tính đã thực hiện"):
                for e, r in calcs:
                    st.code(f"{e} = {r}")
        render_sources(hits, text, len(st.session_state.chat))
        baseline, note = None, ""
        if compare and not llm_failed:
            try:
                with st.spinner("Đang hỏi LLM thuần để so sánh..."):
                    baseline = llm.answer_baseline(q)
                gb = llm.grounding_check(baseline, hits)
                note = (f"Đối chiếu với tài liệu: {len(gb['ungrounded'])}/{gb['checked']} số liệu của LLM thuần KHÔNG có trong ngữ cảnh truy xuất "
                        f"(RAG: {len(grounding['ungrounded'])}/{grounding['checked']}).")
                with st.expander("🔬 LLM thuần (không RAG)", expanded=True):
                    st.markdown(baseline)
                    st.caption(note)
            except Exception as e:
                st.caption(f"Không gọi được LLM thuần: {e}")
    if llm_failed:  # không đưa lỗi vào lịch sử hội thoại
        st.session_state.chat.append({"role": "assistant", "content": text, "hits": hits})
        st.stop()
    h.append({"role": "user", "content": f"CÂU HỎI: {q}"})
    h.append({"role": "assistant", "content": text})
    st.session_state.chat.append({"role": "assistant", "content": text, "hits": hits, "grounding": grounding,
                                  "baseline": baseline, "baseline_note": note})
