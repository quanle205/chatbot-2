"""Pipeline RAG cho BCTC (khớp kế hoạch nghiên cứu):

  Thu thập/tiền xử lý : parse PDF (bảng -> Markdown, OCR trang scan), Word, Excel, CSV, HTML, TXT; loại header/footer lặp
  Chunking            : theo ngữ nghĩa (tiêu đề mục / thuyết minh), kích thước + overlap tinh chỉnh được
  Embedding           : mô hình họ BERT (multilingual-e5, XLM-R) -> vector chuẩn hoá L2
  Vector DB           : FAISS IndexFlatIP (tích vô hướng trên vector chuẩn hoá == Cosine Similarity); dự phòng numpy
  Retrieval           : Cosine (dense) | BM25 | Hybrid (RRF) + Re-ranking bằng cross-encoder (tuỳ chọn)
"""
import csv, glob, hashlib, io, os, re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser

import numpy as np

try:
    import pymupdf
except ImportError:  # chỉ cần cho PDF; các định dạng khác vẫn dùng được
    pymupdf = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # dự phòng gọn để module vẫn chạy/kiểm thử được khi thiếu rank_bm25
    class BM25Okapi:
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.k1, self.b, self.corpus = k1, b, corpus
            self.dl = np.array([len(d) for d in corpus], dtype=float)
            self.avg = max(self.dl.mean(), 1e-9) if len(corpus) else 1.0
            self.tf = [Counter(d) for d in corpus]
            df = Counter(t for d in corpus for t in set(d))
            n = len(corpus)
            self.idf = {t: max(np.log((n - f + 0.5) / (f + 0.5) + 1), 0.0) for t, f in df.items()}

        def get_scores(self, query):
            s = np.zeros(len(self.corpus))
            for t in set(query):
                if t not in self.idf:
                    continue
                for i, tf in enumerate(self.tf):
                    f = tf.get(t, 0)
                    if f:
                        s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
            return s

# Họ BERT: multilingual-e5 (XLM-RoBERTa). Có thể đổi: BAAI/bge-m3, hoặc bkai-foundation-models/vietnamese-bi-encoder (PhoBERT)
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")  # faiss | numpy
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_NO_OCR_INFO = {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0, "header_lines": []}


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    kind: str  # "text" | "table"
    text: str
    section: str = ""


@dataclass
class Hit:
    """Kết quả truy xuất. Unpack được như (chunk, score) để tương thích code cũ."""
    chunk: Chunk
    score: float                  # điểm hợp nhất (RRF) hoặc điểm của chế độ đang dùng
    cos: float = None             # cosine giữa vector truy vấn và đoạn (nếu có embedding)
    bm25: float = 0.0
    rerank: float = None          # điểm cross-encoder (nếu bật re-ranking)

    def __iter__(self):
        return iter((self.chunk, self.score))


# ---------- 1. Parse PDF (có OCR cho trang scan) ----------
OCR_LANG = os.getenv("OCR_LANG", "vie+eng")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))  # cao hơn = nhận diện chữ/số chính xác hơn, nhưng chậm hơn
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "80"))  # giới hạn số trang OCR mỗi file để khỏi chạy quá lâu


def _setup_tesseract():
    """Trả về ngôn ngữ OCR dùng được (vie+eng, eng) hoặc None nếu chưa cài tesseract."""
    if not os.environ.get("TESSDATA_PREFIX"):
        for d in glob.glob("/usr/share/tesseract-ocr/*/tessdata") + glob.glob("/usr/share/tessdata"):
            if os.path.isdir(d):
                os.environ["TESSDATA_PREFIX"] = d
                break
    d = os.environ.get("TESSDATA_PREFIX", "")
    if not d or not os.path.isdir(d):
        return None
    have = {f.split(".")[0] for f in os.listdir(d) if f.endswith(".traineddata")}
    langs = [l for l in OCR_LANG.split("+") if l in have]
    return "+".join(langs) or None


def _needs_ocr(text, tables):
    n = len(text) + sum(len(t) for t in tables)
    if n < 80:
        return True
    bad = sum(1 for ch in text if ch == "�" or 0xE000 <= ord(ch) <= 0xF8FF)  # font mã hoá lỗi
    return bad > 0.15 * max(len(text), 1)


def _ocr_cache_path(file_hash, page_no):
    d = os.path.join(CACHE_DIR, "ocr")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{file_hash}_{page_no}_{OCR_LANG}_{OCR_DPI}.txt")


def _ocr_page(page, lang, cache_key=None):
    if cache_key:
        p = _ocr_cache_path(*cache_key)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    tp = page.get_textpage_ocr(language=lang, dpi=OCR_DPI, full=True)
    text = page.get_text("text", textpage=tp, sort=True).strip()
    if cache_key:
        try:
            open(_ocr_cache_path(*cache_key), "w", encoding="utf-8").write(text)
        except OSError:
            pass
    return text


# --- Tiền xử lý: loại nhiễu header/footer lặp lại ở mỗi trang (kế hoạch, tuần 2) ---
_SAFE_LINE = re.compile(r"đơn vị|vnd|triệu đồng|nghìn đồng|tỷ đồng|\bđồng\b", re.I)  # dòng chứa đơn vị tính: luôn giữ
_AMOUNT = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d{5,}")  # dòng có số tiền (vd. 'Tổng cộng 1.234.567') là dữ liệu, không bao giờ là header
_PAGENUM = re.compile(r"^\s*(?:trang|page)?\s*\d{1,3}(?:\s*[/-]\s*\d{1,3})?\s*$", re.I)


def _norm_line(s):
    return " ".join(re.sub(r"\d+", "#", re.sub(r"[^\w\s]", " ", s.lower())).split())


def strip_noise(pages, min_pages=5, ratio=0.4):
    """Bỏ số trang đứng một mình ở đầu/cuối trang và các dòng ngắn lặp lại ở >= `ratio` số trang (header/footer).
    Trả về (pages, các dòng đã bỏ). Dòng chứa đơn vị tính (VND, triệu đồng...) luôn được giữ lại."""
    n = len(pages)
    if n < min_pages:
        return pages, []
    for p in pages:
        lines = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if lines and _PAGENUM.match(lines[-1]):
            lines = lines[:-1]
        if lines and _PAGENUM.match(lines[0]):
            lines = lines[1:]
        p["text"] = "\n".join(lines)
    cnt = Counter()
    for p in pages:
        cnt.update({k for k in (_norm_line(l) for l in p["text"].split("\n")) if k})
    thr = max(3, ratio * n)
    noisy = {k for k, v in cnt.items() if v >= thr}
    removed = {}
    for p in pages:
        keep = []
        for l in p["text"].split("\n"):
            k = _norm_line(l)
            if k in noisy and len(l) <= 120 and not _SAFE_LINE.search(l) and not _AMOUNT.search(l):
                removed.setdefault(k, l)
            else:
                keep.append(l)
        p["text"] = "\n".join(keep)
    return pages, list(removed.values())


def parse_pdf(data: bytes, name: str, progress=None):
    """Trả về (pages, info). pages: list {page, text, tables[markdown], ocr}. Bảng được tách riêng để không vỡ hàng/cột."""
    if pymupdf is None:
        raise RuntimeError("Thiếu thư viện pymupdf (pip install pymupdf).")
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages, info = [], dict(_NO_OCR_INFO, header_lines=[])
    lang = None
    file_hash = hashlib.md5(data).hexdigest()[:16]
    for i, page in enumerate(doc, 1):
        if progress:
            progress(i, len(doc))
        tables = []
        try:
            for t in page.find_tables().tables:
                md = t.to_markdown().strip()
                if md:
                    tables.append((pymupdf.Rect(t.bbox), md))
        except Exception:
            pass
        blocks = []
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0 or not b[4].strip():
                continue
            r = pymupdf.Rect(b[:4])
            if any((r & tr).get_area() > 0.5 * max(r.get_area(), 1e-6) for tr, _ in tables):
                continue  # block này đã nằm trong bảng
            blocks.append(b[4].strip())
        text, tmd, ocr = "\n".join(blocks), [m for _, m in tables], False
        if _needs_ocr(text, tmd):
            if info["ocr_pages"] >= MAX_OCR_PAGES:
                info["ocr_skipped"] += 1
            else:
                if lang is None:
                    lang = _setup_tesseract() or ""
                if not lang:
                    info["ocr_missing"] = True
                else:
                    try:
                        o = _ocr_page(page, lang, cache_key=(file_hash, i))
                        if len(o) > len(text):
                            text, tmd, ocr = o, [], True
                            info["ocr_pages"] += 1
                    except Exception:
                        info["ocr_missing"] = True
        pages.append({"page": i, "text": text, "tables": tmd, "ocr": ocr})
    pages, info["header_lines"] = strip_noise(pages)
    return pages, info


# ---------- 2. Chunking theo ngữ nghĩa ----------
_NUM_BIG = re.compile(r"\d{1,3}(?:[.,]\d{3})+")  # số tiền dạng 1.234.567 -> dòng dữ liệu, không phải tiêu đề
_HEAD_NUM = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?|[IVXL]+[.)]|[A-Z][.)])\s+\S")
_HEAD_KW = re.compile(r"^(?:thuyết minh|báo cáo|phụ lục|mục lục|kết quả|tình hình|nguyên nhân)\b", re.I)


def is_heading(line):
    s = line.strip()
    if not s or len(s) > 110 or _NUM_BIG.search(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return True  # dòng viết hoa toàn bộ
    return bool(_HEAD_NUM.match(s) or _HEAD_KW.match(s)) and not s.endswith((",", ";"))


def _hard_split(p, size):
    if len(p) <= size * 1.5:
        return [p]
    parts, cur = [], ""
    for s in re.split(r"(?<=[.;!?])\s+", p):
        if cur and len(cur) + len(s) > size:
            parts.append(cur)
            cur = s
        else:
            cur = (cur + " " + s) if cur else s
    return parts + ([cur] if cur else [])


def _split_text(text, size=None, overlap=None):
    """Tách văn bản theo ngữ nghĩa: bắt đầu đoạn mới khi gặp tiêu đề mục (và đoạn hiện tại đủ dài), nếu không thì theo
    kích thước. Trả về list (tiêu đề mục, nội dung); mỗi mảnh sau đó được gắn nhãn mục để truy xuất chính xác hơn."""
    size, overlap = size or CHUNK_SIZE, CHUNK_OVERLAP if overlap is None else overlap
    paras = [q for p in re.split(r"\n+", text) if p.strip() for q in _hard_split(p.strip(), size)]
    out, cur, section, cur_section = [], "", "", ""
    for p in paras:
        head = is_heading(p)
        if cur and (len(cur) + len(p) > size or (head and len(cur) > size * 0.3)):
            out.append((cur_section, cur))
            cur = (cur[-overlap:] + "\n" + p) if (overlap and not head) else p
            cur_section = section
        else:
            cur = (cur + "\n" + p) if cur else p
        if head:
            section = p
            if cur == p:
                cur_section = p
    if cur:
        out.append((cur_section, cur))
    return out


def _split_table(md, max_chars=2500):
    if len(md) <= max_chars:
        return [md]
    lines = md.split("\n")
    header, rows = lines[:2], lines[2:]  # giữ header cho mỗi mảnh
    out, cur = [], list(header)
    for r in rows:
        if sum(len(x) for x in cur) + len(r) > max_chars and len(cur) > 2:
            out.append("\n".join(cur))
            cur = list(header)
        cur.append(r)
    out.append("\n".join(cur))
    return out


def _page_title(text):
    """Tiêu đề trang (vd. 'BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH') để gắn vào bảng — bảng tự nó không có tên."""
    for l in text.split("\n"):
        if is_heading(l) and len(l.strip()) > 8:
            return l.strip()
    return ""


def make_chunks(pages, doc_name, start_id=0, size=None, overlap=None):
    chunks, cid = [], start_id
    for p in pages:
        for sec, t in _split_text(p["text"], size, overlap):
            if len(t) > 30:
                body = t if (not sec or t.lstrip().startswith(sec)) else f"[Mục: {sec}]\n{t}"
                chunks.append(Chunk(cid, doc_name, p["page"], "text", body, sec)); cid += 1
        title = _page_title(p["text"])
        for tb in p["tables"]:
            for part in _split_table(tb):
                chunks.append(Chunk(cid, doc_name, p["page"], "table", f"[{title}]\n{part}" if title else part, title)); cid += 1
    return chunks


# ---------- 3. Index: Vector DB + Cosine + BM25 + Re-ranking ----------
def tokenize(text):
    toks = re.findall(r"\w+", text.lower())
    nums = [re.sub(r"\D", "", t) for t in re.findall(r"\d[\d.,]*\d", text)]  # 1.234.567 -> 1234567
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # bắt cụm từ tiếng Việt: "lợi_nhuận"
    return toks + bigrams + [n for n in nums if len(n) > 2]


def cosine_similarity(q, M):
    """cos(q, m) = (q·m) / (‖q‖·‖m‖) cho mọi hàng m của M (mô hình không gian Vector, mục 6.1.3 của đề cương)."""
    q = np.asarray(q, dtype=np.float32)
    M = np.atleast_2d(np.asarray(M, dtype=np.float32))
    denom = np.linalg.norm(M, axis=1) * np.linalg.norm(q)
    return (M @ q) / np.maximum(denom, 1e-12)


class Index:
    def __init__(self, chunks, use_dense=True):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        self.emb, self.model, self.faiss_index = None, None, None
        self.reranker, self.rerank_error = None, None
        self.backend, self.last_mode = "BM25", "bm25"
        if use_dense and chunks:
            self._build_dense()

    def _prefix(self, kind):
        return {"query": "query: ", "passage": "passage: "}[kind] if "e5" in EMBED_MODEL.lower() else ""

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBED_MODEL)
        return self.model

    def _build_dense(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5((EMBED_MODEL + "".join(c.text for c in self.chunks)).encode()).hexdigest()
        path = os.path.join(CACHE_DIR, f"{key}.npy")
        try:
            if os.path.exists(path):
                self.emb = np.load(path)
            else:
                m = self._load_model()
                self.emb = m.encode([self._prefix("passage") + c.text for c in self.chunks],
                                    normalize_embeddings=True, batch_size=32, show_progress_bar=False)
                np.save(path, self.emb)
            self.emb = np.ascontiguousarray(self.emb, dtype=np.float32)
        except Exception as e:  # không tải được model -> chỉ dùng BM25
            print(f"[WARN] Dense embedding không khả dụng ({e.__class__.__name__}); chỉ dùng BM25.")
            self.emb = None
            return
        self.backend = "numpy cosine"
        if VECTOR_DB == "faiss":
            try:  # vector đã chuẩn hoá L2 nên Inner Product == Cosine Similarity
                import faiss
                self.faiss_index = faiss.IndexFlatIP(self.emb.shape[1])
                self.faiss_index.add(self.emb)
                self.backend = "FAISS IndexFlatIP (cosine)"
            except ImportError:
                print("[WARN] Chưa cài faiss-cpu; dùng numpy cosine.")

    def _dense_search(self, query, pool):
        q = self._load_model().encode([self._prefix("query") + query], normalize_embeddings=True)[0].astype(np.float32)
        k = min(pool, len(self.chunks))
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(q[None, :], k)
            return {int(i): float(d) for d, i in zip(D[0], I[0]) if i >= 0}, q
        sims = cosine_similarity(q, self.emb)
        return {int(i): float(sims[i]) for i in np.argsort(-sims)[:k]}, q

    def _rerank(self, query, hits, top_k):
        try:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
            sc = self.reranker.predict([(query, h.chunk.text[:2000]) for h in hits])
        except Exception as e:  # không có model/RAM -> giữ nguyên thứ tự cũ
            self.rerank_error = f"{e.__class__.__name__}: {e}"
            return hits
        self.rerank_error = None
        for h, s in zip(hits, sc):
            h.rerank = float(s)
        return sorted(hits, key=lambda h: h.rerank, reverse=True)

    def search(self, query, top_k=6, alpha=0.5, pool=30, mode=None, rerank=False):
        """mode: 'dense' (cosine) | 'bm25' | 'hybrid' (RRF; alpha = trọng số dense). rerank: sắp xếp lại bằng cross-encoder.
        Trả về list[Hit]. Mặc định hybrid nếu có embedding, ngược lại BM25."""
        if not self.chunks:
            return []
        has_dense = self.emb is not None
        mode = mode or ("hybrid" if has_dense else "bm25")
        if mode in ("dense", "hybrid") and not has_dense:
            mode = "bm25"
        self.last_mode = mode
        cand_n = max(top_k * 3, 15) if rerank else top_k
        pool = max(pool, cand_n)
        bm = self.bm25.get_scores(tokenize(query))
        cos, qv, scores = {}, None, {}
        if mode in ("dense", "hybrid"):
            cos, qv = self._dense_search(query, pool)
        if mode in ("bm25", "hybrid"):
            w = (1 - alpha) if mode == "hybrid" else 1.0
            has_pos = bool((bm > 0).any())
            for r, i in enumerate(np.argsort(-bm)[:pool]):
                if bm[i] > 0 or not has_pos:  # kho quá nhỏ -> BM25 có thể toàn 0, vẫn xếp theo thứ tự
                    scores[int(i)] = scores.get(int(i), 0) + w / (60 + r)
        if mode in ("dense", "hybrid"):
            w = alpha if mode == "hybrid" else 1.0
            for r, i in enumerate(cos):
                scores[i] = scores.get(i, 0) + w / (60 + r)
        top = sorted(scores, key=scores.get, reverse=True)[:cand_n]
        hits = []
        for i in top:
            c = cos.get(i)
            if c is None and qv is not None:
                c = float(cosine_similarity(qv, self.emb[i])[0])
            hits.append(Hit(self.chunks[i], scores[i], c, float(bm[i])))
        if rerank:
            hits = self._rerank(query, hits, top_k)
        return hits[:top_k]


# ---------- 4. Các định dạng khác ----------
def _md_table(rows):
    width = max(len(r) for r in rows)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def parse_docx(data: bytes, name: str):
    """Word: mỗi heading cấp 1 coi là 1 'trang' ảo để còn dẫn nguồn được; bảng Word -> Markdown."""
    import docx
    d = docx.Document(io.BytesIO(data))
    pages, buf, pno = [], [], 1

    def flush():
        nonlocal buf, pno
        if buf:
            pages.append({"page": pno, "text": "\n".join(buf).strip(), "tables": [], "ocr": False})
            buf, pno = [], pno + 1

    for el in d.element.body:
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            from docx.text.paragraph import Paragraph
            para = Paragraph(el, d)
            if para.style and para.style.name and para.style.name.lower().startswith("heading") and buf:
                flush()
            if para.text.strip():
                buf.append(para.text.strip())
        elif tag == "tbl":
            from docx.table import Table
            t = Table(el, d)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if rows:
                pages.append({"page": pno, "text": "", "tables": [_md_table(rows)], "ocr": False})
                pno += 1
    flush()
    if not pages:
        pages = [{"page": 1, "text": "", "tables": [], "ocr": False}]
    return pages, dict(_NO_OCR_INFO)


def parse_csv(data: bytes, name: str):
    """CSV: toàn bộ thành 1 bảng Markdown (coi là 1 trang)."""
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return [{"page": 1, "text": "", "tables": [], "ocr": False}], dict(_NO_OCR_INFO)
    return [{"page": 1, "text": "", "tables": [_md_table(rows)], "ocr": False}], dict(_NO_OCR_INFO)


def parse_xlsx(data: bytes, name: str):
    """Excel: mỗi sheet là 1 'trang' (số thứ tự), bảng -> Markdown; ô trống ở cuối dòng/cột được cắt bớt."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    pages = []
    for i, ws in enumerate(wb.worksheets, 1):
        rows = []
        for row in ws.iter_rows(values_only=True):
            if any(v not in (None, "") for v in row):
                rows.append(["" if v is None else str(v) for v in row])
        if rows:
            pages.append({"page": i, "text": f"[Sheet: {ws.title}]", "tables": [_md_table(rows)], "ocr": False})
    if not pages:
        pages = [{"page": 1, "text": "(file Excel rỗng)", "tables": [], "ocr": False}]
    return pages, dict(_NO_OCR_INFO)


class _HTMLPages(HTMLParser):
    """HTML (trang tin nhà đầu tư, báo cáo xuất từ web): h1/h2 mở 'trang' mới; <table> -> bảng Markdown."""
    SKIP = {"script", "style", "noscript", "head", "svg"}
    BLOCK = {"p", "div", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pages, self.buf, self.tables, self.skip = [], [], [], 0
        self.tdepth, self.rows, self.row, self.cell = 0, [], [], None

    def _flush(self):
        text = "".join(self.buf)
        text = "\n".join(l.strip() for l in text.split("\n") if l.strip())
        if text or self.tables:
            self.pages.append({"page": len(self.pages) + 1, "text": text, "tables": self.tables, "ocr": False})
        self.buf, self.tables = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif self.skip:
            return
        elif tag == "table":
            self.tdepth += 1
            if self.tdepth == 1:
                self.rows = []
        elif self.tdepth == 1 and tag in ("td", "th"):
            self.cell = []
        elif self.tdepth == 1 and tag == "tr":
            self.row = []
        elif tag in ("h1", "h2") and not self.tdepth:
            self._flush()
        if tag in self.BLOCK and not self.tdepth:
            self.buf.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif self.skip:
            return
        elif self.tdepth == 1 and tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif self.tdepth == 1 and tag == "tr":
            if any(c for c in self.row):
                self.rows.append(self.row)
            self.row = []
        elif tag == "table" and self.tdepth:
            self.tdepth -= 1
            if self.tdepth == 0 and self.rows:
                self.tables.append(_md_table(self.rows))
        if tag in self.BLOCK and not self.tdepth:
            self.buf.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        if self.tdepth and self.cell is not None:
            self.cell.append(data)
        elif not self.tdepth:
            self.buf.append(data)


def parse_html(data: bytes, name: str):
    p = _HTMLPages()
    p.feed(data.decode("utf-8", errors="replace"))
    p._flush()
    pages = p.pages or [{"page": 1, "text": "", "tables": [], "ocr": False}]
    return pages, dict(_NO_OCR_INFO)


def parse_txt(data: bytes, name: str):
    """File .txt: coi là văn bản đã được OCR/làm sạch từ nơi khác. Trang ngăn bằng dòng \f hoặc '---PAGE n---'."""
    text = data.decode("utf-8", errors="replace")
    parts = re.split(r"\f|^-{0,3}\s*PAGE\s+\d+\s*-{0,3}\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    parts = [p for p in parts if p.strip()] or [text]
    return [{"page": i, "text": p.strip(), "tables": [], "ocr": False} for i, p in enumerate(parts, 1)], dict(_NO_OCR_INFO)


SUPPORTED_EXT = (".pdf", ".docx", ".csv", ".xlsx", ".xls", ".html", ".htm", ".txt")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "40"))

_PARSERS = {
    ".pdf": lambda data, name, cb: parse_pdf(data, name, cb),
    ".docx": lambda data, name, cb: parse_docx(data, name),
    ".csv": lambda data, name, cb: parse_csv(data, name),
    ".xlsx": lambda data, name, cb: parse_xlsx(data, name),
    ".xls": lambda data, name, cb: parse_xlsx(data, name),
    ".html": lambda data, name, cb: parse_html(data, name),
    ".htm": lambda data, name, cb: parse_html(data, name),
    ".txt": lambda data, name, cb: parse_txt(data, name),
}


def build_index(files: dict, use_dense=True, progress=None, chunk_size=None, chunk_overlap=None):
    """files: {tên_file: bytes}. progress(msg, fraction) tuỳ chọn. Trả về (Index, cảnh báo, thống kê mỗi file).
    Tự bỏ qua file lỗi/không hỗ trợ/quá lớn thay vì làm hỏng cả mẻ; mỗi file được báo riêng trong `warns`/`stats`."""
    chunks, warns, stats = [], [], []
    names = list(files)
    seen_hash = {}
    for k, name in enumerate(names):
        ext = os.path.splitext(name)[1].lower()
        data = files[name]
        size_mb = len(data) / 1_000_000
        cb = (lambda i, n, k=k, name=name: progress(f"{name}: trang {i}/{n}", (k + i / n) / len(names))) if progress else None

        if ext not in SUPPORTED_EXT:
            warns.append(f"{name}: bỏ qua — định dạng {ext or '(không rõ)'} chưa hỗ trợ (chỉ nhận PDF, DOCX, CSV, XLSX, HTML, TXT).")
            continue
        if size_mb > MAX_FILE_MB:
            warns.append(f"{name}: bỏ qua — file {size_mb:.0f} MB vượt giới hạn {MAX_FILE_MB} MB.")
            continue
        h = hashlib.md5(data).hexdigest()
        if h in seen_hash:
            warns.append(f"{name}: bỏ qua — nội dung trùng với file '{seen_hash[h]}' đã nạp.")
            continue
        seen_hash[h] = name

        try:
            pages, info = _PARSERS[ext](data, name, cb)
        except Exception as e:
            warns.append(f"{name}: lỗi khi đọc file ({e.__class__.__name__}: {e}) — đã bỏ qua, các file khác vẫn được xử lý.")
            continue

        if info["ocr_pages"]:
            warns.append(f"{name}: {info['ocr_pages']} trang là ảnh scan, đã dùng OCR. OCR có thể đọc sai chữ/số — hãy đối chiếu số liệu với file gốc.")
        if info["ocr_missing"]:
            warns.append(f"{name}: có trang scan nhưng chưa cài Tesseract nên chưa đọc được (cần packages.txt: tesseract-ocr, tesseract-ocr-vie).")
        if info["ocr_skipped"]:
            warns.append(f"{name}: bỏ qua {info['ocr_skipped']} trang scan vượt giới hạn OCR ({MAX_OCR_PAGES} trang/file).")
        total_chars = sum(len(p["text"]) + sum(map(len, p["tables"])) for p in pages)
        # Chỉ cảnh báo khi thực sự gần như trống, không áp cho file ngắn hợp lệ — dùng ngưỡng tuyệt đối.
        if ext == ".pdf" and total_chars < 25 and not info["ocr_pages"]:
            warns.append(f"{name}: gần như không có nội dung đọc được — file có thể bị lỗi, mã hoá lạ, hoặc là PDF scan cần OCR.")
        elif ext != ".pdf" and total_chars < 10:
            warns.append(f"{name}: gần như không có nội dung đọc được — file có thể bị lỗi hoặc rỗng.")

        start = len(chunks)
        file_chunks = make_chunks(pages, name, start_id=start, size=chunk_size, overlap=chunk_overlap)
        if info.get("header_lines"):  # giữ lại thông tin đầu trang (tên công ty, 'báo cáo riêng/hợp nhất', kỳ báo cáo) đúng 1 lần
            file_chunks.insert(0, Chunk(start, name, pages[0]["page"], "text",
                                        "[Tiêu đề/đầu trang lặp lại ở các trang của tài liệu]\n" + "\n".join(info["header_lines"]), "Đầu trang"))
            for j, c in enumerate(file_chunks):
                c.id = start + j
        chunks += file_chunks
        stats.append({"name": name, "pages": len(pages), "chunks": len(file_chunks),
                      "tables": sum(c.kind == "table" for c in file_chunks), "ocr_pages": info["ocr_pages"]})
    if progress:
        progress("Đang tạo index...", 1.0)
    return Index(chunks, use_dense), warns, stats
