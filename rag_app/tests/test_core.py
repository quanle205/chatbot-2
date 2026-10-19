"""Kiểm thử logic cốt lõi (không cần mạng/LLM/model). Chạy: python -m pytest -q   hoặc   python tests/test_core.py"""
import os, sys, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag, llm, eval as ev
from rag import Chunk


class FakeModel:  # embedding "bag-of-words băm" 128 chiều, đủ để kiểm tra đường đi dense/cosine/FAISS
    def encode(self, texts, normalize_embeddings=True, **kw):
        out = np.zeros((len(texts), 128), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in rag.tokenize(t.replace("query: ", "").replace("passage: ", "")):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)


def make_index():
    docs = ["Doanh thu thuần quý I năm 2026 đạt 167.923.170.830 đồng",
            "Tổng tài sản tại ngày 31/03/2026 là 33.067.716.327.546 đồng",
            "Vay và nợ thuê tài chính ngắn hạn 6.327.887.440.200 đồng",
            "Giải trình biến động lợi nhuận sau thuế tăng do doanh thu hoạt động tài chính"]
    idx = rag.Index([Chunk(i, "a.pdf", i + 1, "text", t) for i, t in enumerate(docs)], use_dense=False)
    idx.model = FakeModel()
    idx.emb = idx.model.encode(["passage: " + c.text for c in idx.chunks])
    try:
        import faiss
        idx.faiss_index = faiss.IndexFlatIP(128); idx.faiss_index.add(idx.emb)
    except ImportError:
        pass
    return idx


def test_cosine_similarity_matches_formula():
    rng = np.random.default_rng(0)
    q, M = rng.normal(size=16), rng.normal(size=(5, 16))
    exp = [q @ m / (np.linalg.norm(q) * np.linalg.norm(m)) for m in M]
    assert np.allclose(rag.cosine_similarity(q, M), exp, atol=1e-5)
    assert np.allclose(rag.cosine_similarity(q, q * 3), 1.0, atol=1e-5)


def test_search_modes_and_hit_unpacking():
    idx = make_index()
    for mode in ("bm25", "dense", "hybrid"):
        hits = idx.search("tổng tài sản ngày 31/03/2026", top_k=2, mode=mode)
        assert hits[0].chunk.id == 1, mode
        c, s = hits[0]  # tương thích code cũ
        assert c is hits[0].chunk and isinstance(s, float)
    d = idx.search("tổng tài sản", top_k=3, mode="dense")
    assert all(h.cos is not None for h in d) and d[0].cos >= d[-1].cos
    # cosine của FAISS (nếu có) trùng cosine tính tay
    q = idx.model.encode(["query: tổng tài sản"])[0]
    assert abs(d[0].cos - float(rag.cosine_similarity(q, idx.emb[d[0].chunk.id])[0])) < 1e-5


def test_search_falls_back_to_bm25_without_dense():
    idx = rag.Index([Chunk(0, "a", 1, "text", "doanh thu thuần 100.000")], use_dense=False)
    assert idx.search("doanh thu", mode="dense")[0].chunk.id == 0 and idx.last_mode == "bm25"


def test_strip_noise_keeps_units_amounts_and_removes_header_and_pagenum():
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    pages = [{"page": i, "text": f"CÔNG TY CỔ PHẦN ABC\nĐơn vị: VND\nNội dung {words[i-1]}\nTổng cộng {i}.234.567\n{i}", "tables": [], "ocr": False}
             for i in range(1, 8)]
    pages, removed = rag.strip_noise(pages)
    t = pages[2]["text"]
    assert "CÔNG TY CỔ PHẦN ABC" not in t and "Đơn vị: VND" in t and "Nội dung gamma" in t
    assert "Tổng cộng 3.234.567" in t  # dòng số tiền lặp theo mẫu vẫn được giữ
    assert not t.strip().endswith("\n3") and removed == ["CÔNG TY CỔ PHẦN ABC"]


def test_semantic_chunking_splits_on_headings_and_labels_sections():
    text = "THUYẾT MINH 15. DOANH THU\n" + "Doanh thu bán hàng tăng nhẹ. " * 12 + "\n2. Giá vốn hàng bán\n" + "Giá vốn giảm do chi phí. " * 12
    parts = rag._split_text(text, size=500, overlap=50)
    assert len(parts) >= 2 and parts[0][0].startswith("THUYẾT MINH") and any("Giá vốn" in s for s, _ in parts)
    ch = rag.make_chunks([{"page": 1, "text": text, "tables": [], "ocr": False}], "d.pdf", size=500, overlap=50)
    assert all(c.id == i for i, c in enumerate(ch)) and any(c.text.startswith("[Mục:") or c.text.startswith("THUYẾT") for c in ch)
    assert not rag.is_heading("1. Tiền 129.748.721.940")  # dòng số liệu không phải tiêu đề


def test_parse_html_tables_and_headings():
    html = b"<html><head><title>x</title></head><body><h1>Bao cao</h1><p>Doanh thu tang.</p><table><tr><th>Chi tieu</th><th>2026</th></tr><tr><td>Doanh thu</td><td>1.000</td></tr></table><h2>Muc 2</h2><p>Hello</p></body></html>"
    pages, _ = rag.parse_html(html, "a.html")
    assert len(pages) == 2 and "Doanh thu tang" in pages[0]["text"] and "| Doanh thu | 1.000 |" in pages[0]["tables"][0]


def test_ir_metrics():
    m = ev.ir_metrics([5, 1, 2, 9], {1, 2, 3}, k=4)
    assert abs(m["precision"] - 0.5) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9 and abs(m["rr"] - 0.5) < 1e-9 and m["hit"] == 1.0
    assert ev.ir_metrics([7, 8], {1}, 2)["f1"] == 0.0


def test_chunk_matching_and_question_loading():
    c = Chunk(0, "FPT.pdf", 4, "table", "| Tổng cộng tài sản | 33.067.716.327.546 |")
    assert ev.chunk_matches(c, {"pages": [], "snippets": ["33,067,716,327,546"]})  # khác dấu ngăn cách vẫn khớp
    assert ev.chunk_matches(c, {"pages": ["FPT.pdf:4"], "snippets": []}) and not ev.chunk_matches(c, {"pages": ["FPT.pdf:5"], "snippets": []})
    rows = ev.load_questions(os.path.join(os.path.dirname(__file__), "questions_sample.csv"))
    assert len(rows) == 12 and rows[5]["pages"] == ["FPT_Baocaotaichinh_Q1_2026_Congtyme.pdf:7"]
    s, d, skipped = ev.retrieval_eval(make_index(), [{"id": "1", "question": "tổng tài sản", "snippets": ["33.067.716.327.546"], "pages": [], "numbers": [], "gold_answer": ""}],
                                      k=2, modes=("bm25", "dense", "hybrid"))
    assert [r["mode"] for r in s] == ["bm25", "dense", "hybrid"] and all(r["MRR"] == 1.0 for r in s) and not skipped


def test_grounding_check():
    hits = [(Chunk(0, "a", 1, "table", "Doanh thu thuần 167.923.170.830; Lợi nhuận gộp 134.766.471.688"), 1.0)]
    ok = llm.grounding_check("Doanh thu thuần là 167.923.170.830 VND [#1], tức khoảng 167,9 tỷ đồng, kỳ 31/03/2026, năm 2026.", hits)
    assert ok["ungrounded"] == [] and ok["checked"] == 2
    calc = llm.grounding_check("Biên lợi nhuận gộp 80,25% [#1]", hits, [("134766471688/167923170830*100", "80.254255")])
    assert calc["ungrounded"] == []
    bad = llm.grounding_check("Doanh thu là 250.000.000.000 đồng [#3]", hits)
    assert bad["ungrounded"] == ["250.000.000.000"] and bad["invalid_citations"] == [3]
    assert ev.numbers_correct("Tổng tài sản 33.067,7 tỷ đồng", ["33.067.716.327.546"]) == 1.0


def test_safe_calc():
    assert llm.safe_calc("134766471688/167923170830*100").startswith("80.254")
    assert llm.safe_calc("__import__('os')").startswith("Lỗi")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok ", n)
