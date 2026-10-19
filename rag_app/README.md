# Chatbot hỏi đáp BCTC (RAG) — Streamlit

Hệ thống bám theo *Kế hoạch nghiên cứu RAG – BCTC Vinamilk Q1/2026* (đổi `COMPANY` để dùng cho công ty/báo cáo khác).

| Mục kế hoạch | Cài đặt trong code |
|---|---|
| Thu thập & tiền xử lý (PDF/Excel/HTML), loại header/footer (Tuần 2) | `rag.py`: `parse_pdf` (+OCR trang scan), `parse_xlsx`, `parse_html`, `parse_docx`, `parse_csv`, `strip_noise` |
| Chunking theo ngữ nghĩa, tinh chỉnh chunk size (Tuần 3, 8) | `rag.py`: `_split_text` (cắt theo tiêu đề mục/thuyết minh, gắn nhãn `[Mục: …]`), thanh trượt chunk size/overlap trong app |
| Embedding BERT (Tuần 4) | `multilingual-e5-small` (XLM-R); đổi bằng `EMBED_MODEL` (vd. `BAAI/bge-m3`) |
| Vector DB + Cosine Similarity (Tuần 4) | FAISS `IndexFlatIP` trên vector chuẩn hoá L2 (= cosine); hàm `cosine_similarity` cài tay trong `rag.py`; `VECTOR_DB=numpy` để dùng bản thuần numpy |
| Retrieval: Dense / BM25 / Hybrid, Re-ranking (Tuần 8) | `Index.search(mode=…, rerank=…)`; cross-encoder `RERANK_MODEL` |
| Generation + Prompt Engineering chuyên gia tài chính (Tuần 5) | `llm.py`: `SYSTEM` (trích dẫn `[#n]`, phân biệt kỳ/phạm vi, tính toán bằng công cụ, nêu nguyên nhân chỉ khi tài liệu có) |
| Giao diện chat + trích dẫn nguồn (Tuần 6) | `app.py` (hiện điểm cosine, xem trang PDF gốc) |
| Kiểm tra hallucination | `llm.grounding_check` (đối chiếu số liệu với ngữ cảnh) + nút "So sánh với LLM thuần" trong app |
| Đánh giá IR: Precision, Recall, F1, MRR (Tuần 7, mục 4.2) | tab "📈 Đánh giá IR" trong app hoặc `python eval.py` |
| Đánh giá sinh: Faithfulness, Answer Relevance, giảm hallucination (mục 4.3) | `python eval.py --gen --judge` |

## Triển khai (Streamlit Community Cloud)
1. Đưa thư mục này lên repo GitHub **public**, bỏ BCTC (PDF/Excel/HTML) vào `data/`.
2. share.streamlit.io → New app → main file `app.py` → Advanced settings → Secrets:
   ```
   LLM_API_KEY = "khoa-gemini-cua-ban"
   COMPANY = "Công ty Cổ phần Sữa Việt Nam (Vinamilk)"
   # LLM_MODEL = "gemini-3.8-flash"
   # EMBED_MODEL = "intfloat/multilingual-e5-small"
   # APP_PASSWORD = "mat-khau-neu-muon-khoa"
   ```
3. Deploy. Không bao giờ commit API key vào repo.

**Lưu ý RAM:** embedding + torch cần khoảng 0,8–1 GB; gói miễn phí của Streamlit Cloud (≈1 GB) có thể bị quá tải. Nếu app bị khởi động lại, đặt `USE_DENSE = "0"` trong Secrets (chỉ BM25) hoặc chạy trên máy/Hugging Face Spaces. Re-ranking nặng hơn nữa nên mặc định tắt (`USE_RERANK=1` để bật).

## PDF scan (dạng ảnh)
App tự OCR những trang không có text (cần `packages.txt` để Streamlit Cloud cài Tesseract tiếng Việt). OCR chậm và có thể đọc sai chữ/số — luôn đối chiếu với file gốc.
Biến tuỳ chọn: `OCR_DPI` (mặc định 300), `MAX_OCR_PAGES` (80), `CHUNK_SIZE` (900), `CHUNK_OVERLAP` (150).

## Đánh giá (Tuần 7–8)
```
python tests/test_core.py                                              # kiểm thử logic, không cần mạng
python eval.py --data data --questions tests/questions.csv --k 6       # Precision/Recall/F1/MRR: BM25 vs Dense vs Hybrid vs +Re-rank
python eval.py --data data --questions tests/questions.csv --gen --judge --out report
```
`tests/questions_sample.csv` là 12 câu mẫu trên BCTC công ty mẹ FPT Q1/2026 — hãy soạn 50–100 câu cho Vinamilk theo cùng định dạng (cột `relevant_snippets` là số liệu/đoạn chữ phải có trong đoạn đúng). Câu nào không tìm thấy đáp án gold trong kho sẽ được báo riêng (thường do OCR đọc sai số).
`halluc_rate` = tỷ lệ câu trả lời có ≥1 số liệu không khớp ngữ cảnh truy xuất (dung sai 0,5%, chấp nhận đổi đơn vị); "Giảm hallucination" = 1 − RAG/LLM thuần.
