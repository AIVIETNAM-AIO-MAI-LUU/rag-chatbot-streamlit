import streamlit as st
import tempfile
import os
import time
import pypdf
import chromadb
import ollama


# =========================
# 1. Cấu hình model
# =========================

LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "bge-m3"


PROMPT = """
Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:
"""


# =========================
# 2. Khởi tạo session_state
# =========================

for key, value in {
    "collection": None,
    "pdf_name": "",
    "chat_history": []
}.items():
    st.session_state.setdefault(key, value)


# =========================
# 3. Hàm embedding
# =========================

def embed(texts):
    """
    Chuyển danh sách text thành vector embedding.
    """
    return ollama.embed(
        model=EMBED_MODEL,
        input=texts
    )["embeddings"]


# =========================
# 4. Hàm chunking
# =========================

def chunk_text(text, size=1000, overlap=200):
    """
    Cắt văn bản thành nhiều chunk nhỏ.
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    cur = ""

    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())

            if overlap:
                cur = cur[-overlap:] + p + "\n"
            else:
                cur = p + "\n"

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


# =========================
# 5. Xử lý PDF upload
# =========================

def process_pdf(uploaded_file):
    """
    Đọc PDF, chunking, embedding và lưu vào ChromaDB.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    text = "\n".join(
        page.extract_text() or ""
        for page in pypdf.PdfReader(path).pages
    )

    os.unlink(path)

    chunks = chunk_text(text, size=1000, overlap=200)

    client = chromadb.Client()

    collection_name = f"rag_{int(time.time())}"
    collection = client.get_or_create_collection(collection_name)

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks)
    )

    return collection, len(chunks), len(text)


# =========================
# 6. Hàm RAG
# =========================

def rag(question, collection, k=4):
    """
    Tìm context liên quan và gọi LLM trả lời.
    """
    res = collection.query(
        query_embeddings=embed([question]),
        n_results=k
    )

    retrieved_docs = res["documents"][0]
    context = "\n\n".join(retrieved_docs)

    final_prompt = PROMPT.format(
        context=context,
        question=question
    )

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    answer = resp["message"]["content"]

    return answer, retrieved_docs


# =========================
# 7. Giao diện Streamlit
# =========================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("PDF RAG Assistant")

with st.sidebar:
    st.subheader("Upload tài liệu")

    uploaded_pdf = st.file_uploader(
        "Chọn file PDF",
        type=["pdf"]
    )

    if uploaded_pdf and st.button("Xử lý PDF", use_container_width=True):
        with st.spinner("Đang đọc PDF, chunking và tạo vector database..."):
            collection, n_chunks, n_chars = process_pdf(uploaded_pdf)

            st.session_state.collection = collection
            st.session_state.pdf_name = uploaded_pdf.name
            st.session_state.chat_history = []

        st.success(f"Đã xử lý {n_chunks} chunks")
        st.write(f"Tổng ký tự đọc được: {n_chars}")

    if st.session_state.pdf_name:
        st.info(f"File hiện tại: {st.session_state.pdf_name}")
    else:
        st.info("Chưa có tài liệu")

    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []


# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Ô nhập câu hỏi
if st.session_state.collection is None:
    st.info("Vui lòng upload và xử lý PDF trước khi chat.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
else:
    question = st.chat_input("Nhập câu hỏi của bạn...")

    if question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm tài liệu liên quan và trả lời..."):
                answer, sources = rag(
                    question,
                    st.session_state.collection,
                    k=4
                )

                st.write(answer)

                with st.expander("Xem các đoạn tài liệu được dùng làm context"):
                    for i, source in enumerate(sources, start=1):
                        st.markdown(f"### Chunk {i}")
                        st.write(source)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )