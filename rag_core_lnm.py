import pypdf
import chromadb
import ollama
LLM_MODEL = "vicuna:7b-v1.5-q5_1"
#Đọc nội dung file pdf
reader = pypdf.PdfReader("./sample.pdf")

#Ghép nội dung tất cả các trang thành 1 chuỗi text
full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

print("Số trang:", len(reader.pages))
print("Tổng ký tự:", len(full_text))
print(full_text[:1000])     #Kiểm tra text đầu ra có bị lỗi không

#Chunking
def chunk_text(text, size=1000, overlap=200):
    """
    Cắt text thành các đoạn nhỏ.

    text: văn bản đầy đủ từ PDF
    size: độ dài tối đa của mỗi chunk, tính theo ký tự
    overlap: số ký tự lặp lại giữa 2 chunk liên tiếp
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


chunks = chunk_text(full_text, size=1000, overlap=200)

print("Số chunks:", len(chunks))
print("Chunk đầu tiên:")
print(chunks[0][:500])  # Xem 500 ký tự đầu của chunk đầu tiên


#Embedding và lưu vào Vector Database
# Hàm tạo embedding từ danh sách text
def embed(texts):
    """Chuyển danh sách chuỗi text thành danh sách vector."""
    return ollama.embed(model="bge-m3", input=texts)["embeddings"]

# Tạo vector database trong bộ nhớ
client = chromadb.Client()
collection = client.get_or_create_collection("rag")

# Thêm tất cả chunks vào database
collection.add(
    ids=[str(i) for i in range(len(chunks))], # ID duy nhất cho mỗi chunk
    documents=chunks, # Nội dung text gốc
    embeddings=embed(chunks), # Vector tương ứng
)
print("Đã index:", collection.count(), "chunks")

#Tìm kiếm đoạn liên quan (Retrieve)
def retrieve(query, k=4):
    res = collection.query(query_embeddings=embed([query]), n_results=k)
    return res["documents"][0]

#Thử tìm kiếm
query = "Thang điểm MH-STRALP là gì?"

docs = retrieve(query, k=4)

for i, doc in enumerate(docs, start=1):
    print(f"--- Chunk {i} ---")
    print(doc[:500])


#Hỏi đáp với LLM (RAG)
PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

def rag(question, k=4):
    context = "\n\n".join(retrieve(question, k))
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        options={"temperature": 0},
    )
    return resp["message"]["content"]

print(rag("What is this document about?"))