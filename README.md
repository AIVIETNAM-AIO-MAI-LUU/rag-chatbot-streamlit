**PDF RAG Chatbot**

Ứng dụng hỏi đáp tài liệu PDF bằng tiếng Việt, xây dựng theo kiến trúc RAG (Retrieval-Augmented Generation). Bạn upload một file PDF, hệ thống sẽ đọc, cắt nhỏ, tạo vector embedding và lưu vào vector database. Sau đó bạn có thể đặt câu hỏi và mô hình ngôn ngữ sẽ trả lời dựa trên đúng nội dung tài liệu — hạn chế bịa đặt.

Toàn bộ hệ thống chạy offline trên máy của bạn thông qua Ollama, không cần gọi API bên ngoài.

**Tính năng**

Upload và xử lý file PDF ngay trên giao diện web.

Tự động cắt văn bản thành các đoạn (chunk) có phần chồng lấp để giữ ngữ cảnh.

Tạo vector embedding bằng mô hình bge-m3 (hỗ trợ tốt tiếng Việt).

Lưu trữ và truy vấn vector bằng ChromaDB.

Sinh câu trả lời bằng mô hình LLM chạy local qua Ollama.

Giao diện chat trực quan với Streamlit, có lưu lịch sử hội thoại.

Hiển thị các đoạn tài liệu được dùng làm ngữ cảnh cho mỗi câu trả lời (minh bạch nguồn).

**Yêu cầu hệ thống**

Python 3.9 trở lên

Ollama đã được cài đặt và đang chạy

RAM khuyến nghị: 8GB trở lên (tùy mô hình LLM)
