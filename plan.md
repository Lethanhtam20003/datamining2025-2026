# Kế hoạch triển khai - Khai phá dữ liệu (Sentiment Analysis)

**Mục tiêu:** Chuyển đổi các bình luận thô trên mạng xã hội thành tri thức về cảm xúc khách hàng, sử dụng mô hình **Naive Bayes**.

---

## Giai đoạn 1 — Xác định dữ liệu & mục tiêu (Data Definition)

- **Dữ liệu đầu vào:** `comments.csv` (cột `Text` chứa dữ liệu văn bản phi cấu trúc).
- **Mục tiêu:** Phân loại bình luận thành 3 lớp: **Khen (1)**, **Chê (0)**, **Trung tính (2)**.
- **Kỹ thuật chính:** Phân loại có giám sát (Supervised Learning).

---

## Giai đoạn 2 — Tiền xử lý dữ liệu (Data Preprocessing)
Mục tiêu của bước này là làm sạch và chuẩn hóa dữ liệu để nâng cao chất lượng đầu vào.

1. **Làm sạch thô (Data Cleaning)**
   - Xóa URL, thẻ HTML, ký tự đặc biệt, và các thành phần không mang nghĩa.
   - Mục đích: loại bỏ nhiễu trước khi trích xuất đặc trưng.

2. **Chuẩn hóa ngôn ngữ & cảm xúc (Data Transformation)**
   - Xử lý slang và tiếng lóng (ví dụ: `ko`, `đc`, `v`...) — quy về từ điển chuẩn.
   - Chuẩn hóa từ tiếng Anh mang cảm xúc (nice, good, bad, chill...) sang tương đương tiếng Việt để thống nhất tập đặc trưng.
   - **Xử lý emoji:** trích xuất thành các đặc trưng số (ví dụ: `num_emoji_pos`, `num_emoji_neg`) — là chứng cứ quan trọng cho Naive Bayes.

3. **Tách từ tiếng Việt (Tokenization)**
   - Sử dụng `underthesea` để nhận diện từ ghép (ví dụ: `khai_phá_dữ_liệu`), giữ được ý nghĩa cụm từ.

4. **Loại bỏ stopwords (Tiếng Việt & Tiếng Anh)**
   - Xóa các từ ít giá trị phân biệt (ví dụ: là, của, và, the, an, a...).

---

## Giai đoạn 3 — Biến đổi & Vector hóa (Transformation & Vectorization)
Chuyển văn bản thành vector số để mô hình có thể xử lý.

1. **Vector hóa (TF-IDF)**
   - Tính trọng số từ theo tần suất và độ đặc thù.
   - Sử dụng **N-gram (1, 2)** để giữ ngữ cảnh (ví dụ: "không thích" giữ nguyên cụm).

2. **Chuẩn hóa thang đo (Scaling)**
   - Chuẩn hóa các đặc trưng số (ví dụ: số lượng emoji) về khoảng `[0, 1]` để cân bằng với đặc trưng văn bản.

3. **Tích hợp dữ liệu (Data Integration)**
   - Hợp nhất ma trận từ vựng với các đặc trưng emoji thành ma trận đặc trưng cuối cùng `X_final`.

---

## Giai đoạn 4 — Xây dựng mô hình Naive Bayes (Modeling)

1. **Gán nhãn (Labeling)**
   - Sử dụng heuristic (luật dựa trên emoji và từ khóa) để tạo nhãn lớp mục tiêu.

2. **Chia tập dữ liệu (Holdout Method)**
   - Chia 80% train / 20% test để đánh giá khả năng dự đoán.

3. **Huấn luyện (Training)**
   - Chọn biến thể Naive Bayes phù hợp cho dữ liệu rời rạc; áp dụng làm trơn **Laplace** để xử lý từ chưa từng xuất hiện.

---

## Giai đoạn 5 — Đánh giá & tối ưu (Evaluation)

1. **Ma trận nhầm lẫn (Confusion Matrix)** — kiểm tra kiểu nhầm lẫn (ví dụ: mỉa mai bị phân loại là khen).
2. **Các chỉ số:** tính độ chính xác theo lớp và tổng thể (F1-score, v.v.).
3. **Cải tiến:** nếu kết quả chưa đạt, quay lại Giai đoạn 2 để mở rộng từ điển slang hoặc danh sách từ tiếng Anh cần chuyển đổi.

> **Ghi chú (Gia sư Data Mining):** Việc xử lý tốt các từ tiếng Anh thông dụng sẽ giúp thu hẹp không gian thuộc tính và cải thiện xác suất dự đoán đúng! 🎯

