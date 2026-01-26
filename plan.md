
KẾ HOẠCH TRIỂN KHAI KHAI PHÁ DỮ LIỆU (SENTIMENT ANALYSIS)
Dự án này tập trung vào việc chuyển đổi các bình luận thô trên mạng xã hội thành tri thức về cảm xúc khách hàng sử dụng thuật toán Naive Bayes.
Giai đoạn 1: Xác định Dữ liệu & Mục tiêu (Data Definition)
    • Dữ liệu đầu vào: Tệp comments.csv chứa cột Text (dữ liệu phi cấu trúc).
    • Mục tiêu: Phân loại comment thành 3 lớp: Khen (1), Chê (0), Trung tính (2).
    • Kỹ thuật chính: Phân loại có giám sát (Supervised Learning).
Giai đoạn 2: Tiền xử lý dữ liệu (Data Preprocessing)
Đây là bước làm sạch nhiễu (noise) để tăng chất lượng dữ liệu đầu vào.
    1. Làm sạch thô (Data Cleaning): * Xóa bỏ các thành phần không mang nghĩa: URL, thẻ HTML, ký tự đặc biệt.
        ○ Ý nghĩa: Loại bỏ các yếu tố gây nhiễu cho quá trình tính toán xác suất.
    2. Chuẩn hóa ngôn ngữ và cảm xúc (Data Transformation):
        ○ Xử lý Slang & Tiếng Anh thông dụng: * Quy đổi ngôn ngữ chat (ko, đc, v...) về từ điển chuẩn.
            § Chuẩn hóa ngôn ngữ hỗn hợp: Chuyển các từ tiếng Anh mang tính biểu cảm (nice, good, bad, chill...) sang tiếng Việt tương đương để thống nhất tập đặc trưng.
        ○ Xử lý Emoji: Trích xuất các biểu tượng cảm xúc thành các đặc trưng số (num_emoji_pos, num_emoji_neg). Đây là các "chứng cứ" quan trọng cho định lý Bayes.
    3. Tách từ tiếng Việt (Tokenization): * Sử dụng underthesea để nhận diện từ ghép (ví dụ: khai_phá_dữ_liệu). Bước này giúp máy hiểu được ngữ nghĩa cụm từ thay vì các từ đơn rời rạc.
    4. Loại bỏ Stopwords (Tiếng Việt & Tiếng Anh):
        ○ Xóa các từ xuất hiện nhiều nhưng không mang giá trị phân biệt cảm xúc (là, của, và, the, an, a...).
Giai đoạn 3: Biến đổi & Vector hóa (Transformation & Vectorization)
Chuyển đổi dữ liệu văn bản thành các vector số để máy tính có thể tính toán.
    1. Vector hóa văn bản (TF-IDF):
        ○ Tính toán trọng số từ dựa trên tần suất xuất hiện và độ hiếm của từ đó.
        ○ Sử dụng $N-gram (1, 2)$ để giữ lại ngữ cảnh (ví dụ: "không thích" thay vì tách rời "không" và "thích").
    2. Chuẩn hóa thang đo (Scaling):
        ○ Sử dụng kỹ thuật đưa các cột số lượng Emoji về cùng khoảng $[0, 1]$, đảm bảo tính công bằng với các giá trị văn bản trong mô hình toán học.
    3. Tích hợp dữ liệu (Data Integration):
        ○ Hợp nhất ma trận từ vựng và các đặc trưng emoji thành một ma trận đặc trưng cuối cùng X_final.
Giai đoạn 4: Xây dựng Mô hình Naive Bayes (Modeling)
Áp dụng định lý xác suất để xây dựng bộ phân lớp.
    1. Gán nhãn (Labeling): * Sử dụng phương pháp Heuristic (luật dựa trên kinh nghiệm) từ Emoji và từ khóa để tạo nhãn lớp mục tiêu.
    2. Chia tập dữ liệu (Holdout Method):
        ○ Chia dữ liệu thành 80% để học (Train) và 20% để kiểm tra khả năng dự đoán (Test).
    3. Huấn luyện (Training):
        ○ Sử dụng biến thể phù hợp cho dữ liệu rời rạc với kỹ thuật làm trơn Laplace để tránh lỗi khi gặp những từ ngữ mới chưa từng xuất hiện.
Giai đoạn 5: Đánh giá & Tối ưu (Evaluation)
    1. Ma trận nhầm lẫn (Confusion Matrix): Soi xét xem máy đang nhầm lẫn ở đâu (ví dụ: có nhầm lời mỉa mai thành lời khen không?).
    2. Đo lường hiệu quả: * Kiểm tra độ chuẩn xác trên từng lớp cảm xúc và điểm số tổng thể (F1-Score).
    3. Cải tiến: Nếu kết quả thấp, quay lại Giai đoạn 2 để cập nhật thêm từ điển Slang hoặc danh sách từ tiếng Anh cần chuyển đổi.
Gia sư Data Mining nhắc nhở: Việc xử lý tốt các từ tiếng Anh thông dụng sẽ giúp mô hình của bạn thu hẹp được không gian thuộc tính và tăng xác suất dự đoán đúng đáng kể!
