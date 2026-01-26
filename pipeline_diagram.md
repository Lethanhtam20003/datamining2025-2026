# Sơ đồ Quy trình Sentiment Analysis với Naive Bayes

Dựa trên kế hoạch trong `plan.md`, dưới đây là sơ đồ biểu diễn quy trình tổng thể của dự án:

```
[START]
    |
    v
+-------------------+     +-------------------+
| Giai đoạn 1       |     | Xác định Dữ liệu  |
| Data Definition   |     | - comments.csv    |
|                   |     | - Mục tiêu: 3 lớp |
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Giai đoạn 2       | --> | Tiền xử lý        |
| Preprocessing     |     | - Clean text      |
|                   |     | - Slang -> chuẩn  |
|                   |     | - Emoji extract   |
|                   |     | - Tokenize        |
|                   |     | - Remove stopwords|
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Giai đoạn 3       | --> | Vector hóa         |
| Transformation    |     | - TF-IDF (n-gram) |
|                   |     | - Scaling emoji   |
|                   |     | - Data Integration|
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Giai đoạn 4       | --> | Modeling          |
| Naive Bayes       |     | - Labeling        |
|                   |     |   (Heuristic)     |
|                   |     | - Split train/test|
|                   |     | - Train NB        |
|                   |     |   (Laplace)       |
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Giai đoạn 5       | --> | Đánh giá          |
| Evaluation        |     | - Confusion Matrix|
|                   |     | - F1-Score        |
|                   |     | - Tối ưu nếu cần  |
+-------------------+     +-------------------+
    |
    v
[END - Tri thức về cảm xúc khách hàng]
```

## **Giải thích Sơ đồ**:

1. **Luồng tuần tự**: Mỗi giai đoạn phụ thuộc vào giai đoạn trước.
2. **Chi tiết từng bước**: Mô tả ngắn gọn công việc chính.
3. **Đầu ra**: Từ dữ liệu thô → mô hình dự đoán sentiment.

## **Sơ đồ Chi tiết Giai đoạn 4 (Modeling)**:

```
[Data đã xử lý (X_final, df)]
    |
    v
+-------------------+     +-------------------+
| Labeling          | --> | Heuristic Rules   |
| (Gán nhãn)        |     | - Emoji priority  |
|                   |     | - Keyword check   |
|                   |     | - Default: Neutral|
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Split Data        | --> | Holdout Method    |
| (Chia tập)        |     | - 80% Train       |
|                   |     | - 20% Test        |
|                   |     | - Stratified      |
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Training          | --> | MultinomialNB     |
| (Huấn luyện)      |     | - Laplace Smoothing|
|                   |     | - Fit on Train    |
+-------------------+     +-------------------+
    |
    v
+-------------------+     +-------------------+
| Prediction        | --> | Predict on Test   |
| (Dự đoán)         |     | - y_pred          |
+-------------------+     +-------------------+
    |
    v
[Evaluation Metrics]
```

Nếu bạn muốn sơ đồ dạng hình ảnh (PNG/SVG) hoặc chi tiết hơn, tôi có thể tạo code Python để vẽ bằng matplotlib hoặc graphviz. Bạn muốn biểu diễn phần nào cụ thể hơn không?