# Dự Án Phân Tích Cảm Xúc (Sentiment Analysis) Bằng Naive Bayes

## 📋 Tổng Quan Dự Án

Dự án này triển khai hệ thống phân tích cảm xúc từ bình luận trên mạng xã hội bằng thuật toán Naive Bayes. Hệ thống xử lý dữ liệu văn bản tiếng Việt, từ việc thu thập dữ liệu thô đến xây dựng mô hình dự đoán cảm xúc (Khen/Chê/Trung tính).

## 🎯 Mục Tiêu

- **Đầu vào**: Bình luận thô từ YouTube (file `comments.csv`)
- **Đầu ra**: Phân loại cảm xúc thành 3 lớp:
  - `0`: Chê (Negative)
  - `1`: Khen (Positive)  
  - `2`: Trung tính (Neutral)
- **Độ chính xác**: ~71% trên tập test
`
## 🔄 Pipeline Thực Thi

### 1. Thu Thập Dữ Liệu (`getdata.py`)
```python
# Sử dụng YouTube Data API v3
# Lấy tất cả bình luận từ video ID cụ thể
# Xuất ra file CSV với cột: User, Text
```

### 2. Tiền Xử Lý Dữ Liệu (`dataPreprocessing.py`)

#### 2.1 Load & Clean Data
- Đọc file `comments.csv`
- Loại bỏ dòng thiếu nội dung

#### 2.2 Chuẩn Hóa Văn Bản
- **Xử lý Slang**: Chuyển đổi ngôn ngữ chat (ko→không, đc→được)
- **Xử lý Emoji**: Trích xuất và đếm số lượng emoji tích cực/tiêu cực/trung tính
- **Làm sạch**: Xóa URL, HTML, ký tự đặc biệt

#### 2.3 Tokenization & Stopwords
- **Tách từ**: Sử dụng underthesea để tách từ tiếng Việt
- **Negation Transformation**: Ghép từ phủ định với từ tiếp theo (ví dụ: "không hài_lòng" → "không_hài_lòng")
- **Loại stopwords**: Sử dụng danh sách 1943 từ từ file `stopWords_vietnamese.txt`

#### 2.4 Vector hóa
- **TF-IDF**: Với n-gram (1,2), max_features=5000
- **Scaling**: Chuẩn hóa đặc trưng emoji về [0,1]
- **Tích hợp**: Kết hợp thành ma trận đặc trưng cuối cùng

#### 2.5 Gán Nhãn Heuristic
- **Ưu tiên Emoji**: Có emoji → nhãn tương ứng
- **Từ khóa**: Kiểm tra từ tích cực/tiêu cực
- **Phủ định**: Xử lý bigram như "không tốt" → Chê

### 3. Xây Dựng Mô Hình (`dataMining.py`)

#### 3.1 Chia Tập Dữ Liệu
- **Holdout Method**: 80% Train, 20% Test
- **Stratified Split**: Giữ tỷ lệ lớp cân bằng

#### 3.2 Huấn Luyện Naive Bayes
- **Thuật toán**: MultinomialNB với Laplace smoothing (alpha=1.0)
- **Đặc trưng**: TF-IDF + emoji features

#### 3.3 Dự Đoán & Đánh Giá
- **Confusion Matrix**: Ma trận nhầm lẫn    3x3
- **Metrics**: Accuracy, F1-Score (Macro/Weighted)
- **Visualization**: Heatmap confusion matrix

## 📊 Kết Quả Thực Thi

### Phân Bố Dữ Liệu
- **Tổng mẫu**: 2719 bình luận
- **Phân bố nhãn**:
  - Trung tính (2): 65% (1778 mẫu)
  - Khen (1): 29% (831 mẫu)
  - Chê (0): 4% (110 mẫu)

### Hiệu Suất Mô Hình
- **Accuracy**: 71%
- **F1 Macro**: 0.22 (thấp do imbalance)
- **F1 Weighted**: 0.55

### Ma Trận Nhầm Lẫn Cuối Cùng
```
Predicted: 0=Chê, 1=Khen, 2=Trung tính
Actual
[[  0   0  16]  # Chê: Dự đoán sai hoàn toàn
 [  0   8 135]  # Khen: 8/143 đúng (~6%)
 [  0   5 380]] # Trung tính: 380/385 đúng (~99%)
```

## 🔧 Tính Năng Chính

### Xử Lý Ngôn Ngữ
- **Slang Dictionary**: 35+ từ tiếng Anh + slang tiếng Việt
- **Negation Handling**: Ghép từ phủ định (không, chưa, chẳng) với từ cảm xúc
- **Stopwords**: 1943 từ tiếng Việt
- **Tokenization**: underthesea cho ngữ cảnh tiếng Việt

### Tính Năng Đặc Trưng
- **Emoji Analysis**: 3 nhóm emoji (pos/neg/neu)
- **N-gram**: Bigram để giữ ngữ cảnh
- **Normalization**: MinMax scaling cho emoji features

### Mô Hình
- **Naive Bayes**: Multinomial với Laplace smoothing
- **Evaluation**: Confusion matrix + F1-score
- **Persistence**: Lưu mô hình bằng joblib

## 📈 Cải Tiến Tương Lai

1. **Balance Data**: SMOTE oversampling cho lớp thiểu số
2. **Advanced Models**: SVM, BERT cho độ chính xác cao hơn
3. **Feature Engineering**: Thêm POS tagging, sentiment lexicon
4. **Real Labels**: Thu thập dữ liệu labeled thực tế thay vì heuristic

## 👥 Tác Giả

Dự án Data Mining - Phân tích cảm xúc tiếng Việt
