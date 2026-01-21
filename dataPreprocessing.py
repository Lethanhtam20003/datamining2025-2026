# Data Preprocessing Module
# 1. làm sạch dữ liệu
# 2. chuẩn hóa dữ liệu
# 3. chuyển đổi dữ liệu : xử lý tiếng việt 
# 4. loại bỏ stopwords
import pandas as pd
import re

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    # kiểm tra và loại bỏ các dòng bị thiếu
    df.dropna(subset=['Text'], inplace=True)
    return df
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ đường dẫn URL (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Loại bỏ thẻ HTML (nếu có)
    text = re.sub(r'<.*?>', '', text)
    
    # Giữ lại các ký tự chữ cái Tiếng Việt, Tiếng Anh và khoảng trắng
    text = re.sub(r"[^a-zA-Z0-9àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý\s]", '', text)
    
    # Loại bỏ khoảng trắng thừa (ví dụ: "  alo   " -> "alo")
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# chay thu
if __name__ == "__main__":
    # ví dụ sử dụng
    path = "comments.csv"
    df = load_data(path)
    df['cleaned_text'] = df['Text'].apply(clean_text)
    print(df[['Text', 'cleaned_text']].head())