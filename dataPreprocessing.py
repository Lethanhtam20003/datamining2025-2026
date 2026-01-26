# ================= DATA PREPROCESSING MODULE =================
# 1. Làm sạch dữ liệu văn bản
# 2. Chuẩn hóa tiếng Việt
# 3. Xử lý slang (ngôn ngữ chat)
# 4. Tokenize tiếng Việt
# 5. Loại bỏ stopwords
# 6. Trích xuất đặc trưng emoji
# 7. Vector hóa văn bản bằng TF-IDF

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize


# ================= STOPWORDS TIẾNG VIỆT =================
STOPWORDS = set([
    "là", "của", "và", "nhưng", "đã", "đang", "sẽ", "cũng", "cho", "rằng",
    "những", "cái", "con", "thì", "mà", "lại", "với", "tại", "này", "vậy", "ơi", "ạ"
])

# ================= EMOJI GROUPS =================
POS_EMOJIS = [
    '❤️', '❤', '♥️', '🥰', '😍', '😊', '🎉', '🔥', '🤣', '😂', '😅', '👍', '👏',
    '✨', '🌹', '🤩', '🎊', '🎆', '🎇', '🍀', '☘️', '🌟', '😄', '😁', '🙌', '👌',
    '💐', '🌸', '🎤', '🎵', '🎶', '💙', '💗', '💞', '💕', '💖', '💓'
]
NEG_EMOJIS = ['😢', '😭', '😞', '😔', '🥺', '💔', '👎', '😡', '😠', '😤', '💢', '😒', '🙄']
NEU_EMOJIS = ['😐', '😶', '🤔', '🧐', '😬', '😑', '🤫']

# ================= SLANG DICTIONARY =================
SLANG_DICT = {
    "k": "không", "ko": "không", "kh": "không", "v": "vậy",
    "đc": "được", "dc": "được", "r": "rồi", "s": "sao",
    "mn": "mọi người", "ae": "anh em", "mìh": "mình", "mik": "mình",
    "tr": "trời", "j": "gì", "bt": "biết", "kb": "không biết", "h": "giờ"
}


def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV
    Loại bỏ các dòng bị thiếu nội dung văn bản
    """
    df = pd.read_csv(file_path)
    df.dropna(subset=['Text'], inplace=True)
    return df


def extract_emoji(text, emoji_list):
    """
    Trích xuất emoji thuộc một nhóm nhất định từ văn bản
    Dùng để tạo feature: emoji_pos, emoji_neg, emoji_neu
    """
    if not isinstance(text, str):
        return ""
    return "".join([c for c in text if c in emoji_list])


def convert_slang(text):
    """
    Chuyển đổi slang (ngôn ngữ chat) sang tiếng Việt chuẩn
    Ví dụ: 'ko bt j' -> 'không biết gì'
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join([SLANG_DICT.get(w, w) for w in words])


def clean_text(text):
    """
    Làm sạch văn bản:
    - Chuyển về chữ thường
    - Loại bỏ URL
    - Loại bỏ thẻ HTML
    - Giữ lại chữ cái, chữ số và khoảng trắng
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(
        r"[^a-zA-Z0-9àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý\s]",
        '',
        text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_vietnamese(text):
    """
    Tách từ tiếng Việt bằng thư viện underthesea
    Ví dụ: 'rất thích sản phẩm' -> 'rất_thích sản_phẩm'
    """
    if not isinstance(text, str):
        return ""
    return word_tokenize(text, format="text")


def remove_stopwords(text):
    """
    Loại bỏ stopwords tiếng Việt khỏi văn bản
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


# ================= MAIN =================
if __name__ == "__main__":
    # Load dữ liệu
    df = load_data("comments.csv")

    # Trích xuất đặc trưng emoji
    df['emoji_pos'] = df['Text'].apply(lambda x: extract_emoji(x, POS_EMOJIS))
    df['emoji_neg'] = df['Text'].apply(lambda x: extract_emoji(x, NEG_EMOJIS))
    df['emoji_neu'] = df['Text'].apply(lambda x: extract_emoji(x, NEU_EMOJIS))

    # Làm phẳng văn bản gốc
    df['Text'] = df['Text'].apply(
        lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x
    )

    # ===== PIPELINE TIỀN XỬ LÝ =====
    df['cleaned_text'] = df['Text'].apply(convert_slang)
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].apply(tokenize_vietnamese)
    df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

    # ================= TF-IDF =================
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    X_tfidf = tfidf.fit_transform(df['cleaned_text'])
    print("Kích thước TF-IDF:", X_tfidf.shape)

    # ================= EMOJI NUMERIC FEATURES =================
    df['num_emoji_pos'] = df['emoji_pos'].apply(len)
    df['num_emoji_neg'] = df['emoji_neg'].apply(len)
    df['num_emoji_neu'] = df['emoji_neu'].apply(len)

    # ================= FINAL FEATURE MATRIX =================
    X_final = np.hstack([
        X_tfidf.toarray(),
        df[['num_emoji_pos', 'num_emoji_neg', 'num_emoji_neu']].values
    ])

    print("Kích thước feature cuối cùng:", X_final.shape)

    # ================= SAVE FILE =================
    output_name = "comments_final_excel.csv"
    df.to_csv(output_name, index=False, encoding='utf-8-sig')

    print(f"Hoàn thành File '{output_name}' đã sẵn sàng.")
