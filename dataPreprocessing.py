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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# ================= STOPWORDS TIẾNG VIỆT =================
with open('stopWords_vietnamese.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = set(line.strip() for line in f if line.strip())

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
    # Slang tiếng Việt
    "k": "không", "ko": "không", "kh": "không", "v": "vậy",
    "đc": "được", "dc": "được", "r": "rồi", "s": "sao",
    "mn": "mọi người", "ae": "anh em", "mìh": "mình", "mik": "mình",
    "tr": "trời", "j": "gì", "bt": "biết", "kb": "không biết", "h": "giờ",
    # Từ tiếng Anh biểu cảm -> Tiếng Việt
    "good": "tốt", "great": "tuyệt", "awesome": "tuyệt_vời", "excellent": "xuất_sắc",
    "bad": "tệ", "sucks": "tệ", "terrible": "tồi_tệ", "worst": "tồi_tệ",
    "like": "thích", "love": "yêu", "hate": "ghét", "dislike": "không_thích",
    "nice": "tốt", "cool": "ngầu", "chill": "thoải_mái", "fun": "vui",
    "boring": "nhàm_chán", "sad": "buồn", "happy": "vui", "excited": "hào_hứng",
    "amazing": "kinh_ngạc", "wonderful": "tuyệt_vời", "fantastic": "tuyệt_vời",
    "horrible": "kinh_khủng", "awful": "tồi_tệ", "disgusting": "ghê_tởm",
    "perfect": "hoàn_hảo", "super": "siêu", "best": "tốt_nhất",
    "okay": "ổn", "ok": "ổn", "fine": "tốt", "alright": "ổn",
    "shit": "tệ", "damn": "chết_tiệt", "fuck": "chết_tiệt"
}


def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV
    Loại bỏ các dòng bị thiếu nội dung văn bản
    """
    df = pd.read_csv(file_path)
    df.dropna(subset=['Text'], inplace=True)
    return df

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

def tokenize_vietnamese(text):
    """
    Tách từ tiếng Việt bằng thư viện underthesea
    Trả về list các từ đã tách
    """
    if not isinstance(text, str):
        return []
    return word_tokenize(text, format="text").split()


def negation_transformation(word_list):
    """
    Thực hiện Negation Transformation: ghép từ phủ định với từ đứng sau
    Ví dụ: ["không", "hài_lòng"] -> ["không_hài_lòng"]
    """
    if not word_list:
        return word_list
    
    negation_words = ["không", "chưa", "chẳng", "chả", "chớ", "đừng", "không_thích", "chưa_bao_giờ"]
    transformed = []
    i = 0
    while i < len(word_list):
        if i < len(word_list) - 1 and word_list[i] in negation_words:
            # Ghép negation với từ tiếp theo
            transformed.append(word_list[i] + "_" + word_list[i+1])
            i += 2  # Bỏ qua từ tiếp theo vì đã ghép
        else:
            transformed.append(word_list[i])
            i += 1
    return transformed

def remove_stopwords(text):
    """
    Loại bỏ stopwords tiếng Việt khỏi văn bản
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

# ================= LABELING HEURISTIC =================
POS_WORDS = ["tốt", "hay", "thích", "tuyệt", "love", "good", "nice", "excellent", "tuyệt_vời", "ưng", "ok", "oki", "tuyệt", "hài_lòng", "ưng_ý"]
NEG_WORDS = ["tệ", "dở", "ghét", "kém", "bad", "hate", "worst", "terrible", "chán", "buồn", "không_thích", "tồi_tệ", "không_hài_lòng"]

def assign_label(row):
    """
    Gán nhãn dựa trên heuristic: emoji > từ khóa + xử lý phủ định
    1 = Khen, 0 = Chê, 2 = Trung tính
    """
    # Ưu tiên emoji
    if row['num_emoji_pos'] > 0:
        return 1  # Khen
    elif row['num_emoji_neg'] > 0:
        return 0  # Chê
    elif row['num_emoji_neu'] > 0:
        return 2  # Trung tính
    
    # Nếu không có emoji, kiểm tra từ khóa với xử lý phủ định
    text = str(row['cleaned_text']).lower()
    
    # Tách từ
    words = text.split()
    
    # Kiểm tra bigram phủ định
    negation_words = ["không", "không_thích", "chẳng", "chưa", "ko"]
    pos_words = ["tốt", "hay", "thích", "tuyệt", "tuyệt_vời", "ưng", "ổn", "xuất_sắc", "hoàn_hảo"]
    neg_words = ["tệ", "dở", "ghét", "kém", "tồi_tệ", "nhàm_chán", "buồn", "ghê_tởm"]
    
    # Kiểm tra phủ định + từ tích cực = tiêu cực
    for i in range(len(words) - 1):
        if words[i] in negation_words and words[i+1] in pos_words:
            return 0  # Chê (không tốt = tệ)
        if words[i] in negation_words and words[i+1] in neg_words:
            return 1  # Khen (không tệ = tốt)
    
    # Kiểm tra từ khóa đơn giản
    if any(word in pos_words for word in words):
        return 1
    elif any(word in neg_words for word in words):
        return 0
    else:
        return 2  # Mặc định trung tính

# ================= MAIN =================
if __name__ == "__main__":
    df = load_data("data/comments.csv")

    # 1. Trích xuất đặc trưng Emoji (Phải làm TRƯỚC khi clean_text xóa ký tự đặc biệt)
    df['emoji_pos'] = df['Text'].apply(lambda x: extract_emoji(x, POS_EMOJIS))
    df['emoji_neg'] = df['Text'].apply(lambda x: extract_emoji(x, NEG_EMOJIS))
    df['emoji_neu'] = df['Text'].apply(lambda x: extract_emoji(x, NEU_EMOJIS))
    
    # 2. Tính toán đặc trưng số (Numeric Features)
    df['num_emoji_pos'] = df['emoji_pos'].apply(len)
    df['num_emoji_neg'] = df['emoji_neg'].apply(len)
    df['num_emoji_neu'] = df['emoji_neu'].apply(len)

    # 3. Pipeline Tiền xử lý văn bản (Thứ tự tối ưu)
    # Bước a: Xử lý slang trước để chuẩn hóa từ ngữ cho underthesea
    df['cleaned_text'] = df['Text'].apply(convert_slang)
    # Bước b: Làm sạch (Xóa URL, HTML, ký tự đặc biệt...)
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    # Bước c: Tách từ tiếng Việt 
    df['tokenized'] = df['cleaned_text'].apply(tokenize_vietnamese)
    # Bước d: Xử lý phủ định
    df['tokenized'] = df['tokenized'].apply(negation_transformation)
    # Bước e: Ghép lại và loại bỏ Stopwords
    df['cleaned_text'] = df['tokenized'].apply(lambda x: " ".join(x))
    # df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)  # Tạm bỏ để test

    # 4. Gán nhãn (Labeling)
    df['label'] = df.apply(assign_label, axis=1)
    # Lọc bỏ samples trống sau preprocessing
    df = df[df['cleaned_text'].str.len() > 0].copy()
    # 5. Vector hóa văn bản bằng TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, max_df=1.0)
    X_tfidf = tfidf.fit_transform(df['cleaned_text']).toarray()

    # 6. CHUẨN HÓA (NORMALIZATION) - Bước tối ưu quan trọng 
    # Đưa các cột số lượng emoji về cùng thang đo [0, 1] như TF-IDF
    scaler = MinMaxScaler()
    emoji_numeric = df[['num_emoji_pos', 'num_emoji_neg', 'num_emoji_neu']].values
    emoji_scaled = scaler.fit_transform(emoji_numeric)

    # 7. TÍCH HỢP DỮ LIỆU (Data Integration) 
    # Kết hợp vector từ vựng và vector emoji đã chuẩn hóa
    X_final = np.hstack([X_tfidf, emoji_scaled])

    print("Kích thước đặc trưng cuối cùng:", X_final.shape)
    print("Phân bố nhãn:", df['label'].value_counts())
    df.to_csv("data/comments_final_optimized.csv", index=False, encoding='utf-8-sig')    # ================= LABELING HEURISTIC =================
    POS_WORDS = ["tốt", "hay", "thích", "tuyệt", "love", "good", "nice", "excellent", "tuyệt_vời", "ưng", "ok", "oki"]
    NEG_WORDS = ["tệ", "dở", "ghét", "kém", "bad", "hate", "worst", "terrible", "chán", "buồn", "không_thích"]
    
    def assign_label(row):
        """
        Gán nhãn dựa trên heuristic: emoji > từ khóa
        """
        # Ưu tiên emoji
        if row['num_emoji_pos'] > 0:
            return 1  # Khen
        elif row['num_emoji_neg'] > 0:
            return 0  # Chê
        elif row['num_emoji_neu'] > 0:
            return 2  # Trung tính
        
        # Nếu không có emoji, kiểm tra từ khóa
        text = row['cleaned_text'].lower()
        if any(word in text for word in POS_WORDS):
            return 1
        elif any(word in text for word in NEG_WORDS):
            return 0
        else:
            return 2  # Mặc định trung tính
    
    # Thêm vào pipeline chính:
    df['label'] = df.apply(assign_label, axis=1)    
    
    # Sau khi có nhãn và X_final
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Huấn luyện
    nb_model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    nb_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = nb_model.predict(X_test)

    # Đánh giá cơ bản
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Chê (0)', 'Khen (1)', 'Trung tính (2)']))