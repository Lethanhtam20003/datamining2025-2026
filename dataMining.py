# ================= DATA MINING MODULE =================
# Giai đoạn 4: Xây dựng Mô hình Naive Bayes
# Giai đoạn 5: Đánh giá & Tối ưu

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import từ dataPreprocessing
from dataPreprocessing import (
    load_data, extract_emoji, convert_slang, clean_text,
    tokenize_vietnamese, remove_stopwords, assign_label,
    POS_EMOJIS, NEG_EMOJIS, NEU_EMOJIS, STOPWORDS
)

def prepare_data():
    """
    Chuẩn bị dữ liệu: load, preprocess, vectorize, label
    """
    df = load_data("comments.csv")

    # Pipeline tiền xử lý
    df['emoji_pos'] = df['Text'].apply(lambda x: extract_emoji(x, POS_EMOJIS))
    df['emoji_neg'] = df['Text'].apply(lambda x: extract_emoji(x, NEG_EMOJIS))
    df['emoji_neu'] = df['Text'].apply(lambda x: extract_emoji(x, NEU_EMOJIS))

    df['num_emoji_pos'] = df['emoji_pos'].apply(len)
    df['num_emoji_neg'] = df['emoji_neg'].apply(len)
    df['num_emoji_neu'] = df['emoji_neu'].apply(len)

    df['cleaned_text'] = df['Text'].apply(convert_slang)
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].apply(tokenize_vietnamese)
    df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

    # Gán nhãn
    df['label'] = df.apply(assign_label, axis=1)

    # Vector hóa
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_tfidf = tfidf.fit_transform(df['cleaned_text']).toarray()

    scaler = MinMaxScaler()
    emoji_numeric = df[['num_emoji_pos', 'num_emoji_neg', 'num_emoji_neu']].values
    emoji_scaled = scaler.fit_transform(emoji_numeric)

    X_final = np.hstack([X_tfidf, emoji_scaled])

    return X_final, df['label'], df, tfidf, scaler

def train_naive_bayes(X, y):
    """
    Giai đoạn 4.2-4.3: Chia tập và huấn luyện Naive Bayes
    """
    # Chia tập dữ liệu (phương pháp Holdout)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train labels distribution: {y_train.value_counts().to_dict()}")
    print(f"Test labels distribution: {y_test.value_counts().to_dict()}")

    # Huấn luyện Naive Bayes với làm trơn Laplace
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = nb_model.predict(X_test)

    return nb_model, X_train, X_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """
    Giai đoạn 5: Đánh giá mô hình
    """
    print("\n" + "="*50)
    print("PHASE 5: EVALUATION & OPTIMIZATION")
    print("="*50)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    print("\n1. CONFUSION MATRIX:")
    print("Predicted: 0=Chê, 1=Khen, 2=Trung tính")
    print("Actual")
    print(cm)

    # Trực quan hóa
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Chê (0)', 'Khen (1)', 'Trung tính (2)'],
                yticklabels=['Chê (0)', 'Khen (1)', 'Trung tính (2)'])
    plt.title('Ma trận nhầm lẫn - Phân tích cảm xúc')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Báo cáo phân loại
    print("\n2. CLASSIFICATION REPORT:")
    target_names = ['Chê (0)', 'Khen (1)', 'Trung tính (2)']
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    # Các chỉ số tổng thể
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"Độ chính xác: {accuracy:.2f}")
    print(f"F1 Macro: {f1_macro:.2f}")
    print(f"F1 Weighted: {f1_weighted:.2f}")
    # Phân tích theo lớp
    print("\n4. PER-CLASS ANALYSIS:")
    for i, label in enumerate(target_names):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{label}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return accuracy, f1_macro, f1_weighted

def save_model(model, filename='naive_bayes_model.pkl'):
    """
    Lưu mô hình để sử dụng sau
    """
    joblib.dump(model, filename)
    print(f"\nModel saved as {filename}")

# ================= MAIN =================
if __name__ == "__main__":
    print("Starting Sentiment Analysis Pipeline...")

    # Giai đoạn 4: Modeling
    print("\n" + "="*50)
    print("PHASE 4: NAIVE BAYES MODELING")
    print("="*50)

    # Chuẩn bị dữ liệu
    X, y, df, tfidf, scaler = prepare_data()
    print(f"Data shape: {X.shape}")
    print(f"Labels distribution: {y.value_counts()}")

    # Huấn luyện mô hình
    model, X_train, X_test, y_train, y_test, y_pred = train_naive_bayes(X, y)

    # Giai đoạn 5: Đánh giá
    accuracy, f1_macro, f1_weighted = evaluate_model(y_test, y_pred)

    # Lưu mô hình
    save_model(model)

    print("\n" + "="*50)
    print("PIPELINE COMPLETED!")
    print("Files generated: confusion_matrix.png, naive_bayes_model.pkl")
    print("="*50)
