import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load Model dan TF-IDF Vectorizer
model = joblib.load('best_model_svm.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Inisialisasi Stemmer dan Stopword Remover dari Sastrawi
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Fungsi Preprocessing
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove special characters (hanya huruf dan spasi yang disimpan)
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenize (pecah teks menjadi kata-kata)
    tokens = word_tokenize(text)
    # 4. Remove stopwords
    filtered_tokens = [stopword_remover.remove(token) for token in tokens]
    # 5. Stemming (mengubah kata ke bentuk dasar)
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    # 6. Gabungkan kembali token menjadi string
    processed_text = ' '.join(stemmed_tokens)
    # 7. Transformasi ke bentuk fitur numerik menggunakan TF-IDF
    return vectorizer.transform([processed_text])

# Streamlit App Layout
st.set_page_config(page_title="Analisis Sentimen Komentar Mobil Listrik", layout="wide")

# Header
st.markdown(
    """
    <style>
        .header-container {
            background-color: #333333;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header-title {
            color: #ffffff;
            font-family: 'Arial', sans-serif;
            font-size: 28px;
            font-weight: bold;
        }
        .header-subtitle {
            color: #bbbbbb;
            font-family: 'Arial', sans-serif;
            font-size: 16px;
        }
    </style>
    <div class="header-container">
        <h1 class="header-title">Analisis Sentimen Komentar Mobil Listrik</h1>
        <p class="header-subtitle">Aplikasi ini menggunakan Machine Learning untuk menganalisis komentar terkait mobil listrik.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Kolom Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Masukkan Komentar Anda")
    user_input = st.text_area("Tulis komentar di sini...", placeholder="Contoh: Mobil listrik ini sangat inovatif dan ramah lingkungan.")
    
    if st.button("Analisis Sentimen"):
        if user_input.strip():  # Pastikan input tidak kosong
            # Lakukan preprocessing pada input pengguna
            processed_input = preprocess_text(user_input)
            # Prediksi sentimen menggunakan model
            prediction = model.predict(processed_input)
            # Pemetaan label hasil prediksi ke kategori
            sentiment_map = {0: ('Negatif', '#ffcccc', 'red'), 1: ('Netral', '#f0f0f0', 'black'), 2: ('Positif', '#ccffcc', 'green')}
            sentiment_label, sentiment_bg, sentiment_color = sentiment_map[prediction[0]]
            
            # Tampilkan hasil prediksi dengan warna yang sesuai
            st.markdown(
                f"""
                <div style="background-color: {sentiment_bg}; color: {sentiment_color}; 
                            padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;">
                    Hasil Prediksi Sentimen: {sentiment_label}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Silakan masukkan komentar terlebih dahulu.")

with col2:
    st.subheader("Sentimen Kategori")
    st.markdown(
        """
        <ul style="font-size: large;">
            <li><b style="color: red;">Negatif:</b> Komentar dengan penilaian negatif.</li>
            <li><b style="color: black;">Netral:</b> Komentar yang bersifat biasa atau tidak memihak.</li>
            <li><b style="color: green;">Positif:</b> Komentar dengan penilaian positif.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px; color: #888;">
        🚀 Dikembangkan untuk membantu memahami opini masyarakat tentang mobil listrik.
    </div>
    """,
    unsafe_allow_html=True
)
