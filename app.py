import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
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

# Streamlit App
st.title("Analisis Sentimen Komentar Mobil Listrik")

# Input Teks dari Pengguna
user_input = st.text_area("Masukkan komentar:", placeholder="Tulis komentar di sini...")

if st.button("Analisis Sentimen"):
    if user_input.strip():  # Pastikan input tidak kosong
        # Lakukan preprocessing pada input pengguna
        processed_input = preprocess_text(user_input)
        # Prediksi sentimen menggunakan model
        prediction = model.predict(processed_input)
        # Pemetaan label hasil prediksi ke kategori
        sentiment_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        # Tampilkan hasil prediksi ke pengguna
        st.write(f"Hasil Prediksi Sentimen: **{sentiment_map[prediction[0]]}**")
    else:
        st.error("Silakan masukkan teks untuk analisis.")

# Sidebar
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini menggunakan Machine Learning untuk menganalisis sentimen komentar terkait mobil listrik.")
