import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load Model dan Vectorizer
model = joblib.load('./models/best_mode_svm.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

# Inisialisasi Stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Fungsi Preprocessing
def preprocess_text(text):
    tokenized_text = word_tokenize(text.lower())
    stemmed_text = ' '.join([stemmer.stem(word) for word in tokenized_text])
    return vectorizer.transform([stemmed_text])

# Streamlit App
st.title("Analisis Sentimen Komentar Mobil Listrik")

# Input Teks
user_input = st.text_area("Masukkan komentar:", placeholder="Tulis komentar di sini...")
if st.button("Analisis Sentimen"):
    if user_input.strip():
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        sentiment_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        st.write(f"Hasil Prediksi Sentimen: **{sentiment_map[prediction[0]]}**")
    else:
        st.error("Silakan masukkan teks untuk analisis.")

st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini menggunakan Machine Learning untuk menganalisis sentimen komentar terkait mobil listrik.")
