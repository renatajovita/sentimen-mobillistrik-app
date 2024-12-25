import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import zipfile
import os
import nltk

# Ekstrak file punkt.zip dan punkt_tab.zip 
if not os.path.exists('nltk_data/tokenizers/punkt'):
    os.makedirs('nltk_data/tokenizers', exist_ok=True)
    with zipfile.ZipFile('nltk_data/tokenizers/punkt.zip', 'r') as zip_ref:
        zip_ref.extractall('nltk_data/tokenizers/')
    with zipfile.ZipFile('nltk_data/tokenizers/punkt_tab.zip', 'r') as zip_ref:
        zip_ref.extractall('nltk_data/tokenizers/')

# Tambahkan path nltk_data ke nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Load Model dan TF-IDF Vectorizer
model = joblib.load('best_model_svm.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

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
