import streamlit as st
import joblib
import re
import nltk
import pandas as pd  # <-- TAMBAHKAN IMPORT INI

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# FIX NLTK DOWNLOAD
# ===============================
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    """Cache model untuk performa lebih baik"""
    try:
        model = joblib.load("bernoulli_nb_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        return model, tfidf
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.error("Pastikan file berikut ada di folder yang sama:")
        st.error("1. bernoulli_nb_model.pkl")
        st.error("2. tfidf_vectorizer.pkl")
        return None, None

model, tfidf = load_model()

# Cek jika model berhasil dimuat
if model is None or tfidf is None:
    st.stop()

# Cek model info
st.sidebar.markdown("### ℹ️ Info Model")
st.sidebar.write(f"Model type: {type(model).__name__}")
st.sidebar.write(f"Classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")
st.sidebar.write(f"Vocabulary size: {len(tfidf.vocabulary_) if hasattr(tfidf, 'vocabulary_') else 'N/A'}")

# ===============================
# PREPROCESSING
# ===============================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cache stopwords to avoid repeated calls
@st.cache_data
def get_stopwords():
    """Mendapatkan stopwords bahasa Indonesia"""
    try:
        return set(stopwords.words('indonesian'))
    except:
        # Fallback jika stopwords tidak ditemukan
        return set(['yang', 'di', 'dan', 'untuk', 'pada', 'ke', 'dari', 'ini', 'itu', 'dengan'])

stop_words = get_stopwords()

@st.cache_data
def preprocess(text):
    """Preprocessing teks dengan caching"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-z\s]', '', text)
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    
    # Stemming dan filtering
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# Mapping label
label_map = {
    0: "Negatif 😡",
    1: "Netral 😐", 
    2: "Positif 😊"
}

# ===============================
# PREDICTION FUNCTION
# ===============================
def analyze_prediction(text):
    """Analisis lengkap prediksi"""
    if not text.strip():
        return None
    
    clean_text = preprocess(text)
    
    if not clean_text.strip():
        return {
            'original': text,
            'cleaned': clean_text,
            'prediction': 1,  # Default netral jika teks kosong setelah preprocessing
            'label': label_map[1],
            'tokens': [],
            'vocab_status': {},
            'note': 'Teks tidak mengandung kata bermakna setelah preprocessing'
        }
    
    vector = tfidf.transform([clean_text])
    prediction = model.predict(vector)[0]
    
    analysis = {
        'original': text,
        'cleaned': clean_text,
        'prediction': prediction,
        'label': label_map[prediction],
        'tokens': clean_text.split() if clean_text else []
    }
    
    # Cek probabilitas jika ada
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(vector)[0]
            analysis['probabilities'] = {
                'negatif': float(proba[0]),
                'netral': float(proba[1]),
                'positif': float(proba[2])
            }
        except:
            pass
    
    # Cek apakah tokens ada di vocabulary
    vocab_status = {}
    if hasattr(tfidf, 'vocabulary_'):
        for token in analysis['tokens']:
            vocab_status[token] = token in tfidf.vocabulary_
    analysis['vocab_status'] = vocab_status
    
    return analysis

# ===============================
# UI (FRONT END)
# ===============================
st.title("📊 Analisis Sentimen Roblox Indonesia")
st.write("Model: **Bernoulli Naive Bayes + TF-IDF**")
st.markdown("---")

# Sidebar diagnostics
with st.sidebar:
    st.markdown("### 🔍 Diagnosa")
    
    if st.button("Test Model Sederhana", type="secondary"):
        st.markdown("#### Hasil Test:")
        test_cases = [
            ("jelek", "Negatif (0)"),
            ("buruk", "Negatif (0)"),
            ("payah", "Negatif (0)"),
            ("biasa", "Netral (1)"),
            ("bagus", "Positif (2)"),
            ("baik", "Positif (2)")
        ]
        
        for text, expected in test_cases:
            analysis = analyze_prediction(text)
            if analysis:
                # Ekstrak angka dari expected: "Negatif (0)" -> "0"
                expected_num = expected.split('(')[1].split(')')[0]
                match = "✅" if str(analysis['prediction']) == expected_num else "❌"
                st.write(f"{match} **'{text}'**")
                st.write(f"  Prediksi: {analysis['label']}")
                st.write(f"  Expected: {expected}")
                st.write(f"  Cleaned: `{analysis['cleaned']}`")
                st.write("---")

# Main input
st.markdown("### ✏️ Masukkan Teks untuk Dianalisis")
text_input = st.text_area(
    "Komentar Platform X tentang Roblox:",
    placeholder="Contoh: Game roblox makin seru setelah update terbaru!",
    height=120,
    key="input_text"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

if predict_button:
    if not text_input or text_input.strip() == "":
        st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Menganalisis sentimen..."):
            try:
                # Analisis lengkap
                analysis = analyze_prediction(text_input)
                
                if not analysis:
                    st.error("Gagal menganalisis teks.")
                    st.stop()
                
                # Tampilkan hasil utama
                st.markdown("---")
                st.subheader("🎯 Hasil Analisis")
                
                # Tampilkan label dengan warna
                if analysis['prediction'] == 0:
                    st.error(f"## {analysis['label']}")
                elif analysis['prediction'] == 1:
                    st.warning(f"## {analysis['label']}")
                else:
                    st.success(f"## {analysis['label']}")
                
                # Detail analysis
                with st.expander("📊 Detail Analisis Lengkap", expanded=False):
                    st.markdown("**📝 Teks Asli:**")
                    st.info(analysis['original'])
                    
                    st.markdown("**🔧 Teks Hasil Preprocessing:**")
                    st.code(analysis['cleaned'])
                    
                    st.markdown("**🧩 Tokens:**")
                    if analysis['tokens']:
                        st.write(", ".join(f"`{token}`" for token in analysis['tokens']))
                    else:
                        st.write("Tidak ada tokens (teks mungkin hanya berisi stopwords)")
                    
                    # Vocabulary check
                    if analysis['vocab_status']:
                        st.markdown("**📚 Vocabulary Check:**")
                        in_vocab = [t for t, status in analysis['vocab_status'].items() if status]
                        not_in_vocab = [t for t, status in analysis['vocab_status'].items() if not status]
                        
                        if in_vocab:
                            st.success(f"✅ {len(in_vocab)} token ada dalam vocabulary")
                        
                        if not_in_vocab:
                            st.warning(f"⚠️ {len(not_in_vocab)} token tidak ada dalam vocabulary:")
                            st.write(", ".join(f"`{token}`" for token in not_in_vocab))
                    
                    # Probabilities
                    if 'probabilities' in analysis:
                        st.markdown("**📈 Probabilitas Klasifikasi:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            neg_prob = analysis['probabilities']['negatif']
                            st.metric(
                                label="Negatif", 
                                value=f"{neg_prob:.1%}"
                            )
                        
                        with col2:
                            neu_prob = analysis['probabilities']['netral']
                            st.metric(
                                label="Netral", 
                                value=f"{neu_prob:.1%}"
                            )
                        
                        with col3:
                            pos_prob = analysis['probabilities']['positif']
                            st.metric(
                                label="Positif", 
                                value=f"{pos_prob:.1%}"
                            )
                        
                        # Bar chart untuk probabilitas - PERBAIKAN DI SINI
                        prob_data = pd.DataFrame({
                            'Sentimen': ['Negatif', 'Netral', 'Positif'],
                            'Probabilitas': [neg_prob, neu_prob, pos_prob]
                        })
                        
                        # Atur index dan buat chart
                        prob_data_chart = prob_data.set_index('Sentimen')
                        st.bar_chart(prob_data_chart)
                
                # Catatan jika ada
                if 'note' in analysis:
                    st.info(analysis['note'])
                    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
                st.error("Silakan coba lagi atau periksa input teks Anda.")

# Footer
st.markdown("---")
st.caption("Aplikasi Analisis Sentimen menggunakan Bernoulli Naive Bayes • Model dilatih pada data komentar Roblox Indonesia")
