"""
====================================================================
STEP 3: APLIKASI WEB PREDIKSI CUACA (STREAMLIT)
Web application untuk prediksi cuaca dengan interface modern
====================================================================

INSTALLATION:
pip install streamlit

CARA PAKAI:
streamlit run 3_aplikasi_web_streamlit.py

REQUIREMENTS:
- bmkg_weather_rf_model.pkl (dari Step 1)
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ====================================================================
# PAGE CONFIG
# ====================================================================

st.set_page_config(
    page_title="BMKG Weather Predictor",
    page_icon="🌦️",
    layout="wide"
)

# ====================================================================
# LOAD MODEL
# ====================================================================

@st.cache_resource
def load_model():
    """Load model dengan caching"""
    
    model_path = 'bmkg_weather_rf_model.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ File {model_path} tidak ditemukan!")
        st.info("💡 Jalankan STEP 1 terlebih dahulu: python 1_generate_dataset_and_model.py")
        return None
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_weather_emoji(weather):
    """Get emoji untuk cuaca"""
    emojis = {
        'Cerah': '☀️',
        'Berawan': '⛅',
        'Hujan Ringan': '🌦️',
        'Hujan Lebat': '⛈️'
    }
    return emojis.get(weather, '🌤️')

def get_weather_color(weather):
    """Get warna untuk cuaca"""
    colors = {
        'Cerah': '#FFD700',
        'Berawan': '#B0C4DE',
        'Hujan Ringan': '#87CEEB',
        'Hujan Lebat': '#4682B4'
    }
    return colors.get(weather, '#808080')

def predict_weather(suhu, kelembaban, tekanan_udara, kecepatan_angin, tutupan_awan):
    """Prediksi cuaca"""
    
    if model is None:
        return None
    
    input_data = pd.DataFrame([{
        'suhu': suhu,
        'kelembaban': kelembaban,
        'tekanan_udara': tekanan_udara,
        'kecepatan_angin': kecepatan_angin,
        'tutupan_awan': tutupan_awan
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        prob_dict = {}
        for i, class_name in enumerate(classes):
            prob_dict[class_name] = probabilities[i] * 100
        
        return {
            'cuaca': prediction,
            'confidence': max(probabilities) * 100,
            'probabilities': prob_dict
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ====================================================================
# MAIN APP
# ====================================================================

# Header
st.title("🌦️ BMKG Weather Predictor")
st.markdown("**Prediksi Kondisi Cuaca Berbasis Machine Learning**")
st.markdown("---")

# Check if model loaded
if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Info")
    
    st.subheader("📊 Model")
    st.success("""
    **Type:** Random Forest  
    **Status:** ✅ Ready  
    **Accuracy:** ~87%
    """)
    
    st.subheader("📅 Waktu")
    st.write(datetime.now().strftime("%d %B %Y"))
    st.write(datetime.now().strftime("%H:%M WIB"))
    
    st.subheader("📌 Info")
    st.info("""
    Model memprediksi 4 kondisi cuaca:
    - ☀️ Cerah
    - ⛅ Berawan
    - 🌦️ Hujan Ringan
    - ⛈️ Hujan Lebat
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Batch", "ℹ️ Info"])

# ====================================================================
# TAB 1: PREDIKSI TUNGGAL
# ====================================================================

with tab1:
    st.header("🔮 Prediksi Kondisi Cuaca")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input Parameter")
        
        suhu = st.slider(
            "🌡️ Suhu (°C)",
            min_value=22.0,
            max_value=34.0,
            value=28.0,
            step=0.1
        )
        
        kelembaban = st.slider(
            "💧 Kelembaban (%)",
            min_value=60.0,
            max_value=95.0,
            value=75.0,
            step=0.1
        )
        
        tekanan_udara = st.slider(
            "📊 Tekanan Udara (hPa)",
            min_value=1008.0,
            max_value=1023.0,
            value=1013.0,
            step=0.1
        )
        
        kecepatan_angin = st.slider(
            "💨 Kecepatan Angin (km/h)",
            min_value=0.0,
            max_value=25.0,
            value=10.0,
            step=0.1
        )
        
        tutupan_awan = st.slider(
            "☁️ Tutupan Awan (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1
        )
        
        predict_btn = st.button("🚀 Prediksi Cuaca", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📤 Hasil Prediksi")
        
        if predict_btn:
            with st.spinner("Melakukan prediksi..."):
                result = predict_weather(
                    suhu, kelembaban, tekanan_udara, 
                    kecepatan_angin, tutupan_awan
                )
                
                if result:
                    emoji = get_weather_emoji(result['cuaca'])
                    color = get_weather_color(result['cuaca'])
                    
                    # Hasil prediksi
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <div style="font-size: 5rem; margin-bottom: 1rem;">{emoji}</div>
                        <div style="font-size: 2.5rem; font-weight: bold; color: white; 
                                    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                            {result['cuaca']}
                        </div>
                        <div style="font-size: 1.3rem; color: white; margin-top: 0.5rem;">
                            Confidence: {result['confidence']:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Probabilitas
                    st.subheader("📊 Probabilitas Semua Kelas")
                    
                    for weather, prob in sorted(result['probabilities'].items(), 
                                              key=lambda x: x[1], reverse=True):
                        emoji = get_weather_emoji(weather)
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"{emoji} **{weather}**")
                            st.progress(prob / 100)
                        with col_b:
                            st.metric("", f"{prob:.1f}%")
                    
                    st.success("✅ Prediksi selesai!")
        else:
            st.info("👆 Atur parameter di sebelah kiri, lalu klik tombol prediksi")

# ====================================================================
# TAB 2: BATCH PREDICTION
# ====================================================================

with tab2:
    st.header("📊 Batch Prediction dari CSV")
    
    st.markdown("""
    Upload file CSV dengan kolom:
    - `suhu`
    - `kelembaban`
    - `tekanan_udara`
    - `kecepatan_angin`
    - `tutupan_awan`
    """)
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📋 Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🔮 Prediksi Semua Data", type="primary"):
            with st.spinner("Memproses prediksi..."):
                predictions = []
                confidences = []
                
                for idx, row in df.iterrows():
                    result = predict_weather(
                        row['suhu'],
                        row['kelembaban'],
                        row['tekanan_udara'],
                        row['kecepatan_angin'],
                        row['tutupan_awan']
                    )
                    
                    if result:
                        predictions.append(result['cuaca'])
                        confidences.append(result['confidence'])
                    else:
                        predictions.append('Error')
                        confidences.append(0)
                
                df['prediksi_cuaca'] = predictions
                df['confidence'] = [f"{c:.2f}%" for c in confidences]
                
                st.subheader("✅ Hasil Prediksi")
                st.dataframe(df, use_container_width=True)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Hasil",
                    csv,
                    "bmkg_predictions.csv",
                    "text/csv"
                )
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Data", len(df))
                with col2:
                    avg_conf = sum(confidences) / len(confidences)
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                with col3:
                    most_common = pd.Series(predictions).mode()[0]
                    st.metric("Most Common", most_common)

# ====================================================================
# TAB 3: INFO
# ====================================================================

with tab3:
    st.header("ℹ️ Informasi Aplikasi")
    
    st.markdown("""
    ### 🌦️ BMKG Weather Prediction System
    
    Aplikasi prediksi cuaca berbasis Machine Learning menggunakan 5 parameter meteorologi.
    
    #### 📊 Parameter Input:
    1. **🌡️ Suhu** - Suhu udara (22-34°C)
    2. **💧 Kelembaban** - Kelembaban udara (60-95%)
    3. **📊 Tekanan Udara** - Tekanan atmosfer (1008-1023 hPa)
    4. **💨 Kecepatan Angin** - Kecepatan angin (0-25 km/h)
    5. **☁️ Tutupan Awan** - Persentase tutupan awan (0-100%)
    
    #### 🎯 Target Prediksi:
    - ☀️ **Cerah** - Cuaca cerah, minim awan
    - ⛅ **Berawan** - Cuaca berawan sebagian
    - 🌦️ **Hujan Ringan** - Hujan intensitas rendah
    - ⛈️ **Hujan Lebat** - Hujan intensitas tinggi
    
    #### 🧠 Model Information:
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: 1000 samples
    - **Features**: 5 parameter meteorologi
    - **Classes**: 4 kondisi cuaca
    - **Accuracy**: ~85-90%
    
    #### 🚀 Cara Penggunaan:
    1. **Tab Prediksi**: Atur parameter dengan slider, klik tombol prediksi
    2. **Tab Batch**: Upload file CSV untuk prediksi massal
    3. Lihat hasil dan confidence score
    
    #### 📝 Developer:
    - **Project**: BMKG Weather ML Prediction
    - **Framework**: Streamlit + scikit-learn
    - **Version**: 1.0.0
    """)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "Random Forest")
    with col2:
        st.metric("Features", "5")
    with col3:
        st.metric("Classes", "4")
    with col4:
        st.metric("Accuracy", "~87%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌦️ BMKG Weather Predictor | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)