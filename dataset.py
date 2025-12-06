# ====================================================================
# GENERATE DATASET BMKG CUACA - UNTUK GOOGLE COLAB
# ====================================================================

import pandas as pd
import numpy as np

# Set random seed untuk reproducibility
np.random.seed(42)

# Generate dataset BMKG
def generate_bmkg_dataset(n_samples=1000):
    """
    Generate dataset cuaca meteorologi BMKG
    """
    data = []
    
    for i in range(n_samples):
        # Generate fitur meteorologi dengan distribusi realistic
        suhu = np.random.uniform(22, 34)  # Suhu Indonesia: 22-34°C
        kelembaban = np.random.uniform(60, 95)  # Kelembaban: 60-95%
        tekanan_udara = np.random.uniform(1008, 1023)  # Tekanan: 1008-1023 hPa
        kecepatan_angin = np.random.uniform(0, 25)  # Angin: 0-25 km/h
        tutupan_awan = np.random.uniform(0, 100)  # Awan: 0-100%
        
        # Logic untuk menentukan cuaca berdasarkan parameter
        # Aturan berdasarkan meteorologi Indonesia
        if kelembaban > 85 and tekanan_udara < 1012 and tutupan_awan > 70:
            cuaca = 'Hujan Lebat'
        elif kelembaban > 75 and tutupan_awan > 60 and tekanan_udara < 1015:
            cuaca = 'Hujan Ringan'
        elif tutupan_awan > 50 or kelembaban > 70:
            cuaca = 'Berawan'
        else:
            cuaca = 'Cerah'
        
        # Tambahkan sedikit noise untuk variasi
        if np.random.random() < 0.05:  # 5% noise
            cuaca_options = ['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Lebat']
            cuaca = np.random.choice(cuaca_options)
        
        data.append({
            'suhu': round(suhu, 1),
            'kelembaban': round(kelembaban, 1),
            'tekanan_udara': round(tekanan_udara, 1),
            'kecepatan_angin': round(kecepatan_angin, 1),
            'tutupan_awan': round(tutupan_awan, 1),
            'cuaca': cuaca
        })
    
    return pd.DataFrame(data)

# Generate dataset
print("🌦️ Generating BMKG Weather Dataset...")
df = generate_bmkg_dataset(1000)

# Tampilkan informasi dataset
print("\n" + "="*60)
print("📊 INFORMASI DATASET")
print("="*60)
print(f"Total Data: {len(df)}")
print(f"Jumlah Fitur: {len(df.columns) - 1}")
print(f"\nKolom Dataset:")
print(df.columns.tolist())

print("\n" + "="*60)
print("📈 STATISTIK DESKRIPTIF")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("🌤️ DISTRIBUSI CUACA")
print("="*60)
print(df['cuaca'].value_counts())
print("\nPersentase:")
print(df['cuaca'].value_counts(normalize=True) * 100)

print("\n" + "="*60)
print("👀 PREVIEW 10 DATA PERTAMA")
print("="*60)
print(df.head(10))

# Simpan ke CSV
csv_filename = 'bmkg_weather_dataset.csv'
df.to_csv(csv_filename, index=False)
print(f"\n✅ Dataset berhasil disimpan: {csv_filename}")
print(f"📁 File size: {len(df)} baris × {len(df.columns)} kolom")

# Informasi untuk download
print("\n" + "="*60)
print("💾 CARA DOWNLOAD DATASET")
print("="*60)
print("Jika di Google Colab, jalankan kode berikut:")
print(f"from google.colab import files")
print(f"files.download('{csv_filename}')")
print("="*60)