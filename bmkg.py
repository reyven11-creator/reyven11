# ====================================================================
# BMKG WEATHER PREDICTION - COMPLETE MACHINE LEARNING PIPELINE
# Google Colab Ready
# ====================================================================

# STEP 1: INSTALL & IMPORT LIBRARIES
# ====================================================================
print("📦 Installing libraries...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!\n")

# STEP 2: GENERATE DATASET
# ====================================================================
print("="*70)
print("🌦️  STEP 1: GENERATING BMKG WEATHER DATASET")
print("="*70)

np.random.seed(42)

def generate_bmkg_dataset(n_samples=1000):
    data = []
    
    for i in range(n_samples):
        suhu = np.random.uniform(22, 34)
        kelembaban = np.random.uniform(60, 95)
        tekanan_udara = np.random.uniform(1008, 1023)
        kecepatan_angin = np.random.uniform(0, 25)
        tutupan_awan = np.random.uniform(0, 100)
        
        if kelembaban > 85 and tekanan_udara < 1012 and tutupan_awan > 70:
            cuaca = 'Hujan Lebat'
        elif kelembaban > 75 and tutupan_awan > 60 and tekanan_udara < 1015:
            cuaca = 'Hujan Ringan'
        elif tutupan_awan > 50 or kelembaban > 70:
            cuaca = 'Berawan'
        else:
            cuaca = 'Cerah'
        
        if np.random.random() < 0.05:
            cuaca = np.random.choice(['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Lebat'])
        
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
df = generate_bmkg_dataset(1000)

print(f"✅ Dataset generated: {len(df)} samples")
print(f"📊 Features: {len(df.columns) - 1}")
print(f"🎯 Target: cuaca (4 classes)\n")

# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ====================================================================
print("="*70)
print("📊 STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n📋 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n🌤️  Weather Distribution:")
print(df['cuaca'].value_counts())
print("\nPercentage:")
print(df['cuaca'].value_counts(normalize=True) * 100)

print("\n👀 First 5 rows:")
print(df.head())

# Visualisasi
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('🌦️ BMKG Weather Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Distribusi Cuaca
weather_counts = df['cuaca'].value_counts()
axes[0, 0].bar(weather_counts.index, weather_counts.values, color=['#FFD700', '#B0C4DE', '#87CEEB', '#4682B4'])
axes[0, 0].set_title('Distribusi Kondisi Cuaca', fontweight='bold')
axes[0, 0].set_xlabel('Cuaca')
axes[0, 0].set_ylabel('Jumlah')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Distribusi Suhu
axes[0, 1].hist(df['suhu'], bins=30, color='#FF6347', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribusi Suhu', fontweight='bold')
axes[0, 1].set_xlabel('Suhu (°C)')
axes[0, 1].set_ylabel('Frekuensi')

# 3. Distribusi Kelembaban
axes[0, 2].hist(df['kelembaban'], bins=30, color='#4169E1', alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Distribusi Kelembaban', fontweight='bold')
axes[0, 2].set_xlabel('Kelembaban (%)')
axes[0, 2].set_ylabel('Frekuensi')

# 4. Boxplot Suhu per Cuaca
df.boxplot(column='suhu', by='cuaca', ax=axes[1, 0])
axes[1, 0].set_title('Suhu per Kondisi Cuaca', fontweight='bold')
axes[1, 0].set_xlabel('Cuaca')
axes[1, 0].set_ylabel('Suhu (°C)')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# 5. Boxplot Kelembaban per Cuaca
df.boxplot(column='kelembaban', by='cuaca', ax=axes[1, 1])
axes[1, 1].set_title('Kelembaban per Kondisi Cuaca', fontweight='bold')
axes[1, 1].set_xlabel('Cuaca')
axes[1, 1].set_ylabel('Kelembaban (%)')
plt.sca(axes[1, 1])
plt.xticks(rotation=45)

# 6. Boxplot Tutupan Awan per Cuaca
df.boxplot(column='tutupan_awan', by='cuaca', ax=axes[1, 2])
axes[1, 2].set_title('Tutupan Awan per Kondisi Cuaca', fontweight='bold')
axes[1, 2].set_xlabel('Cuaca')
axes[1, 2].set_ylabel('Tutupan Awan (%)')
plt.sca(axes[1, 2])
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation Heatmap
print("\n🔥 Correlation Heatmap...")
plt.figure(figsize=(10, 8))
correlation = df.drop('cuaca', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Matrix - Weather Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# STEP 4: DATA PREPARATION
# ====================================================================
print("\n" + "="*70)
print("🔧 STEP 3: DATA PREPARATION")
print("="*70)

# Pisahkan features dan target
X = df.drop('cuaca', axis=1)
y = df['cuaca']

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")
print(f"📊 Split ratio: 80-20")

# STEP 5: MODEL TRAINING
# ====================================================================
print("\n" + "="*70)
print("🧠 STEP 4: MODEL TRAINING")
print("="*70)

# Model 1: Decision Tree
print("\n🌳 Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
dt_model.fit(X_train, y_train)
print("✅ Decision Tree trained!")

# Model 2: Random Forest
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained!")

# STEP 6: MODEL EVALUATION
# ====================================================================
print("\n" + "="*70)
print("📈 STEP 5: MODEL EVALUATION")
print("="*70)

# Prediksi
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Akurasi
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\n🎯 Decision Tree Accuracy: {dt_accuracy*100:.2f}%")
print(f"🎯 Random Forest Accuracy: {rf_accuracy*100:.2f}%")

# Classification Report
print("\n" + "="*70)
print("📊 DECISION TREE - Classification Report")
print("="*70)
print(classification_report(y_test, dt_pred))

print("\n" + "="*70)
print("📊 RANDOM FOREST - Classification Report")
print("="*70)
print(classification_report(y_test, rf_pred))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree CM
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(df['cuaca'].unique()),
            yticklabels=sorted(df['cuaca'].unique()),
            ax=axes[0])
axes[0].set_title(f'Decision Tree\nAccuracy: {dt_accuracy*100:.2f}%', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Random Forest CM
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=sorted(df['cuaca'].unique()),
            yticklabels=sorted(df['cuaca'].unique()),
            ax=axes[1])
axes[1].set_title(f'Random Forest\nAccuracy: {rf_accuracy*100:.2f}%', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Feature Importance
print("\n" + "="*70)
print("⭐ FEATURE IMPORTANCE")
print("="*70)

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🌲 Random Forest Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='#2E8B57')
plt.xlabel('Importance Score', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')
plt.title('🌲 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# STEP 7: TESTING WITH NEW DATA
# ====================================================================
print("\n" + "="*70)
print("🔮 STEP 6: PREDICTION WITH NEW DATA")
print("="*70)

# Contoh prediksi dengan data baru
test_samples = [
    {
        'suhu': 32.0,
        'kelembaban': 88.0,
        'tekanan_udara': 1010.0,
        'kecepatan_angin': 15.0,
        'tutupan_awan': 85.0,
        'expected': 'Hujan Lebat'
    },
    {
        'suhu': 28.0,
        'kelembaban': 65.0,
        'tekanan_udara': 1018.0,
        'kecepatan_angin': 5.0,
        'tutupan_awan': 20.0,
        'expected': 'Cerah'
    },
    {
        'suhu': 26.0,
        'kelembaban': 78.0,
        'tekanan_udara': 1013.0,
        'kecepatan_angin': 12.0,
        'tutupan_awan': 65.0,
        'expected': 'Berawan'
    }
]

print("\n🧪 Testing with sample data:\n")

for i, sample in enumerate(test_samples, 1):
    test_data = pd.DataFrame([{
        'suhu': sample['suhu'],
        'kelembaban': sample['kelembaban'],
        'tekanan_udara': sample['tekanan_udara'],
        'kecepatan_angin': sample['kecepatan_angin'],
        'tutupan_awan': sample['tutupan_awan']
    }])
    
    dt_prediction = dt_model.predict(test_data)[0]
    rf_prediction = rf_model.predict(test_data)[0]
    rf_proba = rf_model.predict_proba(test_data)[0]
    
    print(f"Sample {i}:")
    print(f"  Input: Suhu={sample['suhu']}°C, Kelembaban={sample['kelembaban']}%, "
          f"Tekanan={sample['tekanan_udara']}hPa, Angin={sample['kecepatan_angin']}km/h, "
          f"Awan={sample['tutupan_awan']}%")
    print(f"  Expected: {sample['expected']}")
    print(f"  Decision Tree: {dt_prediction}")
    print(f"  Random Forest: {rf_prediction}")
    print(f"  Confidence: {max(rf_proba)*100:.1f}%")
    print()

# STEP 8: SAVE MODEL
# ====================================================================
print("="*70)
print("💾 STEP 7: SAVING MODEL")
print("="*70)

import pickle

# Save Random Forest model (lebih akurat)
model_filename = 'bmkg_weather_rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf_model, file)

print(f"\n✅ Model saved as: {model_filename}")

# Save dataset
csv_filename = 'bmkg_weather_dataset.csv'
df.to_csv(csv_filename, index=False)
print(f"✅ Dataset saved as: {csv_filename}")

print("\n" + "="*70)
print("🎉 MACHINE LEARNING PIPELINE COMPLETED!")
print("="*70)
print("\n📝 Summary:")
print(f"  • Dataset: {len(df)} samples, {len(df.columns)-1} features")
print(f"  • Best Model: Random Forest")
print(f"  • Accuracy: {rf_accuracy*100:.2f}%")
print(f"  • Classes: {', '.join(sorted(df['cuaca'].unique()))}")
print("\n💾 Files saved:")
print(f"  • {csv_filename}")
print(f"  • {model_filename}")
print("\n🚀 Model ready for deployment!")
print("="*70)

# BONUS: Download files (untuk Google Colab)
print("\n💡 To download files in Google Colab, run:")
print("=" * 70)
print("from google.colab import files")
print(f"files.download('{csv_filename}')")
print(f"files.download('{model_filename}')")
print("=" * 70)