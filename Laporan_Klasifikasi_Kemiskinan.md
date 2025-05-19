# Laporan Proyek Machine Learning — Klasifikasi Status Kemiskinan

## Domain Proyek  

Kemiskinan masih menjadi tantangan utama di banyak negara berkembang, termasuk Indonesia. Menurut Badan Pusat Statistik (BPS), per Maret 2023, jumlah penduduk miskin di Indonesia mencapai 25,90 juta jiwa atau 9,36% dari total populasi [1]. Pemerintah terus berupaya merancang berbagai program bantuan sosial dan pemberdayaan ekonomi, namun distribusi bantuan sering kali tidak tepat sasaran karena kurangnya sistem pendataan yang akurat dan analisis prediktif berbasis data.

Dengan perkembangan teknologi dan machine learning, klasifikasi status kemiskinan berbasis data dapat menjadi solusi untuk membantu pemerintah atau lembaga sosial dalam memetakan masyarakat miskin secara lebih presisi dan objektif berdasarkan karakteristik sosial-ekonomi.

Beberapa penelitian menunjukkan potensi machine learning dalam klasifikasi status sosial-ekonomi, seperti studi oleh Iqbal et al. (2022) yang menggunakan Decision Tree dan Random Forest untuk memetakan kemiskinan di Pakistan dengan akurasi tinggi [2]. Oleh karena itu, proyek ini penting dilakukan agar dapat diterapkan di Indonesia sebagai alat bantu perencanaan kebijakan yang lebih tepat sasaran.

**Referensi:**  
[1] BPS, *Persentase Penduduk Miskin Maret 2023*, https://www.bps.go.id  
[2] Iqbal, M. et al. (2022). *Predicting Household Poverty Using Machine Learning Approaches: Evidence from Pakistan*. Journal of Artificial Intelligence Research.

---

## Business Understanding

### Problem Statements
- Bagaimana memanfaatkan machine learning untuk memprediksi status kemiskinan masyarakat berdasarkan data sosial-ekonomi?  
- Algoritma machine learning apa yang paling efektif untuk melakukan klasifikasi status kemiskinan?  
- Bagaimana performa model dalam mengklasifikasikan kategori miskin vs tidak miskin berdasarkan metrik evaluasi?  

### Goals
- Membangun model machine learning untuk memprediksi status kemiskinan berdasarkan dataset yang tersedia.  
- Membandingkan performa beberapa algoritma klasifikasi untuk menentukan model terbaik.  
- Mengukur performa model menggunakan metrik klasifikasi yang sesuai.  

### Solution statements
- Menerapkan algoritma **Decision Tree** dan **Random Forest** sebagai baseline model untuk klasifikasi kemiskinan.  
- Melakukan **hyperparameter tuning pada Random Forest** untuk meningkatkan performa baseline.  
- Memilih model terbaik berdasarkan metrik evaluasi **accuracy**, **precision**, **recall**, dan **F1-score**.  

---

## Data Understanding

Dataset yang digunakan berasal dari [UCI Machine Learning Repository - Poverty Prediction Dataset](https://www.kaggle.com/datasets/danofer/poverty-prediction-in-mexico), yang berisi data sosial-ekonomi rumah tangga di Meksiko.

**Jumlah data:** 29.934 baris  
**Jumlah fitur:** 20 kolom  

### Variabel-variabel:
- `household_size` : Jumlah anggota rumah tangga  
- `dependency` : Rasio ketergantungan  
- `age` : Usia kepala keluarga  
- `income` : Pendapatan per bulan  
- `education_years` : Tahun pendidikan kepala keluarga  
- `phone` : Kepemilikan ponsel  
- `refrigerator` : Kepemilikan kulkas  
- `television` : Kepemilikan televisi  
- `poverty_status` : Target (Miskin / Tidak Miskin)  

**EDA**:
- Distribusi kelas target: 47% miskin, 53% tidak miskin  
- Korelasi kuat antara `income` dan `education_years` dengan status kemiskinan  
- Visualisasi: histogram income, education, household size  

---

## Data Preparation

**Tahapan:**
- Handling missing values: tidak ditemukan missing values  
- Encoding target `poverty_status` (0 = Tidak Miskin, 1 = Miskin)  
- Feature selection: memilih 8 fitur relevan berdasarkan korelasi dan domain  
- Data scaling: menggunakan **StandardScaler**  
- Data split: 80% train, 20% test  

**Alasan:**  
- Scaling agar distribusi fitur numerik seragam  
- Feature selection untuk mengurangi overfitting dan mempercepat training  

---

## Modeling

**Model yang digunakan:**
- **Decision Tree Classifier**  
- **Random Forest Classifier**  

**Parameter:**
- **Decision Tree:** default  
- **Random Forest:**  
  - `n_estimators = 100`  
  - `max_depth = 10`  
  - `min_samples_split = 5`  
  - `random_state = 42`  

**Improvement:**  
Hyperparameter tuning Random Forest dengan GridSearchCV:
- `n_estimators = [50, 100, 150]`  
- `max_depth = [5, 10, 20]`  
- `min_samples_split = [2, 5, 10]`  

**Hasil:**  
Model Random Forest terbaik dengan parameter:
- `n_estimators=150, max_depth=20, min_samples_split=5`

**Kelebihan & Kekurangan:**
- Decision Tree: cepat, mudah interpretasi, tapi overfitting  
- Random Forest: akurat, tahan overfitting, tapi lebih lambat  

---

## Evaluation

**Metrik:**
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  

**Formula:**
```
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Hasil akhir:**

| Model            | Accuracy | Precision | Recall | F1-Score |
|:----------------|:----------|:-----------|:---------|:------------|
| Decision Tree    | 84.5%     | 83%        | 85%      | 84%         |
| Random Forest    | **91.7%** | 91%        | 92%      | **91%**     |

**Kesimpulan:**  
Random Forest dengan hyperparameter tuning dipilih sebagai model terbaik karena performa akurasi dan F1-score paling tinggi.

---

**Catatan:**  
EDA visualisasi, confusion matrix, dan proses training dapat dilihat langsung di notebook terlampir.
