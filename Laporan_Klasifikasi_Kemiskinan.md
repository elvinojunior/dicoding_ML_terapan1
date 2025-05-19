# Laporan Proyek Machine Learning — Elvino Junior — Klasifikasi Status Kemiskinan

## Domain Proyek
Domain yang dipilih untuk proyek *machine learning terapan* ini adalah **Sosial-Ekonomi**, dengan judul **Classification Analytics : Status Kemiskinan**  

### Latar Belakang

![image](https://github.com/user-attachments/assets/f9c36556-79e5-46f7-ab3f-34f29935edb8)

Menurut Badan Pusat Statistik (BPS), per Maret 2024, persentase penduduk miskin di Indonesia turun menjadi 9,03% dari total populasi [1]. Meskipun mengalami penurunan, angka ini tetap menjadi tantangan dalam upaya pemerintah meningkatkan kesejahteraan masyarakat[[1](https://www.bps.go.id/id/pressrelease/2024/07/01/2370/persentase-penduduk-miskin-maret-2024-turun-menjadi-9-03-persen-.html)]. Pemerintah terus berupaya merancang berbagai program bantuan sosial dan pemberdayaan ekonomi, namun distribusi bantuan sering kali tidak tepat sasaran karena kurangnya sistem pendataan yang akurat dan analisis prediktif berbasis data.

Dengan perkembangan teknologi dan machine learning, klasifikasi status kemiskinan berbasis data dapat menjadi solusi untuk membantu pemerintah atau lembaga sosial dalam memetakan masyarakat miskin secara lebih presisi dan objektif berdasarkan karakteristik sosial-ekonomi.

Beberapa penelitian menunjukkan potensi machine learning dalam klasifikasi status sosial-ekonomi, seperti studi oleh Obeid et al. (2021) yang menggunakan beberapa algoritma machine learning untuk klasifikasi kemiskinan di Yordania dan memperoleh hasil prediksi yang signifikan[[2](https://www.researchgate.net/publication/348898452_Poverty_Classification_Using_Machine_Learning_The_Case_of_Jordan)]. Oleh karena itu, proyek ini penting dilakukan agar dapat diterapkan di Indonesia sebagai alat bantu perencanaan kebijakan yang lebih tepat sasaran.

---

## Business Understanding

### Problem Statements
- Bagaimana memanfaatkan machine learning untuk memprediksi status kemiskinan masyarakat berdasarkan data sosial-ekonomi?  
- Algoritma machine learning apa yang paling efektif untuk melakukan klasifikasi status kemiskinan?  
- Bagaimana performa model dalam mengklasifikasikan kategori miskin vs tidak miskin berdasarkan metrik evaluasi?
- Fitur-fitur apa saja yang paling berpengaruh terhadap status kemiskinan menurut hasil analisis dan pemodelan?  

### Goals
- Membangun model machine learning untuk memprediksi status kemiskinan berdasarkan dataset yang tersedia.  
- Membandingkan performa beberapa algoritma klasifikasi untuk menentukan model terbaik.  
- Mengukur performa model menggunakan metrik klasifikasi yang sesuai.
- Melakukan Exploratory Data Analysis untuk melihat fitur fitur paling penting.

### Solution statements
- Menerapkan algoritma **K-Nearest Neighbour**, **Decision Tree**, dan **Random Forest** sebagai baseline model untuk klasifikasi kemiskinan.  
- Melakukan **Oversampling SMOTE** untuk mengatasi class minoritas dan meningkatkan performa baseline.  
- Memilih model terbaik berdasarkan metrik evaluasi **accuracy**, **precision**, **recall**, **mse**, dan **F1-score**.
- Menerapkan Analisis Visual secara Univariate dan Multivariate. 

---

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle - Klasikasi Kemiskinan](https://www.kaggle.com/datasets/ermila/klasifikasi-kemiskinan), yang berisi data sosial-ekonomi di Indonesia.

**Jumlah data:** 514 baris  
**Jumlah fitur:** 7 kolom  

### Variabel-variabel:
- `Provinsi` : Nama provinsi asal data
- `Kab/Kota` : Nama kabupaten/kota
- `Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)` : Persentase jumlah penduduk miskin di masing-masing kabupaten/kota
- `Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)` : Jumlah pengeluaran per orang per tahun
- `Indeks Pembangunan Manusia` : Nilai IPM daerah tersebut
- `Tingkat Pengangguran Terbuka` : Persentase pengangguran terbuka di wilayah tersebut
- `Klasifikasi Kemiskinan` : Label target (0 = Tidak Miskin, 1 = Miskin)  

### EDA Univariate:
- Distribusi kelas target: 47% miskin, 53% tidak miskin  
- Korelasi kuat antara `income` dan `education_years` dengan status kemiskinan  
- Visualisasi: histogram income, education, household size

### EDA Multivariate:
- Distribusi kelas target: 47% miskin, 53% tidak miskin  
- Korelasi kuat antara `income` dan `education_years` dengan status kemiskinan  
- Visualisasi: histogram income, education, household size
---

## Data Preparation

### **Tahapan:**
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

### **Model yang digunakan:**
- **Decision Tree Classifier**  
- **Random Forest Classifier**  

### **Parameter:**
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

### **Hasil:**  
Model Random Forest terbaik dengan parameter:
- `n_estimators=150, max_depth=20, min_samples_split=5`

### **Kelebihan & Kekurangan:**
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
| K-Nearest Neighbour    | 91.7% | 91%        | 92%      | 91%     |
| Decision Tree    | 84.5%     | 83%        | 85%      | 84%         |
| Random Forest    |   91.7%   | 91%        | 92%      | 91%     |

**Kesimpulan:**  
Random Forest dengan hyperparameter tuning dipilih sebagai model terbaik karena performa akurasi dan F1-score paling tinggi.

---

**Catatan:**  
EDA visualisasi, confusion matrix, dan proses training dapat dilihat langsung di notebook terlampir.
