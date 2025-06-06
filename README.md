# Laporan Proyek Machine Learning — Elvino Junior

## Domain Proyek
Domain yang dipilih untuk proyek *machine learning terapan* ini adalah **Sosial-Ekonomi**, dengan judul **Klasifikasi : Status Kemiskinan**  

![kemiskinan](https://github.com/user-attachments/assets/f9c36556-79e5-46f7-ab3f-34f29935edb8)

### Latar Belakang

Menurut Badan Pusat Statistik (BPS), per Maret 2024, persentase penduduk miskin di Indonesia turun menjadi 9,03% dari total populasi [[1](https://www.bps.go.id/id/pressrelease/2024/07/01/2370/persentase-penduduk-miskin-maret-2024-turun-menjadi-9-03-persen-.html)]. Meskipun mengalami penurunan, angka ini tetap menjadi tantangan dalam upaya pemerintah meningkatkan kesejahteraan masyarakat. Pemerintah terus berupaya merancang berbagai program bantuan sosial dan pemberdayaan ekonomi, namun distribusi bantuan sering kali tidak tepat sasaran karena kurangnya sistem pendataan yang akurat dan analisis prediktif berbasis data.

Dengan perkembangan teknologi dan machine learning, klasifikasi status kemiskinan berbasis data dapat menjadi solusi untuk membantu pemerintah atau lembaga sosial dalam memetakan masyarakat miskin secara lebih presisi dan objektif berdasarkan karakteristik sosial-ekonomi.

Beberapa penelitian menunjukkan potensi machine learning dalam klasifikasi status sosial-ekonomi, seperti studi oleh Obeid et al. (2021) yang menggunakan beberapa algoritma machine learning untuk klasifikasi kemiskinan di Yordania dan memperoleh hasil prediksi yang signifikan[[2](https://www.researchgate.net/publication/348898452_Poverty_Classification_Using_Machine_Learning_The_Case_of_Jordan)]. Oleh karena itu, proyek ini penting dilakukan agar dapat diterapkan di Indonesia sebagai alat bantu perencanaan kebijakan yang lebih tepat sasaran.

---

## Business Understanding

### Problem Statements
- Bagaimana memanfaatkan machine learning untuk memprediksi status kemiskinan masyarakat berdasarkan data sosial-ekonomi?  
- Algoritma machine learning apa yang paling efektif untuk melakukan klasifikasi status kemiskinan?  
- Bagaimana penerapan performa model dalam menangani ketidakseimbangan kelas(class imbalance) pada Klasifikasi status kemiskinan?
- Fitur-fitur apa saja yang paling berpengaruh terhadap status kemiskinan menurut hasil analisis dan pemodelan?  

### Goals
- Membangun model machine learning untuk memprediksi status kemiskinan berdasarkan dataset yang tersedia.  
- Membandingkan performa beberapa algoritma klasifikasi untuk menentukan model terbaik.  
- Mampu meningkatkan performa model dengan mengatasi ketidakseimbangan kelas(class imbalance).
- Melakukan Exploratory Data Analysis untuk melihat fitur fitur paling penting.

### Solution statements
- Menerapkan algoritma **K-Nearest Neighbour**, **Decision Tree**, dan **Random Forest** sebagai baseline model untuk klasifikasi kemiskinan.   
- Memilih model terbaik berdasarkan metrik evaluasi **accuracy**, **precision**, **recall**, dan **F1-score**.
- Melakukan **Oversampling SMOTE** untuk mengatasi ketidakseimbangan kelas(class imbalance) dan meningkatkan performa baseline. 
- Menerapkan Analisis Visual secara Univariate dan Multivariate. 

---

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle - Klasikasi Kemiskinan](https://www.kaggle.com/datasets/ermila/klasifikasi-kemiskinan), yang berisi data sosial-ekonomi di Indonesia.

**Jumlah data:** 514 baris  
**Jumlah fitur:** 7 kolom  
**Struktur dataset** :
![struktur_dataset](https://github.com/user-attachments/assets/0ec72ca6-6bb1-49ad-b975-8656ed15e046)



### Deskripsi Variable:
- `Provinsi` : Nama provinsi asal data
- `Kab/Kota` : Nama kabupaten/kota
- `Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)` : Persentase jumlah penduduk miskin di masing-masing kabupaten/kota
- `Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)` : Jumlah pengeluaran per orang per tahun
- `Indeks Pembangunan Manusia` : Nilai IPM daerah tersebut
- `Tingkat Pengangguran Terbuka` : Persentase pengangguran terbuka di wilayah tersebut
- `Klasifikasi Kemiskinan` : Label target (0 = Tidak Miskin, 1 = Miskin) 

### EDA Univariate:
Distribusi histogram fitur numerik
![Gambar1a](https://github.com/user-attachments/assets/bb079968-3583-4ae6-b0ca-01f580e8c42e)

Pada tahap EDA univariate, dilakukan analisis distribusi masing-masing variabel numerik dan target untuk memahami pola sebaran data:

- Distribusi Persentase Penduduk Miskin per Kabupaten/Kota (persen_kemiskinan_kota)
Distribusi data bersifat positif skewed dengan mayoritas kabupaten/kota memiliki persentase kemiskinan di rentang 5% hingga 15%, sementara nilai ekstrem di atas 20% cukup jarang ditemukan.
- Distribusi Pengeluaran per Kapita Disesuaikan (pengeluaran_kapita)
Distribusi pengeluaran per kapita juga positif skewed, dengan sebagian besar wilayah memiliki pengeluaran per kapita antara 8.000 hingga 12.000 ribu rupiah/tahun. Nilai ekstrim di atas 14.000 ribu rupiah relatif jarang.
- Distribusi Indeks Pembangunan Manusia (IPM)
Variabel IPM cenderung mengikuti distribusi normal dengan rata-rata di sekitar 70. Sebagian besar daerah memiliki IPM dalam rentang 65–75, sementara nilai di bawah 60 dan di atas 80 jarang terjadi.
- Distribusi Tingkat Pengangguran Terbuka (tingkat_pengangguran)
Distribusi tingkat pengangguran juga bersifat positif skewed, dengan mayoritas kabupaten/kota memiliki tingkat pengangguran di bawah 6%, dan beberapa daerah mencapai di atas 10%.
- Distribusi Klasifikasi Kemiskinan (klasifikasi_kemiskinan)
Distribusi target klasifikasi_kemiskinan sangat imbalance, di mana sebagian besar data diklasifikasikan sebagai tidak miskin (label 0), sedangkan label miskin (label 1) jumlahnya jauh lebih sedikit. Hal ini menunjukkan ketidakseimbangan kelas yang perlu diperhatikan saat pemodelan.

Distribusi kolom kategorik
![gambar1b](https://github.com/user-attachments/assets/3b9a0b9a-567f-48aa-afd3-9d9764fc58c7)
Distribusi data pada kolom provinsi : 
-   Data dengan sedikit kontribusi ada di provinsi DKI Jakarta
-   Kontibusi data paling banyak ada di provinsi Jawa Timur

### EDA Multivariate:
- Klasifikasi Kemiskinan antar provinsi
![gambar2a](https://github.com/user-attachments/assets/a5ee2199-bfb8-47e9-98d2-25951a772a77)
Variasi Pola Kemiskinan Antar Provinsi: Pola distribusi antara kategori "0" dan "1" sangat bervariasi antar provinsi. Ada provinsi di mana jumlah yang tidak miskin jauh lebih banyak dari yang miskin, ada yang perbedaannya tidak terlalu besar, dan bahkan ada beberapa provinsi (meskipun terlihat sedikit) di mana jumlah yang diklasifikasikan sebagai miskin hampir sebanding atau bahkan lebih banyak dari yang tidak miskin.
Beberapa provinsi terlihat memiliki batang kategori "0" yang jauh lebih tinggi daripada batang kategori "1", mengindikasikan proporsi "kemiskinan" yang relatif rendah. Contoh Provinsi Spesifik:
  + Aceh: Terlihat memiliki jumlah kategori 0 yang lebih tinggi dari kategori 1.
  + Papua: Menarik untuk diperhatikan bahwa di Papua, jumlah kategori 1 (miskin) terlihat lebih tinggi dibandingkan kategori 0 (tidak miskin).
  + Nusa tenggara timur & Maluku: memiliki jumlah kategori 0 (tidak miskin) dan kategori 1 (miskin) yang hampir sama
     
- Visualisasi Pairplot
![image](https://github.com/user-attachments/assets/b0212e0c-830b-4ffe-b64f-640fa2a15087)
1. Persentase Penduduk Miskin (persen_kemiskinan_kota) = hubungan negatif dengan pengeluaran per kapita dan IPM. Semakin tinggi pengeluaran dan IPM suatu wilayah, cenderung semakin rendah persentase penduduk miskinnya.
2. Pengeluaran per Kapita vs IPM = Hubungan positif yang cukup kuat. Wilayah dengan pengeluaran per kapita lebih tinggi umumnya memiliki IPM yang lebih baik.
3. Tingkat Pengangguran = Sebaran tingkat pengangguran terlihat menyebar cukup luas, namun tidak menunjukkan pola hubungan yang sangat kuat terhadap variabel lain dalam visualisasi scatter ini.
4. Klasifikasi Kemiskinan (klasifikasi_kemiskinan) = Titik-titik pada sumbu klasifikasi_kemiskinan memperlihatkan kecenderungan:
    + Wilayah dengan persentase kemiskinan tinggi dan pengeluaran rendah lebih banyak masuk ke dalam kategori miskin (label 1).
    + Sementara wilayah dengan pengeluaran tinggi, IPM tinggi, dan persentase kemiskinan rendah didominasi oleh label tidak miskin (label 0).

- Visualisasi heatmap korelasi
![image](https://github.com/user-attachments/assets/5e51dd82-c00b-4c1d-a2ce-3e896585bd0a)
1. Pengeluaran per Kapita memiliki korelasi positif sangat kuat terhadap IPM sebesar 0.85, artinya daerah dengan pengeluaran per kapita yang tinggi umumnya memiliki IPM yang tinggi pula.
2. Persentase Kemiskinan Kota memiliki korelasi negatif sedang terhadap pengeluaran per kapita (-0.52) dan IPM (-0.47). Artinya, semakin tinggi pengeluaran dan IPM suatu wilayah, semakin rendah persentase kemiskinannya.
3. Tingkat Pengangguran menunjukkan korelasi positif lemah terhadap IPM (0.47) dan pengeluaran per kapita (0.43), tetapi hubungan ini cenderung tidak sekuat korelasi antar fitur sebelumnya.
4. Terhadap target klasifikasi_kemiskinan, fitur dengan korelasi paling tinggi adalah:
    + Persentase Kemiskinan Kota (0.57) → korelasi positif cukup kuat.
    + Pengeluaran per Kapita (-0.29) dan IPM (-0.27) → korelasi negatif lemah ke sedang.
    + Tingkat Pengangguran memiliki korelasi paling lemah terhadap klasifikasi kemiskinan (-0.035).
---

## Data Preparation

### **Tahapan:**
- Mengganti nama kolom :
   + `Provinsi` : `provinsi`
   + `Kab/Kota` : `kab/kota`
   + `Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)` : `persen_kemiskinan_kota`
   + `Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)` : `pengeluaran_kapita(ribu/tahun)`
   + `Indeks Pembangunan Manusia` : `IPM`
   + `Tingkat Pengangguran Terbuka` : `tingkat_pengangguran`
   + `Klasifikasi Kemiskinan` : `klasifikasi_kemiskinan`
- Mengubah tipe data pada kolom `persen_kemiskinan_kota`, `IPM`, `tingkat_pengangguran` menjadi numerik (Float)   
- Handling missing values: tidak ditemukan adanya missing values
- Handling duplicate values: tidak ditemukan adanya duplicate values
- Handling Outlier: dilakukan menggunakan teknik IQR Method untuk mengurangi pengaruh data ekstrem yang bisa memengaruhi akurasi model, terutama pada algoritma yang sensitif terhadap nilai outlier seperti Decision Tree dan Random Forest.
- One Hot Encoding untuk kolom provinsi: karena provinsi merupakan data kategorikal, sehingga perlu diubah menjadi representasi numerik biner agar dapat diproses oleh model.
- Data scaling: menggunakan StandardScaler untuk menstandarkan skala fitur numerik sehingga memiliki distribusi mean 0 dan standard deviasi 1. Ini penting karena model tertentu sensitif terhadap perbedaan skala fitur.
- Oversampling SMOTE pada data latih (train): untuk menangani ketidakseimbangan kelas (class imbalance), sehingga jumlah data pada kelas minoritas (label 1 = miskin) diseimbangkan dengan kelas mayoritas. Hal ini dapat meningkatkan sensitivitas model dalam mendeteksi kelas minoritas.
- Data split: membagi data menjadi 80% untuk train dan 20% untuk test agar model dapat dilatih dan dievaluasi secara fair tanpa data leakage.

**Alasan:**  
- Mengganti nama kolom dilakukan untuk memudahkan proses eksplorasi, analisis, dan pemodelan karena nama kolom yang ringkas, konsisten, dan mudah dipanggil dapat mengurangi potensi error saat coding
- Mengubah tipe data agar data numerik dapat digunakan dalam perhitungan statistik, visualisasi, serta sebagai input model machine learning yang hanya menerima data numerik untuk operasi matematis.
- Handling missing values diperiksa untuk memastikan tidak ada data kosong yang dapat mengganggu hasil model atau analisis.
- Handling duplicate values diperiksa agar tidak ada duplikasi data yang bisa membuat bobot informasi menjadi tidak proporsional.
- Handling Outlier (IQR Method) dilakukan untuk mengurangi pengaruh data ekstrem yang bisa mendistorsi parameter model, terutama untuk algoritma berbasis pohon (Decision Tree, Random Forest) yang cukup sensitif terhadap outlier. Dengan membersihkan outlier, model bisa lebih stabil dan akurat.
- One Hot Encoding untuk kolom kategorikal (provinsi) Karena algoritma machine learning tidak dapat memproses data kategorikal dalam format string, maka perlu diubah menjadi format numerik biner agar bisa diproses dengan benar.
- Data Scaling (StandardScaler) penting untuk model-model seperti K-Nearest Neighbour dan algoritma berbasis jarak lainnya, di mana perbedaan skala antar fitur bisa menyebabkan fitur dengan rentang nilai lebih besar mendominasi hasil model.
- Oversampling dengan SMOTE pada data latih dilakukan untuk menangani class imbalance yang bisa membuat model cenderung bias ke kelas mayoritas. Dengan menyeimbangkan jumlah data minoritas, model bisa lebih sensitif dalam mendeteksi kategori miskin. SMOTE hanya diterapkan ke data training, agar evaluasi di data testing tetap mencerminkan kondisi nyata dari distribusi data asli.

---

## Model Development
Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:

### **K-Nearest Neighbour**  
Model pertama yang digunakan yaitu algoritma K-Nearest Neighbour (KNN), yang mengklasifikasikan label dari sebuah data baru berdasarkan label dari K data tetangga terdekatnya dalam ruang fitur. Kedekatan antar data diukur menggunakan jarak, umumnya Euclidean.

Parameter yang digunakan yaitu :
-  `n_neighbors` jumlah tetangga terdekat yang digunakan untuk menentukan kelas prediksi. Dalam proyek ini ditentukan sebanyak 5 tetangga.
  
Kelebihan:
- Mudah dipahami dan diimplementasikan.
- Cocok untuk dataset skala kecil hingga menengah.
- Tidak memerlukan proses training yang kompleks.

Kekurangan:
- Proses prediksi lambat pada dataset besar karena harus menghitung jarak ke seluruh data latih.
- Sensitif terhadap skala fitur (perlu scaling).
- Rentan terhadap outlier dan noise.

### **Decision Tree Classifier**
Model kedua yaitu Decision Tree Classifier, algoritma supervised learning yang dapat digunakan untuk tugas klasifikasi maupun regresi. Model ini bekerja dengan membangun struktur pohon di mana tiap node internal merepresentasikan fitur, tiap cabang merepresentasikan aturan keputusan, dan tiap leaf node berisi hasil akhir klasifikasi.

Parameter yang digunakan yaitu
- `random_state` digunakan untuk memastikan bahwa proses yang bersifat acak (seperti pemilihan subset data dan pemilihan fitur saat membangun node) akan menghasilkan hasil yang sama setiap kali kode dijalankan. Dalam proyek ini diatur sebanyak 42
- `max_depth` kedalaman maksimum masing-masing pohon dalam hutan, untuk mencegah overfitting. Dalam proyek ini diatur maksimal 3.

Kelebihan:
- Mudah dipahami dan divisualisasikan.
- Dapat menangani data numerik maupun kategorikal tanpa perlu normalisasi.
- Dapat mengukur pentingnya tiap fitur (feature importance).

Kekurangan:
- Rentan terhadap overfitting jika tidak dibatasi kedalamannya.
- Performa bisa kurang stabil jika dataset kecil atau banyak noise.
- Cenderung bias ke fitur dengan banyak nilai unik.

### **Random Forest Classifier**  
Model ketiga adalah Random Forest, sebuah algoritma ensemble learning berbasis Decision Tree. Random Forest membangun banyak pohon keputusan secara acak dan independen, lalu menggabungkan prediksinya melalui voting (untuk klasifikasi) atau rata-rata (untuk regresi). Konsep utamanya, "kerumunan pohon yang lemah bisa menjadi hutan yang kuat."

Parameter yang digunakan yaitu
- `random_state` digunakan untuk memastikan bahwa proses yang bersifat acak (seperti pemilihan subset data dan pemilihan fitur saat membangun pohon-pohon) akan menghasilkan hasil yang sama setiap kali kode dijalankan. Dalam proyek ini diatur sebanyak 42
- `max_depth` kedalaman maksimum masing-masing pohon dalam hutan, untuk mencegah overfitting. Dalam proyek ini diatur maksimal 3.

Kelebihan:
- Lebih stabil dan akurat dibanding single Decision Tree.
- Mampu mengatasi overfitting karena melakukan averaging dari banyak model.
- Bisa mengukur feature importance.
- Tahan terhadap outlier dan noise.

Kekurangan:
- Proses training dan prediksi lebih lambat karena banyaknya pohon yang dibuat.
- Model cenderung sulit diinterpretasi karena kompleksitas ensemble.
- Membutuhkan lebih banyak resource memori dan komputasi.



## Evaluation
Dalam tahap evaluasi, metrik yang digunakan adalah

- **Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision**
  
$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Recall**

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **F1-Score**

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Penjelasan
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).



**Hasil akhir:**

| Metric        | KNN (%) | Decision Tree (%) | Random Forest (%) |
| ------------- | :------ | :---------------- | :---------------- |
| **Accuracy**  | 92.22   | 97.78             | 96.67             |
| **Precision** | 45.45   | 75.00             | 66.67             |
| **Recall**    | 83.33   | 100.00            | 100.00            |
| **F1 Score**  | 93.25   | 97.92             | 96.97             |



**Kesimpulan:**  

Berdasarkan hasil analisis dan pemodelan yang telah dilakukan, diperoleh beberapa temuan utama terkait Klasifikasi status kemiskinan :

1. Pemanfaatan Machine Learning untuk Prediksi Kemiskinan
  Model machine learning dapat dimanfaatkan secara efektif untuk Klasifikasi status kemiskinan masyarakat berdasarkan variabel sosial-ekonomi seperti persentase kemiskinan kota, pengeluaran per kapita, IPM, dan tingkat pengangguran. Hasil model menunjukkan adanya pola yang dapat digunakan untuk memisahkan kategori miskin dan tidak miskin.

2. Algoritma Machine Learning yang Paling Efektif
  Dari tiga algoritma baseline yang diuji, yaitu K-Nearest Neighbour, Decision Tree, dan Random Forest, model Decision Tree menunjukkan performa paling stabil dan unggul dalam hal accuracy, precision, recall, dan F1-score. Model ini mampu menangani ketidakseimbangan data lebih baik, apalagi setelah diterapkan SMOTE (Synthetic Minority Oversampling Technique).

3. Performa Model setelah Oversampling SMOTE
  Penerapan SMOTE berhasil meningkatkan performa model secara signifikan, khususnya dalam hal recall untuk kategori miskin yang sebelumnya menjadi class minoritas. Hal ini menunjukkan bahwa teknik oversampling efektif mengatasi ketidakseimbangan kelas dalam dataset.

4. Fitur-Fitur yang Paling Berpengaruh
  Hasil Exploratory Data Analysis (EDA) dan analisis korelasi menunjukkan bahwa fitur **persentase kemiskinan kota** memiliki korelasi paling kuat terhadap status kemiskinan. Disusul oleh pengeluaran per kapita dan IPM yang memiliki hubungan signifikan terhadap klasifikasi kemiskinan. Sebaliknya, tingkat pengangguran memiliki pengaruh yang relatif kecil.

---
**Referensi**
1. Badan Pusat Statistik. (2024, Juli 1). *Persentase Penduduk Miskin Maret 2024 Turun Menjadi 9,03 Persen*. Diakses dari [https://www.bps.go.id/id/pressrelease/2024/07/01/2370/persentase-penduduk-miskin-maret-2024-turun-menjadi-9-03-persen-.html](https://www.bps.go.id/id/pressrelease/2024/07/01/2370/persentase-penduduk-miskin-maret-2024-turun-menjadi-9-03-persen-.html)

2. Obeid, N., Aljarah, I., & Faris, H. (2021). *Poverty Classification Using Machine Learning: The Case of Jordan*. Diakses dari [https://www.researchgate.net/publication/348898452\_Poverty\_Classification\_Using\_Machine\_Learning\_The\_Case\_of\_Jordan](https://www.researchgate.net/publication/348898452_Poverty_Classification_Using_Machine_Learning_The_Case_of_Jordan)

3. Subramanian, D. (2019, November 3). *A Simple Introduction to K-Nearest Neighbors Algorithm*. Towards Data Science. Diakses dari [https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e)

4. IBM. (n.d.). *What are Decision Trees?*. Diakses dari [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)

5. Wood, T. (n.d.). *What is a Random Forest?*. DeepAI. Diakses dari [https://deepai.org/machine-learning-glossary-and-terms/random-forest](https://deepai.org/machine-learning-glossary-and-terms/random-forest)

---
**Catatan:**  
EDA visualisasi, Data Cleaning & Preprocessing ,Confusion Matrix, dan proses training dapat dilihat langsung di notebook terlampir.
Nilai yang didapat : 

![image](https://github.com/user-attachments/assets/40176b47-62d5-4153-8d0a-a00f165437bb)

