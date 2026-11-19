# 🌙 Sistem Analisis Tidur - Platform Cerdas Deteksi Kualitas & Pola Tidur

> **Tugas Mata Kuliah Sistem Cerdas (Intelligent Systems)**  
> Analisis Komprehensif Parameter Gaya Hidup & Pola Tidur Menggunakan Machine Learning dan Generative AI.

---

## 📌 Tentang Proyek

**Sistem Analisis Tidur** adalah aplikasi web berbasis *Machine Learning* dan *Generative AI* yang dirancang untuk menganalisis parameter fisik, kebiasaan tidur, dan aktivitas harian guna mendeteksi secara dini risiko gangguan tidur (**Normal**, **Insomnia**, atau **Sleep Apnea**).

Pengguna cukup memasukkan data fisik harian seperti **Tinggi Badan (cm)** dan **Berat Badan (kg)** (sistem menghitung BMI secara otomatis), durasi tidur, tingkat stres, aktivitas fisik, detak jantung istirahat, dan estimasi langkah kaki. Selanjutnya, sistem akan memberikan hasil klasifikasi berbasis model **Random Forest (Akurasi 96.00%)**, visualisasi hipnogram tahapan tidur, serta penjelasan klinis mendalam didukung **Google Gemini AI**.

---

## ✨ Fitur Utama

| Fitur | Keterangan |
| :--- | :--- |
| **Kalkulator & Diagnostik Cerdas** | Input parameter presisi berbasis *input box* (tinggi, berat badan, durasi tidur, stres, dll). Kalkulasi BMI otomatis tanpa perlu menebak kategori. |
| **Klasifikasi Random Forest (96%)** | Inferensi instan status tidur (*Normal*, *Insomnia*, *Sleep Apnea*) beserta persentase keyakinan model dan peluang tiap kategori. |
| **Visualisasi Hipnogram Tidur** | Ilustrasi visual tahapan siklus tidur (bangun, REM, tidur ringan, tidur dalam) sesuai hasil klasifikasi pola istirahat. |
| **Catatan Klinis Google Gemini AI** | Penjelasan naratif klinis otomatis dari Gemini API (`gemini-3-flash-preview` / `gemini-3.1-flash-lite`) dengan sistem *Smart Fallback Engine* bila offline. |
| **Asisten Konsultasi AI (Chatbot)** | Asisten interaktif *slide-over drawer* untuk konsultasi seputar tips insomnia, gejala apnea, durasi tidur ideal, dan cara kerja model ML. |
| **Dashboard Statistik & Kohort** | Analisis distribusi 374 data kohort klinis, perbandingan durasi tidur vs beban stres, serta korelasi aktivitas fisik harian. |
| **Protokol Sleep Hygiene** | Pedoman intervensi non-farmakologis medis (AASM & WHO): irama sirkadian, suhu lingkungan, paparan cahaya biru, dan waktu paruh kafein. |
| **Audit Log Riwayat** | Seluruh rekam jejak pemeriksaan tersimpan otomatis dan persisten di database SQLite (`db.sqlite3`). |

---

## 🛠️ Teknologi yang Digunakan

- **Backend**: Django 5.x, Python 3.10+
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, `DecisionTreeClassifier`, `LogisticRegression`), Joblib
- **Data Science**: Pandas, NumPy
- **Visualisasi**: SVG Hipnogram Dinamis & Chart.js
- **Generative AI**: Google Gemini REST API (v1beta) + Smart Rule-Based Clinical Engine Fallback
- **Database**: SQLite3 (`db.sqlite3`)
- **Desain UI**: Custom Professional Healthcare & Clinical Workspace 

---

## 🧠 Model Machine Learning

Model dilatih menggunakan dataset klinis **374 responden terverifikasi** dengan pembagian data *80% Data Latih (Train)* dan *20% Data Uji (Test)* secara terstratifikasi (*stratified split*).

### Fitur Input Evaluasi:
- **Demografi & Fisik**: `Gender_num` (Pria: 1, Wanita: 0), `Age` (Usia), `BMI_num` (Kalkulasi dari Tinggi & Berat Badan)
- **Karakteristik Istirahat**: `Sleep Duration` (Jam), `Quality of Sleep` (Skala 1-10)
- **Beban Fisiologis & Kardiovaskular**: `Stress Level` (Skala 1-10), `Physical Activity Level` (Menit/hari), `Heart Rate` (BPM), `Daily Steps` (Langkah/hari)

### Hasil Komparasi Model:

| Algoritma | Accuracy | Precision | Recall | F1-Score | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Classifier** | **96.00%** | **96.06%** | **96.00%** | **95.99%** | **Model Terbaik (Terpilih)** |
| Decision Tree Classifier | 94.67% | 95.02% | 94.67% | 94.75% | Komparasi |
| Logistic Regression | 93.33% | 93.34% | 93.33% | 93.23% | Komparasi |

> Model terbaik disimpan pada file `model_tidur.joblib` dan dimuat secara otomatis oleh server Django saat proses inferensi.

---

## 🚀 Panduan Instalasi & Menjalankan

### 1.
Pastikan Python 3.10+ sudah terpasang. Instal dependensi:
```bash
pip install django pandas scikit-learn joblib requests
```

*(Jika menggunakan Laragon pada Windows, gunakan Python Laragon yang sudah tersedia).*

---

### 2. Konfigurasi File `.env`
Salin template konfigurasi `.env.example` menjadi `.env`:

**Windows:**
```cmd
copy .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

**Isi file `.env`:**
```ini
SECRET_KEY=django-insecure-sistem-analisis-tidur-secret-key-2024
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
GEMINI_API_KEY=masukkan_api_key_gemini_kamu_disini
```

> 💡 **Informasi untuk Penguji / Dosen**:  
> `GEMINI_API_KEY` digunakan untuk fitur penjelasan AI dan Chatbot. Jika dijalankan tanpa API key atau tanpa koneksi internet, sistem akan secara otomatis mengaktifkan **Smart Medical Fallback Engine** sehingga aplikasi tetap berjalan lancar tanpa mengalami kendala/error.

---

### 3. Migrasi Database
Jalankan migrasi untuk memastikan tabel database siap:
```bash
python manage.py migrate
```

---

### 4. Menjalankan Server Aplikasi
Jalankan server lokal Django:
```bash
python manage.py runserver
```

Buka browser dan akses alamat:
👉 **`http://127.0.0.1:8000/`**

*(Di Windows, Anda juga bisa langsung menjalankan aplikasi cukup dengan **klik dua kali `run_django.bat`**).*

---

## 📁 Struktur Direktori Proyek

```plaintext
.
├── Sleep_health_and_lifestyle_dataset.csv  # Dataset klinis kesehatan tidur (374 baris)
├── train.py                                # Skrip pelatihan & evaluasi 3 model ML
├── model_tidur.joblib                      # File bobot model Random Forest terbaik
├── app.py                                  # Opsi antarmuka Streamlit alternatif
├── manage.py                               # CLI entrypoint Django
├── .env.example                            # Template file konfigurasi lingkungan
├── .env                                    # File konfigurasi lokal (tidak di-upload ke git)
├── .gitignore                              # Konfigurasi file yang diabaikan Git
├── run_django.bat                          # File launcher cepat aplikasi Django
├── run_streamlit.bat                       # File launcher cepat aplikasi Streamlit
├── sleep_project/                          # Konfigurasi inti proyek Django
│   ├── settings.py                         # Pengaturan Django
│   ├── urls.py                             # Root routing
│   ├── wsgi.py
│   └── asgi.py
├── predictor/                              # Aplikasi utama analisis tidur
│   ├── models.py                           # Model database SleepPrediction
│   ├── views.py                            # Logika inferensi, kalkulasi BMI, & endpoint API
│   ├── ai_services.py                      # Integrasi Gemini REST API & Smart Fallback
│   ├── urls.py                             # Routing URL modul & API AJAX
│   └── templates/predictor/index.html      # Tampilan UI Kalkulator, Hipnogram, & Chatbot
├── db.sqlite3                              # Database SQLite pencatat riwayat
└── README.md                               # Dokumentasi komprehensif proyek
```

---

## 📝 Lisensi & Catatan Akademik
Proyek ini dikembangkan untuk keperluan akademik mata kuliah **Sistem Cerdas**.  
*Alat bantu skrining awal dan edukasi pola tidur, bukan pengganti pemeriksaan medis profesional.*
