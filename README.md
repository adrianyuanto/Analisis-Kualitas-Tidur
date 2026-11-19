# 🌙 SomnoPulse AI - Sistem Cerdas Analisis & Deteksi Dini Gangguan Tidur

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Django](https://img.shields.io/badge/Django-5.x-darkgreen?logo=django)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Chart.js](https://img.shields.io/badge/Chart.js-Interactive-ff6384?logo=chartdotjs)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API%20%2B%20Fallback-purple?logo=google)

> **Tugas Mata Kuliah Sistem Cerdas (Intelligent Systems)**  
> Analisis Komprehensif Parameter Gaya Hidup & Pola Tidur Menggunakan Machine Learning dan Explainable AI.

---

## 📌 Tentang Proyek

**SomnoPulse AI** adalah aplikasi web sistem cerdas berbasis *Machine Learning* dan *Generative AI* yang dirancang untuk menganalisis parameter kesehatan, kebiasaan tidur, dan aktivitas fisik guna mendeteksi secara dini risiko gangguan tidur (**Normal**, **Insomnia**, atau **Sleep Apnea**).

Pengguna cukup memasukkan parameter gaya hidup harian (seperti durasi tidur, tingkat stres, kategori BMI, aktivitas fisik, denyut jantung istirahat, dan langkah kaki). Selanjutnya, sistem akan memberikan hasil klasifikasi berbasis model **Random Forest (Akurasi 96.00%)**, analisis posisi terhadap pasien sejenis (*peer comparison*), serta visualisasi *Explainable AI* yang transparan mengenai faktor penentu risiko.

---

## ✨ Fitur Utama

| Fitur | Keterangan |
| :--- | :--- |
| **Dashboard Analitik** | Grafik interaktif tren durasi tidur, tingkat stres, dan sebaran data responden per kategori gangguan. |
| **Kalkulator & Deteksi Risiko** | Inferensi instan status tidur (*Normal*, *Insomnia*, *Sleep Apnea*) beserta probabilitas keyakinan model (*probability score*). |
| **Posisi Responden Sejenis** | Analisis komparatif durasi tidur pengguna terhadap kohort usia dan gender serupa dari dataset 374 data. |
| **Explainable AI (XAI)** | Penjelasan faktor penentu yang mendorong (+) atau menahan (-) risiko gangguan tidur secara transparan. |
| **AI Health Analyst** | Narasi analisis otomatis klinis dari Google Gemini API dengan *smart fallback engine* jika kuota habis atau offline. |
| **AI Sleep Assistant (Chatbot)** | Widget asisten interaktif floating di pojok layar untuk konsultasi seputar tips insomnia, sleep apnea, dan sleep hygiene. |
| **Panduan Sleep Hygiene** | Modul edukasi standar medis durasi tidur WHO & Sleep Foundation, 5 pilar higienitas tidur, dan korelasi BMI. |
| **Riwayat Pemeriksaan** | Seluruh simulasi dan rekam jejak parameter tersimpan otomatis di database SQLite (`db.sqlite3`). |

---

## 🛠️ Teknologi yang Digunakan

- **Backend**: Django 5.x, Python 3.10+
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, `DecisionTreeClassifier`, `LogisticRegression`), Joblib
- **Data & Analisis**: Pandas, NumPy
- **Visualisasi**: Chart.js (Grafik interaktif, responsif, & dinamis)
- **Generative AI**: Google Gemini API (REST API + Smart Rule-Based Medical Engine Fallback)
- **Frontend**: Bootstrap 5.3, Bootstrap Icons, Google Fonts (Plus Jakarta Sans), Custom Modern Dark-Sidebar CSS
- **Database**: SQLite3

---

## 🧠 Model Machine Learning

Model dilatih menggunakan dataset klinis **374 data responden** dengan pembagian data *80% Data Latih (Train)* dan *20% Data Uji (Test)* secara terstratifikasi (*stratified split*).

### Fitur Input Evaluasi (Bebas Data Leakage):
- **Demografi**: `Gender_num` (Pria: 1, Wanita: 0), `Age` (Usia)
- **Pola Tidur & Beban Mental**: `Sleep Duration` (Jam), `Quality of Sleep` (Skala 1-10), `Stress Level` (Skala 1-10)
- **Kondisi Fisik & Kardio**: `BMI_num` (Normal: 0, Overweight: 1, Obese: 2), `Physical Activity Level` (Menit/hari), `Heart Rate` (BPM), `Daily Steps` (Langkah/hari)

### Hasil Komparasi Model:

| Algoritma | Accuracy | Precision | Recall | F1-Score | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Classifier** | **96.00%** | **96.06%** | **96.00%** | **95.99%** | **Model Terpilih** |
| Decision Tree Classifier | 94.67% | 95.02% | 94.67% | 94.75% | Komparasi |
| Logistic Regression | 93.33% | 93.34% | 93.33% | 93.23% | Komparasi |

> Model terbaik disimpan ke file `model_tidur.joblib` dan dimuat secara otomatis oleh server Django untuk inferensi waktu nyata (*real-time*).

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat
Pastikan Python 3.10+ sudah terpasang di komputer Anda. Instal dependensi yang diperlukan:
```bash
pip install django pandas scikit-learn joblib requests
```

*(Catatan: Jika menggunakan Laragon pada Windows, gunakan path Python Laragon yang sudah tersedia).*

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
SECRET_KEY=django-insecure-somnopulse-sleep-health-ai-secret-key-2024
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
GEMINI_API_KEY=masukkan_api_key_gemini_kamu_disini
```

> 💡 **Catatan Penting**: `GEMINI_API_KEY` bersifat opsional. Jika dikosongkan, seluruh fitur analisis AI dan Chatbot tetap bekerja 100% normal dengan sistem *Smart Medical Fallback Engine*.

---

### 3. Migrasi Database
Jalankan perintah migrasi Django untuk memastikan skema tabel siap:
```bash
python manage.py migrate
```

---

### 4. Menjalankan Server Aplikasi
Jalankan server pengembangan Django:
```bash
python manage.py runserver
```

Buka peramban (browser) dan akses:
👉 `http://127.0.0.1:8000/`

*(Tersedia pula file peluncur otomatis: cukup dobel klik `run_django.bat` di Windows).*

---

## 📁 Struktur Direktori Proyek

```plaintext
.
├── Sleep_health_and_lifestyle_dataset.csv  # Dataset sumber kesehatan tidur (374 baris)
├── train.py                                # Skrip pelatihan & komparasi 3 algoritma ML
├── model_tidur.joblib                      # File bobot model Random Forest terbaik
├── app.py                                  # Opsi antarmuka Streamlit data science
├── manage.py                               # CLI entrypoint Django
├── .env.example                            # Template file konfigurasi lingkungan
├── .env                                    # File konfigurasi aktif lokal
├── run_django.bat                          # Launcher cepat Django
├── run_streamlit.bat                       # Launcher cepat Streamlit
├── sleep_project/                          # Konfigurasi inti proyek Django
│   ├── settings.py                         # Settings Django (memuat .env)
│   ├── urls.py                             # Root routing
│   ├── wsgi.py
│   └── asgi.py
├── predictor/                              # Modul aplikasi sistem cerdas
│   ├── models.py                           # Model SQLite SleepPrediction
│   ├── views.py                            # Logika inferensi, analytics, & API endpoints
│   ├── ai_services.py                      # Layanan Gemini API, XAI, Peer Comp, & Chatbot
│   ├── urls.py                             # Routing URL modul & API AJAX
│   └── templates/predictor/index.html      # Tampilan Dashboard modern, XAI, & Chatbot
├── db.sqlite3                              # Database SQLite pencatat riwayat
└── README.md                               # Dokumentasi komprehensif proyek
```

---

## 📝 Lisensi & Hak Cipta
Dikembangkan untuk keperluan akademis mata kuliah **Sistem Cerdas**.
