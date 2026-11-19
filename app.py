import os
import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st
import joblib

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Prediksi Kualitas & Gangguan Tidur",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk memperindah tampilan
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. Inisialisasi Database SQLite untuk Riwayat Prediksi
# -------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "riwayat_prediksi.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            gender TEXT,
            age INTEGER,
            sleep_duration REAL,
            quality_of_sleep INTEGER,
            physical_activity INTEGER,
            stress_level INTEGER,
            bmi_category TEXT,
            heart_rate INTEGER,
            daily_steps INTEGER,
            prediction TEXT,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction_history(data, prediction, probability):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO riwayat (
                timestamp, gender, age, sleep_duration, quality_of_sleep,
                physical_activity, stress_level, bmi_category, heart_rate,
                daily_steps, prediction, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["gender"],
            int(data["age"]),
            float(data["sleep_duration"]),
            int(data["quality_of_sleep"]),
            int(data["physical_activity"]),
            int(data["stress_level"]),
            data["bmi_category"],
            int(data["heart_rate"]),
            int(data["daily_steps"]),
            prediction,
            float(probability)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Gagal menyimpan ke database: {e}")

def get_prediction_history(limit=10):
    try:
        conn = sqlite3.connect(DB_PATH)
        df_history = pd.read_sql_query(
            f"SELECT timestamp AS 'Waktu', gender AS 'Gender', age AS 'Usia', "
            f"sleep_duration AS 'Durasi (Jam)', quality_of_sleep AS 'Kualitas', "
            f"stress_level AS 'Stres', bmi_category AS 'BMI', prediction AS 'Hasil Prediksi', "
            f"ROUND(probability, 2) AS 'Probabilitas (%)' "
            f"FROM riwayat ORDER BY id DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df_history
    except Exception:
        return pd.DataFrame()

init_db()

# -------------------------------------------------------------
# 2. Pemuatan Model ML & Dataset dengan Caching
# -------------------------------------------------------------
@st.cache_resource
def load_ml_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_tidur.joblib")
    if not os.path.exists(model_path):
        st.error("File 'model_tidur.joblib' tidak ditemukan! Silakan jalankan 'train.py' terlebih dahulu.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), "Sleep_health_and_lifestyle_dataset.csv")
    df = pd.read_csv(csv_path)
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("Normal").replace({"None": "Normal"})
    return df

model = load_ml_model()
df_dataset = load_dataset()

# -------------------------------------------------------------
# 3. Header Aplikasi
# -------------------------------------------------------------
st.markdown("<div class='main-header'>🌙 Sistem Prediksi Kualitas Tidur & Deteksi Dini Gangguan Tidur</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Aplikasi Sistem Cerdas berbasis Machine Learning (Random Forest) untuk menganalisis pola hidup dan deteksi dini risiko insomnia serta sleep apnea.</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# 4. Form Antarmuka UI Interaktif (9 Fitur Input)
# -------------------------------------------------------------
st.subheader("📋 Masukkan Data Kesehatan & Gaya Hidup Pengguna")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 👤 Profil Demografis")
    gender = st.selectbox(
        "Jenis Kelamin (Gender)",
        options=["Male", "Female"],
        help="Pilih jenis kelamin biologis pengguna"
    )
    age = st.slider(
        "Usia (Tahun)",
        min_value=18,
        max_value=85,
        value=32,
        step=1
    )
    bmi_category = st.selectbox(
        "Kategori BMI (Indeks Massa Tubuh)",
        options=["Normal", "Normal Weight", "Overweight", "Obese"],
        help="Normal/Normal Weight: 18.5-24.9 | Overweight: 25-29.9 | Obese: >=30"
    )

with col2:
    st.markdown("##### 🛏️ Pola & Kualitas Tidur")
    sleep_duration = st.slider(
        "Durasi Tidur Harian (Jam)",
        min_value=3.0,
        max_value=12.0,
        value=7.0,
        step=0.1,
        help="Rata-rata waktu tidur Anda setiap malam dalam hitungan jam"
    )
    quality_of_sleep = st.slider(
        "Kualitas Tidur (Skala 1 - 10)",
        min_value=1,
        max_value=10,
        value=7,
        step=1,
        help="1: Sangat Buruk / Sering Terbangun, 10: Sangat Nyenyak / Segar"
    )
    stress_level = st.slider(
        "Tingkat Stres (Skala 1 - 10)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="1: Bebas Stres / Relaks, 10: Tekanan Sangat Berat"
    )

with col3:
    st.markdown("##### 🏃 Aktivitas & Kondisi Fisik")
    physical_activity = st.slider(
        "Tingkat Aktivitas Fisik (Menit/Hari)",
        min_value=0,
        max_value=180,
        value=60,
        step=5,
        help="Durasi olahraga atau kegiatan fisik aktif per hari"
    )
    heart_rate = st.slider(
        "Denyut Jantung Istirahat (BPM)",
        min_value=50,
        max_value=120,
        value=72,
        step=1,
        help="Denyut nadi istirahat dalam satuan detak per menit"
    )
    daily_steps = st.slider(
        "Jumlah Langkah Harian",
        min_value=1000,
        max_value=20000,
        value=7000,
        step=500,
        help="Estimasi jumlah langkah kaki Anda setiap hari"
    )

# Tombol Prediksi
btn_predict = st.button("🔍 Prediksi Kondisi Tidur", type="primary", use_container_width=True)

# -------------------------------------------------------------
# 5. Logika Prediksi ML & Penjelasan AI
# -------------------------------------------------------------
if btn_predict:
    # Konversi input kategorikal ke numerik sesuai spesifikasi training
    gender_num = 1 if gender == "Male" else 0
    bmi_map = {"Normal": 0, "Normal Weight": 0, "Overweight": 1, "Obese": 2}
    bmi_num = bmi_map.get(bmi_category, 0)

    # Susun DataFrame input dengan urutan kolom persis sama dengan X saat pelatihan
    input_features = [
        "Gender_num",
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "BMI_num",
        "Heart Rate",
        "Daily Steps"
    ]
    
    input_df = pd.DataFrame([[
        gender_num,
        age,
        sleep_duration,
        quality_of_sleep,
        physical_activity,
        stress_level,
        bmi_num,
        heart_rate,
        daily_steps
    ]], columns=input_features)

    # Lakukan Prediksi
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    prob_dict = {cls: prob * 100 for cls, prob in zip(classes, probabilities)}
    current_prob = prob_dict.get(prediction, 0.0)

    # Simpan ke Database SQLite
    save_prediction_history({
        "gender": gender,
        "age": age,
        "sleep_duration": sleep_duration,
        "quality_of_sleep": quality_of_sleep,
        "physical_activity": physical_activity,
        "stress_level": stress_level,
        "bmi_category": bmi_category,
        "heart_rate": heart_rate,
        "daily_steps": daily_steps
    }, prediction, current_prob)

    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi Machine Learning")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if prediction == "Normal":
            st.success(
                f"### ✅ Status: Normal\n"
                f"Model memprediksi pola tidur Anda berada dalam kategori **Normal** dengan tingkat keyakinan **{current_prob:.2f}%**."
            )
        elif prediction == "Insomnia":
            st.warning(
                f"### ⚠️ Status: Terindikasi Insomnia\n"
                f"Model memprediksi adanya indikasi gangguan **Insomnia** dengan tingkat keyakinan **{current_prob:.2f}%**."
            )
        elif prediction == "Sleep Apnea":
            st.error(
                f"### 🚨 Status: Terindikasi Sleep Apnea\n"
                f"Model memprediksi risiko gangguan **Sleep Apnea** dengan tingkat keyakinan **{current_prob:.2f}%**."
            )

        # Tampilkan distribusi probabilitas semua kelas
        st.markdown("##### 📊 Distribusi Probabilitas Kelas:")
        for cls in classes:
            p_val = prob_dict[cls]
            st.write(f"**{cls}**: {p_val:.2f}%")
            st.progress(float(p_val / 100))

    with res_col2:
        # Fitur Penjelasan AI & Rekomendasi Kesehatan Cerdas
        st.markdown("##### 💡 Fitur Penjelasan AI & Rekomendasi Kesehatan")
        saran_box = ""
        if prediction == "Normal":
            saran_box = """
            **Kondisi Tidur Anda Sangat Baik!**
            - **Pertahankan Konsistensi:** Pertahankan jam tidur dan bangun yang teratur setiap hari.
            - **Aktivitas Fisik:** Pola jalan harian dan olahraga Anda mendukung relaksasi ritme sirkadian.
            - **Tips Manajemen:** Tetap jaga hidrasi tubuh, batasi konsumsi kafein di sore hari, dan hindari paparan layar gadget 30 menit sebelum istirahat.
            """
        elif prediction == "Insomnia":
            saran_box = f"""
            **Rekomendasi Pemulihan Insomnia:**
            - **Manajemen Stres:** Tingkat stres Anda tercatat di angka **{stress_level}/10**. Lakukan latihan pernapasan (metode 4-7-8) atau meditasi relaksasi sebelum tidur.
            - **Higienitas Tidur (Sleep Hygiene):** Ciptakan kamar yang sejuk, gelap, dan hening. Jangan gunakan tempat tidur untuk bekerja.
            - **Batasi Stimulan:** Hindari kopi, teh, dan rokok minimal 6 jam sebelum waktu tidur.
            - **Aktivitas Fisik:** Usahakan jalan pagi selama 20-30 menit untuk meregulasi hormon melatonin alami.
            """
        elif prediction == "Sleep Apnea":
            saran_box = f"""
            **Rekomendasi Terindikasi Sleep Apnea:**
            - **Evaluasi Faktor Risiko BMI:** Kategori BMI Anda adalah **{bmi_category}**. Penurunan berat badan secara terprogram sangat efektif membuka saluran pernapasan.
            - **Posisi Tidur Miring (Side-Sleeping):** Hindari tidur telentang karena lidah dan jaringan lunak tenggorokan dapat menyumbat jalan napas.
            - **Perhatikan Gejala Tambahan:** Waspadai tanda dengkuran keras, napas terhenti sejenak, atau terbangun dengan mulut kering.
            - **Konsultasi Medis:** Sangat dianjurkan untuk berkonsultasi dengan dokter spesialis THT atau dokter spesialis paru untuk evaluasi polisomnografi (*sleep study*).
            """
        st.info(saran_box)

# -------------------------------------------------------------
# 6. Visualisasi Grafik Komparasi Dataset
# -------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Visualisasi Grafik: Rata-rata Durasi Tidur Berdasarkan Gangguan Tidur")

chart_col1, chart_col2 = st.columns([2, 1])

# Hitung rata-rata durasi tidur per kategori gangguan tidur
avg_sleep_by_disorder = (
    df_dataset.groupby("Sleep Disorder")["Sleep Duration"]
    .mean()
    .round(2)
)

with chart_col1:
    st.markdown("**Perbandingan Rata-rata Durasi Tidur (Sleep Duration - Jam) di Dataset:**")
    st.bar_chart(avg_sleep_by_disorder)

with chart_col2:
    st.markdown("**📌 Insight Statistik Dataset:**")
    for disorder, avg_dur in avg_sleep_by_disorder.items():
        st.metric(label=f"Rata-rata: {disorder}", value=f"{avg_dur} Jam")
    st.caption("Data diambil dari 374 responden 'Sleep Health and Lifestyle Dataset'.")

# -------------------------------------------------------------
# 7. Riwayat Prediksi Pengguna (SQLite Storage)
# -------------------------------------------------------------
st.markdown("---")
with st.expander("🕒 Riwayat Prediksi Tersimpan (Database SQLite)", expanded=False):
    history_df = get_prediction_history(limit=10)
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.write("Belum ada riwayat prediksi yang tersimpan.")
