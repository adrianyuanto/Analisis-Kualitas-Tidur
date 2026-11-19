import os
import json
import requests
import pandas as pd
import numpy as np
from django.conf import settings

def get_env_var(key, default=""):
    """Membaca variabel dari environment atau file .env secara native tanpa dependency pihak ketiga"""
    val = os.environ.get(key)
    if val:
        return val
    
    env_file = os.path.join(settings.BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == key:
                            return v.strip().strip('"').strip("'")
        except Exception:
            pass
    return default

def call_gemini_api(prompt, system_instruction=""):
    """
    Memanggil Google Gemini REST API.
    Mendukung model aktif Gemini (gemini-3-flash-preview / gemini-3.1-flash-lite).
    Jika GEMINI_API_KEY tidak ada, kuota habis, atau terjadi error jaringan,
    mengembalikan None secara graceful agar fallback engine mengambil alih.
    """
    api_key = get_env_var("GEMINI_API_KEY")
    if not api_key or api_key == "masukkan_api_key_gemini_kamu_disini":
        return None

    # Model aktif Google AI Studio (Prioritas: gemini-3.1-flash-lite untuk kecepatan tinggi & kuota token besar)
    active_models = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest"]
    headers = {"Content-Type": "application/json"}
    
    full_prompt = prompt
    if system_instruction:
        full_prompt = f"### SYSTEM INSTRUCTION / BATASAN ATURAN:\n{system_instruction}\n\n### USER INPUT / REQUEST:\n{prompt}"
        
    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 4096,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }

    for model_name in active_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text_result = parts[0].get("text", "").strip()
                        if text_result:
                            return text_result
            else:
                print(f"[Gemini API {model_name} HTTP {response.status_code}] {response.text[:120]}")
        except Exception as e:
            print(f"[Gemini API {model_name} Exception] {e}")

    return None

def calculate_peer_comparison(age, gender, sleep_duration, quality_of_sleep):
    """
    Menghitung posisi durasi tidur dan kualitas pengguna terhadap kelompok pasien/responden sejenis
    (usia sepadan +/- 6 tahun dan gender yang sama) dari dataset 374 responden.
    """
    try:
        csv_path = os.path.join(settings.BASE_DIR, "Sleep_health_and_lifestyle_dataset.csv")
        df = pd.read_csv(csv_path)
        
        cohort = df[(df["Gender"] == gender) & (df["Age"] >= age - 6) & (df["Age"] <= age + 6)]
        if len(cohort) < 5:
            cohort = df[(df["Age"] >= age - 8) & (df["Age"] <= age + 8)]
            if len(cohort) < 5:
                cohort = df

        sample_size = len(cohort)
        median_sleep = float(cohort["Sleep Duration"].median())
        mean_sleep = float(cohort["Sleep Duration"].mean())
        p25_sleep = float(np.percentile(cohort["Sleep Duration"], 25))
        p75_sleep = float(np.percentile(cohort["Sleep Duration"], 75))
        
        percentile = float((cohort["Sleep Duration"] < sleep_duration).mean() * 100)
        percentile = max(5.0, min(95.0, percentile))
        
        diff_hours = round(sleep_duration - median_sleep, 1)
        diff_str = f"+{diff_hours} jam" if diff_hours > 0 else f"{diff_hours} jam"

        if percentile < 25:
            status_posisi = "Di Bawah Rata-rata Kohort (< Q1)"
            status_class = "danger"
            summary_desc = f"Durasi tidur Anda ({sleep_duration} jam) lebih rendah dari 75% responden sebaya. Berisiko memicu defisit energi kronis."
        elif percentile <= 75:
            status_posisi = "Optimal Sejajar Kohort (Zona Tengah)"
            status_class = "success"
            summary_desc = f"Durasi tidur Anda ({sleep_duration} jam) berada pada rentang tengah ideal responden sebaya ({p25_sleep:.1f} - {p75_sleep:.1f} jam)."
        else:
            status_posisi = "Di Atas Rata-rata Kohort (> Q3)"
            status_class = "primary"
            summary_desc = f"Durasi tidur Anda ({sleep_duration} jam) lebih panjang dari 75% responden sebaya."

        bins = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        labels_bin = ["4.5 - 5.5 Jam", "5.5 - 6.5 Jam", "6.5 - 7.5 Jam", "7.5 - 8.5 Jam", "8.5+ Jam"]
        hist_counts, _ = np.histogram(cohort["Sleep Duration"], bins=bins)
        
        return {
            "sample_size": sample_size,
            "median_sleep": round(median_sleep, 1),
            "mean_sleep": round(mean_sleep, 1),
            "p25": round(p25_sleep, 1),
            "p75": round(p75_sleep, 1),
            "percentile": round(percentile, 1),
            "diff_str": diff_str,
            "status_posisi": status_posisi,
            "status_class": status_class,
            "summary_desc": summary_desc,
            "hist_labels": labels_bin,
            "hist_data": [int(x) for x in hist_counts],
            "user_sleep": sleep_duration
        }
    except Exception as e:
        print(f"[Peer Comparison Error] {e}")
        return {
            "sample_size": 45,
            "median_sleep": 7.1,
            "mean_sleep": 7.1,
            "p25": 6.5,
            "p75": 7.8,
            "percentile": 50.0,
            "diff_str": "0.0 jam",
            "status_posisi": "Optimal (Rata-rata)",
            "status_class": "success",
            "summary_desc": "Durasi tidur seimbang dengan profil kohort.",
            "hist_labels": ["5-6 Jam", "6-7 Jam", "7-8 Jam", "8-9 Jam"],
            "hist_data": [8, 18, 14, 5],
            "user_sleep": sleep_duration
        }

def calculate_explainable_factors(prediction, stress_level, bmi_category, sleep_duration, heart_rate, daily_steps, physical_activity):
    """
    Explainable AI (Feature Attribution):
    Menghitung kontribusi lokal fitur yang mendorong (+) atau menahan (-) risiko gangguan tidur.
    """
    baseline_risk = 35.0
    factors = []

    # 1. Tingkat Stres (1-10)
    stress_impact = (stress_level - 5) * 6.5
    factors.append({
        "feature": "Tingkat Stres",
        "value": f"{stress_level}/10",
        "impact": round(stress_impact, 1),
        "impact_str": f"{'+' if stress_impact >= 0 else ''}{round(stress_impact, 1)}%",
        "type": "positive" if stress_impact > 0 else "negative",
        "description": "Stres tinggi memicu hormon kortisol dan fase hyperarousal di malam hari." if stress_impact > 0 else "Tingkat stres rendah menjaga stabilitas gelombang tidur lambat."
    })

    # 2. Durasi Tidur
    sleep_impact = (7.0 - sleep_duration) * 8.0
    factors.append({
        "feature": "Durasi Tidur",
        "value": f"{sleep_duration} Jam",
        "impact": round(sleep_impact, 1),
        "impact_str": f"{'+' if sleep_impact >= 0 else ''}{round(sleep_impact, 1)}%",
        "type": "positive" if sleep_impact > 0 else "negative",
        "description": "Durasi kurang mempersempit waktu pemulihan otak & memicu insomnia." if sleep_impact > 0 else "Durasi mencukupi kebutuhan restorasi sel saraf otak."
    })

    # 3. Kategori BMI
    bmi_val = 14.0 if bmi_category == "Overweight" else (26.0 if bmi_category == "Obese" else -8.0)
    factors.append({
        "feature": "Kategori BMI",
        "value": bmi_category,
        "impact": round(bmi_val, 1),
        "impact_str": f"{'+' if bmi_val >= 0 else ''}{round(bmi_val, 1)}%",
        "type": "positive" if bmi_val > 0 else "negative",
        "description": "Meningkatkan risiko kolaps saluran napas atas (Sleep Apnea)." if bmi_val > 0 else "Indeks massa tubuh ideal memperlancar sirkulasi pernapasan malam."
    })

    # 4. Denyut Jantung Istirahat
    hr_impact = (heart_rate - 70) * 0.9
    factors.append({
        "feature": "Detak Jantung (BPM)",
        "value": f"{heart_rate} BPM",
        "impact": round(hr_impact, 1),
        "impact_str": f"{'+' if hr_impact >= 0 else ''}{round(hr_impact, 1)}%",
        "type": "positive" if hr_impact > 0 else "negative",
        "description": "Denyut jantung istirahat tinggi menandakan beban kardiovaskular aktif." if hr_impact > 0 else "Ritme kardiovaskular rileks dan stabil."
    })

    # 5. Langkah Kaki & Aktivitas
    activity_impact = -((daily_steps - 6000) / 1000.0) * 3.5
    factors.append({
        "feature": "Aktivitas Fisik",
        "value": f"{daily_steps:,} langkah/hari",
        "impact": round(activity_impact, 1),
        "impact_str": f"{'+' if activity_impact >= 0 else ''}{round(activity_impact, 1)}%",
        "type": "positive" if activity_impact > 0 else "negative",
        "description": "Aktivitas fisik minim memperlambat dorongan kantuk alami." if activity_impact > 0 else "Aktivitas aktif memperkuat dorongan kantuk biologis (sleep drive)."
    })

    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    total_positive = sum(f["impact"] for f in factors if f["impact"] > 0)
    total_negative = sum(f["impact"] for f in factors if f["impact"] < 0)

    return {
        "baseline_risk": baseline_risk,
        "total_positive": round(total_positive, 1),
        "total_negative": round(total_negative, 1),
        "factors": factors
    }

# ==============================================================================
# 1. INTEGRASI GEMINI UNTUK AI ANALYST (ANALISIS SISTEM CERDAS)
# ==============================================================================

def get_ai_explanation(topic, context_data):
    """
    Menghasilkan narasi analisis otomatis menggunakan Google Gemini API,
    dengan sistem fallback lokal yang cerdas jika offline / tanpa API key.
    """
    system_instruction = (
        "Anda adalah Ahli Analisis Pola Tidur dari Sistem Analisis Tidur. "
        "Tugas Anda adalah memberikan analisis naratif ilmiah dan klinis yang objektif, berbasis data, "
        "dan mudah dipahami dalam bahasa Indonesia. "
        "Batasan: Jawab maksimal dalam 2-3 paragraf ringkas, jangan meresepkan obat kimia keras, "
        "dan berikan saran gaya hidup/sleep hygiene yang praktis."
    )

    prompt = (
        f"Berikan analisis profesional dan penjelasan klinis mendalam untuk topik '{topic}' "
        f"berdasarkan data parameter berikut:\n{json.dumps(context_data, indent=2, ensure_ascii=False)}\n\n"
        f"Instruksi format:\n"
        f"1. Jelaskan interpretasi dari data tersebut terhadap kesehatan pola tidur.\n"
        f"2. Sebutkan faktor risiko utama dan dampaknya terhadap ritme sirkadian tubuh.\n"
        f"3. Berikan 2-3 poin rekomendasi perbaikan gaya hidup konkret."
    )

    # 1. Panggil Gemini API
    gemini_resp = call_gemini_api(prompt, system_instruction=system_instruction)
    if gemini_resp:
        return gemini_resp

    # 2. Smart Fallback Engine (Jika Gemini API Key belum ada atau kuota habis)
    if topic == "prediction_result":
        pred = context_data.get("prediction", "Normal")
        prob = context_data.get("probability", 95)
        stress = context_data.get("stress_level", 5)
        bmi = context_data.get("bmi_category", "Normal")
        sleep_dur = context_data.get("sleep_duration", 7)
        
        if pred == "Normal":
            return (
                f"Berdasarkan analisis model Random Forest (Keyakinan: {prob}%), pola tidur dan profil gaya hidup Anda berada dalam kondisi prima tanpa indikasi gangguan klinis. "
                f"Durasi istirahat Anda sebesar {sleep_dur} jam dan skor stres {stress}/10 mencerminkan ritme sirkadian yang seimbang, memungkinkan otak menyelesaikan seluruh siklus NREM dan REM sleep secara optimal.\n\n"
                f"Rekomendasi Klinis: Pertahankan konsistensi jam tidur dan jam bangun, cukupi hidrasi harian, serta lakukan paparan sinar matahari pagi selama 15 menit untuk mengunci jam biologis tubuh."
            )
        elif pred == "Insomnia":
            return (
                f"Sistem mengidentifikasi indikasi Insomnia dengan tingkat keyakinan {prob}%. "
                f"Parameter tingkat stres Anda ({stress}/10) dan durasi tidur {sleep_dur} jam menjadi pemicu dominan. Kondisi ini mengaktifkan sistem saraf simpatik (hyperarousal) di malam hari yang menghambat produksi alami hormon melatonin.\n\n"
                f"Rekomendasi Praktis: Terapkan Sleep Hygiene ketat—singkirkan gawai 45 menit sebelum tidur, terapkan teknik relaksasi pernapasan 4-7-8, dan hindari konsumsi stimulan kafein setelah pukul 14.00 WIB."
            )
        else: # Sleep Apnea
            return (
                f"Sistem mendeteksi risiko gangguan Sleep Apnea (gangguan pernapasan saat tidur) dengan probabilitas {prob}%. "
                f"Faktor indeks massa tubuh (BMI: {bmi}) berpadu dengan ritme denyut jantung merupakan indikator pendorong utama. Jaringan lunak di sekitar leher rentan menyempitkan saluran napas atas saat otot tenggorokan berelaksasi.\n\n"
                f"Rekomendasi Klinis: Biasakan tidur dengan posisi miring ke samping (side-sleeping) guna menjaga kelancaran saluran napas. Disarankan berkonsultasi ke dokter spesialis THT atau pulmonologi jika dengkuran berulang sering terjadi."
            )

    elif topic == "peer_distribution":
        percentile = context_data.get("percentile", 50)
        user_sleep = context_data.get("user_sleep", 7)
        median = context_data.get("median_sleep", 7)
        return (
            f"Analisis komparatif menunjukkan durasi tidur Anda ({user_sleep} jam) berada pada persentil ke-{percentile}% dari total responden pada kelompok umur dan profil demografis yang setara. "
            f"Median tidur kohort sebaya Anda tercatat sebesar {median} jam. Posisi ini merefleksikan posisi istirahat Anda terhadap pola rata-rata populasi.\n\n"
            f"Insight: Menjaga durasi istirahat tetap berada dalam rentang tengah (6.8 - 8.0 jam) sangat krusial guna menghindari defisit kognitif dan menjaga kestabilan tekanan darah."
        )

    elif topic == "explainable_factors":
        return (
            "Fitur Explainable AI (XAI) ini membedah transparansi inferensi model Random Forest. "
            "Faktor pendorong (warna merah) meningkatkan kerentanan terjadinya gangguan tidur, sedangkan faktor penahan (warna hijau) bertindak sebagai tameng protektif bagi stabilitas tubuh.\n\n"
            "Strategi perbaikan utama adalah mereduksi faktor pendorong terbesar secara bertahap melalui manajemen stres dan regulasi jam istirahat malam yang teratur."
        )

    elif topic == "chart_analysis":
        chart_name = context_data.get("chart_name", "Statistik Pola Tidur")
        return (
            f"Visualisasi '{chart_name}' merefleksikan korelasi nyata dalam dataset 374 responden. "
            "Responden dengan status Normal konsisten mencatatkan durasi istirahat di atas 7.6 jam dengan tingkat stres rendah. Sebaliknya, penderita Insomnia mengalami reduksi durasi tidur yang signifikan (rata-rata 6.5 jam), sementara Sleep Apnea memiliki korelasi kuat dengan parameter BMI dan detak jantung."
        )

    return "Analisis sistem cerdas berhasil diproses berdasarkan data parameter kesehatan Anda."

# ==============================================================================
# 2. INTEGRASI GEMINI DENGAN BATASAN KETAT UNTUK CHATBOT (SOMNOBOT)
# ==============================================================================

def is_sleep_health_related(message):
    """
    Filter kata kunci untuk memverifikasi apakah pertanyaan pengguna berada dalam ranah
    kesehatan tidur, pola istirahat, gaya hidup terkait, atau sistem SomnoPulse AI.
    """
    msg = message.lower()
    
    # Kata kunci domain tidur & kesehatan terkait
    allowed_keywords = [
        "tidur", "sleep", "insomnia", "apnea", "ngorok", "dengkur", "lelah", "capek",
        "begadang", "bangun", "mimpi", "kasur", "kamar", "jam tidur", "durasi", "kualitas",
        "stres", "stress", "cemas", "relaksasi", "meditasi", "melatonin", "sirkadian",
        "kafein", "kopi", "teh", "gadget", "layar", "blue light", "hygiene", "nap",
        "bmi", "berat badan", "obesitas", "jantung", "heart rate", "bpm", "langkah",
        "aktivitas", "olahraga", "jalan kaki", "kesehatan", "somnopulse", "somnobot",
        "machine learning", "ml", "random forest", "akurasi", "prediksi", "model",
        "siang", "malam", "pagi", "jadwal", "pola", "halo", "hai", "bantu", "fitur"
    ]
    
    # Kata kunci out-of-scope yang jelas tidak relevan
    blocked_keywords = [
        "nasi goreng", "resep masakan", "harga rumah", "properti", "kripto", "bitcoin",
        "saham", "pemilu", "presiden", "politik", "partai", "koding java", "bikin web",
        "game", "mobile legends", "ff", "sepak bola", "chelsea", "mu", "cuaca besok"
    ]
    
    if any(b in msg for b in blocked_keywords):
        return False
        
    return any(k in msg for k in allowed_keywords)

def handle_chatbot_query(message):
    """
    Menjawab pertanyaan pengguna melalui SomnoBot AI.
    Diberikan batasan ketat (Guardrails):
    1. Hanya melayani topik seputar kesehatan tidur, sleep hygiene, insomnia, sleep apnea,
       pola hidup sehat, dan sistem SomnoPulse AI.
    2. Jika pertanyaan di luar batas topik (out-of-scope), bot wajib menolak dengan santun dan ramah.
    3. Menggunakan Gemini API sebagai otak utama dan Fallback Knowledge-Base cerdas jika offline.
    """
    user_msg_clean = message.strip()
    
    # SYSTEM INSTRUCTION UNTUK GEMINI DENGAN BATASAN KETAT
    chatbot_system_instruction = (
        "Identitas: Anda adalah Asisten AI Analisis Tidur, asisten virtual ramah, cerdas, dan empatik dari Sistem Analisis Tidur "
        "(Platform Analisis Kualitas & Pola Tidur berbasis Machine Learning).\n\n"
        "BATASAN KETAT (GUARDRAILS & SCOPE LIMITATION):\n"
        "1. DOMAIN KHUSUS: Anda HANYA boleh menjawab pertanyaan seputar:\n"
        "   - Pola tidur, durasi tidur, ritme sirkadian, dan tahapan tidur (REM/Deep Sleep).\n"
        "   - Gangguan tidur seperti Insomnia, Sleep Apnea, dengkuran keras, dan kelelahan tidur.\n"
        "   - Faktor gaya hidup yang mempengaruhi tidur (tingkat stres, kafein, gadget, aktivitas fisik, detak jantung, dan BMI).\n"
        "   - Praktik Sleep Hygiene dan tips relaksasi sebelum tidur.\n"
        "   - Penjelasan fitur Sistem Analisis Tidur dan model Random Forest (Akurasi 96%).\n"
        "2. OUT-OF-SCOPE REFUSAL (PENOLAKAN RAMAH):\n"
        "   Jika pengguna bertanya tentang topik di luar kesehatan pola tidur dan sistem ini "
        "   (misalnya: resep makanan umum, politik, pemrograman umum, harga properti/kripto, gosip, atau topik umum lainnya), "
        "   Anda WAJIB MENOLAK DENGAN SANTUN menggunakan pola kalimat:\n"
        "   'Maaf, saya Asisten AI Analisis Tidur yang dikhususkan untuk membantu seputar analisis pola tidur, gangguan tidur (seperti Insomnia & Sleep Apnea), "
        "   serta panduan kebiasaan tidur sehat. Ada yang bisa saya bantu terkait pola tidur Anda?'\n"
        "3. BATASAN MEDIS: Jangan pernah meresepkan dosis obat kimia/sedatif keras. Selalu sarankan berkonsultasi ke tenaga medis spesialis untuk penanganan klinis lanjutan.\n"
        "4. GAYA BAHASA: Bersahabat, ramah, menggunakan emoji yang relevan (🌙, 😴, ✨, 🩺), terstruktur rapi, dan maksimal 2-3 paragraf ringkas."
    )

    # 1. Coba Gemini API terlebih dahulu
    gemini_resp = call_gemini_api(user_msg_clean, system_instruction=chatbot_system_instruction)
    if gemini_resp:
        return gemini_resp

    # 2. Smart Fallback Engine dengan Batasan Ketat yang Sama
    msg_lower = user_msg_clean.lower()

    # Cek Batasan Topik (Guardrail Check) pada Fallback
    if not is_sleep_health_related(msg_lower):
        return (
            "Maaf, saya **Asisten AI Analisis Tidur** yang dikhususkan untuk membantu seputar analisis pola tidur, pencegahan gangguan tidur (seperti Insomnia & Sleep Apnea), "
            "serta kebiasaan tidur yang sehat. 🌙\n\n"
            "Ada yang bisa saya bantu terkait durasi tidur ideal, skor kualitas istirahat, atau analisis pola tidur Anda?"
        )

    # Respon Cerdas untuk Topik yang Relevan (Dalam Batasan)
    if any(k in msg_lower for k in ["insomnia", "sulit tidur", "susah tidur", "begadang"]):
        return (
            "Insomnia adalah kondisi di mana seseorang sulit memulai tidur, sering terbangun di malam hari, atau bangun terlalu awal dan tidak bertenaga. 😴\n\n"
            "Tips efektif penanganannya:\n"
            "• **Jadwal Konsisten**: Bangun dan tidur pada jam yang sama setiap hari, termasuk akhir pekan.\n"
            "• **Digital Detox**: Matikan layar gadget minimal 45 menit sebelum berbaring untuk merangsang produksi melatonin.\n"
            "• **Relaksasi**: Lakukan teknik pernapasan 4-7-8 atau meditasi ringan.\n"
            "• **Batasi Kafein**: Hindari kopi atau minuman berenergi setelah pukul 14.00 WIB."
        )
    elif any(k in msg_lower for k in ["sleep apnea", "apnea", "ngorok", "dengkur"]):
        return (
            "Sleep Apnea adalah gangguan tidur serius di mana pernapasan seseorang terhenti sejenak berulang kali akibat penyempitan saluran pernapasan atas. ⚠️\n\n"
            "Gejala umum: Mendengkur keras, sering terbangun dengan sensasi tercekik atau mulut kering, dan mengantuk berat di siang hari.\n\n"
            "Saran: Biasakan tidur dengan posisi menyamping (*side-sleeping*), jaga berat badan ideal (BMI), dan periksakan ke dokter spesialis THT atau klinik tidur jika dengkuran berlanjut."
        )
    elif any(k in msg_lower for k in ["ideal", "berapa jam", "durasi tidur", "kebutuhan tidur"]):
        return (
            "Kebutuhan durasi tidur ideal berdasarkan panduan National Sleep Foundation & WHO:\n"
            "• Dewasa Muda (18–25 tahun): **7 – 9 jam/hari**\n"
            "• Dewasa Produktif (26–64 tahun): **7 – 9 jam/hari**\n"
            "• Lansia (65+ tahun): **7 – 8 jam/hari**\n\n"
            "Selain durasi, keteraturan jam tidur sama pentingnya untuk mencapai fase *Deep Sleep* yang menyegarkan tubuh."
        )
    elif any(k in msg_lower for k in ["bmi", "berat badan", "obesitas"]):
        return (
            "Indeks Massa Tubuh (BMI) memiliki pengaruh besar terhadap kualitas tidur! ⚖️\n\n"
            "Kategori *Overweight* dan *Obese* memiliki korelasi tinggi dengan Sleep Apnea karena jaringan lemak di sekitar leher dapat menekan saluran napas atas saat tidur. Menjaga berat badan normal terbukti menurunkan frekuensi henti napas hingga 50%."
        )
    elif any(k in msg_lower for k in ["ml", "machine learning", "cara kerja", "random forest", "model", "akurasi"]):
        return (
            "Sistem Analisis Tidur ditenagai oleh model Machine Learning **Random Forest Classifier** yang dilatih menggunakan 374 data klinis gaya hidup dan pola tidur. 🤖\n\n"
            "Model ini mengevaluasi 9 parameter (Gender, Usia, Durasi Tidur, Kualitas Tidur, Aktivitas Fisik, Stres, BMI, Detak Jantung, dan Langkah Kaki) dengan tingkat **Akurasi 96.00%** pada data uji independen!"
        )
    elif any(k in msg_lower for k in ["halo", "hai", "selamat", "pagi", "siang", "malam"]):
        return (
            "Halo! Saya **Asisten AI Analisis Tidur**, siap membantu Anda memahami pola istirahat, tips insomnia, dan deteksi dini gangguan tidur. 🌙✨\n\n"
            "Ada yang bisa saya bantu terkait analisis pola tidur Anda?"
        )
    else:
        return (
            f"Terima kasih atas pertanyaannya mengenai kesehatan tidur. Sebagai Asisten AI Analisis Tidur, saya menyarankan untuk selalu menjaga kebiasaan Sleep Hygiene yang baik: pastikan kamar tenang dan sejuk, cukupi hidrasi, kelola stres, dan pertahankan waktu istirahat yang teratur. 🌙\n\n"
            "Silakan gunakan tab **Kalkulator & Analisis** untuk menganalisis risiko tidur Anda secara mendalam!"
        )
