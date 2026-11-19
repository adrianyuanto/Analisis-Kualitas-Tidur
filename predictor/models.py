from django.db import models

class SleepPrediction(models.Model):
    gender = models.CharField(max_length=10, verbose_name="Jenis Kelamin")
    age = models.IntegerField(verbose_name="Usia (Tahun)")
    sleep_duration = models.FloatField(verbose_name="Durasi Tidur (Jam)")
    quality_of_sleep = models.IntegerField(verbose_name="Kualitas Tidur (1-10)")
    physical_activity_level = models.IntegerField(verbose_name="Aktivitas Fisik (Menit/Hari)")
    stress_level = models.IntegerField(verbose_name="Tingkat Stres (1-10)")
    bmi_category = models.CharField(max_length=20, verbose_name="Kategori BMI")
    heart_rate = models.IntegerField(verbose_name="Denyut Jantung (BPM)")
    daily_steps = models.IntegerField(verbose_name="Langkah Harian")
    
    prediction = models.CharField(max_length=50, verbose_name="Hasil Prediksi")
    probability = models.FloatField(verbose_name="Probabilitas (%)")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Waktu Prediksi")

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Riwayat Prediksi Tidur"
        verbose_name_plural = "Riwayat Prediksi Tidur"

    def __str__(self):
        return f"{self.created_at.strftime('%Y-%m-%d %H:%M')} - {self.prediction} ({self.probability:.1f}%)"
