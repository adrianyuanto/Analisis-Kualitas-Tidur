import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def train_and_compare_models():
    dataset_path = os.path.join(os.path.dirname(__file__), "Sleep_health_and_lifestyle_dataset.csv")
    print(f"[INFO] Membaca dataset dari: {dataset_path}")
    
    # 1. Baca dataset
    df = pd.read_csv(dataset_path)
    
    # 2. Isi nilai kosong (NaN) atau 'None' pada kolom target 'Sleep Disorder' dengan 'Normal'
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("Normal").replace({"None": "Normal"})
    
    # 3. Ubah kolom kategori teks:
    gender_map = {"Male": 1, "Female": 0}
    df["Gender_num"] = df["Gender"].map(gender_map)
    
    bmi_map = {
        "Normal": 0,
        "Normal Weight": 0,
        "Overweight": 1,
        "Obese": 2
    }
    df["BMI_num"] = df["BMI Category"].map(bmi_map)
    
    # Pastikan tidak ada data null pada fitur yang dibuat
    df["Gender_num"] = df["Gender_num"].fillna(0)
    df["BMI_num"] = df["BMI_num"].fillna(0)
    
    # 4. Tentukan fitur X dan target y
    features = [
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
    target = "Sleep Disorder"
    
    X = df[features]
    y = df[target]
    
    print("\n[INFO] Distribusi Target 'Sleep Disorder':")
    print(y.value_counts())
    
    # 5. Split data 80:20 dengan random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Definisi 3 Model untuk Komparasi
    models_to_evaluate = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    comparison_results = []
    trained_models = {}
    
    print("\n" + "="*70)
    print("           KOMPARASI MODEL MACHINE LEARNING (SLEEP HEALTH)")
    print("="*70)
    
    for name, model in models_to_evaluate.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        trained_models[name] = model
        comparison_results.append({
            "Algoritma": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "_raw_acc": acc
        })
        
    comp_df = pd.DataFrame(comparison_results)
    # Urutkan berdasarkan akurasi tertinggi
    comp_df = comp_df.sort_values(by="_raw_acc", ascending=False).reset_index(drop=True)
    comp_df["Keterangan"] = ["Model Terpilih" if i == 0 else "Komparasi" for i in range(len(comp_df))]
    
    print(comp_df[["Algoritma", "Accuracy", "Precision", "Recall", "F1-Score", "Keterangan"]].to_string(index=False))
    print("="*70)
    
    # 7. Pilih Model Terbaik (Random Forest)
    best_model_name = comp_df.iloc[0]["Algoritma"]
    best_model = trained_models[best_model_name]
    print(f"\n[HASIL] Model terbaik terpilih: {best_model_name} dengan Akurasi {comp_df.iloc[0]['Accuracy']}")
    
    # Tampilkan evaluasi mendalam untuk model terbaik
    y_pred_best = best_model.predict(X_test)
    print("\n[LAPORAN KLASIFIKASI MODEL TERBAIK]")
    print(classification_report(y_test, y_pred_best))
    
    # 8. Simpan model ke model_tidur.joblib
    model_output_path = os.path.join(os.path.dirname(__file__), "model_tidur.joblib")
    joblib.dump(best_model, model_output_path)
    print(f"[BERHASIL] Model telah disimpan ke: {model_output_path}\n")

if __name__ == "__main__":
    train_and_compare_models()
