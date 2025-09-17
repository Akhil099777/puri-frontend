import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def normalize_token(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = s.replace(" _", "_").replace("_ ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "datasets" / "DiseaseAndSymptoms.csv"
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)

    # Use only first 6 symptom columns
    symptom_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5", "Symptom_6"]

    # Build symptom sets per row
    X_symptoms = []
    for _, row in df.iterrows():
        symptoms = []
        for col in symptom_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                symptoms.append(normalize_token(val))
        X_symptoms.append(list(set(symptoms)))

    y = df["Disease"].values

    # Encode symptoms into multi-hot vectors
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_symptoms)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model + encoder
    joblib.dump(clf, model_dir / "symptom_disease_model.joblib")
    joblib.dump(mlb, model_dir / "symptom_encoder.joblib")

    print(f"Saved model and encoder in {model_dir}")


if __name__ == "__main__":
    main()
