import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ======================
# Load dataset
# ======================
df = pd.read_csv("DiseaseAndSymptoms.csv")

# Consider only first 6 symptom columns
symptom_cols = [f"Symptom_{i}" for i in range(1, 7)]
df = df[["Disease"] + symptom_cols]

# Fill NaN with "None"
df = df.fillna("None")

# Build symptom list (unique values across first 6 columns, excluding "None")
symptom_list = sorted(set(df[symptom_cols].values.flatten()) - {"None"})

# ======================
# Build training data
# ======================
X = []
y = []

for _, row in df.iterrows():
    symptoms = set(row[symptom_cols].values)
    symptoms = [s for s in symptoms if s != "None"]
    features = [1 if s in symptoms else 0 for s in symptom_list]
    X.append(features)
    y.append(row["Disease"])

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ======================
# Train model
# ======================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

# ======================
# Save model + encoders
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/disease_model.joblib")
joblib.dump(encoder, "models/label_encoder.joblib")
joblib.dump(symptom_list, "models/symptom_list.joblib")

print("✅ Model and files saved in 'models/' directory")
