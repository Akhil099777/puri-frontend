import argparse
from pathlib import Path
import joblib
import json
import re

def normalize_token(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = s.replace(" _", "_").replace("_ ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

ROOT = Path(__file__).resolve().parent
model_path = ROOT / "models" / "symptom_disease_model.joblib"
enc_path = ROOT / "models" / "symptom_encoder.joblib"

clf = joblib.load(model_path)
mlb = joblib.load(enc_path)

parser = argparse.ArgumentParser(description="Predict disease from symptoms")
parser.add_argument("--symptoms", type=str, required=True,
                    help="Comma-separated symptoms, e.g. 'itching, skin_rash, fever'")
args = parser.parse_args()

raw = [normalize_token(s) for s in args.symptoms.split(",")]
raw = [s for s in raw if s]  # drop empties
if not raw:
    print("No valid symptoms were provided.")
    raise SystemExit(1)

unknown = [s for s in raw if s not in mlb.classes_]
if unknown:
    print('Warning: unknown symptoms ignored:', ', '.join(unknown))
known = [s for s in raw if s in mlb.classes_]
if not known:
    print('None of the provided symptoms are in the trained vocabulary.')
    raise SystemExit(1)
X = mlb.transform([known])
pred = clf.predict(X)[0]

# Probabilities if supported
prob_str = ""
if hasattr(clf, "predict_proba"):
    import numpy as np
    probs = clf.predict_proba(X)[0]
    top_idx = int(probs.argmax())
    prob_str = f" (p={probs[top_idx]:.3f})"

    # also show top-5
    order = np.argsort(probs)[::-1][:5]
    top5 = [(clf.classes_[i], float(probs[i])) for i in order]
    print("Top-5 probabilities:")
    for cls, p in top5:
        print(f"  {cls:30s} {p:.3f}")

print(f"Predicted disease: {pred}{prob_str}")