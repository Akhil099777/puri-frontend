from __future__ import annotations
from typing import Dict, List
import pandas as pd


def build_feature_frame(
    age: int,
    gender: str,
    symptoms: List[str],
) -> pd.DataFrame:
    """Minimal placeholder preprocessing that produces a small, generic feature set.

    NOTE: Replace with the exact preprocessing used during model training.
    """
    gender_male = 1 if gender.lower().startswith("m") else 0
    gender_female = 1 if gender.lower().startswith("f") else 0
    symptom_count = len(symptoms)

    data: Dict[str, float] = {
        "age": float(age),
        "gender_male": float(gender_male),
        "gender_female": float(gender_female),
        "symptom_count": float(symptom_count),
    }
    return pd.DataFrame([data])


def parse_batch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure minimal columns exist; try to normalize batch inputs.

    Expected columns (case-insensitive): name, age, gender, symptoms
    """
    normalized = df.copy()
    normalized.columns = [c.strip().lower() for c in normalized.columns]

    # Fill missing required columns
    for col in ["name", "age", "gender", "symptoms"]:
        if col not in normalized.columns:
            if col == "symptoms":
                normalized[col] = ""
            else:
                normalized[col] = None

    # Coerce types
    normalized["age"] = pd.to_numeric(normalized["age"], errors="coerce").fillna(0).astype(int)
    normalized["gender"] = normalized["gender"].fillna("").astype(str)
    normalized["symptoms"] = normalized["symptoms"].fillna("").astype(str)

    return normalized
