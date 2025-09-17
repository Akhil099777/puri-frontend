from __future__ import annotations
import os
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd


class ModelNotAvailableError(Exception):
    pass


def load_models(models_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load models from joblib files defined in config.

    Returns a dict: { disease_name: model or None }
    """
    loaded_models: Dict[str, Any] = {}
    for disease_name, cfg in models_config.items():
        model_path = cfg.get("path")
        if not model_path or not os.path.exists(model_path):
            loaded_models[disease_name] = None
            continue
        try:
            model = joblib.load(model_path)
            loaded_models[disease_name] = model
        except Exception:
            loaded_models[disease_name] = None
    return loaded_models


def predict_probabilities(
    loaded_models: Dict[str, Any],
    input_features: pd.DataFrame,
) -> Dict[str, float]:
    """Predict probabilities for each disease if model is available.

    Returns probabilities in [0,1]. Missing models yield None and are filtered out by caller.
    """
    probabilities: Dict[str, float] = {}
    for disease_name, model in loaded_models.items():
        if model is None:
            continue
        try:
            if hasattr(model, "predict_proba"):
                prob_one = float(model.predict_proba(input_features)[:, 1].item())
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(input_features).item())
                prob_one = 1.0 / (1.0 + np.exp(-score))
            elif hasattr(model, "predict"):
                pred = float(model.predict(input_features).item())
                prob_one = max(0.0, min(1.0, pred))
            else:
                continue
            probabilities[disease_name] = prob_one
        except Exception:
            continue
    return probabilities


def apply_thresholds(
    probabilities: Dict[str, float],
    models_config: Dict[str, Dict[str, Any]],
    default_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """Return structured results with recommendation using per-model or default thresholds."""
    results: Dict[str, Dict[str, Any]] = {}
    for disease_name, prob in probabilities.items():
        model_cfg = models_config.get(disease_name, {})
        threshold = float(model_cfg.get("threshold", default_threshold))
        preventive_measures = model_cfg.get("preventive_measures", [])
        lifestyle_suggestions = model_cfg.get("lifestyle_suggestions", [])
        recommend_doctor = bool(prob >= threshold)
        results[disease_name] = {
            "probability": prob,
            "threshold": threshold,
            "preventive_measures": preventive_measures,
            "lifestyle_suggestions": lifestyle_suggestions,
            "recommend_doctor": recommend_doctor,
        }
    return results
