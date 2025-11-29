from __future__ import annotations
import os, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from ml.LOSclassifier import DataCleanerClassifier 

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/LOS_classifier_pipeline.joblib"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

_model = None

def get_model():
    global _model

    if _model is None:  # If there is no cache
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")
        
        _model = joblib.load(MODEL_PATH)

    return _model


def predict_rows(rows:List[Dict[str, Any]])->Dict[str,Any]:
    
    model= get_model()
    df_raw = pd.DataFrame(rows)
    X, _ = DataCleanerClassifier(df_raw)
    proba_pos = model.predict_proba(X)[:, 1]
    labels = (proba_pos >= THRESHOLD).astype(int)

    return {
        "threshold": THRESHOLD,
        "proba_positive": proba_pos.tolist(),
        "label_int": labels.tolist()
    }

