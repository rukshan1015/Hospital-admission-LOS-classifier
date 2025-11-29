from typing import Dict, List, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from app.model import get_model, predict_rows
from app.shap import feature_importance

app = FastAPI(
    title="LOS Classifier API",
    version="0.1.0",
    description="Simple API to run predictions using a saved LOS pipeline."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

class PredictRequest(BaseModel):
    rows: List[Dict[str,Any]]
"""
@app.get("/debug/ls")
def list_data():
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "files": sorted(p.name for p in DATA_DIR.glob("*"))
    }
"""

app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

@app.get("/healthz")

def health():
    _ = get_model() 

    return {'status':'ok'}


@ app.post("/get_json")  # For single prediction

def predict_row(req: PredictRequest):

    """
    Send features as JSON: {"rows": [ {feature: value, ...}, ... ]}
    """

    return predict_rows(req.rows)

@ app.post("/get_csv") # For batch prediction

async def predict_csv(file : UploadFile = File(...)):

    """
    Upload a CSV file; we'll convert it to rows and predict.
    The CSV must have the same column names the pipeline expects.
    """
    df = pd.read_csv(file.file)
    rows = df.to_dict(orient="records")
    return predict_rows(rows)

@ app.post("/get_shap") # For SHAP values for single inference

def get_shap(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="No rows provided")
    try:
        html = feature_importance(req.rows[0])
        return {"html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
set PYTHONPATH=.
fastapi dev app\main.py  #Once you are in the project folder

"""