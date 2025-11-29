# make_choices.py
import json, pandas as pd
from ml.LOSclassifier import DataCleanerClassifier

RAW_CSV = r"C:\Users\ruksh\Desktop\Fastapi\LOS_medical\data\sampledata.csv"   # your original pre-transform CSV
FEATURES = [
    "age_group","gender","race","ethnicity","type_of_admission",
    "ccsr_diagnosis_description","ccsr_procedure_description",
    "apr_drg_description","apr_mdc_description","apr_severity_of_illness",
    "apr_risk_of_mortality","apr_medical_surgical","payment_typology_1",
    "emergency_department_indicator","payment_typology_2","payment_typology_3",
    "birth_weight","birth_weight_missing",
]

df = pd.read_csv(RAW_CSV, low_memory=False)
X, _ = DataCleanerClassifier(df)

choices = {}
for col in FEATURES:
    vals = X[col].astype(str).dropna().unique().tolist() if col in X.columns else []
    try: vals.sort()
    except: pass
    choices[col] = vals

# sensible fallbacks
choices.setdefault("gender", ["F","M"])
choices.setdefault("emergency_department_indicator", ["Y","N"])
choices.setdefault("apr_severity_of_illness", ["1","2","3","4"])
choices.setdefault("apr_risk_of_mortality", ["1","2","3","4"])
choices.setdefault("apr_medical_surgical", ["Medical","Surgical"])
choices.setdefault("birth_weight_missing", ["0","1"])

with open("choices.json", "w", encoding="utf-8") as f:
    json.dump({"choices": choices}, f, ensure_ascii=False, indent=2)
