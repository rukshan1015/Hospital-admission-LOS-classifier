#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
import pandas as pd
import joblib
import os, requests
from datetime import datetime


# Extracting feature names

API_BASE   = os.getenv("API_BASE", "http://127.0.0.1:8000")   # (We should add to server IP/host when remote)
PRED_URL   = f"{API_BASE}/get_json"                       
CHOICES_URL= f"{API_BASE}/static/choices.json"
SHAP_URL   = F"{API_BASE}/get_shap" 

prediction_log = []

choices = requests.get(CHOICES_URL, timeout=30).json()["choices"]

# Prediction function for single prediction

def single_prediction(*args):
    feature_names = [
        'age_group', 'gender', 'race', 'ethnicity', 'type_of_admission', 'ccsr_diagnosis_description',
        'ccsr_procedure_description','apr_drg_description', 'apr_mdc_description', 'apr_severity_of_illness',
        'apr_risk_of_mortality', 'apr_medical_surgical', 'payment_typology_1','emergency_department_indicator', 
        'payment_typology_2', 'payment_typology_3', 'birth_weight', 'birth_weight_missing'
    ]

    row = dict(zip(feature_names, args))

    ## call FastAPI
    # call pipeline
    resp = requests.post(PRED_URL, json={"rows": [row]}, timeout=30)
    resp.raise_for_status()

    data = resp.json()  # expects {"threshold":..., "proba_positive":[...], "label_int":[...]}

    label_int = int(data["label_int"][0])
    proba_pos = float(data["proba_positive"][0]) if "proba_positive" in data else None

    y_label = "0-2 days" if label_int == 1 else "2+ days"

    if proba_pos is not None:
        result = (
            f"Prediction: {y_label}\n"
            f"Probability of 0-2 days: {proba_pos*100:.1f}%\n"
            f"Probability of 2+ days: {(1.0 - proba_pos)*100:.1f}%"
        )
    else:
        result = f"Prediction: {y_label}"

    # call shap
    shap_resp = requests.post(SHAP_URL, json={"rows":[row]}, timeout=60)
    shap_resp.raise_for_status()
    shap_html = shap_resp.json().get("html", "<em>No SHAP available</em>")

    # Log input + result
    prediction_record = row.copy()
    prediction_record["Prediction Label"] = y_label
    if proba_pos is not None:
        prediction_record["Probability 0-2 days"] = round(proba_pos*100, 2)
        prediction_record["Probability 2+ days"]  = round((1.0 - proba_pos)*100, 2)
    prediction_record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_log.append(prediction_record)

    return result, shap_html


def download_log():
    if not prediction_log:
        return None  # No predictions to download
    
    file_path = "prediction_log.csv"
    
    # Convert current in-memory log to DataFrame
    new_df = pd.DataFrame(prediction_log)
    
    if os.path.exists(file_path):
        # Read existing CSV file (not Excel!)
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Save updated log as CSV
    combined_df.to_csv(file_path, index=False)
    
    # Clear the in-memory log (optional)
    prediction_log.clear()
    
    return file_path

# Gradio UI 
with gr.Blocks() as demo:
    gr.Markdown("## 🏥 Dynamic Patient Input — LOS Classifier")

    with gr.Row():
        age_group = gr.Dropdown(choices=choices.get('age_group',[]), label='Age Group', value=(choices["age_group"][0] if choices.get("age_group") else None))
        gender = gr.Dropdown(choices=[({"M":"Male","F":"Female","U":"Unknown"}.get(c,c), c) for c in choices.get("gender", [])], label="Gender", value=choices.get("gender", [None])[0])
        race = gr.Dropdown(choices=choices.get('race',[]), label='Race', value=(choices["race"][0] if choices.get("race") else None))
        ethnicity = gr.Dropdown(choices=choices.get('ethnicity',[]), label='Ethnicity', value=(choices["ethnicity"][0] if choices.get("ethnicity") else None))

    with gr.Row():
        type_of_admission = gr.Dropdown(choices=choices.get('type_of_admission',[]), label='Admission Type', value=(choices["type_of_admission"][0] if choices.get("type_of_admission") else None))
        ccsr_diagnosis_description = gr.Dropdown(choices=choices.get('ccsr_diagnosis_description',[]), label='Diagnosis Description', value=(choices["ccsr_diagnosis_description"][0] if choices.get("ccsr_diagnosis_description") else None))
        ccsr_procedure_description = gr.Dropdown(choices=choices.get('ccsr_procedure_description',[]), label='Procedure Description', value=(choices["ccsr_procedure_description"][0] if choices.get("ccsr_procedure_description") else None))

    with gr.Row():    
        apr_drg_description = gr.Dropdown(choices=choices.get('apr_drg_description',[]), label='DRG Description', value=(choices['apr_drg_description'][0] if choices.get('apr_drg_description') else None))
        apr_mdc_description = gr.Dropdown(choices=choices.get('apr_mdc_description',[]), label='MDC Description', value=(choices['apr_mdc_description'][0] if choices.get('apr_mdc_description') else None))
        apr_severity_of_illness = gr.Dropdown(choices=choices.get('apr_severity_of_illness',[]), label='Severity', value=(choices['apr_severity_of_illness'][0] if choices.get('apr_severity_of_illness') else None))

    with gr.Row():
        apr_risk_of_mortality = gr.Dropdown(choices=choices.get('apr_risk_of_mortality',[]), label='Risk of Mortality', value=(choices['apr_risk_of_mortality'][0] if choices.get('apr_risk_of_mortality') else None))
        apr_medical_surgical = gr.Dropdown(choices=choices.get('apr_medical_surgical',[]), label='Medical/Surgical', value=(choices['apr_medical_surgical'][0] if choices.get('apr_medical_surgical') else None))
        payment_typology_1 = gr.Dropdown(choices=choices.get('payment_typology_1',[]), label='Primary Payer', value=(choices['payment_typology_1'][0] if choices.get('payment_typology_1') else None))
        emergency_department_indicator = gr.Dropdown(choices=[({"Y":"Yes","N":"No"}.get(c, c), c) for c in choices.get("emergency_department_indicator", [])],label="Emergency Visit?",
                                                     value=choices.get("emergency_department_indicator", [None])[0])


    with gr.Row():
        payment_typology_2 = gr.Dropdown(choices=choices.get('payment_typology_2',[]), label='Secondary Payer', value=(choices['payment_typology_2'][0] if choices.get('payment_typology_2') else None))
        payment_typology_3 = gr.Dropdown(choices=choices.get('payment_typology_3',[]), label='Tertiary Payer', value=(choices['payment_typology_3'][0] if choices.get('payment_typology_3') else None))
        birth_weight = gr.Number(label='Birth Weight')
        birth_weight_missing = gr.Dropdown(choices=[({0:"No",1:"Yes","0":"No","1":"Yes"}.get(c, c), c) for c in choices.get("birth_weight_missing", [])], label="Missing Birth Weight?",
                                            value=choices.get("birth_weight_missing", [None])[0])


    input_components = [
        age_group, gender, race, ethnicity, type_of_admission, ccsr_diagnosis_description,
        ccsr_procedure_description, apr_drg_description, apr_mdc_description, apr_severity_of_illness,
        apr_risk_of_mortality, apr_medical_surgical, payment_typology_1, emergency_department_indicator, 
        payment_typology_2, payment_typology_3, birth_weight, birth_weight_missing
    ]
    
    submit = gr.Button("Predict")
    
    submit.click(
        fn=single_prediction,
        inputs=input_components,
        outputs=[
            gr.Textbox(label="Prediction Output"),
            gr.HTML(label="Feature Importance")
        ]
    )

    download_button = gr.Button("📥 Download Prediction Log")
    download_file = gr.File(label="Download CSV File")

    download_button.click(
        fn=download_log,
        inputs=[],
        outputs=[download_file]
    )

demo.launch(share=True)

