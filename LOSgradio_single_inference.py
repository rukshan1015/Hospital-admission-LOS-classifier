#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
import pandas as pd
import joblib
from LOSclassifier import DataCleanerClassifier
import os
from datetime import datetime

# Extracting feature names
df_original = pd.read_csv(r"C:\Users\ruksh\Desktop\AIagents\llm_engineering\data\nylos2023_500k.csv")
X_original, _ = DataCleanerClassifier(df_original)
prediction_log = []

# Prediction function for single prediction
def single_prediction(*args):
    
    model_pipeline = joblib.load("LOS_classifier_pipeline.joblib")
    
    feature_names = [
        'age_group', 'gender', 'race', 'ethnicity', 'type_of_admission', 'ccsr_diagnosis_description',
        'ccsr_procedure_description','apr_drg_description', 'apr_mdc_description', 'apr_severity_of_illness',
        'apr_risk_of_mortality', 'apr_medical_surgical', 'payment_typology_1','emergency_department_indicator', 
        'payment_typology_2', 'payment_typology_3', 'birth_weight', 'birth_weight_missing'
    ]

    input_data = dict(zip(feature_names, args))
    df = pd.DataFrame([input_data])
    X, _ = DataCleanerClassifier(df)

    y_pred = model_pipeline.predict(X)[0]
    y_label = "0-2 days" if y_pred == 1 else "2+ days"

    y_prob = model_pipeline.predict_proba(X)[0]

    result = (
        f"Prediction: {y_label}\n"
        f"Probability of 0-2 days: {y_prob[1]*100:.1f}%\n"
        f"Probability of 2+ days: {y_prob[0]*100:.1f}%"
    )

    # Simple feature importance without SHAP
    try:
        import shap
        import matplotlib.pyplot as plt
        import io
        import base64
        
        X_transformed = model_pipeline.named_steps['preprocessing'].transform(X)
        model = model_pipeline.named_steps['model']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        
        # If binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # positive class, first instance
        else:
            shap_vals = shap_values[0]  # first instance
        
        # Create a simple bar chart of feature importances
        feature_names_transformed = model_pipeline.named_steps['preprocessing'].get_feature_names_out()
        
        # Get feature importance values
        importance_values = abs(shap_vals)
        
        # Create DataFrame with transformed feature names and their importance
        feature_importance_df = pd.DataFrame({
            'transformed_feature': feature_names_transformed,
            'importance': importance_values
        })
        
        # Map original feature names to their Gradio component labels - DIRECT MAPPING
        feature_to_label_mapping = {
            'age_group': 'Age Group',
            'gender': 'Gender', 
            'race': 'Race',
            'ethnicity': 'Ethnicity',
            'type_of_admission': 'Admission Type',
            'ccsr_diagnosis_description': 'Diagnosis Description',
            'ccsr_procedure_description': 'Procedure Description',
            'apr_drg_description': 'DRG Description',
            'apr_mdc_description': 'MDC Description',
            'apr_severity_of_illness': 'Severity',
            'apr_risk_of_mortality': 'Risk of Mortality',
            'apr_medical_surgical': 'Medical/Surgical',
            'payment_typology_1': 'Primary Payer',
            'emergency_department_indicator': 'Emergency Visit?',
            'payment_typology_2': 'Secondary Payer',
            'payment_typology_3': 'Tertiary Payer',
            'birth_weight': 'Birth Weight',
            'birth_weight_missing': 'Missing Birth Weight'
        }
        
        # Simple approach: directly sum importance by original feature names
        grouped_importance = {}
        
        # Initialize all features with 0
        for feature in feature_names:
            label = feature_to_label_mapping.get(feature, feature)
            grouped_importance[label] = 0
        
        # Sum importance values for each feature
        for i, transformed_name in enumerate(feature_names_transformed):
            importance = importance_values[i]
            
            # Try to match each transformed feature to original features
            matched = False
            for j, original_feature in enumerate(feature_names):
                # Check if transformed name contains the original feature name
                if original_feature in transformed_name:
                    label = feature_to_label_mapping.get(original_feature, original_feature)
                    grouped_importance[label] += importance
                    matched = True
                    break
            
            # If no match found, create a generic label
            if not matched:
                clean_name = transformed_name.replace('target_', '').replace('_', ' ').title()
                if clean_name not in grouped_importance:
                    grouped_importance[clean_name] = importance
        
        # Remove any features with 0 importance
        grouped_importance = {k: v for k, v in grouped_importance.items() if v > 0}
        
        # Convert to DataFrame and get top 10
        importance_df = pd.DataFrame([
            {'feature': label, 'importance': imp} 
            for label, imp in grouped_importance.items()
        ]).sort_values('importance', ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['feature'], fontsize=14)
        plt.xlabel('Feature Importance (absolute SHAP value)', fontsize=16)
        plt.title('Top 10 Most Important Features for This Prediction (Descending Order)', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.gca().invert_yaxis()  # Invert y-axis so highest importance is at top
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plot_base64 = base64.b64encode(plot_data).decode()
        shap_html = f'<img src="data:image/png;base64,{plot_base64}" style="width:100%; max-width:800px;">'
        
    except Exception as e:
        # If SHAP fails, just show basic feature info
        shap_html = f"<p>Feature importance visualization not available. Error: {str(e)}</p>"

    # Log input + result
    prediction_record = input_data.copy()
    prediction_record["Prediction Label"] = y_label
    prediction_record["Probability 0-2 days"] = round(y_prob[1]*100, 2)
    prediction_record["Probability 2+ days"] = round(y_prob[0]*100, 2)
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
        age_group = gr.Dropdown(choices=sorted(X_original['age_group'].astype(str).dropna().unique().tolist()), label='Age Group')
        gender = gr.Dropdown(choices=sorted(X_original['gender'].astype(str).map({"M": "Male", "F": "Female"}).dropna().unique().tolist()), label='Gender')
        race = gr.Dropdown(choices=sorted(X_original['race'].astype(str).dropna().unique().tolist()), label='Race')
        ethnicity = gr.Dropdown(choices=sorted(X_original['ethnicity'].astype(str).dropna().unique().tolist()), label='Ethnicity')

    with gr.Row():
        type_of_admission = gr.Dropdown(choices=sorted(X_original['type_of_admission'].astype(str).dropna().unique().tolist()), label='Admission Type')
        ccsr_diagnosis_description = gr.Dropdown(choices=sorted(X_original['ccsr_diagnosis_description'].astype(str).dropna().unique().tolist()), label='Diagnosis Description')
        ccsr_procedure_description = gr.Dropdown(choices=sorted(X_original['ccsr_procedure_description'].astype(str).dropna().unique().tolist()), label='Procedure Description')

    with gr.Row():    
        apr_drg_description = gr.Dropdown(choices=sorted(X_original['apr_drg_description'].astype(str).dropna().unique().tolist()), label='DRG Description')
        apr_mdc_description = gr.Dropdown(choices=sorted(X_original['apr_mdc_description'].astype(str).dropna().unique().tolist()), label='MDC Description')
        apr_severity_of_illness = gr.Dropdown(choices=sorted(X_original['apr_severity_of_illness'].astype(str).dropna().unique().tolist()), label='Severity')

    with gr.Row():
        apr_risk_of_mortality = gr.Dropdown(choices=sorted(X_original['apr_risk_of_mortality'].astype(str).dropna().unique().tolist()), label='Risk of Mortality')
        apr_medical_surgical = gr.Dropdown(choices=sorted(X_original['apr_medical_surgical'].astype(str).dropna().unique().tolist()), label='Medical/Surgical')
        payment_typology_1 = gr.Dropdown(choices=sorted(X_original['payment_typology_1'].astype(str).dropna().unique().tolist()), label='Primary Payer')
        emergency_department_indicator = gr.Dropdown(choices=sorted(X_original['emergency_department_indicator'].astype(str).dropna().map({'N':'No','Y':'Yes'}).unique().tolist()), label='Emergency Visit?')

    with gr.Row():
        payment_typology_2 = gr.Dropdown(choices=sorted(X_original['payment_typology_2'].astype(str).dropna().unique().tolist()), label='Secondary Payer')
        payment_typology_3 = gr.Dropdown(choices=sorted(X_original['payment_typology_3'].astype(str).dropna().unique().tolist()), label='Tertiary Payer')
        birth_weight = gr.Number(label='Birth Weight')
        birth_weight_missing = gr.Dropdown(choices=sorted(X_original['birth_weight_missing'].astype(str).dropna().map({"0": "No", "1": "Yes"}).unique().tolist()), label='Missing Birth Weight')

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

