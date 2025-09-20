#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gradio as gr
import joblib
from LOSclassifier import DataCleanerClassifier


# In[2]:


def batchinference(input_file):
    
    model_pipeline = joblib.load("LOS_classifier_pipeline.joblib")
    
    df = pd.read_csv(input_file.name)
    X, _ = DataCleanerClassifier(df)
    
    y_pred = model_pipeline.predict(X)
    y_pred = pd.Series(y_pred).replace({1: "0-2 days", 0: "2+ days"})

    y_probs = model_pipeline.predict_proba(X)
    
    confidence = []
    for i, pred in enumerate(y_pred):
        class_index = 1 if pred == "0-2 days" else 0
        confidence.append(round(y_probs[i][class_index], 4))
    
    df_out = df.copy()
    df_out["Predicted LOS"] = y_pred
    df_out["Confidence"] = confidence

    out_path = "los_predictions_output.csv"
    df_out.to_csv(out_path, index=False)
    
    return out_path


# ✨ Flashy Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), css=".gr-button {background-color: #1f6feb !important; color: white;}") as demo:

    gr.Markdown(
        """
        # 🏥 NY Hospital Length of Stay Classifier
        Upload your hospital admissions data below to predict **Length of Stay (LOS)**:
        """,
        elem_id="title"
    )

    with gr.Accordion("📄 How to use this app", open=False):
        gr.Markdown("""
        1. Download and fill the template CSV with your patient-level data.
        2. The model supports predictions for NY hospital admissions.
        3. Output file will include:
            - Predicted class (`0-2 days` or `2+ days`)
            - Prediction confidence (between 0 and 1)

        **Note**: Your uploaded file will not be stored.
        """)

    # For batch inference

    with gr.Row():
        template_file = gr.DownloadButton(
            label="📥 Download Template",
            value="C:/Users/ruksh/Desktop/AIagents/llm_engineering/data/empty.csv"
        )


    with gr.Row():
        input_file = gr.File(label="📤 Upload your CSV file", file_types=[".csv"])
        run_button = gr.Button("🔍 Run Batch Inference")
        output_file = gr.File(label="📥 Download Results")

    run_button.click(fn=batchinference, inputs=input_file, outputs=output_file)        

demo.launch(share=True)

