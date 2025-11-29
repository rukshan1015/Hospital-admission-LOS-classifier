

import pandas as pd
import numpy as np
import gradio as gr
import requests



API_BASE = "http://127.0.0.1:8000"  # FastAPI server base URL

def batchinference(input_file):
    # Read locally ti attach predictions & save a CSV to download
    df = pd.read_csv(input_file.name)

    # Send the uploaded file to FastAPI /predict_csv
    files = {"file": (input_file.name, open(input_file.name, "rb"), "text/csv")}
    r = requests.post(f"{API_BASE}/predict_csv", files=files, timeout=60)
    r.raise_for_status()
    data = r.json()  # {"threshold": ..., "proba_positive": [...], "label_int": [...]}

    # Convert API outputs
    p_pos = pd.Series(data["proba_positive"])
    y_int = pd.Series(data["label_int"])

    # Confidence = probability of the predicted class
    confidence = np.where(y_int == 1, p_pos, 1.0 - p_pos)
    confidence = np.round(confidence.astype(float), 4)

    # Map 1/0 to "0-2 days" and "2+ days
    label_map = {1: "0-2 days", 0: "2+ days"}
    labels_text = y_int.map(label_map)

    df_out = df.copy()
    df_out["Predicted LOS"] = labels_text
    df_out["Confidence"] = confidence

    out_path = "los_predictions_output.csv"
    df_out.to_csv(out_path, index=False)
    return out_path


# ✨ Gradio UI
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

