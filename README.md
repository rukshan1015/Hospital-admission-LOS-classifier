# Length of Stay (LOS) Classification Pipeline

This repository contains an end-to-end machine learning pipeline for predicting **hospital length of stay (LOS) at admission**.  
The model classifies patients into two groups:

- **0–2 days**
- **2+ days**

## Project Structure

All core components are located in the same folder: :contentReference[oaicite:0]{index=0}

- **`data/`** → `choices.json` (values of each attribute), `sampledata.csv`, `empty.csv` (template for batch inference)
- **`app/`** → FastAPI app (`main.py`, `model.py`, `shap.py` for feature importance)
- **`ml/`** → Training scripts (`LOSclassifier.py`)
- **`models/`** → End-to-end pipeline (`LOS_classifier_pipeline.joblib`)
- **`fastapi_UIs/`** → FastAPI + Gradio UI scripts (`fastapi_batch.py`, `fastapi_single_inference.py`)
- **`local_UIs/`** → Local Gradio UI scripts (`batch.py`, `single_inference.py`)

## Features

- Full ML pipeline (data processing → training → evaluation → inference)
- Gradio UI for:
  - Entering patient details manually (single prediction)
  - Uploading a CSV (`empty.csv`) for batch predictions
- Ready-to-use model artifacts and reproducible setup :contentReference[oaicite:1]{index=1}

--- 

## Installation
1. Clone this repo:
   ```bash
   git clone https://github.com/rukshan1015/Hospital-admission-LOS-classifier.git
   cd Hospital-admission-LOS-classifier.git
2. ```bash
   pip install -r requirements.txt

## Local Usage

### 1. Local Gradio UIs

   From the project root:

   #### Batch inference UI
   ```bash
   python local_UIs\batch.py
   ```
   #### Single-patient inference UI
   ```bash
   python local_UIs\single_inference.py
   ```
   Then open the URL Gradio prints (typically http://127.0.0.1:<port>)
   
### 2. FastAPI API + FastAPI UIs

   ```bash
   fastapi dev app\main.py  # Endpoint deployement
   ```
   This exposes the API (e.g., /get_json, /get_shap) as defined in app/main.py.

   The FastAPI-based UIs in fastapi_UIs/ can be run from any folder, and they will call the FastAPI endpoints.

## Docker Usage (Gradio UIs in a Container)

### 1. Building Docker image

   From the project root (where the Dockerfile lives):
   ```bash
   docker build --no-cache -t los1.0 .
   ```
   Feel free to change "los1.0" to any image name/tag you prefer.

### 2. Run the single-inference UI (port 7860)

   This runs local_UIs/single_inference.py inside the container and exposes it on your machine at http://localhost:7860:

   ```bash
   docker run --rm -it -p 7860:7860 los1.0
   ```
   * --rm → remove the container when it stops

   * -it → run interactively and show logs in your terminal

   * -p 7860:7860 → map host port 7860 → container port 7860

   Then open:

      http://localhost:7860


### 3. Run the batch-inference UI (port 7861)

   The batch UI is implemented in local_UIs/batch.py and is configured to listen on port 7861 inside the container. You can run it from the same image by overriding the command at runtime (no rebuild needed):

   ```bash
   docker run --rm -it -p 7861:7861 los1.0 python local_UIs/batch.py
   ```
   Then open:

      http://localhost:7861

   This way:

   * Both UIs share the same image (los1.0)

   * Single UI → default CMD from the Dockerfile (port 7860)

   * Batch UI → same image, different command (port 7861)

## Notes

   * This project is intended for educational and research purposes.

   * The Docker setup is focused on running the local Gradio UIs. The FastAPI app (app/) is kept for API-style use cases and can be run separately without Docker, or integrated into a multi-container setup later if needed.

