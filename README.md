# Length of Stay (LOS) Classification Pipeline

This repository contains an end-to-end machine learning pipeline for predicting **hospital length of stay (LOS) at admission**.  
The model classifies patients into two groups:
- **0–2 days**
- **2+ days**

## Project Structure
All core components are located in the same folder:
- **`data/`** → choices.json (values of each attribute), sampledata.csv, empty.csv (to provide a template)  
- **`app`** → FastAPI - main.py, model.py, shap.py (for feature importance)  
- **`ml`** → Training scripts - LOSClassier.py 
- **`models`**→ End-to-end pipeline - LOS_classifier_pipeline.joblib
- **`fastapi_UIs`** → FastAPI gradio UI scripts - fastapi_batch.py, fastapi_single_inference.py
- **`local_UIs`** → Local gradio UI scripts - batch.py, single_inference.py


## Features
- Full ML pipeline (data processing → training → evaluation → inference)  
- Gradio UI for:
  - Entering patient details manually (single prediction)  
  - Uploading a CSV (`empty.csv`) for batch predictions  
- Ready-to-use model artifacts and reproducible setup  

## Usage
1. Clone this repo:
   ```bash
   git clone https://github.com/rukshan1015/Hospital-admission-LOS-classifier.git
   cd Hospital-admission-LOS-classifier.git
2. ```bash
   pip install -r requirements.txt
3. Inferencing

   local inferencing
   ```bash
   python local_UIs\batch.py  # For batch inferecing 
   python local_UIs\single_inference.py  #For individual predictions
   ```
   FastAPI
   ```bash
   fastapi dev app\main.py  # Endpoint deployement
   ```
   Run Fastapi UIs (located in fastapi_UIs) from any folder


Notes


This project is intended for educational and research purposes.
