# Length of Stay (LOS) Classification Pipeline

This repository contains an end-to-end machine learning pipeline for predicting **hospital length of stay (LOS) at admission**.  
The model classifies patients into two groups:
- **0–2 days**
- **2+ days**

## Project Structure
All core components are located in the same folder:
- **`pipeline/`** → preprocessing, feature engineering, and model training scripts  
- **`empty.csv`** → template CSV for batch inference (users can fill with their own data)  
- **`gradio_ui.py`** → interactive Gradio interface for single-patient and batch predictions  

## Features
- Full ML pipeline (data processing → training → evaluation → inference)  
- Gradio UI for:
  - Entering patient details manually (single prediction)  
  - Uploading a CSV (`empty.csv`) for batch predictions  
- Ready-to-use model artifacts and reproducible setup  

## Usage
1. Clone this repo:
   ```bash
   git clone https://github.com/YourUsername/los-classification-pipeline.git
   cd los-classification-pipeline
2. ```bash
   pip install -r requirements.txt
3. Inferencing
   ```bash
   python LOSgradio_batch.py  # For batch inferecing 
   python LOSgradio_single_inference.py  #For individual predictions

Notes

Keep pipeline/, empty.csv, and gradio UIs in the same folder for smooth execution.

This project is intended for educational and research purposes.
