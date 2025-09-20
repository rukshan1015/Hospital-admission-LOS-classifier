#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import joblib



# Step 1. Data cleaning & bucketing

def DataCleanerClassifier(df):
    # Drop columns not available at admission
    df = df.drop([
        'discharge_year', 'ccsr_diagnosis_code', 'ccsr_procedure_code',
        'apr_drg_code', 'apr_mdc_code', 'patient_disposition',
        'total_charges', 'total_costs',
        'hospital_service_area','hospital_county',
        'operating_certificate_number', 'apr_severity_of_illness_code',
        'permanent_facility_id', 'facility_name', 'zip_code_3_digits'
    ], axis=1, errors="ignore")

    # Drop rows with critical missing values
    df = df.dropna(subset=["apr_severity_of_illness", "apr_risk_of_mortality"])

    # Fill categorical missing where it has meaning
    fill_map = {
        #"zip_code_3_digits": "UNK",
        "ccsr_procedure_description": "NO_PROC",
        "payment_typology_2": "NONE",
        "payment_typology_3": "NONE",
        
    }
    for col, token in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(token).astype(str)

    # Birth weight cleaning + missing indicator
    df["birth_weight_missing"] = df["birth_weight"].isna().astype(int)

    
    to_be_numeric = ['birth_weight']

    for column in to_be_numeric:
        df[column]=(df[column].astype(str)
                    .str.replace(r'[+-,$\/]','',regex=True)
                    .replace('',np.nan)
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                  )

    target_column = 'length_of_stay'

    if target_column in df.columns:
    
        X=df.drop([target_column], axis=1) if target_column in df.columns else df
        y_raw=(df[target_column]
           .astype(str)
           .str.replace(r'[,$\/+-]','',regex=True)
           .pipe(pd.to_numeric, errors="coerce" if target_column in df.columns else None)
           #.pipe(np.log1p)
        )

        y = (y_raw <= 2).astype(int)  # Binary classes Class 1 - 0-2 days, Class 0 - 2+ days

        return X, y

    else:

        return df, None 


def preprocessing():
        
    # Step 1. Load and clean data
    
    df = pd.read_csv(r"C:\Users\ruksh\Desktop\AIagents\llm_engineering\data\nylos2023_500k.csv")
    X, y = DataCleanerClassifier(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    # Step 2. Preprocessing pipeline
    
    num_columns = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_columns = [c for c in X_train.columns if c not in num_columns]
    
    ct = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_columns),
        ("target", TargetEncoder(smooth='auto'), cat_columns)
    ], remainder="drop")

    return ct, X_train, X_test, y_train, y_test
    

def pipeline (ct, X_train, X_test, y_train, y_test):
    
    # Step 1. Base model
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False
    )
    
    pipeline_clf = Pipeline([
        ("preprocessing", ct),
        ("model", model)
    ])
    
    
    # Step 2. Class weights
    
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
    print("Class Weights:", class_weight_dict)
    
    
    sample_weights = np.array([class_weight_dict[cls] for cls in y_train])
    
    
    # Step 3. Bayesian hyperparameter tuning
    
    search_spaces = {
        "model__max_depth": Integer(3, 10),
        "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "model__n_estimators": Integer(200, 600),
        "model__min_child_weight": Integer(1, 10),
        "model__subsample": Real(0.5, 1.0),
        "model__colsample_bytree": Real(0.5, 1.0),
        "model__gamma": Real(0, 5.0)
    }
    
    opt = BayesSearchCV(
        pipeline_clf,
        search_spaces,
        n_iter=25,  # increase if time allows
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print("Training optimized classifier...")
    opt.fit(X_train, y_train, model__sample_weight=sample_weights)

    print("\nBest parameters:\n", opt.best_params_)

    # Step 4. Saving the best model

    joblib.dump(opt.best_estimator_, "LOS_classifier_pipeline.joblib")

    return opt
    
def evaluation(X_test, y_test):

    model_pipeline = joblib.load("LOS_classifier_pipeline.joblib")
    
    y_pred = model_pipeline.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0,1], display_labels=['2+ days', '0-2 days'])

    return model_pipeline, score, report, disp


# In[ ]:





# In[2]:


def shap_global(X_test, model_pipeline):
    ## Global SHAP values and plots 
    
    # Selecting a sample from X_test for SHAP
    
    X_sample = X_test.sample(1000, random_state=42)
    
    X_sample_transformed = model_pipeline.named_steps['preprocessing'].transform(X_sample)
    model = model_pipeline.named_steps['model']
    feature_names = model_pipeline.named_steps['preprocessing'].get_feature_names_out()
    
    explainer=shap.TreeExplainer(model, X_sample_transformed)
    
    shap_values = explainer(X_sample_transformed)
    
    shap_values.feature_names = feature_names
    
    shap.plots.beeswarm(shap_values)

    return shap_values, feature_names
    
def feature_importance(shap_values, feature_names):
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    importance_precentage = mean_abs_shap/mean_abs_shap.sum() * 100
    
    df_shap_percentages = pd.DataFrame(
        {"Feature Names":feature_names,
         "Importance (%)":importance_precentage}
    ).sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)
    
    top_features = df_shap_percentages.head(15)

    return top_features


# In[3]:


if __name__ == "__main__":
    
    ct, X_train, X_test, y_train, y_test= preprocessing()
    opt = pipeline(ct, X_train, X_test, y_train, y_test)
    model_pipeline, accuracy_score, classification_report, disp = evaluation(X_test, y_test)

    print("\nAccuracy:", accuracy_score)
    print("\nClassification report:\n", classification_report)
    disp.plot()
    plt.show()

    
    shap_values, feature_names = shap_global(X_test, model_pipeline)
    top_features = feature_importance(shap_values, feature_names)
    
    print(top_features)

