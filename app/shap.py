import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from ml.LOSclassifier import DataCleanerClassifier
from app.model import get_model 

def feature_importance(row: dict):
    df = pd.DataFrame([row])
    feature_names = [
        'age_group', 'gender', 'race', 'ethnicity', 'type_of_admission', 'ccsr_diagnosis_description',
        'ccsr_procedure_description','apr_drg_description', 'apr_mdc_description', 'apr_severity_of_illness',
        'apr_risk_of_mortality', 'apr_medical_surgical', 'payment_typology_1','emergency_department_indicator', 
        'payment_typology_2', 'payment_typology_3', 'birth_weight', 'birth_weight_missing'
    ]
    
    X, _ = DataCleanerClassifier(df)

    model_pipeline=get_model()

    try:
        
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

        return f'<img src="data:image/png;base64,{plot_base64}" style="width:100%; max-width:800px;">'

        
    except Exception as e:
        # If SHAP fails, just show basic feature info
        return f"<p>Feature importance visualization not available. Error: {str(e)}</p>"