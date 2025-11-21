#!/usr/bin/env python3
"""Test SHAP explainability"""

import pandas as pd
import numpy as np
from src.utils.data_validator import DataValidator
from src.utils.model_utils import ModelPredictor
from src.utils.shap_explainer import SHAPExplainer

print("=" * 70)
print("🧪 TESTING SHAP EXPLAINABILITY")
print("=" * 70)

# Healthy patient
data = {
    'age_category': '55-59', 'sex': 'Male', 'height_cm': 175,
    'weight_kg': 85, 'bmi': 27.8, 'exercise': 'No', 'smoking_history': 'Yes',
    'alcohol_consumption': 10, 'fruit_consumption': 15,
    'green_vegetables_consumption': 10, 'fried_potato_consumption': 8,
    'general_health': 'Fair', 'checkup': 'Within the past year',
    'diabetes': 'No', 'depression': 'No', 'arthritis': 'Yes',
    'skin_cancer': 'No', 'other_cancer': 'No'
}

print("\n📋 Patient: 55yo, BMI 27.8, Smoking, No exercise, Arthritis")

# Preprocess
validator = DataValidator()
input_df = validator.preprocess_for_model(data)

# Load model and background data
print("\n🤖 Loading models and background data...")
predictor = ModelPredictor('models')
predictor.load_models()

# Load background data
train_data = pd.read_csv('data/processed/train_data.csv', nrows=100)
background_data = train_data.drop('Heart_Disease', axis=1).sample(50, random_state=42)

print(f"✅ Background data: {background_data.shape}")

# Get model
model = list(predictor.models.values())[0]
model_name = list(predictor.models.keys())[0]

print(f"✅ Using {model_name} for SHAP")

# Create SHAP explainer
print("\n🔄 Creating SHAP explainer...")
try:
    explainer = SHAPExplainer(model, background_data)
    print("✅ Explainer created")

    print("\n🔬 Generating explanation...")
    explanation = explainer.explain_prediction(input_df)

    if 'error' in explanation:
        print(f"❌ SHAP Error: {explanation['error']}")
    else:
        print("✅ SHAP explanation generated!")

        print("\n📊 Top Risk-Increasing Factors:")
        for feat, val in explanation['top_positive'][:3]:
            print(f"   {feat}: +{val:.4f}")

        print("\n📊 Top Risk-Decreasing Factors:")
        for feat, val in explanation['top_negative'][:3]:
            print(f"   {feat}: {val:.4f}")

        print("\n💡 Recommendations:")
        recs = explainer.get_recommendations(explanation)
        for i, rec in enumerate(recs[:3], 1):
            print(f"   {i}. {rec}")

        print("\n✅ SHAP TEST PASSED!")

except Exception as e:
    print(f"❌ SHAP TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
