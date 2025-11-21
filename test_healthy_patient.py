#!/usr/bin/env python3
"""
Quick test to verify healthy patient bug is fixed
"""

import pandas as pd
from src.utils.data_validator import DataValidator
from src.utils.model_utils import ModelPredictor

print("=" * 70)
print("🧪 TESTING HEALTHY PATIENT PREDICTION")
print("=" * 70)

# Super healthy person - should get LOW risk
healthy_patient = {
    'age_category': '25-29',
    'sex': 'Female',
    'height_cm': 165,
    'weight_kg': 60,
    'bmi': 22.0,
    'exercise': 'Yes',
    'smoking_history': 'No',
    'alcohol_consumption': 0,
    'fruit_consumption': 60,
    'green_vegetables_consumption': 60,
    'fried_potato_consumption': 0,
    'general_health': 'Excellent',
    'checkup': 'Within the past year',
    'diabetes': 'No',
    'depression': 'No',
    'arthritis': 'No',
    'skin_cancer': 'No',
    'other_cancer': 'No'
}

print("\n📋 Patient Profile:")
print(f"   Age: {healthy_patient['age_category']}")
print(f"   BMI: {healthy_patient['bmi']} (Normal)")
print(f"   Exercise: {healthy_patient['exercise']}")
print(f"   Smoking: {healthy_patient['smoking_history']}")
print(f"   General Health: {healthy_patient['general_health']}")
print(f"   Health Conditions: None")

# Validate and preprocess
print("\n🔄 Processing input...")
validator = DataValidator()
is_valid, errors, warnings = validator.validate_input(healthy_patient)

if not is_valid:
    print("❌ Validation failed:")
    for error in errors:
        print(f"   {error}")
    exit(1)

input_df = validator.preprocess_for_model(healthy_patient)

print(f"✅ Validation passed")
print(f"\n📊 Generated {input_df.shape[1]} features:")

# Check if features are being created correctly
feature_check = {
    'Height_(cm)': input_df['Height_(cm)'].iloc[0],
    'Weight_(kg)': input_df['Weight_(kg)'].iloc[0],
    'BMI': input_df['BMI'].iloc[0],
    'FriedPotato_Consumption': input_df['FriedPotato_Consumption'].iloc[0],
    'Exercise_Yes': input_df['Exercise_Yes'].iloc[0],
    'Smoking_History_Yes': input_df['Smoking_History_Yes'].iloc[0],
}

print("\n🔍 Key Features:")
for feat, val in feature_check.items():
    print(f"   {feat}: {val}")

# Load models and predict
print("\n🤖 Loading models...")
predictor = ModelPredictor('models')
predictor.load_models()

print(f"✅ Loaded {len(predictor.models)} models")

# Make prediction
print("\n🔬 Making prediction...")
prediction = predictor.predict(input_df)

# Display results
print("\n" + "=" * 70)
print("📊 PREDICTION RESULTS")
print("=" * 70)
print(f"Risk Score: {prediction['risk_percentage']:.1f}%")
print(f"Prediction: {prediction['prediction_label']}")
print(f"Confidence: {prediction['confidence']*100:.1f}%")

# Evaluate if fix worked
print("\n" + "=" * 70)
if prediction['risk_percentage'] < 30:
    print("✅ TEST PASSED - Healthy patient shows LOW RISK")
    print(f"   Expected: <30%, Got: {prediction['risk_percentage']:.1f}%")
elif prediction['risk_percentage'] < 50:
    print("⚠️  TEST MARGINAL - Risk is moderate")
    print(f"   Expected: <30%, Got: {prediction['risk_percentage']:.1f}%")
else:
    print("❌ TEST FAILED - Healthy patient still shows HIGH RISK")
    print(f"   Expected: <30%, Got: {prediction['risk_percentage']:.1f}%")
    print("\n🔍 Individual model predictions:")
    if 'individual_models' in prediction:
        for model_name, results in prediction['individual_models'].items():
            print(f"   {model_name}: {results['probability_disease']*100:.1f}%")
print("=" * 70)
