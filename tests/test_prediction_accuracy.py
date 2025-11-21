#!/usr/bin/env python3
"""
Comprehensive Prediction Validation Tests
Tests that model predictions are accurate across different risk profiles
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.data_validator import DataValidator
from src.utils.model_utils import ModelPredictor


class TestPredictionAccuracy:
    """Test suite for validating prediction accuracy"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.predictor = ModelPredictor('models')
        print("🤖 Loading models...")
        self.predictor.load_models()
        print(f"✅ Loaded {len(self.predictor.models)} models\n")
    
    def test_healthy_young_patient(self):
        """Test Case 1: Healthy young patient should get LOW risk"""
        print("=" * 70)
        print("TEST 1: Healthy Young Patient")
        print("=" * 70)
        
        patient = {
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
        
        print("\n📋 Profile:")
        print(f"   Age: {patient['age_category']}")
        print(f"   BMI: {patient['bmi']} (Normal)")
        print(f"   Exercise: {patient['exercise']}")
        print(f"   Smoking: {patient['smoking_history']}")
        print(f"   General Health: {patient['general_health']}")
        print(f"   Chronic Conditions: None")
        
        # Preprocess and predict
        input_df = self.validator.preprocess_for_model(patient)
        prediction = self.predictor.predict(input_df)
        
        risk = prediction['risk_percentage']
        print(f"\n📊 Results:")
        print(f"   Risk Score: {risk:.1f}%")
        print(f"   Prediction: {prediction['prediction_label']}")
        print(f"   Confidence: {prediction['confidence']*100:.1f}%")
        
        # Validation
        if risk < 30:
            print(f"\n✅ PASSED - Low risk as expected (<30%)")
            return True
        else:
            print(f"\n❌ FAILED - Expected <30%, got {risk:.1f}%")
            return False
    
    def test_high_risk_elderly_patient(self):
        """Test Case 2: High-risk elderly patient should get HIGH risk"""
        print("\n" + "=" * 70)
        print("TEST 2: High-Risk Elderly Patient")
        print("=" * 70)
        
        patient = {
            'age_category': '70-74',
            'sex': 'Male',
            'height_cm': 175,
            'weight_kg': 95,
            'bmi': 31.0,
            'exercise': 'No',
            'smoking_history': 'Yes',
            'alcohol_consumption': 20,
            'fruit_consumption': 5,
            'green_vegetables_consumption': 5,
            'fried_potato_consumption': 25,
            'general_health': 'Poor',
            'checkup': 'Never',
            'diabetes': 'Yes',
            'depression': 'Yes',
            'arthritis': 'Yes',
            'skin_cancer': 'No',
            'other_cancer': 'Yes'
        }
        
        print("\n📋 Profile:")
        print(f"   Age: {patient['age_category']}")
        print(f"   BMI: {patient['bmi']} (Obese)")
        print(f"   Exercise: {patient['exercise']}")
        print(f"   Smoking: {patient['smoking_history']}")
        print(f"   General Health: {patient['general_health']}")
        print(f"   Chronic Conditions: Diabetes, Depression, Arthritis, Cancer")
        
        # Preprocess and predict
        input_df = self.validator.preprocess_for_model(patient)
        prediction = self.predictor.predict(input_df)
        
        risk = prediction['risk_percentage']
        print(f"\n📊 Results:")
        print(f"   Risk Score: {risk:.1f}%")
        print(f"   Prediction: {prediction['prediction_label']}")
        print(f"   Confidence: {prediction['confidence']*100:.1f}%")
        
        # Validation
        if risk >= 65:
            print(f"\n✅ PASSED - High risk as expected (≥65%)")
            return True
        else:
            print(f"\n❌ FAILED - Expected ≥65%, got {risk:.1f}%")
            return False
    
    def test_moderate_risk_middle_aged(self):
        """Test Case 3: Moderate-risk middle-aged patient"""
        print("\n" + "=" * 70)
        print("TEST 3: Moderate-Risk Middle-Aged Patient")
        print("=" * 70)
        
        patient = {
            'age_category': '50-54',
            'sex': 'Male',
            'height_cm': 178,
            'weight_kg': 85,
            'bmi': 26.8,
            'exercise': 'Yes',
            'smoking_history': 'No',
            'alcohol_consumption': 7,
            'fruit_consumption': 30,
            'green_vegetables_consumption': 20,
            'fried_potato_consumption': 8,
            'general_health': 'Good',
            'checkup': 'Within the past year',
            'diabetes': 'No, pre-diabetes or borderline diabetes',
            'depression': 'No',
            'arthritis': 'Yes',
            'skin_cancer': 'No',
            'other_cancer': 'No'
        }
        
        print("\n📋 Profile:")
        print(f"   Age: {patient['age_category']}")
        print(f"   BMI: {patient['bmi']} (Overweight)")
        print(f"   Exercise: {patient['exercise']}")
        print(f"   Smoking: {patient['smoking_history']}")
        print(f"   General Health: {patient['general_health']}")
        print(f"   Chronic Conditions: Pre-diabetes, Arthritis")
        
        # Preprocess and predict
        input_df = self.validator.preprocess_for_model(patient)
        prediction = self.predictor.predict(input_df)
        
        risk = prediction['risk_percentage']
        print(f"\n📊 Results:")
        print(f"   Risk Score: {risk:.1f}%")
        print(f"   Prediction: {prediction['prediction_label']}")
        print(f"   Confidence: {prediction['confidence']*100:.1f}%")
        
        # Validation
        if 25 <= risk <= 70:
            print(f"\n✅ PASSED - Moderate risk as expected (25-70%)")
            return True
        else:
            print(f"\n❌ FAILED - Expected 25-70%, got {risk:.1f}%")
            return False
    
    def test_feature_scaling_validation(self):
        """Test Case 4: Verify scaler is being applied"""
        print("\n" + "=" * 70)
        print("TEST 4: Feature Scaling Validation")
        print("=" * 70)
        
        # Check that scaler exists
        if self.predictor.scaler is None:
            print("❌ FAILED - No scaler loaded!")
            return False
        
        print("✅ Scaler loaded successfully")
        
        # Check scaler has feature names
        if not hasattr(self.predictor.scaler, 'feature_names_in_'):
            print("❌ FAILED - Scaler missing feature_names_in_")
            return False
        
        scaled_features = self.predictor.scaler.feature_names_in_
        print(f"✅ Scaler has {len(scaled_features)} features")
        print(f"\n📊 Features being scaled:")
        for i, feat in enumerate(scaled_features, 1):
            print(f"   {i}. {feat}")
        
        # Verify expected features
        expected_numerical = [
            'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
            'Fruit_Consumption', 'Green_Vegetables_Consumption',
            'FriedPotato_Consumption', 'Age_Numeric', 
            'Lifestyle_Risk_Score', 'Health_Conditions_Count'
        ]
        
        missing = [f for f in expected_numerical if f not in scaled_features]
        if missing:
            print(f"\n❌ FAILED - Missing features: {missing}")
            return False
        
        print("\n✅ PASSED - All expected numerical features present in scaler")
        return True
    
    def test_prediction_range_validation(self):
        """Test Case 5: Ensure predictions are in valid range"""
        print("\n" + "=" * 70)
        print("TEST 5: Prediction Range Validation")
        print("=" * 70)
        
        test_patients = [
            {'age_category': '18-24', 'sex': 'Female', 'height_cm': 160, 'weight_kg': 55, 
             'bmi': 21.5, 'exercise': 'Yes', 'smoking_history': 'No', 
             'alcohol_consumption': 0, 'fruit_consumption': 50, 
             'green_vegetables_consumption': 50, 'fried_potato_consumption': 2,
             'general_health': 'Excellent', 'checkup': 'Within the past year',
             'diabetes': 'No', 'depression': 'No', 'arthritis': 'No',
             'skin_cancer': 'No', 'other_cancer': 'No'},
            
            {'age_category': '60-64', 'sex': 'Male', 'height_cm': 170, 'weight_kg': 80,
             'bmi': 27.7, 'exercise': 'No', 'smoking_history': 'Yes',
             'alcohol_consumption': 15, 'fruit_consumption': 10,
             'green_vegetables_consumption': 10, 'fried_potato_consumption': 15,
             'general_health': 'Fair', 'checkup': 'Within the past 5 years',
             'diabetes': 'Yes', 'depression': 'No', 'arthritis': 'Yes',
             'skin_cancer': 'No', 'other_cancer': 'No'}
        ]
        
        all_valid = True
        for i, patient in enumerate(test_patients, 1):
            input_df = self.validator.preprocess_for_model(patient)
            prediction = self.predictor.predict(input_df)
            risk = prediction['risk_percentage']
            
            print(f"\nPatient {i}:")
            print(f"   Risk: {risk:.1f}%")
            print(f"   Confidence: {prediction['confidence']*100:.1f}%")
            
            # Validate range
            if not (0 <= risk <= 100):
                print(f"   ❌ Risk outside valid range [0, 100]")
                all_valid = False
            elif not (0.5 <= prediction['confidence'] <= 1.0):
                print(f"   ❌ Confidence outside valid range [0.5, 1.0]")
                all_valid = False
            else:
                print(f"   ✅ Valid")
        
        if all_valid:
            print("\n✅ PASSED - All predictions in valid ranges")
        else:
            print("\n❌ FAILED - Some predictions out of range")
        
        return all_valid
    
    def run_all_tests(self):
        """Run all test cases"""
        print("\n" + "🧪 " * 35)
        print("CARDIO FUSION ML MODEL VALIDATION TEST SUITE")
        print("🧪 " * 35 + "\n")
        
        tests = [
            ("Healthy Young Patient", self.test_healthy_young_patient),
            ("High-Risk Elderly Patient", self.test_high_risk_elderly_patient),
            ("Moderate-Risk Middle-Aged", self.test_moderate_risk_middle_aged),
            ("Feature Scaling", self.test_feature_scaling_validation),
            ("Prediction Ranges", self.test_prediction_range_validation)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                passed = test_func()
                results.append((test_name, passed))
            except Exception as e:
                print(f"\n❌ ERROR in {test_name}: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)
        
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status:12} - {test_name}")
        
        print("=" * 70)
        print(f"Results: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED - MODEL IS PRODUCTION READY!")
        else:
            print("⚠️ SOME TESTS FAILED - REVIEW ISSUES ABOVE")
        
        print("=" * 70)
        
        return passed_count == total_count


if __name__ == "__main__":
    tester = TestPredictionAccuracy()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)
