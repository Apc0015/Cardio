"""
Unit tests for CardioFusion ModelPredictor and clinical risk inference.
"""
import pytest
from src.utils.data_validator import DataValidator
from src.utils.model_utils import ModelPredictor


@pytest.fixture
def predictor():
    mp = ModelPredictor()
    mp.load_models()
    return mp


@pytest.fixture
def preprocessed_features():
    v = DataValidator()
    sample = {
        'general_health': 'Good',
        'checkup': 'Within the past year',
        'exercise': 'Yes',
        'sex': 'Male',
        'age_category': '55-59',
        'height_cm': 175,
        'weight_kg': 80,
        'bmi': 26.1,
        'smoking_history': 'No',
        'alcohol_consumption': 0,
        'fruit_consumption': 30,
        'green_vegetables_consumption': 30,
        'fried_potato_consumption': 4,
        'diabetes': 'No',
        'depression': 'No',
        'arthritis': 'No',
        'other_cancer': 'No',
    }
    return v.preprocess_for_model(sample)


def test_models_load_successfully(predictor):
    available = predictor.get_available_models()
    assert len(available) >= 3
    assert 'Logistic Regression' in available
    assert 'Random Forest' in available


def test_predict_returns_valid_probabilities(predictor, preprocessed_features):
    res = predictor.predict(preprocessed_features)
    assert 'probability_disease' in res
    assert 'risk_percentage' in res
    prob = res['probability_disease']
    assert 0.0 <= prob <= 1.0
    assert 0.0 <= res['risk_percentage'] <= 100.0


def test_risk_category_stratification(predictor):
    low_cat, _, _ = predictor.get_risk_category(15.0)
    assert "Low" in low_cat

    mod_cat, _, _ = predictor.get_risk_category(40.0)
    assert "Moderate" in mod_cat

    high_cat, _, _ = predictor.get_risk_category(85.0)
    assert "High" in high_cat
