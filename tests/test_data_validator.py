"""
Unit tests for CardioFusion DataValidator.
"""
import pytest
from src.utils.data_validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def valid_patient_sample():
    return {
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


def test_valid_input_passes(validator, valid_patient_sample):
    is_valid, errors, warnings = validator.validate_input(valid_patient_sample)
    assert is_valid is True
    assert len(errors) == 0


def test_missing_required_fields_fails(validator, valid_patient_sample):
    bad_sample = valid_patient_sample.copy()
    del bad_sample['age_category']
    del bad_sample['sex']
    is_valid, errors, _ = validator.validate_input(bad_sample)
    assert is_valid is False
    assert any("Required fields missing" in err for err in errors)


def test_male_gestational_diabetes_rejected(validator, valid_patient_sample):
    male_gestational = valid_patient_sample.copy()
    male_gestational['sex'] = 'Male'
    male_gestational['diabetes'] = 'Yes, but female told only during pregnancy'
    is_valid, errors, _ = validator.validate_input(male_gestational)
    assert is_valid is False
    assert any("Gestational diabetes cannot be selected for males" in err for err in errors)


def test_out_of_range_height_rejected(validator, valid_patient_sample):
    bad_height = valid_patient_sample.copy()
    bad_height['height_cm'] = 350
    is_valid, errors, _ = validator.validate_input(bad_height)
    assert is_valid is False
    assert any("Height" in err for err in errors)


def test_preprocessing_produces_exact_27_features(validator, valid_patient_sample):
    df = validator.preprocess_for_model(valid_patient_sample)
    assert df.shape == (1, 27)
    assert 'BMI' in df.columns
    assert 'Age_Numeric' in df.columns
    assert 'Lifestyle_Risk_Score' in df.columns
    assert 'Health_Conditions_Count' in df.columns
