# 🔧 CardioFusion ML Platform - Low-Level Design Document

**Project:** CardioFusion - Cardiovascular Disease Prediction Platform  
**Version:** 1.0  
**Date:** November 21, 2025  
**Author:** Ayush Chhoker

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Module Specifications](#module-specifications)
3. [Data Models](#data-models)
4. [API Specifications](#api-specifications)
5. [Algorithm Details](#algorithm-details)
6. [Database Schema](#database-schema)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)

---

## 🏗️ System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   STREAMLIT WEB APP                          │
│                     (src/app.py)                             │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Sidebar   │  │   Main     │  │   SHAP Display     │    │
│  │   Input    │  │ Prediction │  │   Component        │    │
│  │   Form     │  │  Display   │  │                    │    │
│  └─────┬──────┘  └─────▲──────┘  └──────▲─────────────┘    │
└────────┼──────────────┼─────────────────┼──────────────────┘
         │              │                 │
         ▼              │                 │
┌────────────────────┐  │                 │
│  DataValidator     │  │                 │
│ (data_validator.py)│  │                 │
│                    │  │                 │
│ • validate_input() ├──┘                 │
│ • preprocess_for_  │                    │
│   model()          │                    │
└────────┬───────────┘                    │
         │                                │
         ▼                                │
┌────────────────────┐                    │
│  ModelPredictor    │                    │
│ (model_utils.py)   │                    │
│                    │                    │
│ • load_models()    ├────────────────────┤
│ • predict()        │                    │
└────────┬───────────┘                    │
         │                                │
         ▼                                │
┌────────────────────┐                    │
│  SHAPExplainer     │                    │
│(shap_explainer.py) │                    │
│                    │                    │
│ • explain_         ├────────────────────┘
│   prediction()     │
│ • get_             │
│   recommendations()│
└────────────────────┘
```

---

## 📦 Module Specifications

### 1. Web Application Module (`src/app.py`)

#### Class: N/A (Functional Programming)

#### Functions:

##### `main()`
**Purpose:** Entry point for Streamlit application

**Algorithm:**
```python
1. Set page configuration (title, icon, layout)
2. Display header and description
3. Initialize session state variables
4. Render sidebar input form
5. On prediction button click:
   a. Validate inputs
   b. Display loading spinner
   c. Call prediction pipeline
   d. Display results
   e. Generate SHAP explanations
```

**Dependencies:**
- `streamlit`
- `DataValidator`
- `ModelPredictor`
- `SHAPExplainer`

**Caching:**
```python
@st.cache_resource
def load_predictor():
    """Cache model loading to prevent reloading on every interaction"""
    predictor = ModelPredictor('models')
    predictor.load_models()
    return predictor
```

---

##### `create_input_form()`
**Purpose:** Render patient data input form

**Input:** None  
**Output:** `dict` - Patient data

**Form Fields:**
```python
{
    # Demographics
    'age_category': st.selectbox(...),  # 18 age groups
    'sex': st.selectbox(...),           # Male/Female
    
    # Physical Measurements
    'height_cm': st.number_input(...),  # 120-220 cm
    'weight_kg': st.number_input(...),  # 30-200 kg
    'bmi': st.number_input(...),        # Auto-calculated
    
    # Lifestyle
    'exercise': st.selectbox(...),      # Yes/No
    'smoking_history': st.selectbox(...), # Yes/No
    'alcohol_consumption': st.slider(...), # 0-60 days/month
    
    # Diet
    'fruit_consumption': st.slider(...),  # 0-60 servings/month
    'green_vegetables_consumption': st.slider(...), # 0-60
    'fried_potato_consumption': st.slider(...),     # 0-60
    
    # Health Status
    'general_health': st.selectbox(...),  # 5 categories
    'checkup': st.selectbox(...),        # Last checkup time
    
    # Medical Conditions
    'diabetes': st.selectbox(...),       # Yes/No/Borderline
    'depression': st.selectbox(...),     # Yes/No
    'arthritis': st.selectbox(...),      # Yes/No
    'skin_cancer': st.selectbox(...),    # Yes/No
    'other_cancer': st.selectbox(...)    # Yes/No
}
```

**Validation:**
- BMI auto-calculated from height/weight
- Real-time input validation
- Required field checking

---

##### `display_prediction_results(prediction: dict)`
**Purpose:** Render prediction results with visual indicators

**Input:**
```python
{
    'risk_percentage': float,      # 0-100
    'prediction_label': str,       # 'Disease' or 'No Disease'
    'confidence': float,           # 0.5-1.0
    'individual_models': dict      # Per-model predictions
}
```

**Output:** Streamlit UI components

**Display Logic:**
```python
# Risk Category Classification
if risk_percentage < 30:
    category = "🟢 LOW RISK"
    color = "green"
elif risk_percentage < 60:
    category = "🟡 MODERATE RISK"
    color = "orange"
else:
    category = "🔴 HIGH RISK"
    color = "red"

# Display Components
- st.metric() - Risk percentage with delta
- st.progress() - Visual progress bar
- st.info() / st.warning() / st.error() - Risk category message
- st.expander() - Individual model predictions
```

---

##### `display_shap_explanation(input_df, prediction, predictor)`
**Purpose:** Generate and display SHAP visualizations

**Algorithm:**
```python
1. Load background data (100 samples from training set)
2. Create SHAPExplainer instance
3. Generate SHAP values for prediction
4. Create waterfall plot
5. Display top features
6. Generate recommendations
```

**Visualizations:**
- SHAP waterfall plot (matplotlib)
- Top 5 risk-increasing features
- Top 5 risk-decreasing features
- Actionable recommendations list

---

### 2. Data Validator Module (`src/utils/data_validator.py`)

#### Class: `DataValidator`

**Purpose:** Validate and preprocess patient input data

**Attributes:** None (stateless)

---

##### `validate_input(data: dict) -> tuple[bool, list, list]`
**Purpose:** Validate patient input data

**Input:**
```python
{
    'age_category': str,
    'sex': str,
    'height_cm': float,
    'weight_kg': float,
    'bmi': float,
    # ... 12 more fields
}
```

**Output:**
```python
(
    is_valid: bool,           # True if all validations pass
    errors: list[str],        # Critical errors
    warnings: list[str]       # Non-critical warnings
)
```

**Validation Rules:**

| Field | Type | Range/Values | Error Message |
|-------|------|--------------|---------------|
| `age_category` | str | 18 predefined categories | "Invalid age category" |
| `sex` | str | 'Male', 'Female' | "Sex must be Male or Female" |
| `height_cm` | float | 120-220 | "Height must be between 120-220 cm" |
| `weight_kg` | float | 30-200 | "Weight must be between 30-200 kg" |
| `bmi` | float | 10-60 | "BMI must be between 10-60" |
| `exercise` | str | 'Yes', 'No' | "Exercise must be Yes or No" |
| `smoking_history` | str | 'Yes', 'No' | "Smoking history must be Yes or No" |
| `alcohol_consumption` | int | 0-60 | "Alcohol consumption: 0-60 days/month" |
| `fruit_consumption` | int | 0-60 | "Fruit consumption: 0-60 servings/month" |
| `general_health` | str | 5 categories | "Invalid general health value" |
| `diabetes` | str | 3 categories | "Invalid diabetes value" |

**Algorithm:**
```python
def validate_input(data):
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ['age_category', 'sex', 'height_cm', ...]
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate age category
    valid_ages = ['18-24', '25-29', '30-34', ...]
    if data['age_category'] not in valid_ages:
        errors.append("Invalid age category")
    
    # Validate numerical ranges
    if not (120 <= data['height_cm'] <= 220):
        errors.append("Height must be between 120-220 cm")
    
    # Validate categorical values
    if data['sex'] not in ['Male', 'Female']:
        errors.append("Sex must be Male or Female")
    
    # BMI consistency check
    calculated_bmi = data['weight_kg'] / (data['height_cm']/100)**2
    if abs(calculated_bmi - data['bmi']) > 0.5:
        warnings.append("BMI doesn't match height/weight")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings
```

---

##### `preprocess_for_model(data: dict) -> pd.DataFrame`
**Purpose:** Convert raw input to model-ready features

**Input:** Validated patient data (17 fields)  
**Output:** DataFrame with 27 engineered features

**Feature Engineering Pipeline:**

**Step 1: Direct Numerical Features (7)**
```python
features = {
    'Height_(cm)': data['height_cm'],
    'Weight_(kg)': data['weight_kg'],
    'BMI': data['bmi'],
    'Alcohol_Consumption': data['alcohol_consumption'],
    'Fruit_Consumption': data['fruit_consumption'],
    'Green_Vegetables_Consumption': data['green_vegetables_consumption'],
    'FriedPotato_Consumption': data['fried_potato_consumption']
}
```

**Step 2: Age Encoding (1)**
```python
# Map age category to numeric value
age_mapping = {
    '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
    '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
    '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
    '80+': 82
}
features['Age_Numeric'] = age_mapping[data['age_category']]
```

**Step 3: One-Hot Encoding (12 features)**
```python
# Binary features (Yes=1, No=0)
binary_encodings = {
    'Sex_Male': 1 if data['sex'] == 'Male' else 0,
    'Exercise_Yes': 1 if data['exercise'] == 'Yes' else 0,
    'Smoking_History_Yes': 1 if data['smoking_history'] == 'Yes' else 0,
    'Arthritis_Yes': 1 if data['arthritis'] == 'Yes' else 0,
    'Depression_Yes': 1 if data['depression'] == 'Yes' else 0,
    'Skin_Cancer_Yes': 1 if data['skin_cancer'] == 'Yes' else 0,
    'Other_Cancer_Yes': 1 if data['other_cancer'] == 'Yes' else 0
}

# Multi-category one-hot encoding
# General Health: 5 categories (Excellent, Very Good, Good, Fair, Poor)
# Checkup: 5 categories
# Diabetes: 3 categories (No, Yes, Borderline)
```

**Step 4: Derived Features (2)**
```python
# Lifestyle Risk Score
lifestyle_risk = (
    (1 if data['smoking_history'] == 'Yes' else 0) +
    (1 if data['exercise'] == 'No' else 0) +
    (data['alcohol_consumption'] / 30) +
    (data['fried_potato_consumption'] / 30)
)

# Health Conditions Count
conditions_count = sum([
    1 if data['diabetes'] == 'Yes' else 0,
    1 if data['depression'] == 'Yes' else 0,
    1 if data['arthritis'] == 'Yes' else 0,
    1 if data['skin_cancer'] == 'Yes' else 0,
    1 if data['other_cancer'] == 'Yes' else 0
])

features['Lifestyle_Risk_Score'] = lifestyle_risk
features['Health_Conditions_Count'] = conditions_count
```

**Step 5: Feature Ordering**
```python
# Ensure features match training order (critical!)
feature_order = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
    'Fruit_Consumption', 'Green_Vegetables_Consumption',
    'FriedPotato_Consumption', 'Age_Numeric', 'Sex_Male',
    'Exercise_Yes', 'Smoking_History_Yes', 'General_Health_Excellent',
    'General_Health_Very_Good', 'General_Health_Good',
    'General_Health_Fair', 'Checkup_Within_past_year',
    'Checkup_Within_past_2_years', 'Checkup_Within_past_5_years',
    'Checkup_5_or_more_years_ago', 'Diabetes_Yes',
    'Diabetes_Borderline', 'Arthritis_Yes', 'Depression_Yes',
    'Skin_Cancer_Yes', 'Other_Cancer_Yes', 'Lifestyle_Risk_Score',
    'Health_Conditions_Count'
]

# Create DataFrame with correct column order
df = pd.DataFrame([features])[feature_order]
return df
```

**Output Shape:** `(1, 27)` - Single row with 27 features

---

### 3. Model Predictor Module (`src/utils/model_utils.py`)

#### Class: `ModelPredictor`

**Purpose:** Load models and generate predictions

**Attributes:**
```python
class ModelPredictor:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = {}           # Dict of loaded models
        self.scaler = None         # StandardScaler instance
        self.model_names = [       # Models to load
            'Decision Tree',
            'Random Forest',
            'Logistic Regression',
            'XGBoost',
            'Neural Network'
        ]
```

---

##### `load_models() -> None`
**Purpose:** Load all trained models and scaler from disk

**Algorithm:**
```python
def load_models():
    # 1. Load StandardScaler
    scaler_path = 'models/preprocessing/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        self.scaler = pickle.load(f)
    
    # 2. Load baseline models (Logistic Regression, Decision Tree, Random Forest)
    for model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
        model_path = f'models/baseline_models/{model_name.lower().replace(" ", "_")}.pkl'
        with open(model_path, 'rb') as f:
            self.models[model_name] = pickle.load(f)
    
    # 3. Load XGBoost
    xgb_path = 'models/advanced_models/xgboost_model.pkl'
    with open(xgb_path, 'rb') as f:
        self.models['XGBoost'] = pickle.load(f)
    
    # 4. Load Neural Network (Keras model)
    nn_path = 'models/advanced_models/neural_network_model.h5'
    self.models['Neural Network'] = tf.keras.models.load_model(nn_path)
    
    print(f"✅ Loaded {len(self.models)} models")
```

**Error Handling:**
```python
try:
    # Load models
except FileNotFoundError as e:
    raise Exception(f"Model file not found: {e}")
except Exception as e:
    raise Exception(f"Error loading models: {e}")
```

---

##### `predict(input_df: pd.DataFrame) -> dict`
**Purpose:** Generate ensemble prediction from all models

**Input:** DataFrame (1, 27) - Preprocessed patient features

**Algorithm:**

**Step 1: Identify Numerical Features**
```python
numerical_features = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
    'Fruit_Consumption', 'Green_Vegetables_Consumption',
    'FriedPotato_Consumption', 'Age_Numeric',
    'Lifestyle_Risk_Score', 'Health_Conditions_Count'
]
```

**Step 2: Apply Feature Scaling**
```python
# Extract numerical features
numerical_data = input_df[numerical_features]

# Apply StandardScaler (CRITICAL STEP!)
scaled_numerical = self.scaler.transform(numerical_data)

# Create scaled DataFrame
scaled_df = input_df.copy()
scaled_df[numerical_features] = scaled_numerical
```

**Step 3: Get Individual Model Predictions**
```python
predictions = []
probabilities = []

for model_name, model in self.models.items():
    # Get prediction (0 or 1)
    pred = model.predict(scaled_df)[0]
    
    # Get probability of disease
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(scaled_df)[0][1]  # P(disease)
    else:
        # Neural network returns probabilities directly
        prob = model.predict(scaled_df)[0][0]
    
    predictions.append(pred)
    probabilities.append(prob)
```

**Step 4: Ensemble Prediction (Soft Voting)**
```python
# Average probabilities from all models
ensemble_probability = np.mean(probabilities)

# Binary prediction (threshold = 0.5)
ensemble_prediction = 1 if ensemble_probability >= 0.5 else 0

# Risk percentage (0-100%)
risk_percentage = ensemble_probability * 100

# Confidence score (distance from decision boundary)
confidence = max(ensemble_probability, 1 - ensemble_probability)
```

**Step 5: Format Output**
```python
return {
    'prediction': ensemble_prediction,          # 0 or 1
    'risk_percentage': risk_percentage,        # 0-100
    'prediction_label': 'Disease' if ensemble_prediction == 1 else 'No Disease',
    'confidence': confidence,                  # 0.5-1.0
    'individual_models': {
        model_name: {
            'prediction': int(pred),
            'probability_disease': float(prob),
            'probability_no_disease': float(1 - prob)
        }
        for model_name, pred, prob in zip(self.model_names, predictions, probabilities)
    }
}
```

**Output Example:**
```python
{
    'prediction': 1,
    'risk_percentage': 73.5,
    'prediction_label': 'Disease',
    'confidence': 0.735,
    'individual_models': {
        'Decision Tree': {
            'prediction': 1,
            'probability_disease': 0.82,
            'probability_no_disease': 0.18
        },
        'Random Forest': {
            'prediction': 1,
            'probability_disease': 0.75,
            'probability_no_disease': 0.25
        },
        # ... other models
    }
}
```

---

### 4. SHAP Explainer Module (`src/utils/shap_explainer.py`)

#### Class: `SHAPExplainer`

**Purpose:** Generate model explanations using SHAP values

**Attributes:**
```python
class SHAPExplainer:
    def __init__(self, model, background_data: pd.DataFrame):
        self.model = model
        self.background_data = background_data
        self.explainer = None
```

---

##### `__init__(model, background_data)`
**Purpose:** Initialize SHAP explainer

**Algorithm:**
```python
def __init__(self, model, background_data):
    self.model = model
    self.background_data = background_data
    
    # Select explainer type based on model
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        # Tree-based models: use TreeExplainer
        self.explainer = shap.TreeExplainer(model)
    else:
        # Other models: use KernelExplainer with background data
        self.explainer = shap.KernelExplainer(
            model.predict_proba,
            background_data.sample(min(50, len(background_data)))
        )
```

---

##### `explain_prediction(input_df: pd.DataFrame) -> dict`
**Purpose:** Generate SHAP values for a prediction

**Algorithm:**

**Step 1: Calculate SHAP Values**
```python
# Get SHAP values for input
shap_values = self.explainer.shap_values(input_df)

# For binary classification, extract class 1 (disease) values
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Values for positive class
```

**Step 2: Extract Feature Importance**
```python
# Get feature names
feature_names = input_df.columns.tolist()

# Get SHAP values for first (only) instance
instance_shap = shap_values[0]

# Create feature-value pairs
feature_importance = list(zip(feature_names, instance_shap))

# Sort by absolute impact
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
```

**Step 3: Separate Positive/Negative Contributions**
```python
# Risk-increasing features (positive SHAP values)
positive_features = [(f, v) for f, v in feature_importance if v > 0]

# Risk-decreasing features (negative SHAP values)
negative_features = [(f, v) for f, v in feature_importance if v < 0]

# Get top 5 of each
top_positive = positive_features[:5]
top_negative = negative_features[:5]
```

**Output:**
```python
{
    'shap_values': instance_shap,              # Array of SHAP values
    'feature_names': feature_names,            # Feature names
    'top_positive': top_positive,              # Top risk-increasing
    'top_negative': top_negative,              # Top risk-decreasing
    'base_value': explainer.expected_value,   # Baseline prediction
    'prediction_value': prediction             # Model prediction
}
```

---

##### `create_waterfall_plot(explanation: dict) -> matplotlib.Figure`
**Purpose:** Create SHAP waterfall visualization

**Algorithm:**
```python
def create_waterfall_plot(explanation):
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate waterfall plot using SHAP
    shap.waterfall_plot(
        shap.Explanation(
            values=explanation['shap_values'],
            base_values=explanation['base_value'],
            data=input_df.values[0],
            feature_names=explanation['feature_names']
        ),
        max_display=10,
        show=False
    )
    
    plt.tight_layout()
    return fig
```

---

##### `get_recommendations(explanation: dict) -> list[str]`
**Purpose:** Generate actionable health recommendations

**Algorithm:**
```python
def get_recommendations(explanation):
    recommendations = []
    
    # Analyze top risk-increasing features
    for feature, shap_value in explanation['top_positive']:
        if 'Smoking' in feature and shap_value > 0.1:
            recommendations.append(
                "🚭 Consider smoking cessation programs - smoking significantly increases cardiovascular risk"
            )
        elif 'Exercise' in feature and 'No' in feature:
            recommendations.append(
                "🏃‍♂️ Increase physical activity - aim for 150 minutes of moderate exercise per week"
            )
        elif 'BMI' in feature:
            recommendations.append(
                "⚖️ Work towards healthy weight - BMI in 18.5-24.9 range reduces cardiovascular risk"
            )
        elif 'Alcohol' in feature:
            recommendations.append(
                "🍷 Reduce alcohol consumption - limit to moderate levels (1-2 drinks/day max)"
            )
        elif 'Diabetes' in feature:
            recommendations.append(
                "💉 Manage diabetes carefully - maintain blood sugar control to reduce heart disease risk"
            )
        elif 'Age' in feature:
            recommendations.append(
                "📅 Regular health screenings - age is a risk factor, increase monitoring frequency"
            )
    
    # Analyze protective features
    for feature, shap_value in explanation['top_negative']:
        if 'Fruit' in feature or 'Vegetables' in feature:
            recommendations.append(
                "🥗 Continue healthy diet - fruits and vegetables protect against heart disease"
            )
        elif 'Exercise' in feature and 'Yes' in feature:
            recommendations.append(
                "✅ Maintain exercise routine - current activity level is protective"
            )
    
    # General recommendations
    recommendations.append("🩺 Schedule regular checkups with your healthcare provider")
    recommendations.append("💊 Discuss prevention strategies with your doctor")
    
    return recommendations[:5]  # Return top 5
```

---

## 📊 Data Models

### Input Data Model

**Patient Input Schema:**
```python
PatientInput = {
    # Demographics (2 fields)
    'age_category': str,      # Required, Enum[18 values]
    'sex': str,               # Required, Enum['Male', 'Female']
    
    # Physical Measurements (3 fields)
    'height_cm': float,       # Required, Range[120-220]
    'weight_kg': float,       # Required, Range[30-200]
    'bmi': float,             # Required, Range[10-60]
    
    # Lifestyle (7 fields)
    'exercise': str,          # Required, Enum['Yes', 'No']
    'smoking_history': str,   # Required, Enum['Yes', 'No']
    'alcohol_consumption': int,     # Required, Range[0-60]
    'fruit_consumption': int,       # Required, Range[0-60]
    'green_vegetables_consumption': int,  # Required, Range[0-60]
    'fried_potato_consumption': int,      # Required, Range[0-60]
    
    # Health Status (2 fields)
    'general_health': str,    # Required, Enum[5 values]
    'checkup': str,           # Required, Enum[5 values]
    
    # Medical Conditions (5 fields)
    'diabetes': str,          # Required, Enum[3 values]
    'depression': str,        # Required, Enum['Yes', 'No']
    'arthritis': str,         # Required, Enum['Yes', 'No']
    'skin_cancer': str,       # Required, Enum['Yes', 'No']
    'other_cancer': str       # Required, Enum['Yes', 'No']
}
```

**Total Fields:** 17

---

### Feature Vector Model

**Preprocessed Feature Schema:**
```python
FeatureVector = pd.DataFrame({
    # Numerical Features (10)
    'Height_(cm)': float,                    # Scaled
    'Weight_(kg)': float,                    # Scaled
    'BMI': float,                            # Scaled
    'Alcohol_Consumption': float,            # Scaled
    'Fruit_Consumption': float,              # Scaled
    'Green_Vegetables_Consumption': float,   # Scaled
    'FriedPotato_Consumption': float,        # Scaled
    'Age_Numeric': float,                    # Scaled
    'Lifestyle_Risk_Score': float,           # Scaled
    'Health_Conditions_Count': float,        # Scaled
    
    # Binary Features (7)
    'Sex_Male': int,                         # 0 or 1
    'Exercise_Yes': int,                     # 0 or 1
    'Smoking_History_Yes': int,              # 0 or 1
    'Arthritis_Yes': int,                    # 0 or 1
    'Depression_Yes': int,                   # 0 or 1
    'Skin_Cancer_Yes': int,                  # 0 or 1
    'Other_Cancer_Yes': int,                 # 0 or 1
    
    # One-Hot Encoded Features (10)
    'General_Health_Excellent': int,         # 0 or 1
    'General_Health_Very_Good': int,         # 0 or 1
    'General_Health_Good': int,              # 0 or 1
    'General_Health_Fair': int,              # 0 or 1
    # (General_Health_Poor is reference category)
    
    'Checkup_Within_past_year': int,         # 0 or 1
    'Checkup_Within_past_2_years': int,      # 0 or 1
    'Checkup_Within_past_5_years': int,      # 0 or 1
    'Checkup_5_or_more_years_ago': int,      # 0 or 1
    # (Checkup_Never is reference category)
    
    'Diabetes_Yes': int,                     # 0 or 1
    'Diabetes_Borderline': int               # 0 or 1
    # (Diabetes_No is reference category)
})
```

**Total Features:** 27  
**Shape:** `(1, 27)` for single prediction

---

### Prediction Output Model

**Prediction Schema:**
```python
PredictionOutput = {
    'prediction': int,              # 0 or 1 (binary classification)
    'risk_percentage': float,       # 0.0 to 100.0
    'prediction_label': str,        # 'Disease' or 'No Disease'
    'confidence': float,            # 0.5 to 1.0
    
    'individual_models': {
        'Decision Tree': {
            'prediction': int,                  # 0 or 1
            'probability_disease': float,       # 0.0 to 1.0
            'probability_no_disease': float     # 0.0 to 1.0
        },
        'Random Forest': {...},
        'Logistic Regression': {...},
        'XGBoost': {...},
        'Neural Network': {...}
    }
}
```

---

### SHAP Explanation Model

**SHAP Explanation Schema:**
```python
SHAPExplanation = {
    'shap_values': np.ndarray,              # Shape: (27,)
    'feature_names': list[str],             # 27 feature names
    'top_positive': list[tuple[str, float]], # [(feature, shap_value), ...]
    'top_negative': list[tuple[str, float]], # [(feature, shap_value), ...]
    'base_value': float,                    # Expected value (baseline)
    'prediction_value': float,              # Actual prediction
    'waterfall_plot': matplotlib.Figure,    # Visualization
    'recommendations': list[str]            # Health recommendations
}
```

---

## 🔌 API Specifications

### Internal API (Module Interfaces)

#### DataValidator API

**Method:** `validate_input(data: dict) -> tuple[bool, list, list]`

**Request:**
```python
{
    'age_category': '55-59',
    'sex': 'Male',
    'height_cm': 175,
    'weight_kg': 85,
    # ... 12 more fields
}
```

**Response:**
```python
(
    True,                           # is_valid
    [],                            # errors (empty if valid)
    ['BMI slightly inconsistent']  # warnings
)
```

---

**Method:** `preprocess_for_model(data: dict) -> pd.DataFrame`

**Request:** Validated patient dict (17 fields)

**Response:**
```python
pd.DataFrame({
    'Height_(cm)': [175.0],
    'Weight_(kg)': [85.0],
    'BMI': [27.8],
    # ... 24 more features
}, columns=[...27 features in correct order...])
```

---

#### ModelPredictor API

**Method:** `load_models() -> None`

**Side Effects:**
- Loads 5 models into `self.models` dict
- Loads scaler into `self.scaler`
- Prints loading confirmation

---

**Method:** `predict(input_df: pd.DataFrame) -> dict`

**Request:**
```python
pd.DataFrame with shape (1, 27)
```

**Response:**
```python
{
    'prediction': 1,
    'risk_percentage': 73.5,
    'prediction_label': 'Disease',
    'confidence': 0.735,
    'individual_models': {...}
}
```

---

#### SHAPExplainer API

**Method:** `__init__(model, background_data: pd.DataFrame)`

**Parameters:**
- `model`: Trained scikit-learn or Keras model
- `background_data`: DataFrame with 50-100 samples for baseline

---

**Method:** `explain_prediction(input_df: pd.DataFrame) -> dict`

**Request:**
```python
pd.DataFrame with shape (1, 27)
```

**Response:**
```python
{
    'shap_values': array([0.05, -0.12, 0.31, ...]),
    'feature_names': ['Height_(cm)', 'Weight_(kg)', ...],
    'top_positive': [('Age_Numeric', 0.45), ('Smoking_History_Yes', 0.31), ...],
    'top_negative': [('Exercise_Yes', -0.28), ...],
    'base_value': 0.15,
    'prediction_value': 0.73
}
```

---

**Method:** `get_recommendations(explanation: dict) -> list[str]`

**Request:** SHAP explanation dict

**Response:**
```python
[
    "🚭 Consider smoking cessation programs",
    "🏃‍♂️ Increase physical activity",
    "⚖️ Work towards healthy weight",
    "🩺 Schedule regular checkups",
    "💊 Discuss prevention strategies"
]
```

---

## 🧮 Algorithm Details

### 1. Feature Scaling Algorithm

**StandardScaler Implementation:**

```python
# Training Phase (fit)
def fit(X_train):
    mean = X_train.mean(axis=0)      # Calculate mean for each feature
    std = X_train.std(axis=0)        # Calculate std deviation
    
    return mean, std

# Prediction Phase (transform)
def transform(X, mean, std):
    X_scaled = (X - mean) / std      # Z-score normalization
    return X_scaled
```

**Applied Features:**
- Height_(cm)
- Weight_(kg)
- BMI
- Alcohol_Consumption
- Fruit_Consumption
- Green_Vegetables_Consumption
- FriedPotato_Consumption
- Age_Numeric
- Lifestyle_Risk_Score
- Health_Conditions_Count

**Properties:**
- Mean = 0
- Standard Deviation = 1
- Preserves normal distribution shape

---

### 2. Ensemble Prediction Algorithm

**Soft Voting Classifier:**

```python
def soft_voting_predict(models, X):
    # Step 1: Get probabilities from each model
    probabilities = []
    for model in models:
        prob = model.predict_proba(X)[0]  # [P(class 0), P(class 1)]
        probabilities.append(prob)
    
    # Step 2: Average probabilities
    avg_prob = np.mean(probabilities, axis=0)
    # avg_prob = [avg_P(class 0), avg_P(class 1)]
    
    # Step 3: Predict class with highest average probability
    prediction = np.argmax(avg_prob)
    
    # Step 4: Calculate confidence
    confidence = max(avg_prob)
    
    return {
        'prediction': prediction,
        'probability': avg_prob[1],  # P(disease)
        'confidence': confidence
    }
```

**Advantages:**
- Reduces variance
- Improves generalization
- Handles model disagreements
- Provides probability estimates

---

### 3. SHAP Value Calculation

**TreeExplainer Algorithm (for tree-based models):**

```python
def calculate_shap_values(model, X, background_data):
    # Step 1: Build Tree SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Step 2: Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification:
    # shap_values[0] = SHAP values for class 0 (No Disease)
    # shap_values[1] = SHAP values for class 1 (Disease)
    
    return shap_values[1]  # Return disease class values
```

**SHAP Value Interpretation:**
- **Positive SHAP value:** Feature increases disease risk
- **Negative SHAP value:** Feature decreases disease risk
- **Magnitude:** Impact strength

**Formula:**
```
Prediction = Base Value + Sum(SHAP values)
```

**Example:**
```
Base Value (population average): 0.15 (15% risk)
Age SHAP: +0.30
Smoking SHAP: +0.20
Exercise SHAP: -0.10
...
Total Prediction: 0.15 + 0.30 + 0.20 - 0.10 + ... = 0.73 (73% risk)
```

---

### 4. SMOTE Algorithm (Training Phase)

**Synthetic Minority Oversampling Technique:**

```python
def smote_oversample(X_minority, k=5):
    synthetic_samples = []
    
    for sample in X_minority:
        # Step 1: Find k nearest neighbors
        neighbors = find_k_nearest_neighbors(sample, X_minority, k)
        
        # Step 2: Generate synthetic samples
        for i in range(k):
            neighbor = random.choice(neighbors)
            
            # Step 3: Linear interpolation
            diff = neighbor - sample
            gap = random.uniform(0, 1)
            synthetic = sample + gap * diff
            
            synthetic_samples.append(synthetic)
    
    return synthetic_samples
```

**Result:**
- Original: 308,854 samples (92% No Disease, 8% Disease)
- After SMOTE: 567,606 samples (50% No Disease, 50% Disease)

---

## 🗄️ Database Schema

**Note:** Current version does not use a database. All data is file-based.

**File Storage:**
```
data/
├── raw/
│   └── cardio_train.csv           # Original dataset
├── processed/
│   ├── train_data.csv             # Training set (80%)
│   └── test_data.csv              # Test set (20%)
└── external/
    └── (future external datasets)

models/
├── baseline_models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
├── advanced_models/
│   ├── xgboost_model.pkl
│   └── neural_network_model.h5
└── preprocessing/
    └── scaler.pkl
```

**Future Database Schema (if implemented):**

```sql
-- Users table (for authentication)
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table (for history tracking)
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Input features (JSON or individual columns)
    patient_data JSONB,
    
    -- Prediction results
    risk_percentage FLOAT,
    prediction_label VARCHAR(20),
    confidence FLOAT,
    
    -- Individual model predictions (JSON)
    individual_predictions JSONB,
    
    -- SHAP explanation (JSON)
    shap_explanation JSONB
);

-- Model versions table
CREATE TABLE model_versions (
    version_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    version VARCHAR(20),
    trained_at TIMESTAMP,
    accuracy FLOAT,
    roc_auc FLOAT,
    file_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE
);
```

---

## ⚠️ Error Handling

### Error Categories

#### 1. Input Validation Errors

**Type:** User-facing, recoverable

**Examples:**
```python
errors = [
    "Missing required field: age_category",
    "Height must be between 120-220 cm",
    "Invalid age category",
    "BMI must be between 10-60"
]
```

**Handling:**
```python
is_valid, errors, warnings = validator.validate_input(data)
if not is_valid:
    st.error("❌ Please fix the following errors:")
    for error in errors:
        st.error(f"  • {error}")
    return  # Stop processing
```

---

#### 2. Model Loading Errors

**Type:** System error, non-recoverable

**Examples:**
```python
FileNotFoundError: models/baseline_models/decision_tree.pkl not found
pickle.UnpicklingError: Invalid pickle data
```

**Handling:**
```python
try:
    predictor = ModelPredictor('models')
    predictor.load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file missing: {e}")
    st.info("Please ensure all model files are present")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()
```

---

#### 3. Prediction Errors

**Type:** Runtime error, may be recoverable

**Examples:**
```python
ValueError: Feature mismatch - expected 27, got 26
KeyError: 'Height_(cm)' not found in input
```

**Handling:**
```python
try:
    prediction = predictor.predict(input_df)
except ValueError as e:
    st.error(f"❌ Feature mismatch: {e}")
    st.info("Please ensure all required fields are filled")
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.exception(e)  # Show full traceback in dev mode
```

---

#### 4. SHAP Generation Errors

**Type:** Warning, non-critical

**Examples:**
```python
RuntimeWarning: SHAP values computation timed out
MemoryError: Insufficient memory for background data
```

**Handling:**
```python
try:
    explanation = explainer.explain_prediction(input_df)
    st.pyplot(explanation['waterfall_plot'])
except Exception as e:
    st.warning(f"⚠️ Could not generate SHAP explanation: {e}")
    st.info("Prediction results are still valid")
    # Continue without SHAP
```

---

### Error Logging

**Streamlit Logger Configuration:**
```python
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
try:
    prediction = predictor.predict(input_df)
except Exception as e:
    logger.error(f"Prediction failed: {e}", exc_info=True)
    raise
```

---

## ⚡ Performance Optimization

### 1. Model Caching

**Strategy:** Load models once, cache in Streamlit

```python
@st.cache_resource
def load_predictor():
    """
    Cache resource: Models loaded once per session
    Not re-executed on user interactions
    """
    predictor = ModelPredictor('models')
    predictor.load_models()
    return predictor

# Usage
predictor = load_predictor()  # Only loads on first call
```

**Benefits:**
- Reduces load time from 8s → 0s on subsequent interactions
- Reduces memory usage (single model instance)
- Improves user experience

---

### 2. Data Caching

**Strategy:** Cache background data for SHAP

```python
@st.cache_data
def load_background_data():
    """
    Cache data: Background samples loaded once
    Automatically clears when data file changes
    """
    df = pd.read_csv('data/processed/train_data.csv', nrows=100)
    background = df.drop('Heart_Disease', axis=1).sample(50, random_state=42)
    return background

# Usage
background_data = load_background_data()
```

---

### 3. Vectorized Operations

**Strategy:** Use NumPy/Pandas vectorization instead of loops

**Bad:**
```python
# Slow: Python loop
for i in range(len(df)):
    df.loc[i, 'bmi'] = df.loc[i, 'weight'] / (df.loc[i, 'height']/100)**2
```

**Good:**
```python
# Fast: Vectorized operation
df['bmi'] = df['weight'] / (df['height']/100)**2
```

---

### 4. Lazy Loading

**Strategy:** Only load SHAP when needed

```python
# Don't pre-compute SHAP
if st.checkbox("Show explanation"):  # User opts-in
    with st.spinner("Generating explanation..."):
        explanation = explainer.explain_prediction(input_df)
        display_shap_explanation(explanation)
```

---

### 5. Feature Scaling Optimization

**Strategy:** Pre-compute scaler statistics, apply efficiently

```python
# Efficient: NumPy vectorized operation
numerical_data = input_df[numerical_features].values
scaled = (numerical_data - scaler.mean_) / scaler.scale_

# Instead of:
# scaled = scaler.transform(numerical_data)  # Slightly slower
```

---

### Performance Benchmarks

| Operation | Time | Optimization |
|-----------|------|--------------|
| **Model Loading** | 8s → 0s* | @st.cache_resource |
| **Prediction** | 0.5s | Vectorized operations |
| **SHAP Generation** | 2s | Lazy loading |
| **Background Data Load** | 1s → 0s* | @st.cache_data |

*After first load (cached)

---

## 🔒 Security Considerations

### Input Sanitization

```python
def sanitize_input(data: dict) -> dict:
    """Remove potentially harmful inputs"""
    sanitized = {}
    
    for key, value in data.items():
        # Only allow expected keys
        if key not in ALLOWED_FIELDS:
            continue
        
        # Type validation
        if isinstance(value, str):
            # Remove special characters
            value = re.sub(r'[^\w\s-]', '', value)
            # Limit length
            value = value[:100]
        elif isinstance(value, (int, float)):
            # Ensure numerical bounds
            value = max(min(value, MAX_VALUES[key]), MIN_VALUES[key])
        
        sanitized[key] = value
    
    return sanitized
```

---

### Model Security

```python
# Read-only model loading
def load_model_safe(path: str):
    """Load model with security checks"""
    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    # Check file extension
    if not path.endswith(('.pkl', '.h5')):
        raise ValueError("Invalid model file type")
    
    # Load with restricted permissions
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    return model
```

---

## 📝 Code Standards

### Naming Conventions

- **Classes:** PascalCase (`DataValidator`, `ModelPredictor`)
- **Functions:** snake_case (`validate_input`, `load_models`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_HEIGHT`, `FEATURE_ORDER`)
- **Private methods:** _leading_underscore (`_calculate_bmi`)

### Type Hints

```python
def predict(self, input_df: pd.DataFrame) -> dict:
    """
    Generate prediction from ensemble models
    
    Args:
        input_df: DataFrame with 27 preprocessed features
    
    Returns:
        Dictionary containing prediction results
    """
    pass
```

### Docstrings

```python
def validate_input(self, data: dict) -> tuple[bool, list, list]:
    """
    Validate patient input data
    
    Checks all required fields are present and within valid ranges.
    Returns validation status with detailed error/warning messages.
    
    Args:
        data (dict): Patient input with 17 health parameters
    
    Returns:
        tuple: (is_valid, errors, warnings)
            - is_valid (bool): True if all validations pass
            - errors (list[str]): Critical validation errors
            - warnings (list[str]): Non-critical warnings
    
    Example:
        >>> validator = DataValidator()
        >>> is_valid, errors, warnings = validator.validate_input(patient_data)
        >>> if not is_valid:
        >>>     print(f"Errors: {errors}")
    """
    pass
```

---

## ✅ Testing Specifications

### Unit Test Structure

```python
# tests/test_data_validator.py
class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator()
    
    def test_valid_input(self):
        data = {...}  # Valid patient data
        is_valid, errors, warnings = self.validator.validate_input(data)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_missing_field(self):
        data = {...}  # Missing 'age_category'
        is_valid, errors, warnings = self.validator.validate_input(data)
        assert is_valid == False
        assert "Missing required field: age_category" in errors
    
    def test_invalid_range(self):
        data = {..., 'height_cm': 300}  # Invalid height
        is_valid, errors, warnings = self.validator.validate_input(data)
        assert is_valid == False
        assert any("height" in e.lower() for e in errors)
```

---

## 📚 Dependencies

### Core Dependencies

```txt
# Web Framework
streamlit==1.29.0

# ML Models
scikit-learn==1.3.2
xgboost==2.0.3
tensorflow==2.15.0

# Data Processing
pandas==2.1.4
numpy==1.24.3

# Explainability
shap==0.44.0

# Visualization
matplotlib==3.8.2
plotly==5.18.0

# Utilities
imbalanced-learn==0.11.0  # SMOTE
joblib==1.3.2             # Model serialization
```

---

**Document Status:** ✅ Complete  
**Last Updated:** November 21, 2025  
**Next Review:** February 21, 2026
