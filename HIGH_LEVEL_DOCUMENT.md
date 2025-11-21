# 📋 CardioFusion ML Platform - High-Level Document

**Project:** CardioFusion - Cardiovascular Disease Prediction Platform  
**Version:** 1.0  
**Date:** November 21, 2025  
**Author:** Ayush Chhoker

---

## 🎯 Executive Summary

CardioFusion is a machine learning-powered web application that predicts cardiovascular disease risk based on patient health data. The platform combines multiple ML algorithms into a hybrid ensemble model, achieving 95%+ ROC-AUC accuracy while providing explainable predictions through SHAP analysis.

**Key Achievements:**
- ✅ 95.19% ROC-AUC accuracy on balanced dataset
- ✅ Real-time predictions via web interface
- ✅ SHAP-based explainability for healthcare professionals
- ✅ Production-ready deployment on Streamlit Cloud
- ✅ Comprehensive test suite with 100% pass rate

---

## 🏗️ System Architecture Overview

### Architecture Type
**Monolithic Web Application** with ML pipeline integration

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│              (Streamlit Web Application)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  APPLICATION LAYER                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │   Input      │  │   Model     │  │   SHAP           │   │
│  │   Validation │  │   Predictor │  │   Explainer      │   │
│  └──────────────┘  └─────────────┘  └──────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    MODEL LAYER                               │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │  Logistic  │ │ Decision │ │  Random  │ │   XGBoost   │  │
│  │ Regression │ │   Tree   │ │  Forest  │ │   Model     │  │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────┘  │
│                 ┌────────────────────────┐                   │
│                 │  Hybrid Ensemble       │                   │
│                 │  (Soft Voting)         │                   │
│                 └────────────────────────┘                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    DATA LAYER                                │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │  Raw Data    │  │  Processed  │  │  Model Artifacts │   │
│  │   (CSV)      │  │  Data (CSV) │  │   (.pkl, .h5)    │   │
│  └──────────────┘  └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit 1.29.0 (Python-based web framework) |
| **Backend** | Python 3.8+, scikit-learn, XGBoost, TensorFlow |
| **ML Models** | Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Networks |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Data Processing** | Pandas, NumPy, StandardScaler, SMOTE |
| **Deployment** | Streamlit Cloud, Git LFS for model storage |
| **Version Control** | Git, GitHub |

---

## 📊 Business Requirements

### Primary Objectives
1. **Predict cardiovascular disease risk** with >90% accuracy
2. **Provide explainable predictions** for healthcare professionals
3. **Enable real-time risk assessment** through web interface
4. **Support clinical decision-making** with confidence scores
5. **Ensure accessibility** through cloud deployment

### Functional Requirements

#### FR-1: Patient Data Input
- Accept 17 patient health parameters (demographics, lifestyle, clinical)
- Validate input data for completeness and correctness
- Support multiple input formats (form-based, batch processing)

#### FR-2: Risk Prediction
- Generate cardiovascular disease risk percentage (0-100%)
- Provide binary classification (Disease/No Disease)
- Calculate prediction confidence score
- Return individual model predictions for transparency

#### FR-3: Explainability
- Generate SHAP feature importance visualizations
- Identify top risk-increasing and risk-decreasing factors
- Provide actionable health recommendations
- Display model-agnostic explanations

#### FR-4: Web Interface
- Intuitive patient data entry form
- Real-time prediction display
- Interactive visualizations
- Responsive design for multiple devices

#### FR-5: Model Management
- Support multiple ML models simultaneously
- Enable hybrid ensemble predictions
- Cache models for performance
- Version control for model artifacts

### Non-Functional Requirements

#### NFR-1: Performance
- **Response Time:** <3 seconds for predictions
- **Throughput:** Support 100+ concurrent users
- **Model Loading:** <10 seconds on app startup

#### NFR-2: Accuracy
- **ROC-AUC:** >90% on test dataset
- **Precision:** >85% to minimize false positives
- **Recall:** >85% to minimize false negatives

#### NFR-3: Reliability
- **Uptime:** 99%+ availability (Streamlit Cloud SLA)
- **Error Handling:** Graceful degradation on failures
- **Data Validation:** 100% input validation coverage

#### NFR-4: Usability
- **User Interface:** Intuitive, no ML expertise required
- **Documentation:** Comprehensive README and guides
- **Accessibility:** WCAG 2.1 AA compliance

#### NFR-5: Maintainability
- **Code Quality:** PEP 8 compliance, type hints
- **Testing:** 100% test suite pass rate
- **Documentation:** Inline comments, docstrings

---

## 🔄 Data Flow Architecture

### End-to-End Prediction Flow

```
1. USER INPUT
   ↓
   Patient enters health data via Streamlit form
   (17 parameters: age, sex, BMI, lifestyle, conditions)
   
2. INPUT VALIDATION
   ↓
   DataValidator.validate_input()
   - Check required fields
   - Validate data types
   - Range validation
   - Business rule validation
   
3. PREPROCESSING
   ↓
   DataValidator.preprocess_for_model()
   - Feature encoding (categorical → numeric)
   - Feature engineering (27 total features)
   - One-hot encoding
   - Feature ordering
   
4. FEATURE SCALING
   ↓
   StandardScaler.transform()
   - Scale 10 numerical features
   - Mean = 0, Std = 1
   - Preserve categorical features
   
5. MODEL PREDICTION
   ↓
   ModelPredictor.predict()
   - Load 5 trained models
   - Generate individual predictions
   - Apply soft voting ensemble
   - Calculate risk percentage
   
6. EXPLAINABILITY
   ↓
   SHAPExplainer.explain_prediction()
   - Generate SHAP values
   - Identify top features
   - Create visualizations
   - Generate recommendations
   
7. RESULT DISPLAY
   ↓
   Streamlit renders:
   - Risk percentage
   - Risk category (Low/Moderate/High)
   - Confidence score
   - SHAP visualizations
   - Health recommendations
```

### Data Pipeline (Training Phase)

```
1. RAW DATA
   ↓
   cardio_train.csv (308,854 records)
   
2. DATA CLEANING
   ↓
   - Remove duplicates
   - Handle missing values
   - Outlier detection & removal
   - Feature type conversion
   
3. FEATURE ENGINEERING
   ↓
   - Create derived features (BMI category, age groups)
   - Encode categorical variables
   - Calculate risk scores
   - Count health conditions
   
4. CLASS BALANCING
   ↓
   SMOTE (Synthetic Minority Oversampling)
   - Balance disease/no disease classes
   - Generate 567,606 samples (50/50 split)
   
5. TRAIN/TEST SPLIT
   ↓
   - 80% train (454,084 samples)
   - 20% test (113,522 samples)
   - Stratified sampling
   
6. FEATURE SCALING
   ↓
   StandardScaler fit on training data
   - Save scaler for production
   
7. MODEL TRAINING
   ↓
   Train 5 models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - Neural Network
   
8. MODEL EVALUATION
   ↓
   - Calculate metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Generate performance visualizations
   - Save best models
   
9. MODEL PERSISTENCE
   ↓
   Save artifacts:
   - Trained models (.pkl, .h5)
   - Scaler (.pkl)
   - Metadata (.txt, .json)
```

---

## 🎨 Component Overview

### 1. Web Application (`src/app.py`)
**Responsibility:** User interface and interaction management

**Key Features:**
- Streamlit-based web interface
- Patient data input form with validation
- Real-time prediction display
- SHAP visualization rendering
- Session state management
- Caching for performance

**Technologies:** Streamlit, Plotly, Matplotlib

---

### 2. Data Validator (`src/utils/data_validator.py`)
**Responsibility:** Input validation and preprocessing

**Key Features:**
- Validate 17 input parameters
- Type checking and range validation
- Feature engineering (27 features from 17 inputs)
- Categorical encoding (one-hot, label encoding)
- Feature ordering for model compatibility

**Validation Rules:**
- Age: 18-80 years (categorical)
- BMI: 10-60 range
- Height: 120-220 cm
- Weight: 30-200 kg
- Consumption values: 0-60 range
- Binary fields: Yes/No validation

---

### 3. Model Predictor (`src/utils/model_utils.py`)
**Responsibility:** Model loading and prediction generation

**Key Features:**
- Load 5 trained ML models
- Load StandardScaler for feature scaling
- Generate individual model predictions
- Apply hybrid ensemble (soft voting)
- Calculate risk percentage and confidence
- Return structured prediction results

**Supported Models:**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Neural Network (MLP)

**Prediction Output:**
```python
{
    'prediction': 1,  # Binary (0 or 1)
    'risk_percentage': 73.5,  # 0-100
    'prediction_label': 'Disease',
    'confidence': 0.95,  # 0.5-1.0
    'individual_models': {
        'Decision Tree': {...},
        'Random Forest': {...}
    }
}
```

---

### 4. SHAP Explainer (`src/utils/shap_explainer.py`)
**Responsibility:** Model explainability and interpretability

**Key Features:**
- Generate SHAP values for predictions
- Identify top risk factors (positive/negative)
- Create waterfall and force plots
- Generate actionable recommendations
- Support multiple model types

**Explanation Output:**
- Top 5 risk-increasing features
- Top 5 risk-decreasing features
- Feature importance scores
- Visual plots
- Health recommendations

---

### 5. Jupyter Notebooks
**Responsibility:** Interactive development and experimentation

**Notebooks:**
- `data_preprocessing.ipynb` - Data cleaning, EDA, feature engineering
- `baseline_models.ipynb` - Traditional ML models (LR, DT, RF)
- `advanced_models.ipynb` - Advanced models (XGBoost, NN, Ensemble)
- `prediction_widget.ipynb` - Interactive prediction testing

---

## 🔐 Security & Privacy Considerations

### Data Privacy
- **No Data Storage:** Predictions are not logged or stored
- **Session Isolation:** Each user session is independent
- **No Personal Information:** Patient data is processed in-memory only
- **HIPAA Awareness:** Design allows for HIPAA compliance with additional controls

### Input Security
- **Input Validation:** All inputs validated before processing
- **Type Safety:** Strict type checking prevents injection attacks
- **Range Limits:** Numerical bounds prevent overflow/underflow
- **Sanitization:** Categorical inputs validated against allowed values

### Model Security
- **Model Versioning:** Git LFS tracks model changes
- **Integrity Checks:** Model loading validates file integrity
- **No User Uploads:** Users cannot upload malicious models
- **Read-Only Models:** Production models are not modifiable via UI

---

## 📈 Performance Characteristics

### Model Performance
| Metric | Value | Target |
|--------|-------|--------|
| ROC-AUC | 95.19% | >90% ✅ |
| Accuracy | 87.76% | >85% ✅ |
| Precision | 89.84% | >85% ✅ |
| Recall | 85.15% | >85% ✅ |
| F1-Score | 87.43% | >85% ✅ |

### Application Performance
| Metric | Value | Target |
|--------|-------|--------|
| Model Loading | ~8 seconds | <10s ✅ |
| Prediction Time | ~0.5 seconds | <3s ✅ |
| SHAP Generation | ~2 seconds | <5s ✅ |
| Page Load | ~3 seconds | <5s ✅ |

### Scalability
- **Concurrent Users:** 100+ (Streamlit Cloud tier)
- **Daily Predictions:** Unlimited (no rate limits)
- **Model Size:** ~50MB (compressed)
- **Memory Usage:** ~500MB per instance

---

## 🚀 Deployment Architecture

### Hosting Platform
**Streamlit Cloud** (Community Tier)

**Advantages:**
- Free hosting for public repositories
- Automatic deployment from GitHub
- Built-in SSL/HTTPS
- Auto-scaling
- Zero DevOps overhead

**Configuration:**
- Runtime: Python 3.10
- Repository: `Apc0015/Cardio`
- Main file: `src/app.py`
- Dependencies: `requirements.txt`, `packages.txt`

### CI/CD Pipeline
```
1. Developer commits to GitHub
   ↓
2. GitHub triggers webhook to Streamlit Cloud
   ↓
3. Streamlit Cloud pulls latest code
   ↓
4. Install dependencies (requirements.txt)
   ↓
5. Download model artifacts (Git LFS)
   ↓
6. Build and deploy application
   ↓
7. Health check and go live
   ↓
8. Old version gracefully terminated
```

### Environment Setup
- **Python Version:** 3.10 (specified in runtime.txt if needed)
- **Dependencies:** Auto-installed from requirements.txt
- **System Packages:** Auto-installed from packages.txt
- **Model Files:** Downloaded via Git LFS
- **Configuration:** `.streamlit/config.toml`

---

## 🧪 Testing Strategy

### Test Coverage
1. **Unit Tests** - Individual component validation
2. **Integration Tests** - End-to-end prediction workflow
3. **Performance Tests** - Model accuracy and speed
4. **Validation Tests** - Different risk profiles

### Test Suite (`tests/test_prediction_accuracy.py`)
- ✅ Healthy patient (low risk)
- ✅ High-risk patient (high risk)
- ✅ Moderate-risk patient (moderate risk)
- ✅ Feature scaling validation
- ✅ Prediction range validation

**Results:** 5/5 tests passing (100%)

---

## 📝 Future Enhancements

### Phase 2 Roadmap
1. **Multi-language Support** - Internationalization (i18n)
2. **Batch Predictions** - CSV upload for multiple patients
3. **API Endpoint** - REST API for integration
4. **User Authentication** - Secure login for healthcare providers
5. **Prediction History** - Store and track predictions (with consent)
6. **Advanced Analytics** - Population-level statistics
7. **Model Retraining** - Automated retraining pipeline
8. **A/B Testing** - Compare model versions
9. **Mobile App** - Native iOS/Android applications
10. **FHIR Integration** - Healthcare data standards compliance

### Technical Improvements
- [ ] Add model versioning system
- [ ] Implement feature drift detection
- [ ] Add automated testing in CI/CD
- [ ] Create Docker containerization
- [ ] Add monitoring and logging (Prometheus, Grafana)
- [ ] Implement rate limiting
- [ ] Add database integration (PostgreSQL)
- [ ] Create admin dashboard

---

## 📚 Documentation Structure

### User Documentation
- `README.md` - Project overview, installation, usage
- `DEPLOYMENT_READY.md` - Deployment checklist and guide

### Technical Documentation
- `HIGH_LEVEL_DOCUMENT.md` - This document (architecture, requirements)
- `LOW_LEVEL_DESIGN.md` - Detailed technical specifications
- `DIAGNOSIS_REPORT.md` - ML model diagnostic analysis
- `SYSTEM_STATUS.md` - Validation and testing summary

### Code Documentation
- Inline comments in all Python modules
- Docstrings for all functions and classes
- Type hints for parameter validation
- README files in subdirectories

---

## 🔗 Dependencies & Integrations

### External Dependencies
- **Python Libraries:** scikit-learn, XGBoost, TensorFlow, SHAP, Streamlit
- **Data Sources:** Kaggle cardiovascular disease dataset
- **Hosting:** Streamlit Cloud
- **Version Control:** GitHub with Git LFS

### Internal Dependencies
```
src/app.py
├── src/utils/data_validator.py
├── src/utils/model_utils.py
└── src/utils/shap_explainer.py

models/
├── baseline_models/*.pkl
├── advanced_models/*.pkl
└── preprocessing/scaler.pkl

data/processed/
├── train_data.csv
└── test_data.csv
```

---

## ⚖️ Constraints & Assumptions

### Constraints
1. **Data Quality:** Model accuracy depends on input data quality
2. **Model Interpretability:** Complex ensemble trades some interpretability for accuracy
3. **Deployment Platform:** Limited to Streamlit Cloud capabilities
4. **Dataset Size:** Training requires ~500K records for optimal performance
5. **Computational Resources:** Limited by Streamlit Cloud tier

### Assumptions
1. **Input Data:** Assumes patients provide accurate health information
2. **Feature Availability:** All 17 required features are available
3. **Target Population:** Model trained on general population data
4. **Medical Use:** Tool assists but does not replace medical diagnosis
5. **Internet Access:** Requires active internet connection

---

## 📞 Support & Maintenance

### Maintenance Plan
- **Model Updates:** Quarterly retraining with new data
- **Dependency Updates:** Monthly security patches
- **Bug Fixes:** Within 48 hours of report
- **Feature Requests:** Evaluated monthly

### Support Channels
- **GitHub Issues:** Bug reports and feature requests
- **Documentation:** README and technical docs
- **Code Comments:** Inline documentation

---

## ✅ Success Metrics

### Technical Metrics
- ✅ Model ROC-AUC: 95.19% (Target: >90%)
- ✅ Prediction Time: <1s (Target: <3s)
- ✅ Test Coverage: 100% pass rate
- ✅ Uptime: 99%+ (Streamlit Cloud SLA)

### Business Metrics
- ✅ Deployment: Live on Streamlit Cloud
- ✅ Documentation: Comprehensive and complete
- ✅ Code Quality: PEP 8 compliant
- ✅ Production Ready: All validation tests passing

---

## 📜 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 21, 2025 | Initial production release |

---

**Document Status:** ✅ Approved for Production  
**Last Updated:** November 21, 2025  
**Next Review:** February 21, 2026
