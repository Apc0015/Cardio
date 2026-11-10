# CardioFusion: Cardiovascular Disease Risk Assessment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-8%20passed-10B981.svg)](tests/)

**CardioFusion** is an end-to-end machine learning platform I built to predict cardiovascular disease risk using hybrid ensemble models. It combines traditional ML algorithms with gradient boosting and deep learning to achieve high accuracy, and uses SHAP analysis so you can see exactly which factors drive each prediction.

**Live Demo:** [cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app](https://cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app)

---

## What I Built

- **Hybrid Ensemble Models:** Multiple ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network) working together.
- **Interactive Web Interface:** Streamlit app where you can enter health metrics and get real-time risk predictions.
- **SHAP Explainability:** Local feature attribution showing which factors increase or decrease risk for a patient.
- **What-If Analysis:** Interactive counterfactual tool to test how lifestyle changes (like quitting smoking or lowering BMI) affect risk score.
- **Automated Validation:** Input validation and range checks that engineer 27 features from 17 raw health indicators.
- **Cloud Deployment:** Live on Streamlit Cloud with Git LFS for model weights.

---

## Quick Start

### 1. Try the Live App
You can use it right now without installing anything at [cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app](https://cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app).

### 2. Run Locally

```bash
# Clone the repository
git clone https://github.com/Apc0015/Cardio.git
cd Cardio

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Run the application
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
Cardio/
├── src/
│   ├── app.py                   # Main Streamlit application
│   ├── features/                # What-if analysis, reports, insights, education
│   └── utils/                   # Model loading, validation, and SHAP explainer
├── models/                      # Trained models stored with Git LFS
│   ├── baseline_models/         # Logistic Regression, Decision Tree, Random Forest
│   ├── advanced_models/         # XGBoost, Neural Network, Hybrid Ensemble
│   └── preprocessing/           # Scaler and label encoder
├── data/                        # Datasets (raw and preprocessed)
├── notebooks/                   # Preprocessing, baseline, and advanced training notebooks
├── tests/                       # Unit tests for validator and model inference
├── requirements.txt             # Python dependencies
└── packages.txt                 # System packages for Streamlit Cloud
```

---

## Model Performance

These are the actual test results from training on 567,606 patient samples (50/50 balanced with SMOTE) with an 80/20 train/test split (454,084 train / 113,522 test):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **95.34%** | **98.93%** | 91.67% | **95.16%** | **98.61%** |
| **Hybrid Ensemble** | **91.45%** | 90.53% | **92.58%** | **91.55%** | **97.66%** |
| Logistic Regression | 83.94% | 84.81% | 82.70% | 83.74% | 92.61% |
| Random Forest | 83.40% | 80.16% | 88.76% | 84.24% | 91.75% |
| Neural Network | 83.32% | 85.11% | 80.79% | 82.89% | 92.18% |
| Decision Tree | 82.10% | 82.91% | 80.88% | 81.88% | 90.80% |

XGBoost gave the best precision (98.93%) and overall ROC-AUC (98.61%), while the soft-voting hybrid ensemble balanced sensitivity and recall (92.58%).

---

## How the Pipeline Works

1. **Patient Input:** You enter 17 health metrics (demographics, height/weight/BMI, lifestyle habits, medical history).
2. **Validation & Feature Engineering:** `DataValidator` checks ranges and logical consistency (e.g., bounds, invalid combinations), then generates 27 model-ready numerical and one-hot encoded features.
3. **Inference:** Predictions run through the trained models, returning a combined risk percentage, category (Low, Moderate, High), and individual model confidence scores.
4. **Explainability:** SHAP breaks down which specific factors raised or lowered the patient's risk score.
5. **What-If Simulation:** You can adjust individual habits (like smoking, exercise, or weight) to see how the predicted risk changes in real time.

---

## Privacy Note

- Patient data entered in the app is processed in-memory only and is never saved or logged to any database.
- Sessions are isolated and communication is encrypted over HTTPS via Streamlit Cloud.
- **Medical Disclaimer:** This tool is built for educational and research purposes and does not replace medical advice from a doctor.

---

## Authors

- **Ayush Chhoker** - [Apc0015](https://github.com/Apc0015)
- **Taylor Hunter** - [Taylor-Hunter](https://github.com/Taylor-Hunter)
- **Manaswi Thudi** - [thudimanaswi](https://github.com/thudimanaswi)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
