# CardioFusion: Cardiovascular Disease Risk Assessment

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CardioFusion** is a professional machine learning platform that predicts cardiovascular disease risk using hybrid ensemble models. The platform combines traditional ML algorithms with advanced deep learning to achieve high accuracy while providing explainable predictions through SHAP analysis.

**Live Demo:** [cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app](https://cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app)

---

## Key Features

- **Hybrid Ensemble Models** - Multiple ML algorithms working together for superior accuracy
- **Interactive Web Interface** - Professional Streamlit app for real-time predictions
- **SHAP Explainability** - Visual explanations showing which factors drive predictions
- **Comprehensive Analytics** - Model performance metrics and comparisons
- **Cloud Deployment** - Production-ready on Streamlit Cloud with Git LFS
- **Clinical Focus** - Designed for healthcare professionals with actionable insights

---

## Quick Start

### Option 1: Use the Live App

Visit [cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app](https://cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app) to use the deployed application immediately.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/Apc0015/Cardio.git
cd Cardio

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/app.py
```

Visit **http://localhost:8501** in your browser.

---

## Project Structure

```
Cardio/
├── src/
│   └── app.py                   # Main Streamlit application
├── models/                       # Trained ML models (Git LFS)
│   ├── baseline_models/         # Logistic Regression, Decision Tree, Random Forest
│   ├── advanced_models/         # XGBoost, Neural Network, Hybrid Ensemble
│   └── preprocessing/           # Scalers and encoders
├── data/                         # Dataset files (Git LFS)
│   ├── raw/                     # Original cardiovascular dataset
│   └── processed/               # Cleaned and preprocessed data
├── notebooks/                    # Jupyter notebooks
│   ├── data_preprocessing.ipynb # Data cleaning and feature engineering
│   ├── baseline_models.ipynb    # Traditional ML models
│   └── advanced_models.ipynb    # Advanced models and ensembles
├── requirements.txt             # Python dependencies
└── packages.txt                 # System packages for deployment
```

---

## Model Performance

Based on actual test results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **95.34%** | **98.93%** | 91.67% | **95.16%** | **98.61%** |
| **Hybrid Ensemble** | **91.45%** | 90.53% | **92.58%** | **91.55%** | **97.66%** |
| Logistic Regression | 83.94% | 84.81% | 82.70% | 83.74% | 92.61% |
| Random Forest | 83.40% | 80.16% | 88.76% | 84.24% | 91.75% |
| Neural Network | 83.32% | 85.11% | 80.79% | 82.89% | 92.18% |
| Decision Tree | 82.10% | 82.91% | 80.88% | 81.88% | 90.80% |

**Dataset:** 567,606 samples (50/50 balanced with SMOTE)
**Train/Test Split:** 80/20 (454,084 / 113,522 samples)

---

## How It Works

### 1. Data Input
Patients or healthcare providers enter 17 health parameters including:
- Demographics (age, sex)
- Physical measurements (height, weight, BMI)
- Lifestyle factors (exercise, smoking, alcohol, diet)
- Medical history (diabetes, depression, arthritis, cancer)
- General health status

### 2. Feature Engineering
The system automatically:
- Encodes categorical variables
- Engineers 27 features from 17 inputs
- Scales numerical features
- Validates input ranges

### 3. Ensemble Prediction
6 trained models generate predictions:
- **Logistic Regression** - Baseline linear model
- **Decision Tree** - Interpretable tree-based model
- **Random Forest** - Ensemble of trees
- **XGBoost** - Gradient boosting
- **Neural Network** - Deep learning (Keras)
- **Hybrid Ensemble** - Weighted combination

### 4. Risk Assessment
The system provides:
- **Risk Percentage** (0-100%)
- **Risk Category** (Low, Moderate, High)
- **Confidence Score**
- **Individual Model Predictions**

### 5. Explainability
SHAP analysis reveals:
- Top risk-increasing factors
- Top risk-decreasing factors
- Feature importance scores
- Actionable health recommendations

---

## Streamlit Cloud Deployment

### Current Deployment
- **Live URL:** https://cardio-7ju34z7mh8sbn8fddyj2p8.streamlit.app
- **Platform:** Streamlit Community Cloud
- **Python Version:** 3.13
- **Auto-deploy:** Enabled on push to `main` branch

### Deploy Your Own Instance

#### Step 1: Fork & Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Cardio.git
cd Cardio
```

#### Step 2: Deploy to Streamlit Cloud
1. Visit **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Configure deployment:
   - **Repository:** `YOUR_USERNAME/Cardio`
   - **Branch:** `main`
   - **Main file path:** `src/app.py`
   - **Python version:** 3.13 (auto-detected)
5. Click **"Deploy!"**

#### Step 3: Wait for Deployment
- **Initial deployment:** ~5-8 minutes
- **Subsequent updates:** ~3-5 minutes
- Models and data download automatically via Git LFS

#### Step 4: Access Your App
Your app will be live at: `https://your-app-name.streamlit.app`

### Deployment Configuration

**Files Used:**
- `requirements.txt` - Python dependencies (TensorFlow, scikit-learn, SHAP, etc.)
- `packages.txt` - System packages (libgomp1 for OpenMP support)
- `.streamlit/config.toml` - Streamlit configuration
- `.gitattributes` - Git LFS configuration for large files

**Git LFS Files (Auto-downloaded):**
- All model files in `models/` (~49 MB)
- All data files in `data/` (~257 MB)
- Total: ~306 MB

### Automatic Updates

Every push to `main` branch triggers automatic redeployment:
```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push origin main

# Streamlit Cloud automatically redeploys (~3-5 minutes)
```

### Resource Limits (Free Tier)
- **Memory:** 1 GB (app uses ~800 MB)
- **CPU:** 2 cores
- **Storage:** 50 GB (app uses ~306 MB)
- **Concurrent users:** 100+
- **Monthly hours:** Unlimited
- **Apps per account:** 1 public app

---

## Development

### Local Setup with Jupyter

```bash
# Install development dependencies
pip install -r requirements.txt
pip install jupyter ipykernel

# Start Jupyter Lab
jupyter lab
```

**Available Notebooks:**
- `data_preprocessing.ipynb` - Data cleaning and feature engineering
- `baseline_models.ipynb` - Traditional ML model training
- `advanced_models.ipynb` - Advanced models and ensemble

### Code Quality

```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

---

## Documentation

- **README.md** (this file) - Quick start and deployment guide
- **data/README.md** - Data directory documentation
- **models/README.md** - Models directory documentation
- **notebooks/** - Interactive Jupyter notebooks with detailed workflows

---

## Privacy & Security

- **No Data Storage:** Patient data is processed in-memory only
- **Session Isolation:** Each user session is independent
- **Input Validation:** All inputs validated and sanitized
- **HTTPS:** Encrypted communication via Streamlit Cloud
- **No Logging:** Predictions are not logged or tracked
- **Open Source:** Code is fully transparent and auditable

**Medical Disclaimer:** This tool is for educational and informational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Authors

- **Ayush Chhoker** - *Developer* - [Apc0015](https://github.com/Apc0015)
- **Taylor Hunter** - *Developer* - [Taylor-Hunter](https://github.com/Taylor-Hunter)
- **Manaswi Thudi** - *Developer* - [thudimanaswi](https://github.com/thudimanaswi)

---

## Acknowledgments

- Dataset from [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset/data)
- Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io), and [TensorFlow](https://www.tensorflow.org)
- SHAP library for model explainability
- CDC for cardiovascular health indicators and research

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Apc0015/Cardio/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Apc0015/Cardio/discussions)

---

## Roadmap

### Completed
- [x] Hybrid ensemble model development
- [x] SHAP explainability integration
- [x] Professional Streamlit interface
- [x] Streamlit Cloud deployment
- [x] Git LFS for model storage
- [x] Comprehensive testing suite

### Planned
- [ ] Multi-language support (i18n)
- [ ] Batch prediction via CSV upload
- [ ] REST API endpoint
- [ ] User authentication for healthcare providers
- [ ] Prediction history tracking
- [ ] Mobile-responsive improvements
- [ ] FHIR healthcare data integration
- [ ] Automated model retraining pipeline

---

**Made for better cardiovascular health outcomes**
