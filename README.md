# 🩺 CardioFusion: Hybrid Machine Learning for Heart Disease Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CardioFusion** is a professional machine learning platform that predicts cardiovascular disease risk based on medical and lifestyle data. By combining traditional ML models (Random Forest, Logistic Regression) with advanced deep learning (Neural Networks, XGBoost), CardioFusion achieves high accuracy while remaining explainable to healthcare professionals.

## ✨ Features

- 🎯 **Hybrid Ensemble Models** - Combines multiple ML algorithms for superior accuracy
- 📊 **Interactive Web Application** - Professional Streamlit interface for predictions
- 🔍 **SHAP Explainability** - Visual explanations of prediction factors
- 📈 **Comprehensive Analytics** - Detailed model performance metrics and visualizations
- 🧪 **Jupyter Notebooks** - Complete workflow from data preprocessing to model training
- ⚡ **Production Ready** - Docker support, environment management, and automated workflows

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Apc0015/Cardio.git
cd Cardio

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web application (models and data are included)
streamlit run src/app.py
```

Visit http://localhost:8501 to access the CardioFusion web interface.

---

## 📘 Project Overview

The project leverages a **hybrid ensemble architecture**, integrating multiple models to provide reliable predictions with SHAP-based visual explanations that reveal the most influential health factors.

---

## 📂 Dataset Information

**Dataset Used:** [Cardiovascular Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

**Source:** Kaggle / CDC-inspired health indicators dataset  
**Records:** ~70,000  
**Features:** 12 primary features + derived metrics  

### 🧾 Key Features
| Category | Example Features |
|-----------|------------------|
| **Demographics** | Age, Sex |
| **Physical Health** | Height, Weight, BMI |
| **Lifestyle Factors** | Smoking, Alcohol Intake, Physical Activity |
| **Clinical Indicators** | Blood Pressure, Cholesterol, Glucose Levels |
| **Target Variable** | Presence of Cardiovascular Disease (0 = No, 1 = Yes) |

### ⚙️ Preprocessing Steps
- Missing value imputation  
- Feature encoding (categorical to numeric)  
- Outlier removal and scaling  
- SMOTE for class balancing  
- Train-test split with stratification  

The final cleaned dataset is saved as **`cleaned_data.csv`** for model training.

---

## 🧠 Project Workflow

```plaintext
      ┌────────────────────────┐
      │      Raw Dataset       │
      └──────────┬─────────────┘
                 │
                 ▼
   ┌──────────────────────────────┐
   │  Data Cleaning & EDA         │
   │  - Handle missing data       │
   │  - Feature scaling/encoding  │
   │  - Correlation heatmaps      │
   │  - Baseline models (LogReg,  │
   │    Decision Tree, RandomForest)
   └──────────┬──────────────────┘
              │ Cleaned Data (CSV)
              ▼
   ┌──────────────────────────────┐
   │  Model Development           │
   │  - Train XGBoost, GradBoost  │
   │    and Neural Network (MLP)  │
   │  - Hybrid Ensemble (Soft Vote)
   │  - Model Evaluation          │
   └──────────┬──────────────────┘
              │ Hybrid Model (PKL)
              ▼
   ┌──────────────────────────────┐
   │ Explainability & App         │
   │  - SHAP feature importance   │
   │  - Streamlit web interface   │
   │  - ROC curve, SHAP summary   │
   │  - README & Documentation    │
   └──────────────────────────────┘
```

---

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)

---

## 🔧 Installation

### Prerequisites

- **Python 3.8+** (Python 3.10 recommended)
- **pip** package manager
- **4GB+ RAM** recommended
- **Git** for version control
- **Dataset** - Cardiovascular disease dataset (CSV format)

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/Taylor-Hunter/CardioFusion.git
cd CardioFusion

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n cardiofusion python=3.10 -y
conda activate cardiofusion

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Makefile (Recommended)

```bash
# Complete setup with one command
make setup
```

This will:
- Install all dependencies
- Create `.env` file from template
- Set up the development environment

---

## 📦 Dataset Setup

### Preparing Your Dataset

This project uses a cardiovascular disease dataset. Place your dataset file in the `data/raw/` directory.

**Expected file location:**
```
data/raw/cardio_train.csv
```

**Dataset Requirements:**
- Format: CSV file
- Expected columns: Demographics, lifestyle factors, clinical measurements
- Approximate size: 300K+ records recommended
- Target variable: Heart disease indicator (binary classification)

### Getting the Dataset

**Option 1: Kaggle (Recommended)**
1. Visit [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
2. Download the dataset
3. Extract `cardio_train.csv`
4. Place in `data/raw/` directory

**Option 2: Your Own Dataset**
- Use any compatible cardiovascular disease dataset
- Ensure it follows a similar structure (demographics, lifestyle, clinical features)
- Place CSV file in `data/raw/` directory

**Verify Dataset:**
```bash
# Check if dataset is properly placed
python scripts/download_data.py
```

---

## ⚙️ Configuration

### Environment Variables (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings if needed
nano .env  # or use your preferred editor
```

The `.env` file contains optional configuration for model paths, logging, and other settings.

---

## 🎯 Usage

### Running the Web Application

The **fastest way** to use CardioFusion is through the Streamlit web interface:

**Using Makefile:**
```bash
make run
```

**Using Streamlit directly:**
```bash
streamlit run src/app.py
```

**Using Python:**
```bash
python -m streamlit run src/app.py
```

The application will open in your browser at **http://localhost:8501**

### Complete Workflow (First Time Setup)

If this is your first time running the project, follow these steps:

#### Step 1: Prepare Dataset
```bash
# Ensure dataset is in data/raw/ directory
# Expected file: data/raw/cardio_train.csv

# Verify dataset is present
python scripts/download_data.py
```
✅ **Required:** Dataset must be placed in `data/raw/` before proceeding

#### Step 2: Preprocess Data
```bash
# Using Makefile (runs notebook programmatically)
make preprocess

# Or open in Jupyter manually
jupyter notebook notebooks/data_preprocessing.ipynb
```
⏱️ **Runtime:** ~3-5 minutes
- Cleans and validates data
- Handles missing values and outliers
- Applies SMOTE for class balancing
- Generates train/test splits
- Saves processed data to `data/processed/`

#### Step 3: Train Models
```bash
# Using Makefile (trains all models)
make train

# Or train models individually via Jupyter
jupyter notebook notebooks/baseline_models.ipynb
jupyter notebook notebooks/advanced_models.ipynb
```
⏱️ **Runtime:** ~5-10 minutes
- Trains baseline models (Logistic Regression, Decision Tree, Random Forest)
- Trains advanced models (XGBoost, LightGBM, Neural Networks)
- Generates performance metrics and visualizations
- Saves trained models to `models/`

#### Step 4: Launch Application
```bash
make run
```

### Using Jupyter Notebooks

**Interactive Development:**
```bash
# Start Jupyter server
jupyter notebook

# Or use JupyterLab
jupyter lab
```

**Available Notebooks:**
- [data_preprocessing.ipynb](notebooks/data_preprocessing.ipynb) - Data cleaning and feature engineering
- [baseline_models.ipynb](notebooks/baseline_models.ipynb) - Traditional ML models
- [advanced_models.ipynb](notebooks/advanced_models.ipynb) - Deep learning and ensemble models
- [prediction_widget.ipynb](notebooks/prediction_widget.ipynb) - Interactive prediction interface

### Makefile Commands Reference

```bash
make help           # Show all available commands
make install        # Install dependencies only
make setup          # Complete project setup
make preprocess     # Run data preprocessing
make train          # Train all models
make run            # Launch Streamlit app
make test           # Run tests
make lint           # Run code linting
make format         # Format code with black
make clean          # Clean temporary files
make docker-build   # Build Docker image
make docker-run     # Run in Docker container
```

---

## 📁 Project Structure

```
CardioFusion/
├── 📄 README.md                     # Project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 packages.txt                  # System dependencies for deployment
├── 📄 Makefile                      # Automation commands
├── 📄 .env.example                  # Environment variables template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 .gitattributes                # Git LFS configuration
│
├── 📁 .streamlit/                   # Streamlit configuration
│   └── config.toml                 # App theme and server settings
│
├── 📁 src/                          # Application source code
│   ├── __init__.py
│   ├── app.py                      # Streamlit web application (main entry point)
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       ├── model_utils.py          # Model prediction utilities
│       ├── data_validator.py       # Input data validation
│       └── shap_explainer.py       # SHAP-based explanations
│
├── 📁 notebooks/                    # Jupyter notebooks for development
│   ├── data_preprocessing.ipynb    # Data cleaning & feature engineering
│   ├── baseline_models.ipynb       # Traditional ML models
│   ├── advanced_models.ipynb       # XGBoost, Neural Networks, Ensembles
│   └── prediction_widget.ipynb     # Interactive prediction widget
│
├── 📁 scripts/                      # Automation scripts
│   └── download_data.py            # Kaggle dataset downloader
│
├── 📁 models/                       # Trained model artifacts (Git LFS)
│   ├── baseline_models/            # Logistic Regression, Decision Tree, Random Forest
│   ├── advanced_models/            # XGBoost, Neural Networks, Hybrid Ensemble
│   └── preprocessing/              # Scalers, encoders, transformers
│
├── 📁 data/                         # Data directory (Git LFS)
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Cleaned and preprocessed data
│   └── external/                   # External data sources
│
├── 📁 outputs/                      # Generated outputs
│   ├── figures/                    # Plots and visualizations
│   └── reports/                    # Performance reports
│
├── 📁 logs/                         # Application logs
│
└── 📁 assets/                       # Static assets
    └── images/                     # Images and icons
```

### Key Files

- [src/app.py](src/app.py) - Main Streamlit web application
- [.streamlit/config.toml](.streamlit/config.toml) - Streamlit configuration
- [scripts/download_data.py](scripts/download_data.py) - Automated dataset downloader
- [Makefile](Makefile) - Automation commands for common tasks
- [requirements.txt](requirements.txt) - All Python dependencies
- [packages.txt](packages.txt) - System dependencies for Streamlit Cloud
- [.env.example](.env.example) - Environment configuration template

## 📊 Model Performance

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Original Records** | 308,854 |
| **Class Distribution** | 92% No Disease, 8% Disease |
| **After SMOTE Balancing** | 567,606 records (50/50 split) |
| **Features** | 27 engineered and encoded features |
| **Train/Test Split** | 80% / 20% (454,084 / 113,522 samples) |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Decision Tree** | **87.76%** | **89.84%** | 85.15% | **87.43%** | **95.19%** |
| **Random Forest** | 84.08% | 80.92% | **89.18%** | 84.85% | 92.47% |
| Logistic Regression | 80.11% | 79.37% | 81.37% | 80.35% | 88.68% |
| XGBoost | 86.50% | 88.20% | 84.80% | 86.45% | 94.10% |
| Neural Network | 83.20% | 81.50% | 85.90% | 83.65% | 91.30% |

### Key Insights

- **Decision Tree** achieves best overall performance with **95.19% ROC-AUC**
- **Random Forest** has highest recall (**89.18%**) - catches more cardiovascular disease cases
- All models demonstrate strong performance on the balanced dataset
- Top predictive features: **Age**, **General Health**, **Health Conditions Count**, **BMI**

### Feature Importance (SHAP Analysis)

The SHAP explainability analysis reveals the most influential factors:

1. **Age** - Primary risk factor
2. **General Health Status** - Strong indicator
3. **Health Conditions Count** - Comorbidity impact
4. **BMI Category** - Obesity correlation
5. **Exercise Habits** - Lifestyle factor
6. **Smoking History** - Major risk factor
7. **Blood Pressure** - Clinical indicator

---

## 🛠️ Development

### Running Tests

```bash
# Run all tests
make test

# Or use pytest directly
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Or manually:
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

### Docker Deployment

```bash
# Build Docker image
make docker-build

# Run container
make docker-run

# Application will be available at http://localhost:8501

# Stop container
make docker-stop
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### **"Module not found" or Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install specific missing package
pip install <package_name>
```

#### **Dataset Not Found Errors**
```bash
# Verify dataset is in correct location
ls data/raw/cardio_train.csv  # macOS/Linux
dir data\raw\cardio_train.csv  # Windows

# Run verification script
python scripts/download_data.py

# If missing, download from Kaggle and place in data/raw/
# Visit: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
```

#### **Jupyter Kernel Not Found**
```bash
# Install IPython kernel
pip install ipykernel

# Create kernel for this project
python -m ipykernel install --user --name cardiofusion --display-name "CardioFusion"

# In Jupyter: Kernel > Change Kernel > CardioFusion
```

#### **File Not Found Errors**
- Ensure you're in the project root directory
- Ensure dataset is in `data/raw/` directory before preprocessing
- Run data preprocessing before training models:
  ```bash
  make preprocess
  make train
  ```
- Check that `data/` and `models/` directories exist
- File paths are relative to project root

#### **Memory Errors**
- Dataset contains 567K records after SMOTE balancing
- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM
- Close other applications to free memory
- Consider reducing dataset size for testing

#### **Streamlit App Won't Start**
```bash
# Check if port 8501 is available
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Use different port
streamlit run src/app.py --server.port 8502

# Clear Streamlit cache
streamlit cache clear
```

#### **Model Files Missing**
```bash
# Train all models from scratch
make train

# Or train individually via notebooks
jupyter notebook notebooks/baseline_models.ipynb
jupyter notebook notebooks/advanced_models.ipynb
```

#### **SMOTE/Preprocessing Errors**
- Ensure all features are numeric before SMOTE
- Check for missing values in data
- Verify data types: `df.dtypes`
- Re-run preprocessing notebook from the beginning

### Getting Help

If you encounter issues not covered above:

1. Check existing [GitHub]
2. Search for error messages in the documentation
3. Ensure you're using Python 3.8+ and have all dependencies installed
4. Try cleaning and reinstalling:
   ```bash
   make clean
   pip install -r requirements.txt
   ```
5. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Python version (`python --version`)
   - OS and version

---

## 📋 System Requirements

### Hardware
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for datasets and models
- **CPU**: Modern multi-core processor recommended

### Software
- **Python**: 3.8 or higher (3.10 recommended)
- **OS**:
  - macOS 10.14+
  - Windows 10/11
  - Linux (Ubuntu 20.04+ or equivalent)
- **Git**: For version control
- **pip**: 20.0 or higher

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Taylor Hunter and Ayush Chhoker ** - *Initial work*
- **CardioFusion Development Team**

---

## 🙏 Acknowledgments

- Dataset from [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- Built with Streamlit, scikit-learn, XGBoost, and TensorFlow
- SHAP library for model explainability
- CDC for cardiovascular health indicators

---



---

