"""
CardioFusion Clinical Platform
Professional Web Application for Cardiovascular Disease Risk Assessment

Author: Ayush Chhoker
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.model_utils import ModelPredictor
from src.utils.data_validator import DataValidator
from src.utils.shap_explainer import SHAPExplainer

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CardioFusion | Heart Disease Risk Assessment",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CardioFusion - Professional ML Platform for Cardiovascular Disease Prediction"
    }
)

# ============================================
# PROFESSIONAL MEDICAL STYLING
# ============================================

def load_css():
    """Load custom CSS for professional medical design"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --success-color: #059669;
        --warning-color: #d97706;
        --danger-color: #dc2626;
        --background: #f8fafc;
        --text-dark: #1e293b;
    }

    /* Professional header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 0;
        color: #e0e7ff;
    }

    /* Risk score card styling */
    .risk-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        margin: 1rem 0;
    }

    .risk-low { border-left-color: #059669; }
    .risk-moderate { border-left-color: #d97706; }
    .risk-high { border-left-color: #dc2626; }

    /* Clinical input sections */
    .input-section {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(30, 58, 138, 0.3);
    }

    /* Disclaimer styling */
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        font-size: 0.95rem;
        color: #92400e;
    }

    /* Feature contribution bars */
    .contribution-bar {
        height: 25px;
        border-radius: 4px;
        margin: 5px 0;
        transition: all 0.3s;
    }

    .contribution-positive { background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%); }
    .contribution-negative { background: linear-gradient(90deg, #059669 0%, #10b981 100%); }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize session state variables"""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'simple'
    if 'background_data' not in st.session_state:
        st.session_state.background_data = None

# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        models_path = project_root / 'models'
        predictor = ModelPredictor(str(models_path))
        success = predictor.load_models()
        if success:
            return predictor
        return None
    except Exception as e:
        st.error(f" Error loading models: {str(e)}")
        return None

@st.cache_data
def load_background_data():
    """Load background data for SHAP - optimized for speed"""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        train_data_path = project_root / 'data' / 'processed' / 'train_data.csv'
        # Only load first 1000 rows for faster loading
        train_data = pd.read_csv(train_data_path, nrows=1000)
        X_train = train_data.drop('Heart_Disease', axis=1)
        # Sample for SHAP background - reduced sample size
        return X_train.sample(min(50, len(X_train)), random_state=42)
    except Exception as e:
        print(f" Could not load background data for SHAP: {e}")
        return None

# ============================================
# UI COMPONENTS
# ============================================

def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <h1> CardioFusion Clinical Platform</h1>
        <p>Advanced Cardiovascular Disease Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation and settings"""
    with st.sidebar:
        st.title("Navigation")

        page = st.radio(
            "Select View:",
            [" Risk Assessment", " Model Performance"],
            label_visibility="collapsed"
        )

        st.divider()

        st.subheader(" Settings")
        view_mode = st.radio(
            "Prediction Detail Level:",
            ["Simple View", "Detailed Analysis"],
            index=0 if st.session_state.view_mode == 'simple' else 1
        )
        st.session_state.view_mode = 'simple' if view_mode == "Simple View" else 'detailed'

        st.divider()

        st.markdown("""
        ###  Quick Guide
        1. Enter patient information
        2. Click **Analyze Risk**
        3. Review predictions
        4. Consult healthcare professional

        ---

        **Version**: 1.0.0
        **Models**: 6 Advanced ML Algorithms
        **Accuracy**: 92%+ ROC-AUC
        """)

        return page

def render_patient_input_form():
    """Render comprehensive patient input form"""

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title"> Demographic Information</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age_category = st.selectbox(
                "Age Group",
                ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                 '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
                index=6
            )
        with col2:
            sex = st.selectbox("Biological Sex", ['Male', 'Female'])

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title"> Physical Measurements</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=0.5)
        with col3:
            bmi = weight_kg / ((height_cm/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title"> Lifestyle Factors</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            exercise = st.radio("Regular Physical Exercise", ['Yes', 'No'], horizontal=True)
            smoking_history = st.radio("Smoking History", ['No', 'Yes'], horizontal=True)
        with col2:
            alcohol_consumption = st.slider("Alcohol Consumption (units/month)", 0, 30, 0)
            fruit_consumption = st.slider("Fruit Intake (servings/month)", 0, 120, 30)

        col3, col4 = st.columns(2)
        with col3:
            green_vegetables_consumption = st.slider("Green Vegetables (servings/month)", 0, 128, 12)
        with col4:
            fried_potato_consumption = st.slider("Fried Potato (servings/month)", 0, 128, 4)

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title"> Health Status</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            general_health = st.select_slider(
                "General Health",
                options=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                value='Good'
            )
            checkup = st.selectbox(
                "Last Medical Checkup",
                ['Within the past year', 'Within the past 2 years',
                 'Within the past 5 years', '5 or more years ago', 'Never']
            )
        with col2:
            diabetes = st.selectbox(
                "Diabetes Status",
                ['No', 'Yes', 'No, pre-diabetes or borderline diabetes',
                 'Yes, but female told only during pregnancy']
            )

        st.markdown("**Existing Medical Conditions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            depression = 'Yes' if st.checkbox("Depression") else 'No'
            arthritis = 'Yes' if st.checkbox("Arthritis") else 'No'
        with col2:
            skin_cancer = 'Yes' if st.checkbox("Skin Cancer") else 'No'
            other_cancer = 'Yes' if st.checkbox("Other Cancer") else 'No'

        st.markdown('</div>', unsafe_allow_html=True)

    # Collect all data
    patient_data = {
        'age_category': age_category,
        'sex': sex,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': bmi,
        'exercise': exercise,
        'smoking_history': smoking_history,
        'alcohol_consumption': alcohol_consumption,
        'fruit_consumption': fruit_consumption,
        'green_vegetables_consumption': green_vegetables_consumption,
        'fried_potato_consumption': fried_potato_consumption,
        'general_health': general_health,
        'checkup': checkup,
        'diabetes': diabetes,
        'depression': depression,
        'arthritis': arthritis,
        'skin_cancer': skin_cancer,
        'other_cancer': other_cancer
    }

    return patient_data

def render_risk_gauge(risk_percentage, risk_category, color):
    """Render professional risk gauge visualization"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24, 'color': '#1e293b'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 50], 'color': '#fef3c7'},
                {'range': [50, 70], 'color': '#fed7aa'},
                {'range': [70, 100], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "#1e293b", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, -apple-system, sans-serif"}
    )

    return fig

def render_feature_contributions(explanation):
    """Render SHAP feature contributions"""

    if 'top_positive' not in explanation or 'top_negative' not in explanation:
        return

    st.markdown("###  Feature Contribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("####  Risk-Increasing Factors")
        for feature, value in explanation['top_positive']:
            bar_width = min(abs(value) * 100, 100)
            feature_name = feature.replace('_', ' ').title()
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="font-weight: 500; margin-bottom: 5px;">{feature_name}</div>
                <div style="background: linear-gradient(90deg, #dc2626 0%, #ef4444 {bar_width}%, #f3f4f6 {bar_width}%);
                            height: 25px; border-radius: 4px; padding: 4px 10px; color: white; font-weight: 600;">
                    +{value:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("####  Risk-Decreasing Factors")
        for feature, value in explanation['top_negative']:
            bar_width = min(abs(value) * 100, 100)
            feature_name = feature.replace('_', ' ').title()
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="font-weight: 500; margin-bottom: 5px;">{feature_name}</div>
                <div style="background: linear-gradient(90deg, #059669 0%, #10b981 {bar_width}%, #f3f4f6 {bar_width}%);
                            height: 25px; border-radius: 4px; padding: 4px 10px; color: white; font-weight: 600;">
                    {value:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_simple_prediction(prediction):
    """Render simple prediction view"""

    risk_pct = prediction['risk_percentage']
    category, _, color = st.session_state.predictor.get_risk_category(risk_pct)

    st.markdown(f"""
    <div class="risk-card risk-{category.split()[0].lower()}">
        <h2 style="color: {color}; margin: 0;">{category}</h2>
        <p style="font-size: 1.5rem; margin: 1rem 0 0 0; color: #64748b;">
            Risk Score: <strong style="color: {color};">{risk_pct:.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", prediction['prediction_label'])
    with col2:
        st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
    with col3:
        st.metric("Model", "Ensemble")

    # Risk gauge
    st.plotly_chart(render_risk_gauge(risk_pct, category, color), use_container_width=True)

def render_detailed_prediction(prediction, input_data):
    """Render detailed prediction with multi-tab analysis"""

    # Import new feature modules
    from src.features.what_if_analysis import render_what_if_analysis
    from src.features.patient_insights import render_patient_insights
    from src.features.reports import render_reports_page
    from src.features.education import render_education_hub

    # Simple view first
    render_simple_prediction(prediction)

    st.divider()

    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Model Details",
        " What-If Analysis",
        " Risk Insights",
        " Reports",
        " Model Explainability",
        " Education"
    ])

    with tab1:
        st.markdown("### Individual Model Predictions")

        # Individual model predictions
        if 'individual_models' in prediction:
            models_df = pd.DataFrame([
                {
                    'Model': name,
                    'Risk Probability': f"{results['probability_disease']*100:.1f}%",
                    'Weight': results['weight']
                }
                for name, results in prediction['individual_models'].items()
            ])

            st.dataframe(models_df, use_container_width=True, hide_index=True)

            st.divider()

            # Show ensemble calculation
            st.markdown("###  How Ensemble Works")
            st.info("""
            The final risk prediction is a **weighted average** of all individual models.
            Better-performing models receive higher weights in the final prediction.

            This ensemble approach provides more reliable predictions than any single model.
            """)

    with tab2:
        # Get patient data from session state (you'll need to pass this)
        if 'patient_data' in st.session_state:
            render_what_if_analysis(
                st.session_state.predictor,
                input_data,
                st.session_state.patient_data,
                prediction
            )
        else:
            st.warning("Patient data not available for What-If analysis")

    with tab3:
        if 'patient_data' in st.session_state:
            render_patient_insights(
                prediction,
                st.session_state.patient_data,
                st.session_state.predictor
            )
        else:
            st.warning("Patient data not available for insights")

    with tab4:
        if 'patient_data' in st.session_state:
            render_reports_page(
                prediction,
                st.session_state.patient_data,
                input_data,
                st.session_state.predictor
            )
        else:
            st.warning("Patient data not available for reports")

    with tab5:
        st.markdown("###  Model Explainability (SHAP Analysis)")

        # SHAP explanation
        if st.session_state.background_data is not None:
            try:
                with st.spinner("Generating SHAP explanation..."):
                    # Get best model for SHAP
                    model = list(st.session_state.predictor.models.values())[0]
                    model_name = list(st.session_state.predictor.models.keys())[0]

                    st.info(f" Using {model_name} for SHAP analysis...")

                    explainer = SHAPExplainer(model, st.session_state.background_data)
                    explanation = explainer.explain_prediction(input_data)

                    if 'error' not in explanation:
                        render_feature_contributions(explanation)

                        st.divider()

                        # Clinical recommendations
                        st.markdown("###  Clinical Recommendations")
                        recommendations = explainer.get_recommendations(explanation)

                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"{i}. {rec}")
                    else:
                        st.error(f" SHAP Error: {explanation.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f" SHAP analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning(" SHAP explainability not available - background data could not be loaded")

    with tab6:
        render_education_hub()

def render_disclaimer():
    """Render medical disclaimer"""
    st.markdown("""
    <div class="disclaimer">
        <strong> Medical Disclaimer:</strong><br>
        This tool is for educational and informational purposes only. It does not provide medical advice,
        diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider
        with any questions you may have regarding a medical condition. Never disregard professional medical
        advice or delay in seeking it because of something you have read or seen on this platform.
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application logic"""

    # Load CSS
    load_css()

    # Initialize session state
    init_session_state()

    # Load models with progress indication
    if st.session_state.predictor is None:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text(" Loading ML models...")
        progress_bar.progress(30)
        st.session_state.predictor = load_models()
        
        progress_bar.progress(70)
        progress_text.text(" Loading background data...")
        st.session_state.background_data = load_background_data()
        
        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()

    # Render UI
    render_header()
    page = render_sidebar()

    if page == " Risk Assessment":
        render_risk_assessment_page()
    elif page == " Model Performance":
        render_performance_page()

def render_risk_assessment_page():
    """Render main risk assessment page"""

    if st.session_state.predictor is None:
        st.error(" Models not loaded. Please ensure models are trained and saved.")
        st.info(" Run data_preprocessing.ipynb and baseline_models.ipynb first.")
        return

    st.markdown("##  Patient Risk Assessment")

    # Patient input form
    patient_data = render_patient_input_form()

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(" Analyze Risk Profile", use_container_width=True)

    if analyze_button:
        # Validate input
        validator = DataValidator()
        is_valid, errors, warnings = validator.validate_input(patient_data)

        if not is_valid:
            st.error(" Input Validation Failed:")
            for error in errors:
                st.error(error)
            return

        if warnings:
            for warning in warnings:
                st.warning(warning)

        # Store patient data in session state for other features
        st.session_state.patient_data = patient_data

        # Preprocess
        input_df = validator.preprocess_for_model(patient_data)

        # Make prediction
        with st.spinner(" Analyzing patient data..."):
            prediction = st.session_state.predictor.predict(input_df)

        st.success(" Analysis Complete!")

        # Render results based on view mode
        st.markdown("---")
        st.markdown("##  Risk Assessment Results")

        if st.session_state.view_mode == 'simple':
            render_simple_prediction(prediction)
        else:
            render_detailed_prediction(prediction, input_df)

        # Disclaimer
        render_disclaimer()

@st.cache_data
def load_model_results():
    """Load pre-computed model results from saved CSV files"""
    project_root = Path(__file__).parent.parent

    try:
        # Try to load baseline results
        baseline_path = project_root / 'models/baseline_models/baseline_results.csv'
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
        else:
            baseline_df = pd.DataFrame()

        # Try to load advanced results
        advanced_path = project_root / 'models/advanced_models/advanced_results.csv'
        if advanced_path.exists():
            advanced_df = pd.read_csv(advanced_path)
        else:
            advanced_df = pd.DataFrame()

        # Combine results if both exist
        if not baseline_df.empty and not advanced_df.empty:
            all_results = pd.concat([baseline_df, advanced_df], ignore_index=True)
        elif not baseline_df.empty:
            all_results = baseline_df
        elif not advanced_df.empty:
            all_results = advanced_df
        else:
            return None

        # Normalize metric columns to ensure correct display in UI
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        for col in metric_cols:
            if col in all_results.columns:
                # Convert to numeric and coerce errors
                all_results[col] = pd.to_numeric(all_results[col], errors='coerce')

                # If values appear to be in percentages already (e.g., 95 or '95%'),
                # convert them to fraction (0-1). Also handle string percentages.
                # Detect max value to determine scale.
                if all_results[col].notna().any():
                    max_val = all_results[col].max()
                    # If max value looks like percent in [1, 1000], divide by 100
                    if max_val > 1.0:
                        all_results[col] = all_results[col] / 100.0

                # Clip values to [0,1]
                all_results[col] = all_results[col].clip(lower=0.0, upper=1.0)

        return all_results

    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

def render_performance_overview(results):
    """Render performance overview with key metrics"""
    st.markdown("###  Performance Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_accuracy = results['Accuracy'].max()
        best_model_acc = results.loc[results['Accuracy'].idxmax(), 'Model']
        st.metric("Best Accuracy", f"{best_accuracy:.2%}",
                 help=f"Achieved by {best_model_acc}")

    with col2:
        best_roc = results['ROC-AUC'].max()
        best_model_roc = results.loc[results['ROC-AUC'].idxmax(), 'Model']
        st.metric("Best ROC-AUC", f"{best_roc:.2%}",
                 help=f"Achieved by {best_model_roc}")

    with col3:
        best_recall = results['Recall'].max()
        best_model_recall = results.loc[results['Recall'].idxmax(), 'Model']
        st.metric("Best Recall", f"{best_recall:.2%}",
                 help=f"Achieved by {best_model_recall}")

    with col4:
        total_models = len(results)
        st.metric("Models Trained", total_models)

    st.divider()

    # Performance comparison chart
    st.markdown("###  Model Performance Comparison")

    # Prepare data for visualization
    metrics_df = results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].copy()

    # Create grouped bar chart
    fig = go.Figure()

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, metric in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            marker_color=colors[i]
        ))

    fig.update_layout(
        title='Model Performance Across All Metrics',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Model recommendations
    st.markdown("###  Model Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        best_overall = results.loc[results['ROC-AUC'].idxmax(), 'Model']
        best_overall_score = results['ROC-AUC'].max()
        st.success(f"**Best Overall Performance**\n\n{best_overall}\n\nROC-AUC: {best_overall_score:.2%}")

    with col2:
        best_recall_model = results.loc[results['Recall'].idxmax(), 'Model']
        best_recall_score = results['Recall'].max()
        st.info(f"**Best for Disease Detection**\n\n{best_recall_model}\n\nRecall: {best_recall_score:.2%}")

    with col3:
        best_f1 = results.loc[results['F1-Score'].idxmax(), 'Model']
        best_f1_score = results['F1-Score'].max()
        st.warning(f"**Best Balanced Model**\n\n{best_f1}\n\nF1-Score: {best_f1_score:.2%}")

def render_detailed_metrics(results):
    """Render detailed metrics table"""
    st.markdown("###  Detailed Performance Metrics")

    # Format results for display
    display_df = results.copy()

    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

    # Highlight best values
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Model comparison selector
    st.markdown("###  Model Deep Dive")

    selected_models = st.multiselect(
        "Select models to compare:",
        options=results['Model'].tolist(),
        default=results['Model'].tolist()[:3] if len(results) >= 3 else results['Model'].tolist()
    )

    if selected_models:
        filtered_results = results[results['Model'].isin(selected_models)]

        # Radar chart for selected models
        fig = go.Figure()

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        for model in selected_models:
            model_data = filtered_results[filtered_results['Model'] == model].iloc[0]
            values = [model_data[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='Model Performance Comparison (Radar Chart)',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

def render_visualizations(results):
    """Render interactive visualizations"""
    st.markdown("###  Interactive Visualizations")

    # Metric selector
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_metric = st.selectbox(
            "Select Metric to Visualize:",
            options=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            index=4  # Default to ROC-AUC
        )

    with col2:
        chart_type = st.radio(
            "Chart Type:",
            options=['Bar Chart', 'Line Chart'],
            horizontal=True
        )

    # Create visualization based on selection
    if chart_type == 'Bar Chart':
        fig = px.bar(
            results,
            x='Model',
            y=selected_metric,
            title=f'{selected_metric} by Model',
            color=selected_metric,
            color_continuous_scale='Blues',
            text=selected_metric
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    else:
        fig = px.line(
            results,
            x='Model',
            y=selected_metric,
            title=f'{selected_metric} by Model',
            markers=True,
            text=selected_metric
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='top center')

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Performance heatmap
    st.markdown("###  Performance Heatmap")

    # Prepare data for heatmap
    heatmap_data = results.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

    fig = px.imshow(
        heatmap_data.T,
        labels=dict(x="Model", y="Metric", color="Score"),
        x=heatmap_data.index,
        y=heatmap_data.columns,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        text_auto='.2%'
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_model_comparison(results):
    """Render side-by-side model comparison"""
    st.markdown("###  Side-by-Side Model Comparison")

    # Model selectors
    col1, col2 = st.columns(2)

    with col1:
        model1 = st.selectbox(
            "Select First Model:",
            options=results['Model'].tolist(),
            index=0
        )

    with col2:
        model2 = st.selectbox(
            "Select Second Model:",
            options=results['Model'].tolist(),
            index=1 if len(results) > 1 else 0
        )

    if model1 and model2:
        # Get model data
        model1_data = results[results['Model'] == model1].iloc[0]
        model2_data = results[results['Model'] == model2].iloc[0]

        # Create comparison
        st.divider()

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        for metric in metrics:
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                value1 = model1_data[metric]
                st.metric(f"{model1}", f"{value1:.2%}")

            with col2:
                st.markdown(f"**{metric}**")

            with col3:
                value2 = model2_data[metric]
                delta = value2 - value1
                st.metric(f"{model2}", f"{value2:.2%}",
                         delta=f"{delta:+.2%}")

        # Winner summary
        st.divider()
        st.markdown("###  Winner Summary")

        wins = {model1: 0, model2: 0}

        for metric in metrics:
            if model1_data[metric] > model2_data[metric]:
                wins[model1] += 1
            elif model2_data[metric] > model1_data[metric]:
                wins[model2] += 1

        if wins[model1] > wins[model2]:
            st.success(f"**{model1}** wins with {wins[model1]}/{len(metrics)} better metrics!")
        elif wins[model2] > wins[model1]:
            st.success(f"**{model2}** wins with {wins[model2]}/{len(metrics)} better metrics!")
        else:
            st.info(f"**Tie!** Both models perform similarly.")

def render_performance_page():
    """Render comprehensive model performance page"""
    st.markdown("##  Model Performance Dashboard")

    # Load results
    results = load_model_results()

    if results is None or results.empty:
        st.warning(" No model performance data found. Please train models first.")
        st.markdown("""
        **To train models and generate performance metrics:**

        ```bash
        # Train all models
        make train

        # Or train individually
        jupyter notebook notebooks/baseline_models.ipynb
        jupyter notebook notebooks/advanced_models.ipynb
        ```
        """)

        # Show available models from predictor
        if st.session_state.predictor:
            st.divider()
            st.markdown("###  Available Loaded Models")
            models = st.session_state.predictor.get_available_models()
            for model in models:
                st.markdown(f"-  {model}")

        return

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        " Overview",
        " Detailed Metrics",
        " Visualizations",
        " Model Comparison"
    ])

    with tab1:
        render_performance_overview(results)

    with tab2:
        render_detailed_metrics(results)

    with tab3:
        render_visualizations(results)

    with tab4:
        render_model_comparison(results)


if __name__ == "__main__":
    main()
