"""
Patient Insights - CardioFusion
Risk context, percentiles, and comparisons
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict

def calculate_percentile(risk_score: float, age_category: str) -> Dict:
    """
    Calculate risk percentile compared to population

    Args:
        risk_score: Patient's risk percentage
        age_category: Age category

    Returns:
        Dictionary with percentile information
    """
    # Simulated population distribution (replace with actual data)
    age_risk_distributions = {
        '18-24': {'mean': 5, 'std': 3},
        '25-29': {'mean': 7, 'std': 4},
        '30-34': {'mean': 10, 'std': 5},
        '35-39': {'mean': 15, 'std': 7},
        '40-44': {'mean': 22, 'std': 10},
        '45-49': {'mean': 30, 'std': 12},
        '50-54': {'mean': 40, 'std': 15},
        '55-59': {'mean': 50, 'std': 17},
        '60-64': {'mean': 60, 'std': 18},
        '65-69': {'mean': 68, 'std': 17},
        '70-74': {'mean': 75, 'std': 15},
        '75-79': {'mean': 80, 'std': 12},
        '80+': {'mean': 85, 'std': 10}
    }

    dist = age_risk_distributions.get(age_category, {'mean': 45, 'std': 20})

    # Calculate percentile (assuming normal distribution)
    from scipy import stats
    percentile = stats.norm.cdf(risk_score, dist['mean'], dist['std']) * 100

    return {
        'percentile': percentile,
        'age_mean': dist['mean'],
        'age_std': dist['std'],
        'comparison': 'higher' if risk_score > dist['mean'] else 'lower'
    }


def render_patient_insights(prediction, patient_data, predictor):
    """Render patient insights and context"""

    st.markdown("## 📊 Your Risk in Context")
    st.markdown("**Understand how your risk compares to others**")

    st.divider()

    # Risk Percentile
    st.markdown("### 📈 Risk Percentile Analysis")

    age_cat = patient_data.get('age_category', '50-54')
    risk_score = prediction['risk_percentage']

    percentile_info = calculate_percentile(risk_score, age_cat)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Your Risk",
            f"{risk_score:.1f}%",
            delta=None
        )

    with col2:
        st.metric(
            f"Average for Age {age_cat}",
            f"{percentile_info['age_mean']:.1f}%",
            delta=None
        )

    with col3:
        percentile = percentile_info['percentile']
        st.metric(
            "Your Percentile",
            f"{percentile:.0f}th",
            delta=f"{percentile_info['comparison']} than average"
        )

    # Visual percentile indicator
    st.markdown("#### Where You Stand")

    # Create percentile visualization
    percentile = percentile_info['percentile']

    if percentile < 25:
        status = "🟢 LOW RISK GROUP"
        message = "You're in the lowest 25% of risk for your age group. Excellent!"
    elif percentile < 50:
        status = "🟡 BELOW AVERAGE RISK"
        message = "Your risk is below average for your age group. Good job!"
    elif percentile < 75:
        status = "🟠 ABOVE AVERAGE RISK"
        message = "Your risk is above average for your age group. Consider interventions."
    else:
        status = "🔴 HIGH RISK GROUP"
        message = "You're in the highest 25% of risk for your age group. Take action!"

    st.info(f"""
    **{status}**

    {message}

    **What this means:** Out of 100 people your age, you have higher risk than approximately **{int(percentile)}** of them.
    """)

    # Progress bar visualization
    st.progress(percentile / 100, text=f"{percentile:.0f}th percentile")

    st.divider()

    # Feature Comparison
    st.markdown("### 📋 Your Health Metrics vs. Healthy Ranges")

    # Define healthy ranges
    healthy_ranges = {
        'BMI': {
            'healthy': (18.5, 25),
            'value': patient_data.get('bmi', 25),
            'unit': '',
            'lower_better': False
        },
        'Age': {
            'risk_threshold': 45,
            'value': get_age_from_category(patient_data.get('age_category', '50-54')),
            'unit': 'years',
            'lower_better': True
        },
        'Alcohol (per week)': {
            'healthy': (0, 7),
            'value': patient_data.get('alcohol_consumption', 0),
            'unit': 'drinks',
            'lower_better': True
        },
        'Fruit (per week)': {
            'healthy': (21, 60),
            'value': patient_data.get('fruit_consumption', 0),
            'unit': 'servings',
            'lower_better': False
        },
        'Vegetables (per week)': {
            'healthy': (21, 60),
            'value': patient_data.get('green_vegetables_consumption', 0),
            'unit': 'servings',
            'lower_better': False
        }
    }

    for metric, info in healthy_ranges.items():
        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            st.markdown(f"**{metric}**")

        with col2:
            value = info['value']
            unit = info['unit']
            st.markdown(f"{value:.1f} {unit}")

        with col3:
            if 'healthy' in info:
                low, high = info['healthy']
                if low <= value <= high:
                    st.success("✅ Healthy range")
                elif value < low:
                    st.warning(f"⚠️ Below healthy range ({low}-{high})")
                else:
                    st.error(f"❌ Above healthy range ({low}-{high})")
            else:
                threshold = info.get('risk_threshold', 0)
                if value < threshold:
                    st.success("✅ Low risk")
                else:
                    st.warning(f"⚠️ Risk increases after {threshold}")

    st.divider()

    # Model Confidence Analysis
    st.markdown("### 🎯 Prediction Confidence")

    confidence = prediction['confidence'] * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Confidence", f"{confidence:.1f}%")

        # Confidence interpretation
        if confidence >= 90:
            conf_level = "Very High"
            conf_color = "green"
            conf_message = "The model is very confident in this prediction"
        elif confidence >= 75:
            conf_level = "High"
            conf_color = "blue"
            conf_message = "The model has high confidence in this prediction"
        elif confidence >= 60:
            conf_level = "Moderate"
            conf_color = "orange"
            conf_message = "The model has moderate confidence in this prediction"
        else:
            conf_level = "Low"
            conf_color = "red"
            conf_message = "The model has lower confidence - consider additional testing"

        st.markdown(f"**Confidence Level:** :{conf_color}[{conf_level}]")
        st.caption(conf_message)

    with col2:
        # Show confidence interval (approximate)
        margin = (100 - confidence) / 2
        lower_bound = max(0, risk_score - margin)
        upper_bound = min(100, risk_score + margin)

        st.metric("Estimated Range", f"{lower_bound:.1f}% - {upper_bound:.1f}%")
        st.caption("95% confidence interval")

    st.divider()

    # Risk Factors Summary
    st.markdown("### ⚠️ Your Risk Factors")

    risk_factors = []
    protective_factors = []

    # Analyze patient data
    if patient_data.get('smoking_history') == 'Yes':
        risk_factors.append("🚭 Smoking")

    if patient_data.get('exercise') == 'No':
        risk_factors.append("🏃 No regular exercise")

    bmi = patient_data.get('bmi', 25)
    if bmi >= 30:
        risk_factors.append(f"⚖️ Obesity (BMI {bmi:.1f})")
    elif bmi >= 25:
        risk_factors.append(f"⚖️ Overweight (BMI {bmi:.1f})")

    if patient_data.get('diabetes') != 'No':
        risk_factors.append("💉 Diabetes")

    if patient_data.get('general_health') in ['Poor', 'Fair']:
        risk_factors.append(f"❤️ {patient_data.get('general_health')} health status")

    # Protective factors
    if patient_data.get('smoking_history') == 'No':
        protective_factors.append("✅ Non-smoker")

    if patient_data.get('exercise') == 'Yes':
        protective_factors.append("✅ Regular exercise")

    if 18.5 <= bmi < 25:
        protective_factors.append("✅ Healthy weight")

    if patient_data.get('fruit_consumption', 0) >= 21:
        protective_factors.append("✅ Good fruit intake")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Risk Factors")
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("No major risk factors identified!")

    with col2:
        st.markdown("#### Protective Factors")
        if protective_factors:
            for factor in protective_factors:
                st.markdown(f"- {factor}")
        else:
            st.warning("Few protective factors - room for improvement")

    st.divider()

    # Timeline recommendations
    st.markdown("### 📅 Recommended Timeline")

    if risk_score >= 70:
        st.error("""
        **HIGH RISK - Immediate Action Required**
        - 🏥 Schedule doctor visit: This week
        - 🔄 Reassess risk: Every 1-2 months
        - 💊 Consider medication: Consult physician
        """)
    elif risk_score >= 50:
        st.warning("""
        **MODERATE-HIGH RISK - Action Recommended**
        - 🏥 Schedule doctor visit: Within 2 weeks
        - 🔄 Reassess risk: Every 3 months
        - 🏃 Start lifestyle changes: Immediately
        """)
    elif risk_score >= 30:
        st.info("""
        **MODERATE RISK - Monitor and Improve**
        - 🏥 Schedule doctor visit: Within 1 month
        - 🔄 Reassess risk: Every 6 months
        - 🥗 Focus on prevention: Start today
        """)
    else:
        st.success("""
        **LOW RISK - Maintain Healthy Habits**
        - 🏥 Regular checkups: Annually
        - 🔄 Reassess risk: Yearly
        - ✅ Continue healthy lifestyle
        """)


def get_age_from_category(age_category: str) -> float:
    """Convert age category to numeric value"""
    age_mapping = {
        '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
        '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
        '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77, '80+': 82
    }
    return age_mapping.get(age_category, 52)
