"""
What-If Analysis - CardioFusion
Interactive risk reduction calculator
"""

import streamlit as st
import pandas as pd
from typing import Dict

def calculate_risk_change(predictor, base_data: pd.DataFrame, patient_data: Dict, changes: Dict) -> Dict:
    """
    Calculate how risk changes with lifestyle modifications

    Args:
        predictor: ModelPredictor instance
        base_data: Original preprocessed patient data
        patient_data: Original patient input
        changes: Dictionary of changes to apply

    Returns:
        Dictionary with risk changes
    """
    from src.utils.data_validator import DataValidator

    # Create modified patient data
    modified_data = patient_data.copy()
    modified_data.update(changes)

    # Preprocess modified data
    validator = DataValidator()
    modified_df = validator.preprocess_for_model(modified_data)

    # Get predictions
    base_prediction = predictor.predict(base_data)
    modified_prediction = predictor.predict(modified_df)

    risk_reduction = base_prediction['risk_percentage'] - modified_prediction['risk_percentage']

    return {
        'original_risk': base_prediction['risk_percentage'],
        'new_risk': modified_prediction['risk_percentage'],
        'risk_reduction': risk_reduction,
        'risk_reduction_pct': (risk_reduction / base_prediction['risk_percentage']) * 100
    }


def render_what_if_analysis(predictor, input_data, patient_data, base_prediction):
    """Render interactive What-If analysis"""

    st.markdown("## 🎯 What-If Analysis: Risk Reduction Calculator")
    st.markdown("**Explore how lifestyle changes could reduce your cardiovascular risk**")

    st.divider()

    # Current risk display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Risk", f"{base_prediction['risk_percentage']:.1f}%")
    with col2:
        st.metric("Classification", base_prediction['prediction_label'])
    with col3:
        risk_cat, emoji, _ = predictor.get_risk_category(base_prediction['risk_percentage'])
        st.metric("Risk Level", f"{emoji} {risk_cat}")

    st.divider()

    # Interactive scenarios
    st.markdown("### 📊 Scenario Analysis")
    st.markdown("*Adjust the sliders to see how changes affect your risk*")

    # Create tabs for different intervention categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚭 Lifestyle Changes",
        "🏃 Physical Activity",
        "🍎 Nutrition",
        "🔄 Combined Interventions"
    ])

    with tab1:
        st.markdown("#### Lifestyle Modifications")

        col1, col2 = st.columns(2)

        with col1:
            # Smoking cessation
            if patient_data.get('smoking_history') == 'Yes':
                if st.checkbox("✅ Quit Smoking", key="quit_smoking"):
                    changes = {'smoking_history': 'No'}
                    result = calculate_risk_change(predictor, input_data, patient_data, changes)

                    st.success(f"""
                    **Impact of Quitting Smoking:**
                    - Risk reduction: **{result['risk_reduction']:.1f}%**
                    - New risk: **{result['new_risk']:.1f}%**
                    - Improvement: **{result['risk_reduction_pct']:.1f}%**
                    """)

                    st.progress(result['risk_reduction'] / 100, text=f"Risk reduced by {result['risk_reduction']:.1f}%")
            else:
                st.info("✅ You don't smoke - excellent!")

        with col2:
            # Weight loss
            current_bmi = patient_data.get('bmi', 25)
            if current_bmi > 25:
                target_bmi = st.slider(
                    "Target BMI",
                    min_value=18.5,
                    max_value=current_bmi,
                    value=25.0,
                    step=0.5,
                    key="target_bmi"
                )

                if target_bmi < current_bmi:
                    # Calculate new weight
                    height_m = patient_data.get('height_cm', 170) / 100
                    new_weight = target_bmi * (height_m ** 2)

                    changes = {
                        'bmi': target_bmi,
                        'weight_kg': new_weight
                    }
                    result = calculate_risk_change(predictor, input_data, patient_data, changes)

                    st.success(f"""
                    **Impact of Weight Loss:**
                    - BMI: {current_bmi:.1f} → {target_bmi:.1f}
                    - Risk reduction: **{result['risk_reduction']:.1f}%**
                    - New risk: **{result['new_risk']:.1f}%**
                    """)

    with tab2:
        st.markdown("#### Physical Activity Impact")

        if patient_data.get('exercise') == 'No':
            exercise_level = st.select_slider(
                "Exercise Commitment",
                options=["No Exercise", "Light (2x/week)", "Moderate (3-4x/week)", "Regular (5+x/week)"],
                key="exercise_level"
            )

            if exercise_level != "No Exercise":
                changes = {'exercise': 'Yes'}
                result = calculate_risk_change(predictor, input_data, patient_data, changes)

                st.success(f"""
                **Impact of {exercise_level}:**
                - Risk reduction: **{result['risk_reduction']:.1f}%**
                - New risk: **{result['new_risk']:.1f}%**
                """)

                # Show additional benefits
                st.info(f"""
                **Additional Benefits:**
                - Improved cardiovascular fitness
                - Better blood pressure control
                - Enhanced mood and mental health
                - Reduced risk of diabetes
                """)
        else:
            st.success("✅ You already exercise - keep it up!")

    with tab3:
        st.markdown("#### Nutritional Improvements")

        col1, col2 = st.columns(2)

        with col1:
            current_fruits = patient_data.get('fruit_consumption', 0)
            new_fruits = st.slider(
                "Fruit servings per week",
                min_value=0,
                max_value=60,
                value=min(current_fruits + 20, 60),
                step=5,
                key="fruits"
            )

            current_veggies = patient_data.get('green_vegetables_consumption', 0)
            new_veggies = st.slider(
                "Vegetable servings per week",
                min_value=0,
                max_value=60,
                value=min(current_veggies + 20, 60),
                step=5,
                key="veggies"
            )

        with col2:
            current_fried = patient_data.get('fried_potato_consumption', 0)
            if current_fried > 0:
                new_fried = st.slider(
                    "Fried food servings per week",
                    min_value=0,
                    max_value=int(current_fried),
                    value=max(0, int(current_fried - 5)),
                    step=1,
                    key="fried"
                )
            else:
                st.success("✅ Already at 0 fried foods - excellent!")
                new_fried = 0

            current_alcohol = patient_data.get('alcohol_consumption', 0)
            if current_alcohol > 0:
                new_alcohol = st.slider(
                    "Alcohol drinks per week",
                    min_value=0,
                    max_value=int(current_alcohol),
                    value=max(0, int(current_alcohol - 5)),
                    step=1,
                    key="alcohol"
                )
            else:
                st.success("✅ Already at 0 alcohol - excellent!")
                new_alcohol = 0

        if st.button("Calculate Nutrition Impact", key="calc_nutrition"):
            changes = {
                'fruit_consumption': new_fruits,
                'green_vegetables_consumption': new_veggies,
                'fried_potato_consumption': new_fried,
                'alcohol_consumption': new_alcohol
            }
            result = calculate_risk_change(predictor, input_data, patient_data, changes)

            st.success(f"""
            **Impact of Dietary Changes:**
            - Risk reduction: **{result['risk_reduction']:.1f}%**
            - New risk: **{result['new_risk']:.1f}%**
            """)

    with tab4:
        st.markdown("#### 🔥 Maximum Impact Scenario")
        st.markdown("*What if you made ALL recommended changes?*")

        # Build comprehensive changes
        all_changes = {}

        if patient_data.get('smoking_history') == 'Yes':
            all_changes['smoking_history'] = 'No'

        if patient_data.get('exercise') == 'No':
            all_changes['exercise'] = 'Yes'

        current_bmi = patient_data.get('bmi', 25)
        if current_bmi > 25:
            height_m = patient_data.get('height_cm', 170) / 100
            all_changes['bmi'] = 24.0
            all_changes['weight_kg'] = 24.0 * (height_m ** 2)

        all_changes['fruit_consumption'] = 40
        all_changes['green_vegetables_consumption'] = 40
        all_changes['fried_potato_consumption'] = 0
        all_changes['alcohol_consumption'] = max(0, patient_data.get('alcohol_consumption', 0) - 10)

        if st.button("🚀 Calculate Maximum Potential", type="primary"):
            result = calculate_risk_change(predictor, input_data, patient_data, all_changes)

            st.balloons()

            st.success(f"""
            ### 🎉 Maximum Risk Reduction Potential

            **By making ALL recommended changes:**

            📊 **Current Risk:** {result['original_risk']:.1f}%

            📉 **Potential New Risk:** {result['new_risk']:.1f}%

            ⬇️ **Total Reduction:** {result['risk_reduction']:.1f}% ({result['risk_reduction_pct']:.0f}% improvement)
            """)

            # Visualize the change
            st.markdown("#### Risk Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Before", f"{result['original_risk']:.1f}%", delta=None)
            with col2:
                st.metric("After", f"{result['new_risk']:.1f}%",
                         delta=f"-{result['risk_reduction']:.1f}%",
                         delta_color="inverse")

            # Recommendations
            st.markdown("#### 📋 Action Plan")
            st.markdown("""
            To achieve this risk reduction:

            1. **This Week:**
               - Schedule doctor appointment
               - Join smoking cessation program (if applicable)
               - Plan weekly exercise schedule

            2. **This Month:**
               - Start walking 30 minutes daily
               - Switch to Mediterranean diet
               - Reduce alcohol and fried foods

            3. **3 Months:**
               - Reassess your risk
               - Track BMI progress
               - Maintain new habits
            """)

    st.divider()

    # Summary card
    st.markdown("### 💡 Key Takeaways")
    st.info("""
    **Most Impactful Changes** (in order):
    1. 🚭 **Quit Smoking** - Highest impact on risk reduction
    2. 🏃 **Regular Exercise** - Significant protective effect
    3. ⚖️ **Healthy Weight** - Reduces multiple risk factors
    4. 🍎 **Nutritious Diet** - Cumulative long-term benefits

    **Remember:** Small, consistent changes add up! Start with one change at a time.
    """)
