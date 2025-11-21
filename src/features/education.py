"""
Education Hub - CardioFusion
Educational content about cardiovascular health
"""

import streamlit as st

def render_education_hub():
    """Render education and information hub"""

    st.markdown("## 📚 Education Hub")
    st.markdown("**Learn about cardiovascular disease and risk factors**")

    st.divider()

    # Tabs for different topics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "❤️ About CVD",
        "⚠️ Risk Factors",
        "🛡️ Prevention",
        "❓ FAQ",
        "📖 Guidelines"
    ])

    with tab1:
        st.markdown("### What is Cardiovascular Disease?")

        st.markdown("""
        **Cardiovascular Disease (CVD)** refers to a group of disorders affecting the heart and blood vessels.

        #### Types of CVD:

        1. **Coronary Heart Disease (CHD)**
           - Affects blood vessels supplying the heart muscle
           - Most common type of heart disease
           - Can lead to heart attacks

        2. **Cerebrovascular Disease**
           - Affects blood vessels supplying the brain
           - Can cause strokes

        3. **Peripheral Arterial Disease**
           - Affects blood vessels in the limbs
           - Reduces blood flow to arms and legs

        4. **Heart Failure**
           - Heart doesn't pump blood effectively
           - Can result from other forms of CVD
        """)

        st.divider()

        st.markdown("#### 📊 CVD Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Global Deaths/Year", "17.9 million")
            st.caption("CVD is #1 cause of death")

        with col2:
            st.metric("% of All Deaths", "31%")
            st.caption("Nearly 1 in 3 deaths")

        with col3:
            st.metric("Preventable", "~80%")
            st.caption("Through lifestyle changes")

    with tab2:
        st.markdown("### ⚠️ Major Risk Factors")

        st.markdown("""
        #### Modifiable Risk Factors (You can change these!)

        **1. 🚭 Smoking**
        - Increases risk by 2-4x
        - Damages blood vessel walls
        - **Action:** Quit smoking - benefits start immediately!

        **2. 🏃 Physical Inactivity**
        - Sedentary lifestyle doubles risk
        - Weakens cardiovascular system
        - **Action:** 150 minutes moderate exercise/week

        **3. ⚖️ Obesity & Overweight**
        - BMI ≥30 significantly increases risk
        - Strains heart and blood vessels
        - **Action:** Achieve healthy weight (BMI 18.5-25)

        **4. 🍔 Poor Diet**
        - High saturated fat, salt, and sugar
        - Leads to other risk factors
        - **Action:** Mediterranean diet recommended

        **5. 🍺 Excessive Alcohol**
        - >14 drinks/week increases risk
        - Damages heart muscle
        - **Action:** Limit to 0-7 drinks/week

        **6. 💊 High Blood Pressure**
        - "Silent killer" - often no symptoms
        - Damages arteries over time
        - **Action:** Monitor and manage (<130/80)

        **7. 💉 Diabetes**
        - Doubles cardiovascular risk
        - Damages blood vessels
        - **Action:** Control blood sugar levels

        **8. 🧈 High Cholesterol**
        - LDL "bad" cholesterol builds up in arteries
        - Blocks blood flow
        - **Action:** Diet, exercise, medication if needed
        """)

        st.divider()

        st.markdown("""
        #### Non-Modifiable Risk Factors (Awareness is key!)

        **1. 🎂 Age**
        - Risk increases with age
        - Men: 45+, Women: 55+

        **2. 👨👩 Sex**
        - Men at higher risk until ~65
        - Women's risk increases after menopause

        **3. 🧬 Family History**
        - Genetic predisposition
        - If parent had early CVD (<55 for men, <65 for women)

        **4. 🏥 Previous CVD Events**
        - Prior heart attack or stroke
        - Existing heart conditions
        """)

    with tab3:
        st.markdown("### 🛡️ Prevention Strategies")

        st.success("""
        **Good News:** 80% of cardiovascular disease is preventable through lifestyle changes!
        """)

        st.markdown("#### The 7 Pillars of Heart Health")

        pillars = {
            "1. 🚭 Don't Smoke": {
                "why": "Smoking damages blood vessels and accelerates atherosclerosis",
                "how": """
                - Set a quit date
                - Use nicotine replacement therapy
                - Join support groups
                - Consult your doctor for medications
                - Benefits start within 20 minutes of quitting!
                """
            },
            "2. 🏃 Stay Active": {
                "why": "Exercise strengthens heart and improves circulation",
                "how": """
                - **Goal:** 150 minutes/week moderate activity
                - **Examples:** Brisk walking, swimming, cycling
                - Start slow and build up gradually
                - Find activities you enjoy
                - Track your progress
                """
            },
            "3. 🥗 Eat Well": {
                "why": "Nutrition directly impacts all cardiovascular risk factors",
                "how": """
                **Mediterranean Diet (Recommended):**
                - ✅ Fruits, vegetables, whole grains
                - ✅ Fish, nuts, olive oil
                - ✅ Moderate red wine (optional)
                - ❌ Reduce: Red meat, processed foods
                - ❌ Limit: Salt, sugar, saturated fats
                """
            },
            "4. ⚖️ Maintain Healthy Weight": {
                "why": "Excess weight strains heart and increases other risk factors",
                "how": """
                - **Target:** BMI 18.5-25
                - Combine diet and exercise
                - Lose weight gradually (1-2 lbs/week)
                - Focus on sustainable changes
                - Track progress weekly
                """
            },
            "5. 🩺 Control Blood Pressure": {
                "why": "High BP damages arteries and forces heart to work harder",
                "how": """
                - **Target:** <130/80 mmHg
                - Monitor at home regularly
                - Reduce sodium (<2,300mg/day)
                - DASH diet recommended
                - Medication if needed
                """
            },
            "6. 🧪 Manage Cholesterol": {
                "why": "LDL cholesterol clogs arteries",
                "how": """
                - **Target:** LDL <100 mg/dL
                - Reduce saturated fats
                - Increase fiber intake
                - Consider plant sterols
                - Statins if high risk
                """
            },
            "7. 💉 Control Blood Sugar": {
                "why": "Diabetes accelerates cardiovascular disease",
                "how": """
                - **Target:** HbA1c <7% (if diabetic)
                - Monitor glucose levels
                - Carbohydrate counting
                - Regular meals, avoid skipping
                - Medication as prescribed
                """
            }
        }

        for pillar, info in pillars.items():
            with st.expander(pillar):
                st.markdown(f"**Why it matters:**\n{info['why']}")
                st.markdown(f"**How to do it:**{info['how']}")

    with tab4:
        st.markdown("### ❓ Frequently Asked Questions")

        faqs = {
            "How accurate is this risk calculator?": """
            CardioFusion uses machine learning models trained on 308,000+ patient records,
            achieving 87-95% accuracy. However, this is a screening tool, not a diagnostic test.
            Always consult a healthcare provider for definitive diagnosis.
            """,

            "Can I trust AI for health decisions?": """
            AI is a powerful tool for risk assessment but should NEVER replace professional
            medical advice. Use this tool to:
            - Understand your risk factors
            - Track your progress
            - Have informed discussions with your doctor

            Do NOT use it to:
            - Self-diagnose
            - Make treatment decisions
            - Ignore professional medical advice
            """,

            "What should I do if my risk is high?": """
            1. **Don't panic** - Risk assessment is about prevention, not diagnosis
            2. **See your doctor** - Schedule an appointment for comprehensive evaluation
            3. **Start small** - Pick ONE lifestyle change to start with
            4. **Track progress** - Reassess in 3 months to see improvements
            5. **Stay consistent** - Lifestyle changes take time to show benefits
            """,

            "How often should I reassess my risk?": """
            **High Risk (>70%):** Every 1-2 months
            **Moderate Risk (30-70%):** Every 3-6 months
            **Low Risk (<30%):** Annually

            Also reassess whenever you make significant lifestyle changes.
            """,

            "Can young people get cardiovascular disease?": """
            Yes, though it's less common. Risk factors like:
            - Family history
            - Obesity
            - Smoking
            - Diabetes

            Can cause CVD even in younger adults. Prevention should start early!
            """,

            "Is cardiovascular disease reversible?": """
            While existing damage may be permanent, you can:
            - **Stop progression** through lifestyle changes
            - **Reduce risk** of future events significantly
            - **Improve symptoms** with proper management
            - **Reverse some damage** (e.g., arterial plaque can reduce)

            The earlier you start, the better the outcomes!
            """
        }

        for question, answer in faqs.items():
            with st.expander(question):
                st.markdown(answer)

    with tab5:
        st.markdown("### 📖 Clinical Guidelines")

        st.info("""
        This tool aligns with recommendations from leading cardiovascular health organizations.
        """)

        st.markdown("#### 🏥 American Heart Association (AHA) Guidelines")

        st.markdown("""
        **Life's Essential 8™** (Updated 2022)

        1. **Diet** - Mediterranean or DASH diet
        2. **Physical Activity** - 150 min/week moderate or 75 min/week vigorous
        3. **Nicotine Exposure** - Avoid all tobacco and e-cigarettes
        4. **Sleep Duration** - 7-9 hours per night for adults
        5. **Body Mass Index** - Target 18.5-24.9
        6. **Blood Lipids** - LDL <100 mg/dL
        7. **Blood Glucose** - Fasting <100 mg/dL
        8. **Blood Pressure** - <120/80 mmHg (ideal)

        [Learn more at heart.org](https://www.heart.org/)
        """)

        st.markdown("#### 🌍 World Health Organization (WHO)")

        st.markdown("""
        **WHO Cardiovascular Disease Prevention:**

        - **Reduce salt** - <5g (1 teaspoon) per day
        - **Reduce saturated fat** - <10% of total energy intake
        - **Eliminate trans fats** - 0% of total energy intake
        - **Physical activity** - At least 150-300 min/week
        - **No tobacco** - In any form
        - **Moderate alcohol** - If any

        [WHO CVD Fact Sheet](https://www.who.int/health-topics/cardiovascular-diseases)
        """)

        st.markdown("#### 🔬 Evidence-Based Interventions")

        st.markdown("""
        **Proven Risk Reduction:**

        | Intervention | Risk Reduction |
        |--------------|----------------|
        | Smoking cessation | -30 to -50% |
        | Blood pressure control | -20 to -30% |
        | Cholesterol management | -25 to -30% |
        | Regular exercise | -20 to -35% |
        | Healthy diet | -15 to -30% |
        | Weight loss (if overweight) | -10 to -20% |
        | Diabetes control | -20 to -25% |

        **Combined effects can reduce risk by 70-80%!**
        """)

    st.divider()

    # Additional resources
    st.markdown("### 🔗 Additional Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📚 Learn More:**
        - [American Heart Association](https://www.heart.org/)
        - [CDC Heart Disease](https://www.cdc.gov/heartdisease/)
        - [WHO Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)
        """)

    with col2:
        st.markdown("""
        **🏥 Find Care:**
        - [Find a Cardiologist](https://www.heart.org/en/find-a-doctor)
        - [Cardiac Rehabilitation](https://www.heart.org/en/health-topics/cardiac-rehab)
        - [Support Groups](https://www.heart.org/en/get-involved/support-network)
        """)

    with col3:
        st.markdown("""
        **🍎 Healthy Living:**
        - [Heart-Healthy Recipes](https://www.heart.org/en/healthy-living/healthy-eating)
        - [Exercise Plans](https://www.heart.org/en/healthy-living/fitness)
        - [Quit Smoking](https://www.heart.org/en/healthy-living/healthy-lifestyle/quit-smoking-tobacco)
        """)
