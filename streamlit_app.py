"""
TargetCall — Bank Marketing Term Deposit Predictor
Machine Learning for Developers (CAI2C08)

Streamlit web app that predicts whether a bank customer is likely to
subscribe to a term deposit, so the marketing team can prioritise its calls.

The app loads the trained Random Forest model (bank_marketing_rf_model.pkl)
produced by MLDP_Program_Codes.ipynb, rebuilds the same engineered features
from the user's inputs, aligns the columns to the model, and predicts.
"""

import joblib
import streamlit as st
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="TargetCall — Term Deposit Predictor",
    page_icon="📞",
    layout="centered",
)

# ------------------------------------------------------------------
# Load the trained model
# ------------------------------------------------------------------
# The model file must be in the same folder as this app.
# We wrap this in a try/except so the app shows a friendly message
# instead of crashing if the file is missing.
try:
    model = joblib.load("bank_marketing_rf_model.pkl")
except FileNotFoundError:
    st.error(
        "Model file 'bank_marketing_rf_model.pkl' not found. "
        "Please run MLDP_Program_Codes.ipynb first to train and save the model."
    )
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📞 TargetCall")
st.subheader("Term Deposit Subscription Predictor for Bank Marketing")
st.write(
    "Enter a customer's details below. The tool predicts how likely they are "
    "to subscribe to a term deposit, helping the marketing team decide **who to call first**."
)
st.divider()

# ------------------------------------------------------------------
# Input options — these MUST match the categories the model was trained on
# ------------------------------------------------------------------
jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown']
maritals = ['divorced', 'married', 'single']
educations = ['primary', 'secondary', 'tertiary', 'unknown']
contacts = ['cellular', 'telephone', 'unknown']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcomes = ['failure', 'other', 'success', 'unknown']

# ------------------------------------------------------------------
# User inputs — organised into sections with friendly, non-technical labels
# ------------------------------------------------------------------
st.markdown("### 👤 Customer profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Customer's age", min_value=18, max_value=95, value=40)
    job = st.selectbox("Occupation", jobs)
    marital = st.selectbox("Marital status", maritals)
    education = st.selectbox("Education level", educations)

with col2:
    balance = st.number_input(
        "Average yearly account balance (€)",
        min_value=-10000, max_value=110000, value=1000, step=100,
        help="The customer's average yearly balance in euros. Can be negative (overdraft)."
    )
    housing = st.radio("Has a housing loan?", ['no', 'yes'], horizontal=True)
    loan = st.radio("Has a personal loan?", ['no', 'yes'], horizontal=True)
    default = st.radio("Has credit in default?", ['no', 'yes'], horizontal=True)

st.markdown("### 📇 Campaign contact details")

col3, col4 = st.columns(2)

with col3:
    contact = st.selectbox("How was the customer contacted?", contacts)
    month = st.selectbox("Month of last contact", months)
    day = st.slider("Day of month of last contact", min_value=1, max_value=31, value=15)

with col4:
    campaign = st.number_input(
        "Number of contacts during this campaign",
        min_value=1, max_value=60, value=2, step=1,
        help="How many times this customer has been contacted in the current campaign."
    )
    previous = st.number_input(
        "Number of contacts before this campaign",
        min_value=0, max_value=60, value=0, step=1,
        help="Contacts in previous campaigns. Set to 0 if never contacted before."
    )
    poutcome = st.selectbox("Outcome of previous campaign", poutcomes)

st.divider()

# ------------------------------------------------------------------
# Predict button
# ------------------------------------------------------------------
if st.button("🔮 Predict subscription likelihood", type="primary", use_container_width=True):

    # ----------------------------------------------------------
    # INPUT VALIDATION (Validate inputs before predicting so bad data doesn't reach the model.)
    # ----------------------------------------------------------
    errors = []

    # Logical check: if previous campaign outcome is known, there must be previous contacts
    if poutcome in ['success', 'failure', 'other'] and previous == 0:
        errors.append(
            "You selected a previous campaign outcome, but 'contacts before this "
            "campaign' is 0. Please set the number of previous contacts to at least 1, "
            "or set the previous outcome to 'unknown'."
        )

    # Sanity check on balance
    if balance < -8019 or balance > 102127:
        errors.append(
            "Account balance is outside the range seen in the training data "
            "(-8,019 to 102,127). The prediction may be unreliable."
        )

    if errors:
        for e in errors:
            st.warning(e)

    # Only predict if there are no blocking errors (balance warning is non-blocking)
    blocking = [e for e in errors if "unreliable" not in e]
    if not blocking:
        # pdays: -1 means "never contacted before". When the customer WAS contacted
        # before, we don't ask the user for the exact gap, so we substitute the median
        # pdays of previously-contacted customers from the training data (194 days).
        # This is a representative value the model actually saw, rather than an
        # arbitrary placeholder.
        pdays = -1 if previous == 0 else 194    

        # Engineered feature 1: previously_contacted
        previously_contacted = 0 if pdays == -1 else 1

        # Engineered feature 2: previous_success
        previous_success = 1 if poutcome == 'success' else 0

        # Engineered feature 3: age_band (same bins as the notebook)
        if age <= 30:
            age_band = "<=30"
        elif age <= 40:
            age_band = "31-40"
        elif age <= 50:
            age_band = "41-50"
        elif age <= 60:
            age_band = "51-60"
        else:
            age_band = "60+"

        # ------------------------------------------------------
        # Build a one-row DataFrame with the raw inputs
        # ------------------------------------------------------
        df_input = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'day': [day],
            'month': [month],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'previously_contacted': [previously_contacted],
            'previous_success': [previous_success],
            'age_band': [age_band],
        })

        # ------------------------------------------------------
        # One-hot encode (same as notebook: drop_first=True)
        # ------------------------------------------------------
        df_input = pd.get_dummies(df_input, drop_first=True)

        # ------------------------------------------------------
        # Align columns to the model's expected features
        # (This is the key pattern from the HDB class example.)
        # Any column the model expects but the user's input lacks
        # gets filled with 0.
        # ------------------------------------------------------
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        # ------------------------------------------------------
        # Predict
        # ------------------------------------------------------
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]  # probability of "yes"

        # ------------------------------------------------------
        # Display the result — business-friendly output
        # ------------------------------------------------------
        st.divider()
        st.markdown("### 📊 Prediction result")

        # Show the probability as a metric and progress bar
        st.metric(label="Likelihood of subscribing", value=f"{probability:.0%}")
        st.progress(float(probability))

        # Colour-coded recommendation
        if prediction == 1:
            st.success(
                f"✅ **RECOMMEND CALLING** — this customer has a **{probability:.0%}** "
                f"chance of subscribing, which is above the calling threshold. "
                f"Prioritise this lead."
            )
        else:
            st.error(
                f"⛔ **RECOMMEND SKIPPING** — this customer has only a **{probability:.0%}** "
                f"chance of subscribing. Focus call-centre time on higher-probability leads."
            )

        # Context for the user
        st.caption(
            "Note: the average subscription rate across all customers is about 12%. "
            "This tool helps the bank focus on customers well above that baseline."
        )

# ------------------------------------------------------------------
# Sidebar — context about the tool
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About TargetCall")
    st.write(
        "This tool uses a **Random Forest** model trained on 45,211 real bank "
        "marketing records (UCI Bank Marketing dataset) to predict term deposit "
        "subscriptions."
    )
    st.write(
        "**Business value:** only ~12% of cold calls succeed. By calling the "
        "customers this tool flags as high-probability, the bank cuts wasted "
        "calls and lifts campaign ROI."
    )
