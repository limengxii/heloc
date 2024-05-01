import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

####theme
# Define a function to set up theme before any other Streamlit calls
def setup_theme():
    # Create or modify the theme
    st.set_page_config(
        page_title="HELOC Estimator",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🏘️"
    )

# Call the function to set up the theme
setup_theme()

# Placeholder for model loading
model = pickle.load(open('modelXGB.pkl', 'rb'))


# Human-readable to technical feature name mapping
feature_mapping = {
    'Consolidated Risk Score': 'ExternalRiskEstimate',
    'Months Since First Credit Line': 'MSinceOldestTradeOpen',
    'Months Since Latest Credit Line': 'MSinceMostRecentTradeOpen',
    'Average Credit History Length (Months)': 'AverageMInFile',
    'Number of Positive Credit Lines': 'NumSatisfactoryTrades',
    'Number of 60+ Days Late Payments or Derogatory Public Records': 'NumTrades60Ever2DerogPubRec',
    'Number of 90+ Days Late Payments or Derogatory Public Records': 'NumTrades90Ever2DerogPubRec',
    'Percentage of Never Delinquent Trades': 'PercentTradesNeverDelq',
    'Months Since Last Delinquency': 'MSinceMostRecentDelq',
    'Maximum Delinquency on Public Records in Last 12 Months': 'MaxDelq2PublicRecLast12M',
    'Worst Delinquency on Record': 'MaxDelqEver',
    'Total Number of Credit Lines': 'NumTotalTrades',
    'Number of Credit Lines Opened in Last 12 Months': 'NumTradesOpeninLast12M',
    'Percentage of Installment Credit Lines': 'PercentInstallTrades',
    'Months Since Last Credit Inquiry (Excluding Last 7 Days)': 'MSinceMostRecentInqexcl7days',
    'Number of Credit Inquiries in Last 6 Months': 'NumInqLast6M',
    'Number of Credit Inquiries in Last 6 Months (Excluding Last 7 Days)': 'NumInqLast6Mexcl7days',
    'Ratio of Revolving Credit Used to Credit Limit': 'NetFractionRevolvingBurden',
    'Ratio of Installment Credit Used to Original Loan Amount': 'NetFractionInstallBurden',
    'Number of Revolving Trades with Balance': 'NumRevolvingTradesWBalance',
    'Number of Installment Trades with Balance': 'NumInstallTradesWBalance',
    'Number of Bank/National Trades with High Credit Utilization': 'NumBank2NatlTradesWHighUtilization',
    'Percentage of Credit Lines with Balance': 'PercentTradesWBalance'
}

#special conditions
special_conditions = {
    'MSinceMostRecentDelq': {
        '-7': 'Condition not Met (e.g. No Inquiries, No Delinquencies)',
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'MSinceMostRecentInqexcl7days': {
        '-7': 'Condition not Met (e.g. No Inquiries, No Delinquencies)',
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    # Ensure each feature's conditions are defined as dictionaries
    'MSinceOldestTradeOpen': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'NetFractionRevolvingBurden': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'NetFractionInstallBurden': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'NumRevolvingTradesWBalance': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'NumInstallTradesWBalance': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'NumBank2NatlTradesWHighUtilization': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
    'PercentTradesWBalance': {
        '-8': 'No Usable/Valid Trades or Inquiries'
    },
}


# Data setup with technical names for data processing and mappings for display
ranges = pd.DataFrame({
    'Feature': list(feature_mapping.values()),
    'Min Value': [-9, -8, 0, 4, 0, 0, 0, 0, -8, 2, 2, 0, 0, 0, 0, -8, 0, 0, -8, -8, -8, -8, -8],
    'Max Value': [94, 803, 383, 383, 79, 19, 19, 100, 83, 9, 8, 104, 19, 100, 24, 66, 66, 232, 471, 32, 23, 18, 100],
    'Rounded Max Value': [100, 810, 390, 390, 80, 20, 20, 100, 90, 10, 10, 110, 20, 100, 30, 70, 70, 240, 480, 40, 30, 20, 100]
})

ranges['Human Readable Name'] = ranges['Feature'].map({v: k for k, v in feature_mapping.items()})  # Map technical names to human-readable
ranges.set_index('Feature', inplace=True)

st.title("Home Equity Line of Credit Estimator")
st.write("Please enter the applicant's details to evaluate the risk of the application.")

# Prepare for model feature expectation, as order is important
model_features = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 
                  'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 
                  'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 
                  'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 
                  'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 
                  'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 
                  'PercentTradesWBalance', 'MSinceMostRecentDelq=-7', 'MSinceMostRecentInqexcl7days=-7', 
                  'MSinceOldestTradeOpen=-8', 'MSinceMostRecentDelq=-8', 'MSinceMostRecentInqexcl7days=-8', 
                  'NetFractionRevolvingBurden=-8', 'NetFractionInstallBurden=-8', 'NumRevolvingTradesWBalance=-8', 
                  'NumInstallTradesWBalance=-8', 'NumBank2NatlTradesWHighUtilization=-8', 'PercentTradesWBalance=-8']

def create_risk_scale(score):
    cmap = LinearSegmentedColormap.from_list('risk_scale', ['green', 'red'])
    normalized_score = score * 255
    gradient = np.linspace(1, 0, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    ax.text(0, -1.5, 'Bad', verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, color='black')
    ax.text(1, -1.5, 'Good', verticalalignment='center', horizontalalignment='right', transform=ax.transAxes, color='black')
    ax.axvline(x=normalized_score, color='yellow', linewidth=4)
    return fig

with st.sidebar:
    with st.form("input_form"):
        input_data = {feature: 0 for feature in model_features}  # Initialize all features to zero

        for feature_name, row in ranges.iterrows():
            human_readable_name = row['Human Readable Name']
            min_val = int(row['Min Value'])
            max_val = int(row['Rounded Max Value'])
            default_val = int((min_val + max_val) / 2)

            slider_key = f"slider_{feature_name}"
            num_key = f"num_{feature_name}"

            if feature_name in special_conditions:
                for code, label in special_conditions[feature_name].items():
                    checkbox_key = f"special_{feature_name}_{code}"
                    checked = st.checkbox(label, key=checkbox_key)
                    input_data[f"{feature_name}={code}"] = 1 if checked else 0

            slider_val = st.slider(human_readable_name, min_value=min_val, max_value=max_val, 
                                   value=st.session_state.get(slider_key, default_val), key=slider_key)
            number_val = st.number_input("", min_value=min_val, max_value=max_val, 
                                         value=st.session_state.get(num_key, slider_val), key=num_key)

            input_data[feature_name] = number_val

        submit_button = st.form_submit_button("Estimate")

if submit_button:
    input_data_df = pd.DataFrame([input_data])
    prediction = model.predict(input_data_df)
    st.write("Prediction:", "Good" if prediction[0] == 1 else "Bad")

    user_score = prediction[0]  # Assuming prediction is a number between 0 and 1
    st.pyplot(create_risk_scale(user_score))



