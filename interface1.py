import streamlit as st
import pandas as pd
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Home Equity Line of Credit Estimator")
st.write("Please enter the applicant's details on the sidebar to evaluate the risk of the application.")

# Placeholder for model loading
# model = pickle.load(open('your_model.pkl', 'rb'))

with st.sidebar.form("input_form"):
    st.sidebar.subheader("Credit History")
    st.sidebar.caption("Overview of the applicant's credit account longevity and management over time.")
    external_risk_estimate = st.sidebar.number_input("Consolidated Risk Score", value=75, format="%d")
    msince_oldest_trade_open = st.sidebar.number_input("Months Since First Credit Line", value=120, format="%d")
    msince_most_recent_trade_open = st.sidebar.number_input("Months Since Latest Credit Line", value=2, format="%d")
    average_m_in_file = st.sidebar.number_input("Average Credit History Length (Months)", value=30, format="%d")

    st.sidebar.subheader("Credit Line Information")
    st.sidebar.caption("Details regarding the applicant's current and past credit lines.")
    num_satisfactory_trades = st.sidebar.number_input("Number of Positive Credit Lines", value=20, format="%d")
    num_total_trades = st.sidebar.number_input("Total Number of Credit Lines", value=23, format="%d")
    num_trades_openin_last_12m = st.sidebar.number_input("Number of Credit Lines Opened in Last 12 Months", value=2, format="%d")
    percent_install_trades = st.sidebar.number_input("Percentage of Installment Credit Lines", value=50, format="%d")

    st.sidebar.subheader("Delinquency Information")
    st.sidebar.caption("Records of past payment issues and defaults.")
    num_trades_60ever2_derog_pub_rec = st.sidebar.number_input("Number of 60+ Days Late Payments or Derogatory Public Records", value=1, format="%d")
    num_trades_90ever2_derog_pub_rec = st.sidebar.number_input("Number of 90+ Days Late Payments or Derogatory Public Records", value=0, format="%d")
    percent_trades_never_delq = st.sidebar.number_input("Percentage of Never Delinquent Trades", value=95, format="%d")
    msince_most_recent_delq = st.sidebar.number_input("Months Since Last Delinquency", value=12, format="%d")
    max_delq2_public_rec_last_12m = st.sidebar.number_input("Maximum Delinquency on Public Records in Last 12 Months", value=2, format="%d")
    max_delq_ever = st.sidebar.number_input("Worst Delinquency on Record", value=5, format="%d")

    st.sidebar.subheader("Credit Inquiries")
    st.sidebar.caption("Tracks the number of recent inquiries into the applicant's credit report.")
    msince_most_recent_inq_excl7days = st.sidebar.number_input("Months Since Last Credit Inquiry (Excluding Last 7 Days)", value=1, format="%d")
    num_inq_last_6m = st.sidebar.number_input("Number of Credit Inquiries in Last 6 Months", value=0, format="%d")
    num_inq_last_6m_excl7days = st.sidebar.number_input("Number of Credit Inquiries in Last 6 Months (Excluding Last 7 Days)", value=1, format="%d")

    st.sidebar.subheader("Credit Utilization and Balances")
    st.sidebar.caption("Assesses how much of the available credit is currently utilized.")
    net_fraction_revolving_burden = st.sidebar.number_input("Ratio of Revolving Credit Used to Credit Limit", value=45, format="%d")
    net_fraction_install_burden = st.sidebar.number_input("Ratio of Installment Credit Used to Original Loan Amount", value=65, format="%d")
    num_revolving_trades_w_balance = st.sidebar.number_input("Number of Revolving Trades with Balance", value=4, format="%d")
    num_install_trades_w_balance = st.sidebar.number_input("Number of Installment Trades with Balance", value=3, format="%d")
    num_bank2_natl_trades_w_high_utilization = st.sidebar.number_input("Number of Bank/National Trades with High Credit Utilization", value=1, format="%d")
    percent_trades_w_balance = st.sidebar.number_input("Percentage of Credit Lines with Balance", value=75, format="%d")

     # Submit buttons at the top and bottom of the sidebar
    submit_button_top = st.form_submit_top("Evaluate")

if submit_button:
    input_data = pd.DataFrame([[
        external_risk_estimate, msince_oldest_trade_open, msince_most_recent_trade_open,
        average_m_in_file, num_satisfactory_trades, num_trades_60ever2_derog_pub_rec,
        num_trades_90ever2_derog_pub_rec, num_total_trades, num_trades_openin_last_12m,
        percent_trades_never_delq, msince_most_recent_delq, max_delq2_public_rec_last_12m,
        max_delq_ever, percent_install_trades, net_fraction_install_burden, num_install_trades_w_balance,
        msince_most_recent_inq_excl7days, num_inq_last_6m, num_inq_last_6m_excl7days,
        net_fraction_revolving_burden, num_revolving_trades_w_balance, num_bank2_natl_trades_w_high_utilization,
        percent_trades_w_balance
    ]], columns=[
       'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
        'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
        'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
        'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
        'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
        'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance''ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
        'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
        'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
        'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
        'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
        'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'
    ])
    
    # Once the model is integrated:
    # prediction = model.predict(input_data)
    # st.write(f"Performance: {'Bad' if prediction[0] == 1 else 'Good'}")

    # For now, show input data (for testing)
    st.write("Data submitted:")
    st.write(input_data)


