import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Risk", page_icon="🏦")
model = joblib.load("model.pkl")

st.title("🏦 Loan Default Risk Scoring")
st.write("Customer ki details bharo aur default hone ki probability dekho.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual income", 1000, 5000000, 50000, step=1000)
    home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp = st.number_input("Employment length (years)", 0.0, 60.0, 3.0)
    intent = st.selectbox("Loan intent", ["EDUCATION", "MEDICAL", "VENTURE",
                                          "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
with col2:
    grade = st.selectbox("Loan grade", list("ABCDEFG"))
    amount = st.number_input("Loan amount", 500, 100000, 10000, step=500)
    rate = st.number_input("Interest rate (%)", 1.0, 30.0, 11.0)
    default_file = st.selectbox("Previous default on file", ["N", "Y"])
    hist = st.number_input("Credit history length (years)", 1, 40, 4)

if st.button("Check Risk"):
    row = pd.DataFrame([{
        "person_age": age, "person_income": income, "person_home_ownership": home,
        "person_emp_length": emp, "loan_intent": intent, "loan_grade": grade,
        "loan_amnt": amount, "loan_int_rate": rate,
        "loan_percent_income": amount / income,
        "cb_person_default_on_file": default_file,
        "cb_person_cred_hist_length": hist,
        "cred_hist_to_age": hist / age,
    }])
    prob = model.predict_proba(row)[0][1]
    st.metric("Default Probability", f"{prob:.1%}")
    if prob >= 0.5:
        st.error("High Risk — loan reject karne ki salah.")
    else:
        st.success("Low Risk — loan approve kiya ja sakta hai.")
