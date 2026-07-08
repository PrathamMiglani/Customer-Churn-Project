import streamlit as st
import pandas as pd
# import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
# @st.cache_resource
# def load_model():
#     with open("customer_churn_model.pkl", "rb") as f:
#         data = pickle.load(f)
#     return data["model"], data["columns"]

# model, columns = load_model()
import json
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("customer_churn_model.json")

with open("columns.json") as f:
    columns = json.load(f)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below.")

st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure", 0, 100, 10)
citytier = st.sidebar.selectbox("City Tier", [1, 2, 3])

warehousetohome = st.sidebar.number_input(
    "Warehouse To Home Distance", 0, 200, 15
)

hourspendonapp = st.sidebar.slider(
    "Hours Spent on App", 0, 10, 3
)

numberofdeviceregistered = st.sidebar.slider(
    "Registered Devices", 1, 10, 3
)

satisfactionscore = st.sidebar.slider(
    "Satisfaction Score", 1, 5, 3
)

numberofaddress = st.sidebar.slider(
    "Number of Addresses", 1, 20, 2
)

complain = st.sidebar.selectbox(
    "Complaint Registered?",
    ["No", "Yes"]
)

orderamounthikefromlastyear = st.sidebar.slider(
    "Order Amount Hike (%)",
    0,
    100,
    15
)

couponused = st.sidebar.slider(
    "Coupons Used",
    0,
    20,
    2
)

ordercount = st.sidebar.slider(
    "Order Count",
    0,
    20,
    5
)

daysincelastorder = st.sidebar.slider(
    "Days Since Last Order",
    0,
    60,
    5
)

cashbackamount = st.sidebar.number_input(
    "Cashback Amount",
    0.0,
    5000.0,
    150.0
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

# -----------------------------
# Feature Vector
# -----------------------------
input_df = pd.DataFrame(0, index=[0], columns=columns)

input_df["tenure"] = tenure
input_df["citytier"] = citytier
input_df["warehousetohome"] = warehousetohome
input_df["hourspendonapp"] = hourspendonapp
input_df["numberofdeviceregistered"] = numberofdeviceregistered
input_df["satisfactionscore"] = satisfactionscore
input_df["numberofaddress"] = numberofaddress
input_df["complain"] = 1 if complain == "Yes" else 0
input_df["orderamounthikefromlastyear"] = orderamounthikefromlastyear
input_df["couponused"] = couponused
input_df["ordercount"] = ordercount
input_df["daysincelastorder"] = daysincelastorder
input_df["cashbackamount"] = cashbackamount

# Gender Encoding
if gender == "Male":
    input_df["gender_Male"] = 1
else:
    input_df["gender_Female"] = 1

# Marital Status Encoding
if marital == "Single":
    input_df["maritalstatus_Single"] = 1
elif marital == "Married":
    input_df["maritalstatus_Married"] = 1
else:
    input_df["maritalstatus_Divorced"] = 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))