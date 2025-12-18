import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model & features
# -----------------------------
model = joblib.load("flight_delay_model.pkl")
features = joblib.load("model_features.pkl")

st.title("✈️ Flight Delay Prediction (60+ Minutes)")
st.write("Predict the risk of long flight delays using ML")

# -----------------------------
# Extract dropdown options
# -----------------------------
airlines = sorted({col.replace("Airline_", "") 
                   for col in features if col.startswith("Airline_")})

origins = sorted({col.replace("Origin_", "") 
                  for col in features if col.startswith("Origin_")})

destinations = sorted({col.replace("Dest_", "") 
                       for col in features if col.startswith("Dest_")})

# -----------------------------
# User Inputs
# -----------------------------
day_of_week = st.selectbox("Day of Week (1=Mon, 7=Sun)", [1,2,3,4,5,6,7])
dep_hour = st.slider("Departure Hour", 0, 23, 12)
month = st.selectbox("Month", list(range(1,13)))

airline = st.selectbox("Airline", airlines)
origin = st.selectbox("Origin Airport", origins)
dest = st.selectbox("Destination Airport", destinations)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Delay"):
    # Base numeric features
    input_data = pd.DataFrame({
        "DayOfWeek": [day_of_week],
        "DepHour": [dep_hour],
        "month": [month]
    })

    # One-hot encoding for categorical values
    input_data[f"Airline_{airline}"] = 1
    input_data[f"Origin_{origin}"] = 1
    input_data[f"Dest_{dest}"] = 1

    # Fill missing columns
    for col in features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns
    input_data = input_data[features]

    # Predict probability
    prob = model.predict_proba(input_data)[0][1]

    # Threshold (conservative)
    threshold = 0.30

    st.subheader("Prediction Result")

    if prob >= threshold:
        st.error(f"⚠️ High Delay Risk\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Likely On Time\n\nDelay Risk: {prob:.2f}")
