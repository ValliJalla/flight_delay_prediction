# Flight Delay Prediction Web App ✈️

This project is a machine learning–based web application that predicts whether a flight is likely to be delayed by more than 60 minutes.  
I built this project as part of my learning in machine learning and model deployment concepts.

The model uses historical flight-related features and the application is built using Streamlit for user interaction.

---

## Problem Overview

Flight delays are a common issue and can affect passengers as well as airline operations.  
The objective of this project is to predict **long delays (60+ minutes)** using a classification model.

---

## Machine Learning Details

- **Problem type:** Binary classification  
- **Target variable:**
  - `1` → Delay greater than 60 minutes  
  - `0` → On time or delay less than 60 minutes  

- **Model used:** XGBoost Classifier  
- **Imbalanced data handling:** Class weights  

### Evaluation metrics
- ROC–AUC score  
- Confusion matrix  
- ROC curve  

---

## Features Used

- Day of week  
- Month  
- Departure hour  
- Airline  
- Origin airport  
- Destination airport  

Categorical variables were converted using one-hot encoding.

---

## Streamlit Application

The web app allows users to:
- Select airline, source, and destination airports
- Choose date and departure time
- Get a prediction along with probability score

To avoid unrealistic inputs:
- No free-text fields are used  
- Dropdown options are generated from trained model features  

---

## Project Structure

flight_delay_prediction/
│
├── app.py
├── flight_delay_model.pkl
├── model_features.pkl
├── requirements.txt
└── README.md


---

## How to Run the Project Locally

1. Clone the repository  
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py

   
---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  

---

## Notes

This project helped me understand:
- Handling imbalanced datasets
- Feature consistency during deployment-ready applications
- Building machine learning models with real-world constraints

