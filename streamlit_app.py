# -*- coding: utf-8 -*-
"""
Streamlit Web Application
------------------------------------
Title  : Heart Disease Prediction using ML Models
Author : Lokesh N Rao
ID: 2025AA05309
Course : Machine Learning – Assignment 2

This app allows the user to:
1. Upload a CSV file with patient data
2. Select a trained ML model
3. Run predictions
4. View results, confusion matrix, and evaluation metrics
5. Download the prediction output as CSV
"""

# ----------- Import Required Libraries -----------

import streamlit as st                
import joblib                          
import pandas as pd                  
from sklearn.metrics import (         
    confusion_matrix,
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# ----------- WebPage Customisation -----------

# Set browser tab title and layout style
st.set_page_config(page_title="Machine Learning Assignment 2", layout="centered")


col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("bits_logo.png", width=350)

# Custom HTML & CSS for attractive heading
st.markdown(
    """
    <style>
    .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #4CAF50; }
    .sub-title { text-align: center; font-size: 24px; color: #2196F3; margin-bottom: 30px; }
    .name { text-align: center; font-size: 18px; color: #888888; margin-top: 20px; }
    </style>

    <div class="main-title">Machine Learning Assignment 2</div>
    <div class="sub-title">Heart Disease Prediction – Different Models</div>
    <div class="name">Created by: <b>Lokesh N Rao</b></div>
    """,
    unsafe_allow_html=True
)

# ----------- Load Trained Models and scalar obtained from the training of models -----------


models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# ----------- Feature Engineering Function -----------

def feature_engineering(df, scaler):
    """
    This function prepares raw input data for prediction.
    Steps:
    1. Handle missing values
    2. Create new derived features
    3. Remove target column if present
    4. Scale numeric features using the trained scaler
    """

    df = df.copy()

    # Fill missing values for each column
    for col in df.columns:
        if df[col].dtype == "object":
            # For categorical columns -> fill with most frequent value
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # For numeric columns -> fill with median
            df[col] = df[col].fillna(df[col].median())

    # Create new meaningful features
    # Ratio between cholesterol and age
    df["chol_age_ratio"] = df["chol"] / df["age"]

    # Binary feature indicating high-risk patients
    df["high_risk"] = ((df["trestbps"] > 140) & (df["chol"] > 240)).astype(int)

    # Separate input features from target (if target exists)
    if "target" in df.columns:
        X = df.drop("target", axis=1)
    else:
        X = df

    # Select only numeric columns for scaling
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # Apply the same scaling used during training
    X[num_cols] = scaler.transform(X[num_cols])

    return X

# ----------- Main App Logic -----------

# Dropdown for model selection
selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Use session_state to preserve results after rerun
if "predicted_df" not in st.session_state:
    st.session_state.predicted_df = None
    st.session_state.metrics = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data (Preview):", df.head())

    if st.button("Predict"):
        X_processed = feature_engineering(df, scaler)
        predictions = model.predict(X_processed)

        df["Prediction"] = predictions
        df["Prediction_Label"] = df["Prediction"].map({0: "No Disease", 1: "Disease"})

        st.session_state.predicted_df = df.copy()
        st.session_state.metrics = None

        # --------- THIS BLOCK MUST STAY HERE ---------
        if "target" in df.columns:

            valid_target_df = df[df["target"].notna()]

            if len(valid_target_df) == 0:
                st.warning(
                    "The uploaded file contains a 'target' column, but it is empty. "
                    "Please upload a CSV file with valid target values (0/1) to view model metrics."
                )
            else:
                y_true = valid_target_df["target"]
                y_pred = valid_target_df["Prediction"]

                acc = accuracy_score(y_true, y_pred) * 100
                prec = precision_score(y_true, y_pred) * 100
                rec = recall_score(y_true, y_pred) * 100
                f1 = f1_score(y_true, y_pred) * 100
                mcc = matthews_corrcoef(y_true, y_pred) * 100

                try:
                    X_valid = X_processed.loc[valid_target_df.index]
                    y_prob = model.predict_proba(X_valid)[:, 1]
                    auc = roc_auc_score(y_true, y_prob) * 100
                except:
                    auc = None

                cm = confusion_matrix(y_true, y_pred)

                st.session_state.metrics = {
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "mcc": mcc,
                    "auc": auc,
                    "cm": cm
                }

        else:
            st.info(
                "To view model evaluation metrics, please upload a CSV file "
                "that includes a 'target' column with actual values (0 or 1)."
            )

       
        st.success("Prediction Completed")


# ----------- Display Stored Results -----------

if st.session_state.predicted_df is not None:

    result_df = st.session_state.predicted_df
    st.subheader("Prediction Result")
    st.write(result_df)

    if st.session_state.metrics is not None:
        m = st.session_state.metrics

        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            m["cm"],
            index=["Actual No Disease", "Actual Disease"],
            columns=["Predicted No Disease", "Predicted Disease"]
        )
        st.dataframe(cm_df)

        st.subheader("Model Evaluation Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['acc']:.2f}%")
        c2.metric("Precision", f"{m['prec']:.2f}%")
        c3.metric("Recall", f"{m['rec']:.2f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("F1 Score", f"{m['f1']:.2f}%")
        c5.metric("MCC Score", f"{m['mcc']:.2f}%")
        c6.metric("AUC Score", f"{m['auc']:.2f}%" if m["auc"] is not None else "N/A")

    # Convert results to CSV for download
    csv_data = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="heart_disease_predictions.csv",
        mime="text/csv"
    )
