import streamlit as st
import joblib
import pandas as pd


# Load saved model pipelines (make sure these files are in your project folder)
model_A = joblib.load('model_A_balanced_pipeline.pkl')
model_B = joblib.load('model_B_imbalanced_pipeline.pkl')

st.title("Automated Review Rating System")

# Input widgets for user review and numeric features
review_text = st.text_area("Enter your product review:")

helpfulness_num = st.number_input("Helpfulness Numerator", min_value=0, step=1)
helpfulness_den = st.number_input("Helpfulness Denominator", min_value=0, step=1)
review_length = len(review_text.split()) if review_text else 0
time = st.number_input("Review Timestamp (Unix time)", min_value=0, step=1)

if st.button("Get Predictions"):
    if not review_text.strip():
        st.warning("Please enter a review text to proceed.")
    else:
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame({
            "Text": [review_text],
            "HelpfulnessNumerator": [helpfulness_num],
            "HelpfulnessDenominator": [helpfulness_den],
            "review_length": [review_length],
            "Time": [time]
        })

        # Get predictions from both models
        pred_A = model_A.predict(input_df)[0]
        pred_B = model_B.predict(input_df)[0]

        # Display results
        st.subheader("Prediction Results")
        st.write(f"Model A (Balanced Dataset) predicted score: **{pred_A}**")
        st.write(f"Model B (Imbalanced Dataset) predicted score: **{pred_B}**")
        st.write("Note: Model A is trained on a balanced dataset, while Model B is trained on an imbalanced dataset.")
        st.write("Consider the context of your review when interpreting the scores.")