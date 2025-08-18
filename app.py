import streamlit as st
import joblib
import tensorflow as tf
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load deep learning model
deep_model = tf.keras.models.load_model("deep_model_balanced.h5")

# Load tokenizer
with open("tokenizer_balanced.pkl", "rb") as f:
    deep_tokenizer = pickle.load(f)

# ✅ Verify tokenizer integrity
if not hasattr(deep_tokenizer, "texts_to_sequences"):
    st.error("Tokenizer is corrupted or not properly loaded.")
    st.stop()

# Load label mapping
with open("label_to_int.json", "r") as f:
    label_to_int = json.load(f)

# Ensure keys are integers
int_to_label = {i: int(k) if k.isdigit() else k for k, i in label_to_int.items()}

# Load ML models and vectorizers
model_A = joblib.load("model_A_balanced.pkl")
vectorizer_A = joblib.load("vectorizer_model_A.pkl")
model_B = joblib.load("model_B_imbalanced.pkl")
vectorizer_B = joblib.load("vectorizer_model_B.pkl")

# Streamlit UI
st.title("Automated Review Rating System")
review_text = st.text_area("Enter your product review:")

if st.button("Get Predictions"):
    if not review_text.strip():
        st.warning("Please enter a review text to proceed.")
    else:
        input_list = [review_text.strip()]

        # --- ML Model A ---
        try:
            X_A = vectorizer_A.transform(input_list)
            pred_A = model_A.predict(X_A)[0]
        except Exception as e:
            pred_A = "Error"
            st.error(f"Model A prediction failed: {e}")

        # --- ML Model B ---
        try:
            X_B = vectorizer_B.transform(input_list)
            pred_B = model_B.predict(X_B)[0]
        except Exception as e:
            pred_B = "Error"
            st.error(f"Model B prediction failed: {e}")

        # --- Deep Learning Model ---
        try:
            import re
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', review_text.strip())

         # ✅ Fallback if cleaned text is empty
            if not cleaned_text:
                cleaned_text = "<OOV>"

         # ✅ Tokenize safely
            seq = deep_tokenizer.texts_to_sequences([cleaned_text])
            if not seq or not seq[0]:
                seq = [[deep_tokenizer.word_index.get("<OOV>", 1)]]

            pad = pad_sequences(seq, maxlen=100)
            deep_pred_dist = deep_model.predict(pad)
            deep_pred_class = int(deep_pred_dist.argmax(axis=1))
            deep_pred_label = int_to_label.get(deep_pred_class, "Unknown")
        except Exception as e:
            deep_pred_label = "Error"
            st.error(f"Deep model prediction failed: {e}")


        # --- Display Results ---
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model A (Balanced ML)", str(pred_A))
        with col2:
            st.metric("Model B (Imbalanced ML)", str(pred_B))
        with col3:
            st.metric("Deep Model (Keras)", str(deep_pred_label))