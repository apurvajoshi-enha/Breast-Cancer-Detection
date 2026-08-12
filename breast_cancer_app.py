import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

# ------------------------------------------------------------
# Load model and scaler
# ------------------------------------------------------------
model = keras.models.load_model("breast_cancer_model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("🩺 About")
    st.markdown("**Breast Cancer Detection**")
    st.write(
        "This app uses a **Neural Network** to predict whether a breast tumor is:"
    )
    st.markdown("- 🟢 **Benign** (Non-cancerous)")
    st.markdown("- 🟠 **Malignant** (Cancerous)")

    st.markdown("---")
    st.markdown("**Model Accuracy:** >95%")
    st.markdown("**Features Used:** 30 cell nuclei measurements")

    st.markdown("---")
    st.markdown("**How to Use**")
    st.markdown(
        "1. Enter values for all 30 features\n"
        "2. Click Predict\n"
        "3. View the result and confidence score"
    )

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🩺 Breast Cancer Detection Predictor")
st.markdown("Enter the cell nuclei measurements below")

# Feature names and average values (for sensible default inputs)
feature_defaults = {
    "mean radius": 14.127, "mean texture": 19.29, "mean perimeter": 91.969,
    "mean area": 654.889, "mean smoothness": 0.096, "mean compactness": 0.104,
    "mean concavity": 0.089, "mean concave points": 0.049, "mean symmetry": 0.181,
    "mean fractal dimension": 0.063, "radius error": 0.405, "texture error": 1.217,
    "perimeter error": 2.866, "area error": 40.337, "smoothness error": 0.007,
    "compactness error": 0.025, "concavity error": 0.032, "concave points error": 0.012,
    "symmetry error": 0.021, "fractal dimension error": 0.004, "worst radius": 16.269,
    "worst texture": 25.677, "worst perimeter": 107.261, "worst area": 880.583,
    "worst smoothness": 0.132, "worst compactness": 0.254, "worst concavity": 0.272,
    "worst concave points": 0.115, "worst symmetry": 0.29, "worst fractal dimension": 0.084,
}

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 Input Features", "ℹ️ Model Info"])

inputs = {}

with tab1:
    col1, col2, col3 = st.columns(3)

    mean_keys = [k for k in feature_defaults if k.startswith("mean")]
    error_keys = [k for k in feature_defaults if "error" in k]
    worst_keys = [k for k in feature_defaults if k.startswith("worst")]

    with col1:
        st.subheader("📊 Mean Values")
        for key in mean_keys:
            label = key.replace("mean ", "").title()
            inputs[key] = st.number_input(
                f"Mean {label}", value=float(feature_defaults[key]), format="%.5f", key=key
            )

    with col2:
        st.subheader("📈 Standard Error")
        for key in error_keys:
            label = key.replace(" error", "").title()
            inputs[key] = st.number_input(
                f"{label} Error", value=float(feature_defaults[key]), format="%.5f", key=key
            )

    with col3:
        st.subheader("🔴 Worst Values")
        for key in worst_keys:
            label = key.replace("worst ", "").title()
            inputs[key] = st.number_input(
                f"Worst {label}", value=float(feature_defaults[key]), format="%.5f", key=key
            )

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        input_data = np.array([[inputs[k] for k in feature_defaults]])
        input_std = scaler.transform(input_data)
        prediction = model.predict(input_std)
        label = np.argmax(prediction)

        if label == 0:
            st.error(f"⚠️ Result: **Malignant** (confidence: {prediction[0][0]*100:.2f}%)")
        else:
            st.success(f"✅ Result: **Benign** (confidence: {prediction[0][1]*100:.2f}%)")

        st.caption("This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")

with tab2:
    st.subheader("Model Information")
    st.markdown("""
    - **Model Type:** Feedforward Neural Network (Keras/TensorFlow)
    - **Architecture:** 30 → 20 (ReLU) → 2 (Sigmoid)
    - **Dataset:** Wisconsin Breast Cancer Dataset (built into scikit-learn)
    - **Training:** 80/20 train-test split, standardized features
    - **Test Accuracy:** >95%
    """)
    st.info("This model was trained on the classic Wisconsin Diagnostic Breast Cancer dataset, which contains digitized measurements from breast mass images.")
st.markdown(
    """
    # ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Spam Mail Detection System
    <br>
    Developed By Agrima Saxena
    </div>
    """
    unsafe_allow_html=True
)
