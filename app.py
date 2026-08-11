import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# ------------------------------------------------------------
# Load model and scaler
# ------------------------------------------------------------
model = keras.models.load_model("breast_cancer_model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺")
st.title("🩺 Breast Cancer Detection")
st.write(
    "Enter the tumor's cell nuclei measurements below. "
    "Default values are pre-filled with dataset averages for reference."
)

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

groups = {
    "Mean values": [k for k in feature_defaults if k.startswith("mean")],
    "Error values": [k for k in feature_defaults if "error" in k],
    "Worst values": [k for k in feature_defaults if k.startswith("worst")],
}

inputs = {}
for group_name, keys in groups.items():
    with st.expander(group_name, expanded=(group_name == "Mean values")):
        cols = st.columns(2)
        for i, key in enumerate(keys):
            with cols[i % 2]:
                inputs[key] = st.number_input(
                    key, value=float(feature_defaults[key]), format="%.5f"
                )

if st.button("Predict"):
    input_data = np.array([[inputs[k] for k in feature_defaults]])
    input_std = scaler.transform(input_data)
    prediction = model.predict(input_std)
    label = np.argmax(prediction)

    if label == 0:
        st.error(f"Result: Malignant (confidence: {prediction[0][0]*100:.2f}%)")
    else:
        st.success(f"Result: Benign (confidence: {prediction[0][1]*100:.2f}%)")

    st.caption("This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")
