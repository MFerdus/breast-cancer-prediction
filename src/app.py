\
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Breast Cancer Prediction (PyCaret)", layout="wide")

st.title("Breast Cancer Prediction (PyCaret)")
st.write(
    "Upload a CSV of patient records (same feature columns as training data). "
    "The app will load a saved PyCaret model and return predicted class + probability."
)

with st.sidebar:
    st.header("Model")
    model_base = st.text_input("Model path (without .pkl)", value="models/best_model")
    st.caption("Example: `models/best_model` (expects `models/best_model.pkl`).")

@st.cache_resource
def _load(model_base_name: str):
    return load_model(model_base_name)

try:
    model = _load(model_base.replace(".pkl", ""))
    st.success(f"Loaded model: {model_base.replace('.pkl','')}")
except Exception as e:
    st.error("Could not load model. Train first (see README) or check the model path.")
    st.exception(e)
    st.stop()

st.header("Predict")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if st.button("Run prediction"):
        with st.spinner("Predicting..."):
            pred = predict_model(model, data=df)

        st.subheader("Results")
        st.dataframe(pred.head(50), use_container_width=True)

        csv = pred.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Upload a CSV file to begin.")
