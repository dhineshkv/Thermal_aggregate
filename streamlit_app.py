import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io
import time

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_PATH = Path("models/best_rf_pipeline.joblib")

st.set_page_config(
    page_title="Lightweight Thermal Aggregate – AI Predictor",
    layout="centered",
)

st.title("🧱 Civil Lightweight Thermal Aggregate – AI Prediction System")
st.write("Predict thermal & mechanical performance of lightweight aggregates using a trained Random Forest model.")

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
st.success("Model loaded successfully ✔")

# -------------------------------------------------------
# INPUT FIELDS
# -------------------------------------------------------
st.header("📝 Enter Input Parameters")

INPUT_ORDER = [
    "PCM_%", "Nano_%", "Density_kgm3", "Porosity_%",
    "Thermal_Conductivity_WmK", "Specific_Heat_JkgK",
    "Latent_Heat_kJkg", "Orientation", "Ambient_Temp_C"
]

DEFAULTS = {
    "PCM_%": 0.0,
    "Nano_%": 0.0,
    "Density_kgm3": 2300.0,
    "Porosity_%": 5.0,
    "Thermal_Conductivity_WmK": 1.45,
    "Specific_Heat_JkgK": 900.0,
    "Latent_Heat_kJkg": 0.0,
    "Orientation": "East",
    "Ambient_Temp_C": 35.0,
}

with st.form("input_form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs["PCM_%"] = st.number_input("PCM %", value=DEFAULTS["PCM_%"], step=0.1)
        inputs["Nano_%"] = st.number_input("Nano %", value=DEFAULTS["Nano_%"], step=0.1)
        inputs["Density_kgm3"] = st.number_input("Density (kg/m³)", value=DEFAULTS["Density_kgm3"])
        inputs["Porosity_%"] = st.number_input("Porosity %", value=DEFAULTS["Porosity_%"])
        inputs["Orientation"] = st.selectbox("Orientation", ["East", "West", "North", "South"])

    with col2:
        inputs["Thermal_Conductivity_WmK"] = st.number_input(
            "Thermal Conductivity (W/mK)", value=DEFAULTS["Thermal_Conductivity_WmK"]
        )
        inputs["Specific_Heat_JkgK"] = st.number_input(
            "Specific Heat (J/kg·K)", value=DEFAULTS["Specific_Heat_JkgK"]
        )
        inputs["Latent_Heat_kJkg"] = st.number_input(
            "Latent Heat (kJ/kg)", value=DEFAULTS["Latent_Heat_kJkg"]
        )
        inputs["Ambient_Temp_C"] = st.number_input(
            "Ambient Temperature (°C)", value=DEFAULTS["Ambient_Temp_C"]
        )

    submitted = st.form_submit_button("🔮 Predict")

# -------------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------------
if submitted:
    X = pd.DataFrame([inputs], columns=INPUT_ORDER)

    st.subheader("✨ Predictions")
    with st.spinner("Running model..."):
        start = time.time()
        preds = model.predict(X)[0]
        end = time.time()

    output_labels = [
        "Thermal Delay (min)",
        "Attenuation Rate (%)",
        "Energy Saving (%)",
        "Compressive Strength (MPa)"
    ]

    result_df = pd.DataFrame(
        [preds], columns=output_labels
    ).round(4)

    st.metric(output_labels[0], f"{preds[0]:.2f} min")
    st.metric(output_labels[1], f"{preds[1]:.2f} %")
    st.metric(output_labels[2], f"{preds[2]:.2f} %")
    st.metric(output_labels[3], f"{preds[3]:.2f} MPa")

    st.caption(f"⏱ Prediction time: {end - start:.3f} seconds")

# -------------------------------------------------------
# BATCH CSV PREDICTION
# -------------------------------------------------------
st.header("📂 Batch Prediction (Upload CSV)")
st.write("Upload a CSV file with the same input columns to generate predictions for multiple rows.")

uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)

    missing_cols = [col for col in INPUT_ORDER if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
    else:
        with st.spinner("Predicting for all rows..."):
            preds = model.predict(df[INPUT_ORDER])
            pred_cols = [
                "pred_Thermal_Delay_min",
                "pred_Attenuation_Rate_pct",
                "pred_Energy_Saving_pct",
                "pred_Compressive_Strength_MPa",
            ]
            pred_df = pd.DataFrame(preds, columns=pred_cols).round(4)
            final_df = pd.concat([df, pred_df], axis=1)

        st.subheader("Preview of Results")
        st.dataframe(final_df.head())

        # Download button
        csv_buf = io.StringIO()
        final_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download Predictions CSV",
            csv_buf.getvalue(),
            "batch_predictions.csv",
            "text/csv",
        )

# -------------------------------------------------------
# TEMPLATE DOWNLOAD
# -------------------------------------------------------
st.write("📄 Download template for batch prediction:")
template_df = pd.DataFrame([DEFAULTS])
st.download_button(
    "⬇ Download Input Template",
    template_df.to_csv(index=False),
    "input_template.csv",
    "text/csv",
)

st.caption("Built with ❤️ using Streamlit + RandomForest + Civil Engineering insights.")
