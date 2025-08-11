# app.py
import streamlit as st
import pandas as pd
import os
from src.models.predict_model import FitnessTrackerPredictor

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="ML Fitness Activity Predictor",
    page_icon="🏋️",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
        }
    </style>
    <div class="main-title">🏋️ ML Fitness Activity Predictor</div>
    <div class="subtitle">Upload sensor data and get real-time predictions with detailed model insights</div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# SIDEBAR - MODEL SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Model Configuration")
model_path = st.sidebar.text_input("Main Model Path", "models/final_model.pkl")
cluster_model_path = st.sidebar.text_input("Cluster Model Path", "models/Clustering_model.pkl")
st.sidebar.markdown("📌 **Tip:** Ensure model files exist before running predictions.")

# -----------------------------
# STEP 1: FILE UPLOAD
# -----------------------------
st.subheader("📂 Step 1: Upload Sensor Data")
col1, col2 = st.columns(2)

with col1:
    acc_file = st.file_uploader("**Accelerometer CSV**", type=["csv"], help="Upload the accelerometer readings in CSV format.")
with col2:
    gyr_file = st.file_uploader("**Gyroscope CSV**", type=["csv"], help="Upload the gyroscope readings in CSV format.")

# -----------------------------
# STEP 2: DATA PREVIEW
# -----------------------------
if acc_file:
    st.markdown("#### 📊 Accelerometer Data Preview")
    acc_df = pd.read_csv(acc_file)
    st.dataframe(acc_df.head(), use_container_width=True)

if gyr_file:
    st.markdown("#### 📊 Gyroscope Data Preview")
    gyr_df = pd.read_csv(gyr_file)
    st.dataframe(gyr_df.head(), use_container_width=True)

st.markdown("---")

# -----------------------------
# STEP 3: PREDICTION
# -----------------------------
st.subheader("🤖 Step 2: Run Prediction")
predict_button = st.button("🔮 Predict Activity", use_container_width=True)

if predict_button:
    # Validation
    if not acc_file or not gyr_file:
        st.error("⚠️ Please upload **both** Accelerometer and Gyroscope CSV files.")
    elif not os.path.exists(model_path):
        st.error(f"❌ Model file not found: `{model_path}`")
    elif not os.path.exists(cluster_model_path):
        st.error(f"❌ Cluster model file not found: `{cluster_model_path}`")
    else:
        # Save temporary files
        acc_path = "temp_acc.csv"
        gyr_path = "temp_gyr.csv"
        acc_file.seek(0)
        gyr_file.seek(0)
        with open(acc_path, "wb") as f:
            f.write(acc_file.getbuffer())
        with open(gyr_path, "wb") as f:
            f.write(gyr_file.getbuffer())

        try:
            # Create predictor instance
            predictor = FitnessTrackerPredictor(
                acc_path=acc_path,
                gyr_path=gyr_path,
                model_path=model_path,
                cluster_model_path=cluster_model_path
            )

            # Run prediction
            prediction = predictor.predict_activity()

            # -----------------------------
            # RESULT CARD
            # -----------------------------
            st.markdown(
                f"""
                <div style="
                    background-color:#f1f8e9;
                    padding:30px;
                    border-radius:12px;
                    border-left: 6px solid #4caf50;
                    text-align:center;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                ">
                    <h2 style="color:#2e7d32; margin-bottom:10px;">✅ Prediction Successful</h2>
                    <p style="font-size:18px; color:#555;">Your predicted activity is:</p>
                    <h1 style="color:#1b5e20; font-size:42px;">🏆 {prediction}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -----------------------------
            # EXTRA DETAILS
            # -----------------------------
            st.markdown("### 📌 Model Details")
            st.info(f"**Main Model:** `{model_path}`\n\n**Cluster Model:** `{cluster_model_path}`")

            st.markdown("### 📈 Data Summary")
            st.write("**Accelerometer Shape:**", acc_df.shape if acc_file else "Not provided")
            st.write("**Gyroscope Shape:**", gyr_df.shape if gyr_file else "Not provided")

        except Exception as e:
            st.error(f"💥 Error during prediction: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color: #95a5a6; font-size: 14px;">
        Built with ❤️ using Streamlit | ML Fitness Activity Predictor
    </div>
    """,
    unsafe_allow_html=True
)
