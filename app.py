import streamlit as st
import pandas as pd
import os
import tempfile
from src.models.predict_model import FitnessTrackerPredictor

def initialize_session_state():
    """Initialize session state variables."""
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'error' not in st.session_state:
        st.session_state.error = None

def validate_files(acc_file, gyr_file, model_path, cluster_model_path):
    """Validate input files and model paths."""
    if not acc_file or not gyr_file:
        return False, "Please upload both accelerometer and gyroscope CSV files."
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    if not os.path.exists(cluster_model_path):
        return False, f"Cluster model file not found: {cluster_model_path}"
    return True, None

def main():
    # Initialize session state
    initialize_session_state()

    # App styling and title
    st.set_page_config(page_title="Fitness Activity Predictor", page_icon="🏋️")
    st.title("🏋️ ML Fitness Activity Predictor")
    st.markdown("Upload accelerometer and gyroscope CSV files to predict physical activities with our ML model.")

    # Sidebar configuration
    st.sidebar.header("⚙️ Model Configuration")
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="models/final_model.pkl",
        help="Path to the trained ML model file"
    )
    cluster_model_path = st.sidebar.text_input(
        "Cluster Model Path", 
        value="models/Clustering_model.pkl",
        help="Path to the clustering model file"
    )

    # File upload section
    st.subheader("📂 Upload Sensor Data")
    col1, col2 = st.columns(2)
    with col1:
        acc_file = st.file_uploader(
            "Accelerometer CSV", 
            type=["csv"],
            help="Upload CSV file containing accelerometer data"
        )
    with col2:
        gyr_file = st.file_uploader(
            "Gyroscope CSV", 
            type=["csv"],
            help="Upload CSV file containing gyroscope data"
        )

    # Predict button
    if st.button("🔮 Predict Activity", use_container_width=True):
        # Reset previous results
        st.session_state.prediction = None
        st.session_state.error = None

        # Validate inputs
        is_valid, error_msg = validate_files(acc_file, gyr_file, model_path, cluster_model_path)
        if not is_valid:
            st.session_state.error = error_msg
        else:
            # Create temporary directory for file handling
            with tempfile.TemporaryDirectory() as tmpdirname:
                acc_path = os.path.join(tmpdirname, "temp_acc.csv")
                gyr_path = os.path.join(tmpdirname, "temp_gyr.csv")

                # Save uploaded files
                with open(acc_path, "wb") as f:
                    f.write(acc_file.getbuffer())
                with open(gyr_path, "wb") as f:
                    f.write(gyr_file.getbuffer())

                # Progress bar
                with st.spinner("Processing data and predicting activity..."):
                    try:
                        # Initialize predictor
                        predictor = FitnessTrackerPredictor(
                            acc_path=acc_path,
                            gyr_path=gyr_path,
                            model_path=model_path,
                            cluster_model_path=cluster_model_path
                        )

                        # Make prediction
                        prediction = predictor.predict_activity()
                        st.session_state.prediction = prediction

                    except Exception as e:
                        st.session_state.error = f"Prediction failed: {str(e)}"

    # Display results
    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.prediction:
        st.success(f"**Predicted Activity:** {st.session_state.prediction}")
        st.balloons()

    # Display file previews if uploaded
    if acc_file:
        st.subheader("📊 Accelerometer Data Preview")
        acc_df = pd.read_csv(acc_file)
        st.dataframe(acc_df.head(), use_container_width=True)
    
    if gyr_file:
        st.subheader("📊 Gyroscope Data Preview")
        gyr_df = pd.read_csv(gyr_file)
        st.dataframe(gyr_df.head(), use_container_width=True)

if __name__ == "__main__":
    main()