import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# Dictionary to map model names to their file paths
model_paths = {
    'ffnn_model_top_10_features': 'saved_models/ffnn_model_top_10_features.h5',
    'ffnn_model_top_20_features': 'saved_models/ffnn_model_top_20_features.h5',
    'ffnn_model_top_30_features': 'saved_models/ffnn_model_top_30_features.h5',
    'ffnn_model_top_40_features': 'saved_models/ffnn_model_top_40_features.h5',
    'ffnn_model_top_50_features': 'saved_models/ffnn_model_top_50_features.h5',
}

def load_trained_model(model_path):
    return load_model(model_path)

def load_test_data(uploaded_file):
    data = pd.read_pickle(uploaded_file)
    print(data.shape)
    return data

def predict_intrusion_attack(model, test_data):
    # Define label mapping
    label_mapping = {'BENIGN': 0, 'DoS Hulk': 1, 'DoS GoldenEye': 2, 'DoS slowloris': 3, 'DoS Slowhttptest': 4, 'Heartbleed': 5}
    
    # Perform prediction using the model
    predictions = model.predict(test_data)
    
    # Convert numerical predictions to attack types using label mapping
    predicted_labels = [list(label_mapping.keys())[np.argmax(prediction)] for prediction in predictions]
    
    return predicted_labels

def main():
    st.title("SPYUS AI: IDS")
    
    # Select model
    selected_model = st.selectbox("Select Model", list(model_paths.keys()))

    # Load selected model
    model = load_trained_model(model_paths[selected_model])

    # File uploader for test data
    st.subheader("Upload Test Data")
    uploaded_file = st.file_uploader("Choose a PKL file", type="pkl")

    if uploaded_file is not None:
        test_data = load_test_data(uploaded_file)

        if st.button("Predict"):
            # Ensure test data matches the input shape of the model
            # Then preprocess the data as necessary (scaling, encoding, etc.)
            # Here I assume the test_data is already preprocessed and in the correct format
            test_data = np.array(test_data)  # Convert to numpy array if not already
            predictions = predict_intrusion_attack(model, test_data)
            st.write("Predictions:", predictions)

if __name__ == "__main__":
    main()
