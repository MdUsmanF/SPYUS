import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model

import subprocess

# Function to check if a module is installed
def is_module_installed(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

# Install joblib if it's not already installed
if not is_module_installed('joblib'):
    subprocess.check_call(['pip', 'install', 'joblib'])


# Load models
models = {
    # 'Decision Tree': 'D:/Major_Project/Streamlit_App/Major_Project/saved_models/Decision_Tree.joblib',
    'KNN': 'D:/Major_Project/Streamlit_App/Major_Project/saved_models/KNN.joblib',
    'Quadratic Discriminant Analysis': 'D:/Major_Project/Streamlit_App/Major_Project/saved_models/Quadratic Discriminant Analysis.joblib',
    'Perceptron': 'D:/Major_Project/Streamlit_App/Major_Project/saved_models/Perceptron.joblib',
    # 'Random Forest': 'D:/Major_Project/Streamlit_App/Major_Project/saved_models/Random_Forest.joblib',
}
from tensorflow.keras.models import load_model as load_keras_model

def load_model_from_h5(model_path):
    return load_keras_model(model_path)

# Function to load selected model
def load_model(model_name):
    model_path = models.get(model_name)
    print("Selected model:", model_name)
    print("Model path:", model_path)

    if model_path:
        if model_path.endswith('.joblib'):
            return joblib.load(model_path)
        elif model_path.endswith('.h5'):  # Handling .h5 files for FFNN models
            return load_model_from_h5(model_path)
        else:
            return None
    else:
        return None

# Function to predict attack type for each record
def predict_attack(model, data):
    predictions = []

    for index, row in data.iterrows():
        features = row.to_numpy().reshape(1, -1)
        prediction = model.predict(features)[0]
        predictions.append(prediction)

    return predictions

csv = pd.read_csv("test_50.csv")

# Main function
def main():
    st.title('🕸️SPYUS AI: Attack Type Prediction')

    # Dropdown to select model
    selected_model = st.selectbox('Select Model', list(models.keys()))

    # Load selected model
    model = load_model(selected_model)

    if model is None:
        st.error("Model not found!")
    else:
        st.sidebar.markdown('### Model Info')
        st.sidebar.text(f"Selected Model: {selected_model}")
        st.sidebar.text("Trained with 69000+ Records")
        st.sidebar.text("Data template")
        st.sidebar.download_button(
            label="Download Template",
            data=csv.to_csv(index=False).encode(),
            file_name='test_50.csv',
            mime='text/csv',
        )

        # File uploader for CSV file
        st.markdown('### Upload CSV File')
        uploaded_file = st.file_uploader("Choose a CSV or PKL file", type=["csv", "pkl"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write('### Uploaded Data')
            st.write(data)

            if st.button('Predict'):
                try:
                    predictions = predict_attack(model, data)
                    results_df = pd.DataFrame({'Predicted Attack Type': predictions})
                    st.write('### Prediction Results')
                    st.write(results_df)
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
