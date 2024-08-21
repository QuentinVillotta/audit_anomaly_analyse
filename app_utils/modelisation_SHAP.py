import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import shap
import matplotlib.pyplot as plt
import seaborn as sns


# Function to train the model and make predictions
def train_and_predict(model_name, data):
    if model_name == 'IsolationForest':
        model = IsolationForest()
    elif model_name == 'OneClassSVM':
        model = OneClassSVM()
    elif model_name == 'LOF':
        model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    else:
        st.error("Unknown model")
        return None, None
    
    if model_name == 'LOF':
        data = data.copy()
        data['anomaly'] = model.fit_predict(data)
        data['anomaly']  = (data['anomaly']  == -1).astype(int)
        data['anomaly'] = data['anomaly'] == -1
        return model, data
    else:
        model.fit(data)
        predictions = model.predict(data)
        anomalies = predictions == -1
        return model, anomalies

# Function to plot SHAP visualizations
def plot_shap(model, data, predictions):
    explainer = shap.KernelExplainer(model)
    shap_values = explainer.shap_values(data)

    # Global interpretation
    st.subheader("Global Interpretation")
    shap.summary_plot(shap_values, data)
    st.pyplot()

    # Local interpretation
    st.subheader("Local Interpretation")
    selected_index = st.selectbox("Select an instance", range(len(data)))
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[selected_index], data.iloc[selected_index])
    st.pyplot()