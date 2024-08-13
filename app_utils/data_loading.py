import pandas as pd
import os
import streamlit as st

def data_loader(uploaded_file):
    """
    Load data from a CSV, Excel, Parquet.

    Parameters:
    uploaded_file (UploadedFile): Streamlit UploadedFile object.

    Returns:
    DataFrame or None: A pandas DataFrame containing the loaded data, or None if loading fails.
    """
    try:
        # Check file extension to determine the appropriate loader
        _, file_extension = os.path.splitext(uploaded_file.name)

        if file_extension.lower() == '.csv':
            # Load data from CSV
            data = pd.read_csv(uploaded_file, index_col=None)
            return data
        elif file_extension.lower() in ['.xls', '.xlsx']:
            # Load data from Excel
            data = pd.read_excel(uploaded_file, index_col=None)
            return data
        elif file_extension.lower() in ['.pq', '.parquet']:
            # Load data from Parquet
            data = pd.read_parquet(uploaded_file)
            return data
        else:
            st.caption(":x: :red[Unsupported file type. Please provide a CSV, Excel or Parquet file]")
            return None

    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def classify_variable_types(df, threshold=20):
    """
    Classify the variables of a DataFrame as continuous or discrete.

    :param df: DataFrame to analyze
    :param threshold: Threshold to distinguish between continuous and discrete variables
    :return: Dictionary with variable names as keys and their type ('continuous' or 'discrete') as values
    """
    variable_types = {}

    for column in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].nunique()
            if unique_values <= threshold:  # Threshold to distinguish continuous from discrete
                variable_types[column] = 'discrete'
            else:
                variable_types[column] = 'continuous'
        else:
            variable_types[column] = 'non-numeric'

    return variable_types
