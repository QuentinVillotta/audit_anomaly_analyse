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

        st.caption(file_extension)
        if file_extension.lower() == '.csv':
            # Load data from CSV
            data = pd.read_csv(uploaded_file)
            return data
        elif file_extension.lower() in ['.xls', '.xlsx']:
            st.caption("est ce qu'on rentre la")
            # Load data from Excel
            data = pd.read_excel(uploaded_file)
            st.write(data)
            return data
        elif file_extension.lower() in ['.pq', '.parquet']:
            # Load data from Parquet
            data = pd.read_parquet(uploaded_file)
            return data
        else:
            st.caption(":x: :red[Unsupported file type. Please provide a CSV, Excel or Parquet file]")
            return None

    except Exception as e:
        st.caption("Error loading data")
        print(f"Error loading data: {e}")
        return None