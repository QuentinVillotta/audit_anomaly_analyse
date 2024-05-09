from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
 
# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)
# Import Features data
features_path = "data/HSOS_RD09/model_input/features.xlsx"
features = pd.read_excel(features_path)

# Visualizer
pyg_app = StreamlitRenderer(features)
pyg_app.explorer()