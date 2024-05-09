from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
 
# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)

# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Features Explorer",
    layout="wide"
)

# Title
st.title("Features Explorer")

# You should cache your pygwalker renderer, if you don't want your memory to explode
@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    # Import Features data
    features_path = "app_data/features.xlsx"
    features = pd.read_excel(features_path)
    # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
    return StreamlitRenderer(features, spec="./gw_config.json", spec_io_mode="rw")

renderer = get_pyg_renderer()

renderer.explorer()